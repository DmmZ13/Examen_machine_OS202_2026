import argparse
import time
import numpy as np
from mpi4py import MPI
from numba import njit, prange, set_num_threads, get_num_threads

from nbodies_grid_numba_parallel import (
    SpatialGrid,
    compute_acceleration,
    verlet_position_update,
    verlet_velocity_update,
    generate_star_color,
)


def load_system(filename: str):
    positions = []
    velocities = []
    masses = []
    max_mass = 0.0
    box = np.array([[-1.0e-6, -1.0e-6, -1.0e-6], [1.0e-6, 1.0e-6, 1.0e-6]], dtype=np.float64)

    with open(filename, "r", encoding="utf-8") as fich:
        for line in fich:
            data = line.split()
            m = float(data[0])
            p = [float(data[1]), float(data[2]), float(data[3])]
            v = [float(data[4]), float(data[5]), float(data[6])]
            masses.append(m)
            positions.append(p)
            velocities.append(v)
            max_mass = max(max_mass, m)
            for i in range(3):
                box[0, i] = min(box[0, i], p[i] - 1.0e-6)
                box[1, i] = max(box[1, i], p[i] + 1.0e-6)

    return (
        np.array(positions, dtype=np.float32),
        np.array(velocities, dtype=np.float32),
        np.array(masses, dtype=np.float32),
        max_mass,
        box,
    )


@njit(parallel=True, fastmath=True, cache=True)
def select_rows(src: np.ndarray, start: int, end: int) -> np.ndarray:
    n = end - start
    out = np.empty((n, src.shape[1]), dtype=src.dtype)
    for i in prange(n):
        out[i, 0] = src[start + i, 0]
        out[i, 1] = src[start + i, 1]
        out[i, 2] = src[start + i, 2]
    return out


@njit(parallel=True, fastmath=True, cache=True)
def place_rows(dst: np.ndarray, start: int, src: np.ndarray):
    for i in prange(src.shape[0]):
        dst[start + i, 0] = src[i, 0]
        dst[start + i, 1] = src[i, 1]
        dst[start + i, 2] = src[i, 2]


def split_range(n: int, rank: int, size: int):
    base = n // size
    rem = n % size
    start = rank * base + min(rank, rem)
    count = base + (1 if rank < rem else 0)
    end = start + count
    return start, end


def gather_full_array(comm: MPI.Comm, local_block: np.ndarray, n_rows: int) -> np.ndarray:
    size = comm.Get_size()
    rank = comm.Get_rank()

    counts_rows = np.array([split_range(n_rows, r, size)[1] - split_range(n_rows, r, size)[0] for r in range(size)], dtype=np.int32)
    counts = counts_rows * local_block.shape[1]
    displs = np.zeros(size, dtype=np.int32)
    displs[1:] = np.cumsum(counts[:-1])

    recv = np.empty((n_rows, local_block.shape[1]), dtype=local_block.dtype)
    comm.Allgatherv(local_block.ravel(), [recv.ravel(), counts, displs, MPI.FLOAT])
    return recv


def parse_args():
    parser = argparse.ArgumentParser(description="MPI + Numba parallel body updates")
    parser.add_argument("dataset", nargs="?", default="data/galaxy_1000")
    parser.add_argument("dt", nargs="?", type=float, default=0.001)
    parser.add_argument("ni", nargs="?", type=int, default=20)
    parser.add_argument("nj", nargs="?", type=int, default=20)
    parser.add_argument("nk", nargs="?", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--visual", action="store_true", help="Affiche sur rank 0")
    return parser.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.threads is not None:
        set_num_threads(args.threads)

    positions, velocities, masses, max_mass, box = load_system(args.dataset)
    n_bodies = positions.shape[0]

    start, end = split_range(n_bodies, rank, size)

    grid = SpatialGrid(positions, (args.ni, args.nj, args.nk))
    grid.update(positions, masses)

    def one_step(dt: float):
        nonlocal positions, velocities

        acc_global = compute_acceleration(
            positions,
            masses,
            grid.cell_start_indices,
            grid.body_indices,
            grid.cell_masses,
            grid.cell_com_positions,
            grid.min_bounds,
            grid.cell_size,
            grid.n_cells,
        )
        acc_local = select_rows(acc_global, start, end)

        pos_local = select_rows(positions, start, end)
        vel_local = select_rows(velocities, start, end)
        verlet_position_update(pos_local, vel_local, acc_local, dt)

        positions = gather_full_array(comm, pos_local, n_bodies)

        grid.update(positions, masses)
        acc_global_new = compute_acceleration(
            positions,
            masses,
            grid.cell_start_indices,
            grid.body_indices,
            grid.cell_masses,
            grid.cell_com_positions,
            grid.min_bounds,
            grid.cell_size,
            grid.n_cells,
        )
        acc_local_new = select_rows(acc_global_new, start, end)

        verlet_velocity_update(vel_local, acc_local, acc_local_new, dt)
        velocities = gather_full_array(comm, vel_local, n_bodies)

    for _ in range(args.warmup):
        one_step(args.dt)

    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        one_step(args.dt)
    comm.Barrier()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    max_elapsed = comm.reduce(elapsed, op=MPI.MAX, root=0)

    if rank == 0:
        print(f"mpi_parallel ranks={size} threads={get_num_threads()} steps={args.steps}")
        print(f"wall_total_s={max_elapsed:.6f}")
        print(f"avg_step_wall_s={max_elapsed / args.steps:.6f}")

    if args.visual and rank == 0:
        import visualizer3d

        colors = [generate_star_color(float(m)) for m in masses]
        intensity = np.clip(masses / max_mass, 0.5, 1.0)
        visu = visualizer3d.Visualizer3D(
            positions,
            colors,
            intensity,
            [[box[0, 0], box[1, 0]], [box[0, 1], box[1, 1]], [box[0, 2], box[1, 2]]],
        )

        current_positions = positions.copy()

        def updater(dt: float):
            nonlocal current_positions
            one_step(dt)
            current_positions = positions
            return current_positions

        visu.run(updater=updater, dt=args.dt)


if __name__ == "__main__":
    main()
