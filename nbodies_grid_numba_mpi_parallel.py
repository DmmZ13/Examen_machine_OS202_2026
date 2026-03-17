import argparse
import time
import numpy as np
from mpi4py import MPI
from numba import njit, prange, set_num_threads, get_num_threads

from nbodies_grid_numba_parallel import (
    update_stars_in_grid,
    verlet_position_update,
    verlet_velocity_update,
    generate_star_color,
)


G = 1.560339e-13
GHOST_WIDTH = 2
STAR_BLOCK_WIDTH = 7


def load_system(filename: str):
    positions = []
    velocities = []
    masses = []
    max_mass = 0.0
    box = np.array([[-1.0e-6, -1.0e-6, -1.0e-6], [1.0e-6, 1.0e-6, 1.0e-6]], dtype=np.float64)

    with open(filename, "r", encoding="utf-8") as fich:
        for line in fich:
            data = line.split()
            mass = float(data[0])
            position = [float(data[1]), float(data[2]), float(data[3])]
            velocity = [float(data[4]), float(data[5]), float(data[6])]
            masses.append(mass)
            positions.append(position)
            velocities.append(velocity)
            max_mass = max(max_mass, mass)
            for i in range(3):
                box[0, i] = min(box[0, i], position[i] - 1.0e-6)
                box[1, i] = max(box[1, i], position[i] + 1.0e-6)

    return (
        np.array(positions, dtype=np.float32),
        np.array(velocities, dtype=np.float32),
        np.array(masses, dtype=np.float32),
        max_mass,
        box,
    )


def split_range(n: int, rank: int, size: int):
    base = n // size
    rem = n % size
    start = rank * base + min(rank, rem)
    count = base + (1 if rank < rem else 0)
    end = start + count
    return start, end


def build_cell_owner_map(n_i: int, size: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    owner = np.empty(n_i, dtype=np.int32)
    ranges: list[tuple[int, int]] = []
    for rank in range(size):
        start, end = split_range(n_i, rank, size)
        ranges.append((start, end))
        if end > start:
            owner[start:end] = rank
    return owner, ranges


def empty_star_block() -> np.ndarray:
    return np.empty((0, STAR_BLOCK_WIDTH), dtype=np.float32)


def pack_star_block(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    n_stars = positions.shape[0]
    if n_stars == 0:
        return empty_star_block()

    block = np.empty((n_stars, STAR_BLOCK_WIDTH), dtype=np.float32)
    block[:, 0:3] = positions
    block[:, 3:6] = velocities
    block[:, 6] = masses
    return block


def unpack_star_block(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if block.size == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return block[:, 0:3].copy(), block[:, 3:6].copy(), block[:, 6].copy()


def concat_star_blocks(blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    non_empty = [block for block in blocks if block.size > 0]
    if not non_empty:
        return unpack_star_block(empty_star_block())
    return unpack_star_block(np.concatenate(non_empty, axis=0))


def compute_cell_indices(positions: np.ndarray, grid_min: np.ndarray, cell_size: np.ndarray, n_cells: np.ndarray) -> np.ndarray:
    if positions.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)

    cell_idx = np.floor((positions - grid_min) / cell_size).astype(np.int64)
    for axis in range(3):
        np.clip(cell_idx[:, axis], 0, int(n_cells[axis]) - 1, out=cell_idx[:, axis])
    return cell_idx


@njit(cache=True)
def accumulate_cell_moments(
    masses: np.ndarray,
    positions: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    n_total_cells = np.prod(n_cells)
    cell_masses = np.zeros(n_total_cells, dtype=np.float32)
    cell_moments = np.zeros((n_total_cells, 3), dtype=np.float32)

    for ibody in range(positions.shape[0]):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0

        morse_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        mass = masses[ibody]
        cell_masses[morse_idx] += mass
        cell_moments[morse_idx, 0] += positions[ibody, 0] * mass
        cell_moments[morse_idx, 1] += positions[ibody, 1] * mass
        cell_moments[morse_idx, 2] += positions[ibody, 2] * mass

    return cell_masses, cell_moments


@njit(parallel=True, fastmath=True, cache=True)
def compute_acceleration_local(
    owned_positions: np.ndarray,
    local_positions: np.ndarray,
    local_masses: np.ndarray,
    local_cell_start_indices: np.ndarray,
    local_body_indices: np.ndarray,
    global_cell_masses: np.ndarray,
    global_cell_com_positions: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    n_owned = owned_positions.shape[0]
    accelerations = np.zeros_like(owned_positions)

    for ibody in prange(n_owned):
        pos_x = owned_positions[ibody, 0]
        pos_y = owned_positions[ibody, 1]
        pos_z = owned_positions[ibody, 2]

        cell_idx = np.floor((owned_positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0

        ax = 0.0
        ay = 0.0
        az = 0.0

        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]
                    is_far = (
                        (abs(ix - cell_idx[0]) > GHOST_WIDTH)
                        or (abs(iy - cell_idx[1]) > GHOST_WIDTH)
                        or (abs(iz - cell_idx[2]) > GHOST_WIDTH)
                    )

                    if is_far:
                        cell_mass = global_cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            dx = global_cell_com_positions[morse_idx, 0] - pos_x
                            dy = global_cell_com_positions[morse_idx, 1] - pos_y
                            dz = global_cell_com_positions[morse_idx, 2] - pos_z
                            dist2 = dx * dx + dy * dy + dz * dz
                            if dist2 > 1.0e-20:
                                inv_dist = 1.0 / np.sqrt(dist2)
                                inv_dist3 = inv_dist * inv_dist * inv_dist
                                coeff = G * cell_mass * inv_dist3
                                ax += coeff * dx
                                ay += coeff * dy
                                az += coeff * dz
                    else:
                        start_idx = local_cell_start_indices[morse_idx]
                        end_idx = local_cell_start_indices[morse_idx + 1]
                        for j in range(start_idx, end_idx):
                            jbody = local_body_indices[j]
                            if jbody != ibody:
                                dx = local_positions[jbody, 0] - pos_x
                                dy = local_positions[jbody, 1] - pos_y
                                dz = local_positions[jbody, 2] - pos_z
                                dist2 = dx * dx + dy * dy + dz * dz
                                if dist2 > 1.0e-20:
                                    inv_dist = 1.0 / np.sqrt(dist2)
                                    inv_dist3 = inv_dist * inv_dist * inv_dist
                                    coeff = G * local_masses[jbody] * inv_dist3
                                    ax += coeff * dx
                                    ay += coeff * dy
                                    az += coeff * dz

        accelerations[ibody, 0] = ax
        accelerations[ibody, 1] = ay
        accelerations[ibody, 2] = az

    return accelerations


def build_global_cell_summary(
    comm: MPI.Comm,
    owned_positions: np.ndarray,
    owned_masses: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_masses, local_moments = accumulate_cell_moments(owned_masses, owned_positions, grid_min, cell_size, n_cells)
    global_masses = np.empty_like(local_masses)
    global_moments = np.empty_like(local_moments)

    comm.Allreduce(local_masses, global_masses, op=MPI.SUM)
    comm.Allreduce(local_moments, global_moments, op=MPI.SUM)

    global_com = np.zeros_like(global_moments)
    non_zero = global_masses > 0.0
    global_com[non_zero] = global_moments[non_zero] / global_masses[non_zero][:, None]
    return global_masses, global_com


def build_local_interaction_grid(
    local_positions: np.ndarray,
    local_masses: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_total_cells = int(np.prod(n_cells))
    cell_start_indices = np.full(n_total_cells + 1, -1, dtype=np.int64)
    body_indices = np.empty((local_positions.shape[0],), dtype=np.int64)
    dummy_masses = np.zeros((n_total_cells,), dtype=np.float32)
    dummy_com = np.zeros((n_total_cells, 3), dtype=np.float32)

    update_stars_in_grid(
        cell_start_indices,
        body_indices,
        dummy_masses,
        dummy_com,
        local_masses,
        local_positions,
        grid_min,
        cell_size,
        n_cells,
    )
    return cell_start_indices, body_indices


def exchange_ghost_stars(
    comm: MPI.Comm,
    rank: int,
    owned_positions: np.ndarray,
    owned_velocities: np.ndarray,
    owned_masses: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
    owner_ranges: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = comm.Get_size()
    send_lists: list[list[np.ndarray]] = [[] for _ in range(size)]
    cell_indices = compute_cell_indices(owned_positions, grid_min, cell_size, n_cells)

    for i_star in range(owned_positions.shape[0]):
        cell_i = int(cell_indices[i_star, 0])
        ghost_i_min = max(0, cell_i - GHOST_WIDTH)
        ghost_i_max = min(int(n_cells[0]) - 1, cell_i + GHOST_WIDTH)

        star_record = np.empty((STAR_BLOCK_WIDTH,), dtype=np.float32)
        star_record[0:3] = owned_positions[i_star]
        star_record[3:6] = owned_velocities[i_star]
        star_record[6] = owned_masses[i_star]

        for dest_rank, (i_start, i_end) in enumerate(owner_ranges):
            if dest_rank == rank or i_end <= i_start:
                continue
            if i_start <= ghost_i_max and (i_end - 1) >= ghost_i_min:
                send_lists[dest_rank].append(star_record.copy())

    send_buffers = [empty_star_block() for _ in range(size)]
    for dest_rank in range(size):
        if send_lists[dest_rank]:
            send_buffers[dest_rank] = np.vstack(send_lists[dest_rank]).astype(np.float32, copy=False)

    recv_buffers = comm.alltoall(send_buffers)
    return concat_star_blocks(recv_buffers)


def migrate_owned_stars(
    comm: MPI.Comm,
    rank: int,
    owned_positions: np.ndarray,
    owned_velocities: np.ndarray,
    owned_masses: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
    cell_owner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = comm.Get_size()
    send_lists: list[list[np.ndarray]] = [[] for _ in range(size)]
    keep_mask = np.ones(owned_positions.shape[0], dtype=bool)

    if owned_positions.shape[0] > 0:
        cell_indices = compute_cell_indices(owned_positions, grid_min, cell_size, n_cells)
        destination_ranks = cell_owner[cell_indices[:, 0]]

        for i_star, dest_rank in enumerate(destination_ranks):
            if int(dest_rank) == rank:
                continue
            keep_mask[i_star] = False
            star_record = np.empty((STAR_BLOCK_WIDTH,), dtype=np.float32)
            star_record[0:3] = owned_positions[i_star]
            star_record[3:6] = owned_velocities[i_star]
            star_record[6] = owned_masses[i_star]
            send_lists[int(dest_rank)].append(star_record)

    send_buffers = [empty_star_block() for _ in range(size)]
    for dest_rank in range(size):
        if send_lists[dest_rank]:
            send_buffers[dest_rank] = np.vstack(send_lists[dest_rank]).astype(np.float32, copy=False)

    recv_buffers = comm.alltoall(send_buffers)
    local_block = pack_star_block(owned_positions[keep_mask], owned_velocities[keep_mask], owned_masses[keep_mask])
    return concat_star_blocks([local_block] + recv_buffers)


def gather_visual_state(
    comm: MPI.Comm,
    owned_positions: np.ndarray,
    owned_masses: np.ndarray,
    max_mass: float,
):
    if owned_positions.shape[0] == 0:
        local_block = np.empty((0, 4), dtype=np.float32)
    else:
        local_block = np.empty((owned_positions.shape[0], 4), dtype=np.float32)
        local_block[:, 0:3] = owned_positions
        local_block[:, 3] = owned_masses

    gathered = comm.gather(local_block, root=0)
    if comm.Get_rank() != 0:
        return None

    non_empty = [block for block in gathered if block.size > 0]
    if not non_empty:
        return np.empty((0, 3), dtype=np.float32), [], np.empty((0,), dtype=np.float32)

    merged = np.concatenate(non_empty, axis=0)
    positions = merged[:, 0:3].copy()
    masses = merged[:, 3].copy()
    colors = [generate_star_color(float(mass)) for mass in masses]
    intensity = np.clip(masses / max_mass, 0.5, 1.0)
    return positions, colors, intensity


def parse_args():
    parser = argparse.ArgumentParser(description="MPI + Numba parallel body updates with ghost cells")
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

    if size > args.ni:
        if rank == 0:
            print("Erreur: le nombre de processus MPI ne doit pas depasser Ni pour cette decomposition en slabs.")
        return

    if args.threads is not None:
        set_num_threads(args.threads)

    positions_all, velocities_all, masses_all, max_mass, box = load_system(args.dataset)
    n_cells = np.array((args.ni, args.nj, args.nk), dtype=np.int64)
    grid_min = np.min(positions_all, axis=0) - 1.0e-6
    grid_max = np.max(positions_all, axis=0) + 1.0e-6
    cell_size = (grid_max - grid_min) / n_cells

    cell_owner, owner_ranges = build_cell_owner_map(args.ni, size)
    initial_cells = compute_cell_indices(positions_all, grid_min, cell_size, n_cells)
    initial_mask = cell_owner[initial_cells[:, 0]] == rank

    owned_positions = positions_all[initial_mask].copy()
    owned_velocities = velocities_all[initial_mask].copy()
    owned_masses = masses_all[initial_mask].copy()

    def one_step(dt: float):
        nonlocal owned_positions, owned_velocities, owned_masses

        global_cell_masses, global_cell_com_positions = build_global_cell_summary(
            comm, owned_positions, owned_masses, grid_min, cell_size, n_cells
        )

        ghost_positions, ghost_velocities, ghost_masses = exchange_ghost_stars(
            comm,
            rank,
            owned_positions,
            owned_velocities,
            owned_masses,
            grid_min,
            cell_size,
            n_cells,
            owner_ranges,
        )

        local_positions, _, local_masses = concat_star_blocks(
            [pack_star_block(owned_positions, owned_velocities, owned_masses), pack_star_block(ghost_positions, ghost_velocities, ghost_masses)]
        )
        local_cell_start_indices, local_body_indices = build_local_interaction_grid(
            local_positions, local_masses, grid_min, cell_size, n_cells
        )

        acc = compute_acceleration_local(
            owned_positions,
            local_positions,
            local_masses,
            local_cell_start_indices,
            local_body_indices,
            global_cell_masses,
            global_cell_com_positions,
            grid_min,
            cell_size,
            n_cells,
        )

        verlet_position_update(owned_positions, owned_velocities, acc, dt)

        global_cell_masses_new, global_cell_com_positions_new = build_global_cell_summary(
            comm, owned_positions, owned_masses, grid_min, cell_size, n_cells
        )

        ghost_positions_new, ghost_velocities_new, ghost_masses_new = exchange_ghost_stars(
            comm,
            rank,
            owned_positions,
            owned_velocities,
            owned_masses,
            grid_min,
            cell_size,
            n_cells,
            owner_ranges,
        )

        local_positions_new, _, local_masses_new = concat_star_blocks(
            [
                pack_star_block(owned_positions, owned_velocities, owned_masses),
                pack_star_block(ghost_positions_new, ghost_velocities_new, ghost_masses_new),
            ]
        )
        local_cell_start_indices_new, local_body_indices_new = build_local_interaction_grid(
            local_positions_new, local_masses_new, grid_min, cell_size, n_cells
        )

        acc_new = compute_acceleration_local(
            owned_positions,
            local_positions_new,
            local_masses_new,
            local_cell_start_indices_new,
            local_body_indices_new,
            global_cell_masses_new,
            global_cell_com_positions_new,
            grid_min,
            cell_size,
            n_cells,
        )

        verlet_velocity_update(owned_velocities, acc, acc_new, dt)
        owned_positions, owned_velocities, owned_masses = migrate_owned_stars(
            comm,
            rank,
            owned_positions,
            owned_velocities,
            owned_masses,
            grid_min,
            cell_size,
            n_cells,
            cell_owner,
        )

    for _ in range(args.warmup):
        one_step(args.dt)

    if args.visual:
        if rank == 0:
            import visualizer3d

            positions_vis, colors_vis, intensity_vis = gather_visual_state(comm, owned_positions, owned_masses, max_mass)
            visu = visualizer3d.Visualizer3D(
                positions_vis,
                colors_vis,
                intensity_vis,
                [[box[0, 0], box[1, 0]], [box[0, 1], box[1, 1]], [box[0, 2], box[1, 2]]],
            )
            visu._render()
        else:
            gather_visual_state(comm, owned_positions, owned_masses, max_mass)

        while True:
            if rank == 0:
                running = visu._handle_events()
            else:
                running = None
            running = comm.bcast(running, root=0)
            if not running:
                break

            one_step(args.dt)
            state = gather_visual_state(comm, owned_positions, owned_masses, max_mass)
            if rank == 0:
                positions_vis, colors_vis, intensity_vis = state
                visu.update_points(positions_vis, colors_vis, intensity_vis)
                visu._render()

        if rank == 0:
            visu.cleanup()
        return

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


if __name__ == "__main__":
    main()
