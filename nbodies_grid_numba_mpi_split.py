import argparse
import time
import numpy as np
from mpi4py import MPI

from nbodies_grid_numba_parallel import NBodySystem


TAG_DT = 10
TAG_POS = 11
TAG_STOP = 12


def parse_args():
    parser = argparse.ArgumentParser(description="MPI split: rank0 display, rank1 compute")
    parser.add_argument("dataset", nargs="?", default="data/galaxy_1000")
    parser.add_argument("dt", nargs="?", type=float, default=0.001)
    parser.add_argument("ni", nargs="?", type=int, default=20)
    parser.add_argument("nj", nargs="?", type=int, default=20)
    parser.add_argument("nk", nargs="?", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true", help="Mesure sans affichage")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def worker_loop(comm: MPI.Comm, args):
    system = NBodySystem(args.dataset, (args.ni, args.nj, args.nk))

    while True:
        status = MPI.Status()
        payload = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.tag == TAG_STOP:
            break

        dt = float(payload)
        system.step(dt)
        comm.send(system.positions, dest=0, tag=TAG_POS)


def rank0_visual_loop(comm: MPI.Comm, args):
    import visualizer3d

    local_model = NBodySystem(args.dataset, (args.ni, args.nj, args.nk))
    intensity = np.clip(local_model.masses / local_model.max_mass, 0.5, 1.0)

    visu = visualizer3d.Visualizer3D(
        local_model.positions,
        local_model.colors,
        intensity,
        [
            [local_model.box[0, 0], local_model.box[1, 0]],
            [local_model.box[0, 1], local_model.box[1, 1]],
            [local_model.box[0, 2], local_model.box[1, 2]],
        ],
    )

    def updater(dt: float):
        comm.send(float(dt), dest=1, tag=TAG_DT)
        positions = comm.recv(source=1, tag=TAG_POS)
        return positions

    try:
        visu.run(updater=updater, dt=args.dt)
    finally:
        comm.send(None, dest=1, tag=TAG_STOP)


def rank0_benchmark_loop(comm: MPI.Comm, args):
    for _ in range(args.warmup):
        comm.send(float(args.dt), dest=1, tag=TAG_DT)
        _ = comm.recv(source=1, tag=TAG_POS)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        comm.send(float(args.dt), dest=1, tag=TAG_DT)
        _ = comm.recv(source=1, tag=TAG_POS)
    t1 = time.perf_counter()

    comm.send(None, dest=1, tag=TAG_STOP)

    elapsed = t1 - t0
    print(f"mpi_split ranks=2 steps={args.steps}")
    print(f"wall_total_s={elapsed:.6f}")
    print(f"avg_step_wall_s={elapsed / args.steps:.6f}")


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        if rank == 0:
            print("Erreur: lancer avec au moins 2 processus MPI.")
        return

    if rank == 1:
        worker_loop(comm, args)
        return

    if rank == 0:
        if args.benchmark:
            rank0_benchmark_loop(comm, args)
        else:
            rank0_visual_loop(comm, args)
        return

    # Rangs > 1 inutilises dans cette etape
    if rank > 1:
        pass


if __name__ == "__main__":
    main()
