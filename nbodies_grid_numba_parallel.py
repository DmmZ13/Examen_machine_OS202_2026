import argparse
import time
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

G = 1.560339e-13


def generate_star_color(mass: float) -> tuple[int, int, int]:
    if mass > 5.0:
        return (150, 180, 255)
    if mass > 2.0:
        return (255, 255, 255)
    if mass >= 1.0:
        return (255, 255, 200)
    return (255, 150, 100)


@njit(cache=True)
def update_stars_in_grid(
    cell_start_indices: np.ndarray,
    body_indices: np.ndarray,
    cell_masses: np.ndarray,
    cell_com_positions: np.ndarray,
    masses: np.ndarray,
    positions: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    n_bodies = positions.shape[0]
    n_total_cells = np.prod(n_cells)
    cell_start_indices.fill(-1)

    cell_counts = np.zeros(shape=(n_total_cells,), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        cell_counts[morse_idx] += 1

    running_index = 0
    for i in range(n_total_cells):
        cell_start_indices[i] = running_index
        running_index += cell_counts[i]
    cell_start_indices[n_total_cells] = running_index

    current_counts = np.zeros(shape=(n_total_cells,), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        index_in_cell = cell_start_indices[morse_idx] + current_counts[morse_idx]
        body_indices[index_in_cell] = ibody
        current_counts[morse_idx] += 1

    for i in range(n_total_cells):
        cell_mass = 0.0
        com_x = 0.0
        com_y = 0.0
        com_z = 0.0
        start_idx = cell_start_indices[i]
        end_idx = cell_start_indices[i + 1]
        for j in range(start_idx, end_idx):
            ibody = body_indices[j]
            m = masses[ibody]
            cell_mass += m
            com_x += positions[ibody, 0] * m
            com_y += positions[ibody, 1] * m
            com_z += positions[ibody, 2] * m
        if cell_mass > 0.0:
            inv_mass = 1.0 / cell_mass
            cell_com_positions[i, 0] = com_x * inv_mass
            cell_com_positions[i, 1] = com_y * inv_mass
            cell_com_positions[i, 2] = com_z * inv_mass
        else:
            cell_com_positions[i, 0] = 0.0
            cell_com_positions[i, 1] = 0.0
            cell_com_positions[i, 2] = 0.0
        cell_masses[i] = cell_mass


@njit(parallel=True, fastmath=True, cache=True)
def compute_acceleration(
    positions: np.ndarray,
    masses: np.ndarray,
    cell_start_indices: np.ndarray,
    body_indices: np.ndarray,
    cell_masses: np.ndarray,
    cell_com_positions: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    n_bodies = positions.shape[0]
    a = np.zeros_like(positions)

    for ibody in prange(n_bodies):
        pos_x = positions[ibody, 0]
        pos_y = positions[ibody, 1]
        pos_z = positions[ibody, 2]

        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0

        ax = 0.0
        ay = 0.0
        az = 0.0

        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]
                    if (
                        (abs(ix - cell_idx[0]) > 2)
                        or (abs(iy - cell_idx[1]) > 2)
                        or (abs(iz - cell_idx[2]) > 2)
                    ):
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            dx = cell_com_positions[morse_idx, 0] - pos_x
                            dy = cell_com_positions[morse_idx, 1] - pos_y
                            dz = cell_com_positions[morse_idx, 2] - pos_z
                            dist2 = dx * dx + dy * dy + dz * dz
                            if dist2 > 1.0e-20:
                                inv_dist = 1.0 / np.sqrt(dist2)
                                inv_dist3 = inv_dist * inv_dist * inv_dist
                                coeff = G * cell_mass * inv_dist3
                                ax += coeff * dx
                                ay += coeff * dy
                                az += coeff * dz
                    else:
                        start_idx = cell_start_indices[morse_idx]
                        end_idx = cell_start_indices[morse_idx + 1]
                        for j in range(start_idx, end_idx):
                            jbody = body_indices[j]
                            if jbody != ibody:
                                dx = positions[jbody, 0] - pos_x
                                dy = positions[jbody, 1] - pos_y
                                dz = positions[jbody, 2] - pos_z
                                dist2 = dx * dx + dy * dy + dz * dz
                                if dist2 > 1.0e-20:
                                    inv_dist = 1.0 / np.sqrt(dist2)
                                    inv_dist3 = inv_dist * inv_dist * inv_dist
                                    coeff = G * masses[jbody] * inv_dist3
                                    ax += coeff * dx
                                    ay += coeff * dy
                                    az += coeff * dz

        a[ibody, 0] = ax
        a[ibody, 1] = ay
        a[ibody, 2] = az

    return a


@njit(parallel=True, fastmath=True, cache=True)
def verlet_position_update(positions: np.ndarray, velocities: np.ndarray, acc: np.ndarray, dt: float):
    for i in prange(positions.shape[0]):
        positions[i, 0] += velocities[i, 0] * dt + 0.5 * acc[i, 0] * dt * dt
        positions[i, 1] += velocities[i, 1] * dt + 0.5 * acc[i, 1] * dt * dt
        positions[i, 2] += velocities[i, 2] * dt + 0.5 * acc[i, 2] * dt * dt


@njit(parallel=True, fastmath=True, cache=True)
def verlet_velocity_update(velocities: np.ndarray, acc: np.ndarray, acc_new: np.ndarray, dt: float):
    for i in prange(velocities.shape[0]):
        velocities[i, 0] += 0.5 * (acc[i, 0] + acc_new[i, 0]) * dt
        velocities[i, 1] += 0.5 * (acc[i, 1] + acc_new[i, 1]) * dt
        velocities[i, 2] += 0.5 * (acc[i, 2] + acc_new[i, 2]) * dt


class SpatialGrid:
    def __init__(self, positions: np.ndarray, nb_cells_per_dim: tuple[int, int, int]):
        self.min_bounds = np.min(positions, axis=0) - 1.0e-6
        self.max_bounds = np.max(positions, axis=0) + 1.0e-6
        self.n_cells = np.array(nb_cells_per_dim, dtype=np.int64)
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells

        self.cell_start_indices = np.full(np.prod(self.n_cells) + 1, -1, dtype=np.int64)
        self.body_indices = np.empty(shape=(positions.shape[0],), dtype=np.int64)
        self.cell_masses = np.zeros(shape=(np.prod(self.n_cells),), dtype=np.float32)
        self.cell_com_positions = np.zeros(shape=(np.prod(self.n_cells), 3), dtype=np.float32)

    def update(self, positions: np.ndarray, masses: np.ndarray):
        update_stars_in_grid(
            self.cell_start_indices,
            self.body_indices,
            self.cell_masses,
            self.cell_com_positions,
            masses,
            positions,
            self.min_bounds,
            self.cell_size,
            self.n_cells,
        )


class NBodySystem:
    def __init__(self, filename: str, ncells_per_dir: tuple[int, int, int] = (20, 20, 1)):
        positions = []
        velocities = []
        masses = []

        self.max_mass = 0.0
        self.box = np.array([[-1.0e-6, -1.0e-6, -1.0e-6], [1.0e-6, 1.0e-6, 1.0e-6]], dtype=np.float64)

        with open(filename, "r", encoding="utf-8") as fich:
            for line in fich:
                data = line.split()
                masses.append(float(data[0]))
                positions.append([float(data[1]), float(data[2]), float(data[3])])
                velocities.append([float(data[4]), float(data[5]), float(data[6])])
                self.max_mass = max(self.max_mass, masses[-1])
                for i in range(3):
                    self.box[0, i] = min(self.box[0, i], positions[-1][i] - 1.0e-6)
                    self.box[1, i] = max(self.box[1, i], positions[-1][i] + 1.0e-6)

        self.positions = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32)
        self.masses = np.array(masses, dtype=np.float32)
        self.colors = [generate_star_color(m) for m in masses]

        self.grid = SpatialGrid(self.positions, ncells_per_dir)
        self.grid.update(self.positions, self.masses)

    def step(self, dt: float) -> float:
        t0 = time.perf_counter()
        acc = compute_acceleration(
            self.positions,
            self.masses,
            self.grid.cell_start_indices,
            self.grid.body_indices,
            self.grid.cell_masses,
            self.grid.cell_com_positions,
            self.grid.min_bounds,
            self.grid.cell_size,
            self.grid.n_cells,
        )
        verlet_position_update(self.positions, self.velocities, acc, dt)
        self.grid.update(self.positions, self.masses)
        acc_new = compute_acceleration(
            self.positions,
            self.masses,
            self.grid.cell_start_indices,
            self.grid.body_indices,
            self.grid.cell_masses,
            self.grid.cell_com_positions,
            self.grid.min_bounds,
            self.grid.cell_size,
            self.grid.n_cells,
        )
        verlet_velocity_update(self.velocities, acc, acc_new, dt)
        t1 = time.perf_counter()
        return t1 - t0


def run_benchmark(filename: str, dt: float, ncells: tuple[int, int, int], steps: int, warmup: int):
    system = NBodySystem(filename, ncells)

    for _ in range(warmup):
        system.step(dt)

    start = time.perf_counter()
    calc_total = 0.0
    for _ in range(steps):
        calc_total += system.step(dt)
    end = time.perf_counter()

    wall = end - start
    print(f"threads={get_num_threads()} steps={steps} bodies={system.positions.shape[0]}")
    print(f"calc_total_s={calc_total:.6f}")
    print(f"wall_total_s={wall:.6f}")
    print(f"avg_step_calc_s={calc_total / steps:.6f}")
    print(f"avg_step_wall_s={wall / steps:.6f}")


def run_visual(filename: str, dt: float, ncells: tuple[int, int, int]):
    import visualizer3d

    system = NBodySystem(filename, ncells)
    intensity = np.clip(system.masses / system.max_mass, 0.5, 1.0)
    visu = visualizer3d.Visualizer3D(
        system.positions,
        system.colors,
        intensity,
        [[system.box[0, 0], system.box[1, 0]], [system.box[0, 1], system.box[1, 1]], [system.box[0, 2], system.box[1, 2]]],
    )

    def updater(local_dt: float):
        system.step(local_dt)
        return system.positions

    visu.run(updater=updater, dt=dt)


def parse_args():
    parser = argparse.ArgumentParser(description="N-Body grille + Numba parallel")
    parser.add_argument("dataset", nargs="?", default="data/galaxy_1000")
    parser.add_argument("dt", nargs="?", type=float, default=0.001)
    parser.add_argument("ni", nargs="?", type=int, default=20)
    parser.add_argument("nj", nargs="?", type=int, default=20)
    parser.add_argument("nk", nargs="?", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None, help="Fixe NUMBA_NUM_THREADS")
    parser.add_argument("--benchmark", action="store_true", help="Mode sans affichage")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.threads is not None:
        set_num_threads(args.threads)

    ncells = (args.ni, args.nj, args.nk)
    if args.benchmark:
        run_benchmark(args.dataset, args.dt, ncells, args.steps, args.warmup)
    else:
        run_visual(args.dataset, args.dt, ncells)


if __name__ == "__main__":
    main()
