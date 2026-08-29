#!/usr/bin/env python3
"""Heat-geodesic distance matrix followed by Isomap classical MDS.

The heat and Poisson matrices are factorized once and reused for every sampled
surface source. MDS uses CuPy when CUDA is available.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

RUNTIME_CACHE = Path(__file__).resolve().parents[2] / ".runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("CUPY_CACHE_DIR", str(RUNTIME_CACHE / "cupy"))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

from Geodesics import build_cotangent_laplacian_and_mass, mean_edge_length
from Isomap import (
    classical_mds_cpu,
    classical_mds_gpu,
    cut_two_boundary_loft,
    gpu_is_available,
    load_stl,
    ordered_boundary_loops,
)


EPS = 1.0e-12


@dataclass
class SurfaceSamples:
    points: np.ndarray
    face_ids: np.ndarray
    barycentric: np.ndarray


@dataclass
class HeatSystem:
    mesh: trimesh.Trimesh
    faces: np.ndarray
    areas: np.ndarray
    basis_gradients: np.ndarray
    heat_solve: object
    poisson_solve: object
    poisson_mask: np.ndarray
    time_step: float


def select_mesh_file() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("tkinter is unavailable") from exc
    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title="Select STL mesh for Heat-Geodesic Isomap",
        filetypes=[("STL models", "*.stl"), ("Mesh files", "*.stl *.obj *.ply")],
    )
    root.destroy()
    return Path(selected) if selected else None


def area_uniform_samples(
    mesh: trimesh.Trimesh,
    count: int,
    rng: np.random.Generator,
) -> SurfaceSamples:
    count = max(3, int(count))
    areas = np.asarray(mesh.area_faces, dtype=float)
    valid = np.isfinite(areas) & (areas > EPS)
    face_pool = np.flatnonzero(valid)
    valid_areas = areas[valid]
    if float(valid_areas.sum()) <= EPS:
        raise ValueError("Mesh has no non-degenerate faces")
    face_ids = rng.choice(face_pool, size=count, p=valid_areas / valid_areas.sum())

    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    barycentric = np.column_stack([
        1.0 - r1,
        r1 * (1.0 - r2),
        r1 * r2,
    ])
    triangles = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces)[face_ids]]
    points = np.einsum("ni,nij->nj", barycentric, triangles)
    return SurfaceSamples(points, face_ids.astype(np.int64), barycentric)


def triangle_basis_gradients(mesh: trimesh.Trimesh):
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_areas = np.linalg.norm(normals, axis=1)
    valid = double_areas > EPS
    unit_normals = np.zeros_like(normals)
    unit_normals[valid] = normals[valid] / double_areas[valid, None]
    areas = 0.5 * double_areas
    denominator = np.maximum(double_areas, EPS)[:, None]
    basis = np.empty((len(faces), 3, 3), dtype=float)
    basis[:, 0] = np.cross(unit_normals, triangles[:, 2] - triangles[:, 1]) / denominator
    basis[:, 1] = np.cross(unit_normals, triangles[:, 0] - triangles[:, 2]) / denominator
    basis[:, 2] = np.cross(unit_normals, triangles[:, 1] - triangles[:, 0]) / denominator
    basis[~valid] = 0.0
    return faces, areas, basis


def prepare_heat_system(mesh: trimesh.Trimesh, time_scale: float) -> HeatSystem:
    stiffness, mass = build_cotangent_laplacian_and_mass(mesh)
    stiffness = stiffness.tocsr()
    h = mean_edge_length(mesh)
    time_step = max(float(time_scale), EPS) * h * h
    heat_matrix = (diags(mass) + time_step * stiffness).tocsc()
    heat_solve = factorized(heat_matrix)

    # One fixed anchor removes the Poisson null space. Every solution is later
    # shifted at its own source, so the chosen anchor has no geometric meaning.
    poisson_mask = np.ones(len(mesh.vertices), dtype=bool)
    poisson_mask[0] = False
    poisson_matrix = stiffness[poisson_mask][:, poisson_mask].tocsc()
    poisson_solve = factorized(poisson_matrix)
    faces, areas, basis = triangle_basis_gradients(mesh)
    return HeatSystem(
        mesh=mesh,
        faces=faces,
        areas=areas,
        basis_gradients=basis,
        heat_solve=heat_solve,
        poisson_solve=poisson_solve,
        poisson_mask=poisson_mask,
        time_step=time_step,
    )


def one_heat_distance_field(
    system: HeatSystem,
    source_face: int,
    source_barycentric: np.ndarray,
) -> np.ndarray:
    vertex_count = len(system.mesh.vertices)
    source_vertices = system.faces[int(source_face)]
    heat_rhs = np.zeros(vertex_count, dtype=float)
    np.add.at(heat_rhs, source_vertices, source_barycentric)
    heat = np.asarray(system.heat_solve(heat_rhs), dtype=float)

    gradients = np.einsum(
        "fi,fij->fj", heat[system.faces], system.basis_gradients
    )
    norms = np.linalg.norm(gradients, axis=1)
    vector_field = np.zeros_like(gradients)
    valid = norms > EPS
    vector_field[valid] = -gradients[valid] / norms[valid, None]

    divergence = np.zeros(vertex_count, dtype=float)
    for local in range(3):
        contribution = -system.areas * np.einsum(
            "fi,fi->f", vector_field, system.basis_gradients[:, local]
        )
        np.add.at(divergence, system.faces[:, local], contribution)

    phi = np.zeros(vertex_count, dtype=float)
    phi[system.poisson_mask] = system.poisson_solve(divergence[system.poisson_mask])
    source_value = float(source_barycentric @ phi[source_vertices])
    phi -= source_value
    if float(np.mean(phi)) < 0.0:
        phi *= -1.0
    return np.maximum(phi, 0.0)


def heat_geodesic_distance_matrix(
    system: HeatSystem,
    samples: SurfaceSamples,
) -> np.ndarray:
    count = len(samples.points)
    matrix = np.empty((count, count), dtype=float)
    destination_faces = system.faces[samples.face_ids]
    for source in range(count):
        field = one_heat_distance_field(
            system,
            int(samples.face_ids[source]),
            samples.barycentric[source],
        )
        values = np.einsum(
            "ni,ni->n", samples.barycentric, field[destination_faces]
        )
        values[source] = 0.0
        matrix[source] = values
        print(f"  heat sources: {source + 1}/{count}", end="\r", flush=True)
    print()

    # Discretization and interpolation make independently evaluated heat
    # distances slightly asymmetric; classical MDS requires a symmetric matrix.
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    if not np.isfinite(matrix).all():
        raise ValueError("Heat-geodesic distance matrix contains non-finite values")
    return matrix


def save_csv(path: str | Path, samples: SurfaceSamples, uv: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "face_index", "u", "v", "x", "y", "z"])
        for index, (face, coord, point) in enumerate(
            zip(samples.face_ids, uv, samples.points)
        ):
            writer.writerow([index, int(face), coord[0], coord[1], *point])


def visualize(samples: SurfaceSamples, uv: np.ndarray, title: str) -> None:
    color = uv[:, 0]
    figure = plt.figure(figsize=(12, 5))
    ax3d = figure.add_subplot(1, 2, 1, projection="3d")
    ax3d.scatter(*samples.points.T, c=color, cmap="viridis", s=4)
    ax3d.set_title("Area-uniform surface samples")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax2d = figure.add_subplot(1, 2, 2)
    ax2d.scatter(uv[:, 0], uv[:, 1], c=color, cmap="viridis", s=4)
    ax2d.set_title("Heat-Geodesic Isomap")
    ax2d.set_xlabel("u")
    ax2d.set_ylabel("v")
    ax2d.set_aspect("equal", adjustable="datalim")
    figure.suptitle(title)
    figure.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", nargs="?", default="", help="Input STL/OBJ/PLY")
    parser.add_argument("--points", type=int, default=1000, help="Area-uniform sample count")
    parser.add_argument("--time-scale", type=float, default=1.0, help="Heat time multiplier")
    parser.add_argument("--seed", type=int, default=0, help="Sampling random seed")
    parser.add_argument("--cpu-mds", action="store_true", help="Force CPU classical MDS")
    parser.add_argument("--out", default="geodesics_isomap.csv", help="Output CSV")
    parser.add_argument("--no-show", action="store_true", help="Do not show visualization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = Path(args.mesh) if args.mesh else select_mesh_file()
    if mesh_path is None:
        raise SystemExit("No mesh selected")
    mesh = load_stl(mesh_path)
    boundary_count = len(ordered_boundary_loops(np.asarray(mesh.faces, dtype=np.int64)))
    print(f"Boundary loops before preparation: {boundary_count}")
    if boundary_count == 2:
        mesh, seam = cut_two_boundary_loft(mesh)
        print(f"Two-boundary loft cut: {len(seam)} seam vertices")
        print(
            "Boundary loops after cutting: "
            f"{len(ordered_boundary_loops(np.asarray(mesh.faces, dtype=np.int64)))}"
        )

    samples = area_uniform_samples(mesh, args.points, np.random.default_rng(args.seed))
    print(f"Mesh vertices / faces: {len(mesh.vertices)} / {len(mesh.faces)}")
    print(f"Area-uniform samples: {len(samples.points)}")
    start = perf_counter()
    system = prepare_heat_system(mesh, args.time_scale)
    print(f"Shared heat/Poisson factorization: {perf_counter() - start:.3f} s")
    print(f"Heat time step: {system.time_step:.6g}")
    start = perf_counter()
    distances = heat_geodesic_distance_matrix(system, samples)
    print(f"Geodesic distance matrix: {perf_counter() - start:.3f} s")

    available, device = gpu_is_available()
    use_gpu = available and not args.cpu_mds
    print(f"MDS backend: {'GPU (' + device + ')' if use_gpu else 'CPU'}")
    start = perf_counter()
    uv = classical_mds_gpu(distances) if use_gpu else classical_mds_cpu(distances)
    print(f"Classical MDS: {perf_counter() - start:.3f} s")
    save_csv(args.out, samples, uv)
    print(f"Saved: {Path(args.out).resolve()}")
    if not args.no_show:
        visualize(samples, uv, mesh_path.name)


if __name__ == "__main__":
    main()
