#!/usr/bin/env python3
"""Generate a saddle surface, sample a point cloud, and unfold it with Isomap."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from Isomap import (
    estimate_boundary_minimum_gradient_directions,
    point_cloud_boundary_mask,
    run_isomap,
    sample_surface_uniform_density,
    save_boundary_directions,
)


def create_saddle_mesh(
    extent: float = 2.5,
    height: float = 1.0,
    resolution: int = 81,
) -> trimesh.Trimesh:
    """Create z = height * (x^2 - y^2) / extent^2 on a square domain."""
    if extent <= 0.0:
        raise ValueError("extent must be greater than zero")
    if resolution < 3:
        raise ValueError("resolution must be at least 3")

    axis = np.linspace(-extent, extent, resolution)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    z = height * (x * x - y * y) / (extent * extent)
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    faces: list[tuple[int, int, int]] = []
    for row in range(resolution - 1):
        for col in range(resolution - 1):
            lower_left = row * resolution + col
            lower_right = lower_left + 1
            upper_left = lower_left + resolution
            upper_right = upper_left + 1
            # Alternate diagonals to avoid a directional triangulation bias.
            if (row + col) % 2 == 0:
                faces.append((lower_left, lower_right, upper_right))
                faces.append((lower_left, upper_right, upper_left))
            else:
                faces.append((lower_left, lower_right, upper_left))
                faces.append((lower_right, upper_right, upper_left))

    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def sample_uniform_point_cloud(
    mesh: trimesh.Trimesh,
    point_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-uniform Poisson-like sampling of the triangle surface."""
    if point_count < 3:
        raise ValueError("point_count must be at least 3")
    rng = np.random.default_rng(seed)
    density = point_count / float(mesh.area)
    return sample_surface_uniform_density(
        mesh,
        density=density,
        rng=rng,
        max_points=point_count,
        boundary_multiplier=0.0,
    )


def save_point_cloud(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_index", "x", "y", "z"])
        for index, point in enumerate(points):
            writer.writerow([index, *point])


def save_embedding(path: Path, points: np.ndarray, uv: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_index", "u", "v", "x", "y", "z"])
        for index, (point, coord) in enumerate(zip(points, uv)):
            writer.writerow([index, *coord, *point])


def plot_demo(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    uv: np.ndarray,
    boundary_mask: np.ndarray,
    directions: np.ndarray,
    arrow_count: int,
    figure_path: Path,
    show: bool,
) -> None:
    color = points[:, 0]
    figure = plt.figure(figsize=(20, 5.2))
    boundary_indices = np.flatnonzero(boundary_mask)
    if arrow_count > 0 and len(boundary_indices) > arrow_count:
        selection = np.linspace(0, len(boundary_indices) - 1, arrow_count, dtype=np.int64)
        arrow_indices = boundary_indices[selection]
    else:
        arrow_indices = boundary_indices
    # A tangent vector only approximates a curved surface locally.  Scale the
    # arrows by point spacing rather than model size, otherwise a long straight
    # quiver visually leaves the saddle even though its origin is tangent.
    nearest_distances, _ = cKDTree(points).query(points, k=2)
    local_spacing = float(np.median(nearest_distances[:, 1]))
    arrow_length = 1.5 * max(local_spacing, np.finfo(float).eps)

    surface_axis = figure.add_subplot(1, 4, 1, projection="3d")
    vertices = np.asarray(mesh.vertices)
    surface_axis.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=np.asarray(mesh.faces),
        cmap="coolwarm",
        linewidth=0.05,
        antialiased=True,
        alpha=0.9,
    )
    surface_axis.set_title("Generated saddle surface")
    surface_axis.set_xlabel("x")
    surface_axis.set_ylabel("y")
    surface_axis.set_zlabel("z")
    surface_axis.set_box_aspect(np.ptp(vertices, axis=0))

    cloud_axis = figure.add_subplot(1, 4, 2, projection="3d")
    cloud_axis.scatter(*points.T, c=color, cmap="viridis", s=6)
    cloud_axis.scatter(*points[boundary_indices].T, c="black", s=10)
    cloud_axis.quiver(
        points[arrow_indices, 0], points[arrow_indices, 1], points[arrow_indices, 2],
        directions[arrow_indices, 0], directions[arrow_indices, 1],
        directions[arrow_indices, 2],
        length=arrow_length, normalize=False, color="crimson", linewidth=0.8,
    )
    cloud_axis.set_title("Cloud-boundary minimum-gradient directions")
    cloud_axis.set_xlabel("x")
    cloud_axis.set_ylabel("y")
    cloud_axis.set_zlabel("z")
    cloud_axis.set_box_aspect(np.ptp(points, axis=0))

    direction_axis = figure.add_subplot(1, 4, 3)
    direction_axis.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap="coolwarm", s=5)
    direction_axis.scatter(
        points[boundary_indices, 0], points[boundary_indices, 1], c="black", s=10
    )
    direction_axis.quiver(
        points[arrow_indices, 0], points[arrow_indices, 1],
        directions[arrow_indices, 0], directions[arrow_indices, 1],
        color="black", angles="xy", scale_units="xy", scale=1.0 / arrow_length,
        width=0.004, headwidth=3.5,
    )
    direction_axis.set_title("Cloud boundary only: one direction per point")
    direction_axis.set_xlabel("x")
    direction_axis.set_ylabel("y")
    direction_axis.set_aspect("equal", adjustable="box")

    isomap_axis = figure.add_subplot(1, 4, 4)
    isomap_axis.scatter(uv[:, 0], uv[:, 1], c=color, cmap="viridis", s=7)
    isomap_axis.set_title("Standard Isomap embedding")
    isomap_axis.set_xlabel("u")
    isomap_axis.set_ylabel("v")
    isomap_axis.set_aspect("equal", adjustable="datalim")

    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extent", type=float, default=2.5, help="Half-width of the surface")
    parser.add_argument("--height", type=float, default=1.0, help="Saddle height coefficient")
    parser.add_argument(
        "--mesh-resolution", type=int, default=81,
        help="Vertices along each side of the generated mesh",
    )
    parser.add_argument("--points", type=int, default=1200, help="Point-cloud target size")
    parser.add_argument("--neighbors", type=int, default=12, help="Isomap k-neighbor count")
    parser.add_argument(
        "--gradient-neighbors", type=int, default=24,
        help="Point neighbors used for each local gradient fit",
    )
    parser.add_argument(
        "--gradient-arrows", type=int, default=100,
        help="Maximum boundary arrows shown; 0 draws every boundary direction",
    )
    parser.add_argument(
        "--boundary-angle", type=float, default=135.0,
        help="Empty tangent-plane angle used to detect cloud boundary points",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling random seed")
    parser.add_argument("--cpu", action="store_true", help="Disable GPU kNN/MDS")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("saddle_isomap_output"),
        help="Directory for STL, CSV, and PNG outputs",
    )
    parser.add_argument("--no-show", action="store_true", help="Save without opening a window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1/3 Generating saddle triangle mesh ...")
    mesh = create_saddle_mesh(args.extent, args.height, args.mesh_resolution)
    stl_path = output_dir / "saddle_surface.stl"
    mesh.export(stl_path)
    print(f"    mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    print("2/3 Sampling an area-uniform point cloud ...")
    points, _ = sample_uniform_point_cloud(mesh, args.points, args.seed)
    cloud_path = output_dir / "saddle_point_cloud.csv"
    save_point_cloud(cloud_path, points)
    print(f"    point cloud: {len(points)} points")

    print("3/3 Running standard kNN Isomap ...")
    uv, backend = run_isomap(
        points,
        neighbors=args.neighbors,
        prefer_gpu=not args.cpu,
    )
    embedding_path = output_dir / "saddle_isomap.csv"
    save_embedding(embedding_path, points, uv)

    print("Detecting the boundary from point-cloud neighborhoods only ...")
    boundary_mask = point_cloud_boundary_mask(
        points,
        neighbors=args.gradient_neighbors,
        angle_threshold_degrees=args.boundary_angle,
    )
    normals, directions, gradient_norms, minimum_directional_gradients = (
        estimate_boundary_minimum_gradient_directions(
            points, boundary_mask, neighbors=args.gradient_neighbors
        )
    )
    print(f"    point-cloud boundary: {np.count_nonzero(boundary_mask)} points")
    gradient_path = output_dir / "saddle_boundary_minimum_gradient_directions.csv"
    save_boundary_directions(
        gradient_path,
        points,
        boundary_mask,
        normals,
        directions,
        gradient_norms,
        minimum_directional_gradients,
    )

    if args.no_show:
        plt.switch_backend("Agg")
    figure_path = output_dir / "saddle_isomap_result.png"
    plot_demo(
        mesh, points, uv, boundary_mask, directions, args.gradient_arrows,
        figure_path, show=not args.no_show,
    )

    print(f"Isomap backend: {backend}")
    print(f"STL:         {stl_path}")
    print(f"Point cloud: {cloud_path}")
    print(f"Embedding:   {embedding_path}")
    print(f"Directions:  {gradient_path}")
    print(f"Figure:      {figure_path}")


if __name__ == "__main__":
    main()
