#!/usr/bin/env python3
"""Uniformly sample an STL surface and unfold it with standard LTSA.

LTSA (Local Tangent Space Alignment) estimates a two-dimensional tangent
space in every k-neighborhood and then aligns all local tangent coordinates
through one global sparse eigenproblem.  With no input argument, a native STL
file-selection dialog is opened.

Dependencies:
    numpy scipy trimesh matplotlib
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

RUNTIME_CACHE = Path(__file__).resolve().parents[2] / ".runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix

import Isomap as isomap_reference
import LLE as lle_reference


def local_tangent_alignment_matrix(
    points: np.ndarray,
    neighbor_indices: np.ndarray,
    output_dimensions: int = 2,
) -> tuple[coo_matrix, np.ndarray]:
    """Assemble the standard LTSA global alignment matrix.

    For neighborhood Xi, centered SVD supplies the local tangent basis Ui.
    Gi=[1/sqrt(k), Ui] spans its affine tangent coordinates and
    Bi=I-Gi Gi^T measures components that cannot be explained by that local
    tangent space.  Each Bi is accumulated into the corresponding rows and
    columns of the global alignment matrix.
    """
    points = np.asarray(points, dtype=float)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    if points.ndim != 2:
        raise ValueError("points must be a 2-D array")
    if neighbor_indices.ndim != 2 or len(neighbor_indices) != len(points):
        raise ValueError("neighbor_indices must have shape (N, k)")
    if output_dimensions < 1:
        raise ValueError("output_dimensions must be positive")
    sample_count, neighbors = neighbor_indices.shape
    if neighbors <= output_dimensions:
        raise ValueError(
            "LTSA requires neighbors greater than the output dimension "
            f"({output_dimensions})"
        )

    block_size = neighbors * neighbors
    rows = np.empty(sample_count * block_size, dtype=np.int64)
    columns = np.empty(sample_count * block_size, dtype=np.int64)
    values = np.empty(sample_count * block_size, dtype=float)
    tangent_singular_values = np.empty((sample_count, output_dimensions), dtype=float)
    constant = np.full((neighbors, 1), 1.0 / np.sqrt(neighbors), dtype=float)

    for point_index, local_indices in enumerate(neighbor_indices):
        neighborhood = points[local_indices]
        centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
        left_vectors, singular_values, _right_vectors = np.linalg.svd(
            centered, full_matrices=False
        )
        available = min(output_dimensions, len(singular_values))
        if available < output_dimensions:
            raise ValueError(
                f"Neighborhood {point_index} has insufficient ambient rank for "
                f"a {output_dimensions}-D tangent space"
            )
        tangent_singular_values[point_index] = singular_values[:output_dimensions]
        tangent_basis = left_vectors[:, :output_dimensions]

        # QR removes tiny numerical non-orthogonality between the constant
        # vector and SVD tangent vectors before forming I-GG^T.
        affine_basis, _ = np.linalg.qr(
            np.column_stack((constant, tangent_basis)), mode="reduced"
        )
        local_alignment = np.eye(neighbors) - affine_basis @ affine_basis.T

        start = point_index * block_size
        stop = start + block_size
        rows[start:stop] = np.repeat(local_indices, neighbors)
        columns[start:stop] = np.tile(local_indices, neighbors)
        values[start:stop] = local_alignment.reshape(-1)

    alignment = coo_matrix(
        (values, (rows, columns)), shape=(sample_count, sample_count)
    )
    return alignment, tangent_singular_values


def run_ltsa(
    points: np.ndarray,
    neighbors: int = 12,
    output_dimensions: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Embed points using standard Local Tangent Space Alignment."""
    points = np.asarray(points, dtype=float)
    if len(points) <= output_dimensions + 1:
        raise ValueError("Not enough points for the requested embedding dimension")
    if neighbors <= output_dimensions:
        raise ValueError(
            f"neighbors must be greater than output_dimensions ({output_dimensions})"
        )

    neighbor_indices = lle_reference.nearest_neighbors(points, neighbors)
    lle_reference.check_neighbor_graph(neighbor_indices, len(points))
    alignment, singular_values = local_tangent_alignment_matrix(
        points, neighbor_indices, output_dimensions
    )
    alignment = alignment.tocsr()
    # Symmetrize only against floating-point accumulation noise.
    alignment = (0.5 * (alignment + alignment.T)).tocsr()
    eigenvalues, eigenvectors, solver = lle_reference._smallest_eigenvectors(
        alignment, output_dimensions + 1
    )
    eigenvalues = np.einsum(
        "ij,ij->j", eigenvectors, alignment @ eigenvectors
    )

    embedding = eigenvectors[:, 1:output_dimensions + 1] * np.sqrt(len(points))
    for axis in range(embedding.shape[1]):
        anchor = int(np.argmax(np.abs(embedding[:, axis])))
        if embedding[anchor, axis] < 0.0:
            embedding[:, axis] *= -1.0
    alignment_error = np.asarray(eigenvalues[1:output_dimensions + 1])
    return embedding, alignment_error, singular_values, solver


def select_stl_file() -> Path | None:
    """Open a native file chooser and return the selected STL path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("No STL path was provided and tkinter is unavailable") from exc

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title="Select an STL model for standard LTSA",
        filetypes=[("STL models", "*.stl"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(selected) if selected else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stl", nargs="?", default="",
        help="Input STL file; when omitted, a file-selection dialog opens",
    )
    parser.add_argument(
        "--density", type=float, default=1.0,
        help="Samples per square model unit (default: 1.0)",
    )
    parser.add_argument(
        "--neighbors", type=int, default=12,
        help="Number of points in each local tangent neighborhood (default: 12)",
    )
    parser.add_argument(
        "--max-points", type=int, default=4000,
        help="Maximum uniformly sampled surface points; 0 disables the cap",
    )
    parser.add_argument(
        "--gradient-neighbors", type=int, default=24,
        help="Neighbors for boundary and tangent estimation (default: 24)",
    )
    parser.add_argument(
        "--boundary-angle", type=float, default=135.0,
        help="Boundary empty-sector threshold in degrees (default: 135)",
    )
    parser.add_argument(
        "--gradient-arrows", type=int, default=150,
        help="Maximum boundary direction arrows in the 3-D view",
    )
    parser.add_argument(
        "--gradient-out", default="ltsa_boundary_directions.csv",
        help="Output CSV for point-cloud boundary directions",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--out", default="ltsa_points.csv", help="Output CSV")
    parser.add_argument("--save-figure", default="", help="Optional output PNG")
    parser.add_argument("--no-show", action="store_true", help="Do not show the plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stl_path = Path(args.stl) if args.stl else select_stl_file()
    if stl_path is None:
        raise SystemExit("No STL model selected")
    if stl_path.suffix.lower() != ".stl":
        raise SystemExit("The selected input must be an STL model")

    print(f"Input STL: {stl_path.resolve()}")
    mesh = isomap_reference.load_stl(stl_path)
    boundary_count = len(
        isomap_reference.ordered_boundary_loops(
            np.asarray(mesh.faces, dtype=np.int64)
        )
    )
    print(f"Boundary loops in input mesh: {boundary_count}")
    rng = np.random.default_rng(args.seed)
    requested_count = max(3, int(round(float(mesh.area) * args.density)))
    points, face_ids = isomap_reference.sample_surface_uniform_density(
        mesh,
        args.density,
        rng,
        max_points=args.max_points,
        boundary_multiplier=0.0,
    )
    print(f"Mesh area: {mesh.area:.6g}")
    print(f"Interior target: {requested_count} ({args.density:g} per square model unit)")
    print(f"Total sampled points: {len(points)}")

    print("Detecting point-cloud boundary and local tangent directions ...")
    boundary_mask = isomap_reference.point_cloud_boundary_mask(
        points,
        neighbors=args.gradient_neighbors,
        angle_threshold_degrees=args.boundary_angle,
    )
    normals, directions, gradient_norms, minimum_gradients = (
        isomap_reference.estimate_boundary_minimum_gradient_directions(
            points, boundary_mask, neighbors=args.gradient_neighbors
        )
    )
    isomap_reference.save_boundary_directions(
        args.gradient_out,
        points,
        boundary_mask,
        normals,
        directions,
        gradient_norms,
        minimum_gradients,
    )
    print(f"Boundary samples: {np.count_nonzero(boundary_mask)}")
    print(f"Boundary directions: {Path(args.gradient_out).resolve()}")

    print(f"Running standard LTSA with k={args.neighbors} ...")
    uv, eigenvalues, tangent_singular_values, solver = run_ltsa(
        points,
        neighbors=args.neighbors,
        output_dimensions=2,
    )
    tangent_ratio = tangent_singular_values[:, 1] / np.maximum(
        tangent_singular_values[:, 0], np.finfo(float).eps
    )
    print(f"LTSA solver: {solver}")
    print(f"Non-trivial eigenvalues: {eigenvalues.tolist()}")
    print(f"Alignment cost: {float(eigenvalues.sum()):.9g}")
    print(
        "Local tangent s2/s1 ratio: "
        f"min={float(tangent_ratio.min()):.6g}, "
        f"median={float(np.median(tangent_ratio)):.6g}, "
        f"max={float(tangent_ratio.max()):.6g}"
    )

    isomap_reference.save_csv(args.out, points, uv, face_ids)
    print(f"Saved: {Path(args.out).resolve()}")
    if not args.no_show or args.save_figure:
        if args.no_show:
            plt.switch_backend("Agg")
        isomap_reference.plot_result(
            points,
            uv,
            boundary_mask,
            directions,
            args.gradient_arrows,
            args.save_figure,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
