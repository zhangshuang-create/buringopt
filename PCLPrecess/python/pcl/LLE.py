#!/usr/bin/env python3
"""Uniformly sample an STL surface and unfold it with standard LLE.

When no input path is supplied, a file-selection dialog is opened.  The
output CSV uses the same columns as Isomap.py so the two embeddings can be
compared directly.

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
from scipy.sparse import coo_matrix, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from scipy.spatial import cKDTree

# Reuse the STL loader, uniform surface sampler, CSV layout, boundary
# diagnostics and side-by-side visualization from the reference program.
import Isomap as isomap_reference


def nearest_neighbors(points: np.ndarray, neighbors: int) -> np.ndarray:
    """Return the indices of each point's k nearest *other* points."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2-D array")
    sample_count = len(points)
    if sample_count < 4:
        raise ValueError("Standard 2-D LLE requires at least four points")
    if neighbors < 2:
        raise ValueError("neighbors must be at least 2")
    if neighbors >= sample_count:
        raise ValueError(
            f"neighbors must be smaller than the sample count ({sample_count})"
        )
    _distances, indices = cKDTree(points).query(points, k=neighbors + 1)
    return np.asarray(indices[:, 1:], dtype=np.int64)


def check_neighbor_graph(indices: np.ndarray, sample_count: int) -> None:
    """Reject disconnected kNN graphs, which make the LLE nullspace ambiguous."""
    rows = np.repeat(np.arange(sample_count, dtype=np.int64), indices.shape[1])
    columns = indices.reshape(-1)
    graph = coo_matrix(
        (
            np.ones(2 * len(rows), dtype=np.uint8),
            (
                np.concatenate([rows, columns]),
                np.concatenate([columns, rows]),
            ),
        ),
        shape=(sample_count, sample_count),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    if component_count != 1:
        sizes = np.bincount(labels)
        raise ValueError(
            f"The {indices.shape[1]}-neighbor graph has {component_count} connected "
            f"components (sizes: {sizes.tolist()}). Increase --neighbors or improve "
            "the surface sampling density."
        )


def reconstruction_weights(
    points: np.ndarray,
    neighbor_indices: np.ndarray,
    regularization: float = 1.0e-3,
) -> coo_matrix:
    """Compute standard LLE barycentric reconstruction weights.

    For every point Xi, solve Cw=1 in its centered neighborhood and normalize
    w so sum(w)=1.  Trace-scaled diagonal regularization handles locally flat
    or rank-deficient neighborhoods without changing translation invariance.
    """
    if regularization <= 0.0:
        raise ValueError("regularization must be positive")
    sample_count, neighbors = neighbor_indices.shape
    rows = np.repeat(np.arange(sample_count, dtype=np.int64), neighbors)
    columns = neighbor_indices.reshape(-1)
    values = np.empty(sample_count * neighbors, dtype=float)
    ones = np.ones(neighbors, dtype=float)

    for point_index, local_indices in enumerate(neighbor_indices):
        centered = points[local_indices] - points[point_index]
        covariance = centered @ centered.T
        trace = float(np.trace(covariance))
        diagonal = regularization * trace if trace > 1.0e-15 else regularization
        covariance.flat[:: neighbors + 1] += diagonal
        try:
            weights = np.linalg.solve(covariance, ones)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(covariance, ones, rcond=None)[0]
        weight_sum = float(weights.sum())
        if abs(weight_sum) <= np.finfo(float).eps:
            raise ValueError(f"Degenerate reconstruction weights at point {point_index}")
        values[point_index * neighbors:(point_index + 1) * neighbors] = (
            weights / weight_sum
        )

    return coo_matrix(
        (values, (rows, columns)), shape=(sample_count, sample_count)
    )


def _smallest_eigenvectors(matrix, count: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Compute the smallest eigenpairs with a dense fallback for small inputs."""
    sample_count = matrix.shape[0]
    if sample_count <= 700:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix.toarray())
        return eigenvalues[:count], eigenvectors[:, :count], "dense symmetric eigensolver"

    # LLE always has a zero eigenvalue.  A tiny positive diagonal shift makes
    # shift-invert factorization stable while leaving all eigenvectors intact.
    scale = max(float(matrix.diagonal().mean()), 1.0)
    shifted = matrix + eye(sample_count, format="csr") * (1.0e-10 * scale)
    try:
        eigenvalues, eigenvectors = eigsh(
            shifted,
            k=count,
            sigma=0.0,
            which="LM",
            tol=1.0e-8,
            maxiter=max(5000, 20 * sample_count),
        )
    except (ArpackNoConvergence, RuntimeError) as exc:
        converged_values = getattr(exc, "eigenvalues", None)
        converged_vectors = getattr(exc, "eigenvectors", None)
        if converged_values is None or len(converged_values) < count:
            raise RuntimeError(
                "LLE eigenvalue solver did not converge. Reduce --max-points or "
                "adjust --neighbors."
            ) from exc
        eigenvalues, eigenvectors = converged_values, converged_vectors
    order = np.argsort(eigenvalues)
    return (
        np.asarray(eigenvalues[order], dtype=float),
        np.asarray(eigenvectors[:, order], dtype=float),
        "sparse ARPACK shift-invert eigensolver",
    )


def run_lle(
    points: np.ndarray,
    neighbors: int = 12,
    output_dimensions: int = 2,
    regularization: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Embed points using the original/standard LLE algorithm."""
    points = np.asarray(points, dtype=float)
    if output_dimensions < 1:
        raise ValueError("output_dimensions must be positive")
    if len(points) <= output_dimensions + 1:
        raise ValueError("Not enough points for the requested embedding dimension")
    indices = nearest_neighbors(points, neighbors)
    check_neighbor_graph(indices, len(points))
    weights = reconstruction_weights(points, indices, regularization)
    difference = eye(len(points), format="csr") - weights.tocsr()
    cost = (difference.T @ difference).tocsr()
    eigenvalues, eigenvectors, solver = _smallest_eigenvectors(
        cost, output_dimensions + 1
    )
    # Report the Rayleigh quotients of the original, unshifted LLE cost
    # matrix.  The sparse solver internally adds a tiny diagonal shift only
    # to stabilize factorization; it must not appear in the reconstruction
    # error presented to the user.
    eigenvalues = np.einsum("ij,ij->j", eigenvectors, cost @ eigenvectors)

    # Discard the constant null-space eigenvector. Multiplication by sqrt(N)
    # only gives convenient plotting scale and does not alter the embedding.
    embedding = eigenvectors[:, 1:output_dimensions + 1] * np.sqrt(len(points))
    for axis in range(embedding.shape[1]):
        anchor = int(np.argmax(np.abs(embedding[:, axis])))
        if embedding[anchor, axis] < 0.0:
            embedding[:, axis] *= -1.0
    reconstruction_error = np.asarray(eigenvalues[1:output_dimensions + 1])
    return embedding, reconstruction_error, solver


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
        title="Select an STL model for standard LLE",
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
        help="Number of LLE reconstruction neighbors (default: 12)",
    )
    parser.add_argument(
        "--regularization", type=float, default=1.0e-3,
        help="Trace-scaled local covariance regularization (default: 1e-3)",
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
        "--gradient-out", default="lle_boundary_directions.csv",
        help="Output CSV for point-cloud boundary directions",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--out", default="lle_points.csv", help="Output CSV")
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

    print(f"Running standard LLE with k={args.neighbors} ...")
    uv, eigenvalues, solver = run_lle(
        points,
        neighbors=args.neighbors,
        output_dimensions=2,
        regularization=args.regularization,
    )
    print(f"LLE solver: {solver}")
    print(f"Non-trivial eigenvalues: {eigenvalues.tolist()}")
    print(f"Embedding cost: {float(eigenvalues.sum()):.9g}")

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
