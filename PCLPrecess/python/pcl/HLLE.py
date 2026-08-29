#!/usr/bin/env python3
"""Area-uniformly sample an STL surface and embed it in 2D with Hessian Locally Linear Embedding (HLLE).

Ref:
    David L. Donoho and Carrie Grimes, "Hessian eigenmaps: Locally linear embedding
    techniques for high-dimensional data", PNAS 2003.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

# Ensure the local path is available to import Isomap reuse functions
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

# Reuse the exact STL, uniform sampling, boundary, CSV, and plotting pipeline.
# A silent simplified fallback would no longer match Isomap.py.
from Isomap import (
    estimate_boundary_minimum_gradient_directions,
    load_stl,
    ordered_boundary_loops,
    plot_result,
    point_cloud_boundary_mask,
    sample_surface_uniform_density,
    save_boundary_directions,
    save_csv,
    select_stl_file,
)


def run_hessian_lle(
        points: np.ndarray,
        neighbors: int,
        n_components: int = 2,
        prefer_gpu: bool = True,
        tol: float = 1e-4,
        max_iter: int = 1000,
) -> tuple[np.ndarray, str]:
    """Hessian Locally Linear Embedding (HLLE) core solver.

    Mathematical stages mapped to the paper "Hessian eigenmaps: Locally linear embedding
    techniques for high-dimensional data" (Donoho & Grimes, PNAS 2003).
    """
    N = len(points)
    d = n_components
    dp = d * (d + 1) // 2
    _ = prefer_gpu  # Kept for command/API compatibility; the solver is CPU-only.

    # Paper Sect. 5: X^i has 1+d+d(d+1)/2 columns, all of which
    # must be orthonormalized in a k-point neighbourhood.
    required_columns = 1 + d + dp
    if neighbors < required_columns:
        raise ValueError(
            f"HLLE needs neighbors >= 1+d+d(d+1)/2 = {required_columns}; "
            f"received k={neighbors}."
        )
    if neighbors > N:
        raise ValueError(f"neighbors (k={neighbors}) cannot exceed point count N={N}")
    if points.ndim != 2 or points.shape[1] < d:
        raise ValueError(f"HLLE needs at least {d} ambient coordinates")

    # Stage 1: Local Neighborhood Query (Page 3: "Identify neighbors")
    # The query point itself is part of its k-point neighbourhood, matching the
    # original reference implementation.
    _, indices = cKDTree(points).query(points, k=neighbors)

    # Neighborhood graph connectivity verification (consistent with Isomap checks)
    rows_knn = np.repeat(np.arange(N), neighbors)
    cols_knn = indices.ravel()
    knn_graph = coo_matrix((np.ones_like(rows_knn), (rows_knn, cols_knn)), shape=(N, N))
    n_components_connected, _ = connected_components(knn_graph, directed=False)
    if n_components_connected != 1:
        raise ValueError(
            f"The kNN graph has {n_components_connected} connected components; "
            "increase --neighbors or improve sampling density."
        )

    # Stage 2: SVD & Local Coordinate Orthonormalization (Page 3: "Obtain tangent coordinates")
    # For each local neighborhood, project coordinates onto local orthogonal tangent planes
    rows, cols, data = [], [], []

    for i in range(N):
        idx = indices[i]
        X_idx = points[idx]

        # Center the neighborhood and perform SVD (Page 3)
        M_i = X_idx - X_idx.mean(axis=0)
        U, _, _ = np.linalg.svd(M_i, full_matrices=False)
        U_local = U[:, :d]  # Tangent coordinates spanned by the first d singular vectors

        # Develop Hessian estimator: Design matrix construction (Page 3, Equation [1])
        # Column 0: constant 1; Columns 1..d: linear local coordinates; Columns d+1..: quadratic cross-products
        design_cols = [np.ones(neighbors)]
        for j in range(d):
            design_cols.append(U_local[:, j])
        for j in range(d):
            for h in range(j, d):
                design_cols.append(U_local[:, j] * U_local[:, h])
        X_i = np.column_stack(design_cols)
        if np.linalg.matrix_rank(X_i) < required_columns:
            raise ValueError(
                f"Neighbourhood {i} has a rank-deficient Hessian design matrix; "
                "increase --neighbors or sampling density."
            )

        # Gram-Schmidt Orthonormalization (Page 3: "yielding a matrix with orthonormal columns")
        # Performed via QR decomposition for numerical robustness
        Q, R = np.linalg.qr(X_i)

        # Define H^i by extracting the last d(d+1)/2 columns and transposing (Page 3)
        H_i = Q[:, 1 + d:].T  # Shape: (dp, k)

        # Local quadratic form: W_i = (H^i).T @ H^i (Page 3, "Develop quadratic form")
        W_i = H_i.T @ H_i

        # Assemble index grids for global sparse representation
        idx_grid_r, idx_grid_c = np.meshgrid(idx, idx, indexing="ij")
        rows.append(idx_grid_r.ravel())
        cols.append(idx_grid_c.ravel())
        data.append(W_i.ravel())

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    H_matrix = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    H_matrix = 0.5 * (H_matrix + H_matrix.T)  # Force exact symmetry

    # Stage 3: Spectral Analysis (paper p. 5593: "Find approximate null
    # space"). The actual implementation is NumPy/SciPy CPU throughout.
    backend = "CPU sparse/dense eigensolver"

    # We retrieve the d + 1 smallest eigenvalues of H_matrix.
    # The smallest eigenvalue corresponds to the constant null space (eigenvalue ~ 0).
    # The next d eigenvectors span the low-dimensional embedding coordinates.
    if N <= 1000:
        # Dense LAPACK is numerically equivalent and robust for small data.
        H_dense = H_matrix.toarray()
        eigenvals, eigenvectors = la.eigh(H_dense)
        idx_sorted = np.argsort(eigenvals)
        eigenvals = eigenvals[idx_sorted[:d + 1]]
        eigenvectors = eigenvectors[:, idx_sorted[:d + 1]]
        solver_status = "Converged (Dense LAPACK solver)"
    else:
        try:
            # H is positive semidefinite with a theoretical null space.  A tiny
            # negative shift makes H-sigma*I positive definite while preserving
            # the ordering of its smallest eigenvalues.  This is the stable
            # shift-invert realization of the paper's sparse Arnoldi analysis.
            spectral_scale = max(float(np.max(np.abs(H_matrix.diagonal()))), 1.0)
            negative_shift = -1.0e-8 * spectral_scale
            eigenvals, eigenvectors = eigsh(
                H_matrix,
                k=d + 1,
                sigma=negative_shift,
                which="LM",
                tol=tol,
                maxiter=max_iter,
            )
            idx_sorted = np.argsort(eigenvals)
            eigenvals = eigenvals[idx_sorted]
            eigenvectors = eigenvectors[:, idx_sorted]
            solver_status = "Converged (ARPACK sparse eigsh)"
        except Exception as e:
            raise RuntimeError(
                "Sparse HLLE eigensolver failed. Reduce --max-points, increase "
                "--iterations, or improve neighbourhood sampling; refusing to "
                "convert a large Hessian matrix to dense form."
            ) from e

    # Numerically, an eigensolver may return an arbitrary orthonormal rotation
    # of the (d+1)-dimensional approximate null space.  Explicitly remove the
    # constant function from that subspace instead of assuming it is column 0.
    constant = np.ones(N, dtype=float) / np.sqrt(float(N))
    constant_coefficients = eigenvectors.T @ constant
    constant_subspace_overlap = float(np.linalg.norm(constant_coefficients))
    if constant_subspace_overlap < 1.0 - 1.0e-5:
        raise RuntimeError(
            "The d+1 dimensional Hessian eigenspace does not contain the "
            "constant null vector; the local fits are degenerate."
        )
    nonconstant_coefficients = la.null_space(constant_coefficients[None, :])
    V = eigenvectors @ nonconstant_coefficients

    # Stage 4: Local Orthonormal Alignment (Page 3: "Find basis for null space")
    # Coordinates are recovered up to a linear isometry. Perform local orthonormal normalization.
    # W = V * R^(-1/2) where R is the local correlation matrix evaluated over an arbitrary neighborhood.
    idx_0 = indices[0]
    V_local = V[idx_0]
    R_mat = V_local.T @ V_local

    val, vec = la.eigh(R_mat)
    if np.min(val) <= 1.0e-10 * max(float(np.max(val)), 1.0):
        raise RuntimeError(
            "The selected null-space basis is rank deficient on the fixed "
            "reference neighbourhood; increase --neighbors or sampling density."
        )
    R_inv_sqrt = vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T
    W = V @ R_inv_sqrt

    # Print algorithm specifics
    print(f"  kNN graph connectivity components: {n_components_connected}")
    print(f"  Smallest eigenvalues (H-functional curviness values): {eigenvals}")
    print(f"  Eigen solver convergence status: {solver_status}")

    return W, backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hessian Locally Linear Embedding (HLLE) on STL surfaces")
    parser.add_argument(
        "stl", nargs="?", default="",
        help="Input STL file; when omitted, a file dialog opens",
    )
    parser.add_argument(
        "--density", type=float, default=1.0,
        help="Samples per square model unit (default: 1.0)",
    )
    parser.add_argument("--neighbors", type=int, default=30, help="Hessian LLE k-neighbor count (default: 30)")
    parser.add_argument(
        "--max-points", type=int, default=4000,
        help="Maximum total samples used by HLLE (default: 4000)",
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="ARPACK solver maximum iterations (default: 1000)",
    )
    parser.add_argument("--tolerance", type=float, default=1e-4, help="ARPACK solver numerical tolerance")
    parser.add_argument(
        "--cpu", action="store_true",
        help="Compatibility option; the paper-faithful HLLE solver is CPU-only",
    )
    parser.add_argument(
        "--gradient-neighbors", type=int, default=24,
        help="Point-cloud neighbors for boundary and tangent estimation (default: 24)",
    )
    parser.add_argument(
        "--boundary-angle", type=float, default=135.0,
        help="Minimum empty tangent-plane angle for a cloud boundary point, degrees",
    )
    parser.add_argument(
        "--gradient-arrows", type=int, default=150,
        help="Maximum boundary direction arrows to draw; 0 draws all",
    )
    parser.add_argument(
        "--gradient-out", default="point_cloud_boundary_directions.csv",
        help="Output CSV for point-cloud boundary minimum-gradient directions",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out", default="hlle_points.csv", help="Output CSV path")
    parser.add_argument("--save-figure", default="", help="Optional output PNG path")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stl_path = Path(args.stl) if args.stl else select_stl_file()
    if stl_path is None:
        raise SystemExit("No STL model selected")

    print(f"Input STL path: {stl_path.resolve()}")
    mesh = load_stl(stl_path)
    print(f"STL surface area: {mesh.area:.6g}")

    # Boundary extraction diagnostics
    boundary_count = len(ordered_boundary_loops(np.asarray(mesh.faces, dtype=np.int64)))
    print(f"Number of boundary loops in mesh: {boundary_count}")

    rng = np.random.default_rng(args.seed)

    # Model surface uniform sampling
    points, face_ids = sample_surface_uniform_density(
        mesh, args.density, rng, max_points=args.max_points,
        boundary_multiplier=0.0,
    )
    print(f"Total points generated: {len(points)}")

    # Point-cloud-based boundary detection (copied logic/functions from Isomap)
    print("Executing point-cloud based boundary analysis...")
    cloud_boundary_mask = point_cloud_boundary_mask(
        points,
        neighbors=args.gradient_neighbors,
        angle_threshold_degrees=args.boundary_angle,
    )
    normals, minimum_directions, height_gradient_norms, minimum_directional_gradients = (
        estimate_boundary_minimum_gradient_directions(
            points, cloud_boundary_mask, neighbors=args.gradient_neighbors
        )
    )
    print(
        f"Detected point-cloud boundary samples: {np.count_nonzero(cloud_boundary_mask)} "
        f"using empty-angle threshold {args.boundary_angle:g} degrees."
    )
    save_boundary_directions(
        args.gradient_out,
        points,
        cloud_boundary_mask,
        normals,
        minimum_directions,
        height_gradient_norms,
        minimum_directional_gradients,
    )

    # Run core HLLE solver
    print("Beginning HLLE dimensional projection...")
    uv, backend = run_hessian_lle(
        points,
        args.neighbors,
        n_components=2,
        prefer_gpu=not args.cpu,
        tol=args.tolerance,
        max_iter=args.iterations
    )
    print(f"HLLE projection completed with backend: {backend}")

    save_csv(args.out, points, uv, face_ids)
    print(f"Saved output coordinates to: {Path(args.out).resolve()}")

    # Display visualization (identical layout)
    if not args.no_show or args.save_figure:
        if args.no_show:
            plt.switch_backend("Agg")
        plot_result(
            points,
            uv,
            cloud_boundary_mask,
            minimum_directions,
            args.gradient_arrows,
            args.save_figure,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
