#!/usr/bin/env python3
"""Flatten an STL point cloud with Topologically Constrained Isometric Embedding.

This is an implementation of Rosman, Bronstein, Bronstein and Kimmel,
"Nonlinear Dimensionality Reduction by Topologically Constrained Isometric
Embedding", IJCV 89 (2010), 56-68.  STL loading, surface sampling, point-cloud
boundary estimation, CSV output, and plotting are shared with ``Isomap.py``.

Dependencies are the same as Isomap.py:
    py -m pip install numpy scipy trimesh matplotlib cupy-cuda11x
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags, triu
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import cg

import Isomap as isomap


def _local_graph_pairs(graph) -> tuple[np.ndarray, np.ndarray]:
    upper = triu(graph, k=1).tocoo()
    return (
        upper.row.astype(np.int64, copy=False),
        upper.col.astype(np.int64, copy=False),
    )


def criterion2_consistent_pairs(
    geodesic: np.ndarray,
    boundary_mask: np.ndarray,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper criterion 2: delta_ij <= d(i,boundary)+d(j,boundary)."""
    point_count = len(geodesic)
    boundary_indices = np.flatnonzero(boundary_mask)
    if not len(boundary_indices):
        boundary_distance = np.full(point_count, np.inf, dtype=float)
    else:
        boundary_distance = np.min(geodesic[:, boundary_indices], axis=1)

    first_parts: list[np.ndarray] = []
    second_parts: list[np.ndarray] = []
    tolerance = 32.0 * np.finfo(float).eps * max(
        1.0, float(np.max(geodesic[np.isfinite(geodesic)]))
    )
    columns = np.arange(point_count, dtype=np.int64)
    for start in range(0, point_count, block_size):
        stop = min(start + block_size, point_count)
        rows = np.arange(start, stop, dtype=np.int64)
        threshold = boundary_distance[rows, None] + boundary_distance[None, :]
        consistent = geodesic[rows] <= threshold + tolerance
        consistent &= columns[None, :] > rows[:, None]
        local_row, local_column = np.nonzero(consistent)
        if len(local_row):
            first_parts.append(rows[local_row])
            second_parts.append(local_column.astype(np.int64, copy=False))

    first = np.concatenate(first_parts) if first_parts else np.empty(0, dtype=np.int64)
    second = np.concatenate(second_parts) if second_parts else np.empty(0, dtype=np.int64)
    return first, second, boundary_distance


def criterion1_consistent_pairs(
    geodesic: np.ndarray,
    predecessors: np.ndarray,
    boundary_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper criterion 1: retain shortest paths with no boundary midpoint.

    SciPy's Dijkstra predecessor forest is traversed in increasing-distance
    order.  A boundary source or target is allowed; a boundary predecessor in
    the interior of a path invalidates that source-target pair.
    """
    point_count = len(geodesic)
    first_parts: list[np.ndarray] = []
    second_parts: list[np.ndarray] = []
    for source in range(point_count):
        valid = np.zeros(point_count, dtype=bool)
        valid[source] = True
        order = np.argsort(geodesic[source], kind="stable")
        for vertex in order[1:]:
            if not np.isfinite(geodesic[source, vertex]):
                break
            parent = int(predecessors[source, vertex])
            valid[vertex] = (
                parent >= 0
                and valid[parent]
                and not (boundary_mask[parent] and parent != source)
            )
        targets = np.flatnonzero(valid & (np.arange(point_count) > source))
        if len(targets):
            first_parts.append(np.full(len(targets), source, dtype=np.int64))
            second_parts.append(targets.astype(np.int64, copy=False))
        if point_count >= 500 and (source + 1) % max(1, point_count // 10) == 0:
            print(f"  criterion 1 paths: {source + 1}/{point_count} sources")
    first = np.concatenate(first_parts) if first_parts else np.empty(0, dtype=np.int64)
    second = np.concatenate(second_parts) if second_parts else np.empty(0, dtype=np.int64)
    return first, second


def build_weight_matrix(
    point_count: int,
    consistent_i: np.ndarray,
    consistent_j: np.ndarray,
    local_i: np.ndarray,
    local_j: np.ndarray,
):
    """Binary TCIE weights, with short graph edges retained as in the paper."""
    first = np.concatenate((consistent_i, local_i))
    second = np.concatenate((consistent_j, local_j))
    rows = np.concatenate((first, second))
    columns = np.concatenate((second, first))
    weights = coo_matrix(
        (np.ones(len(rows), dtype=float), (rows, columns)),
        shape=(point_count, point_count),
    ).tocsr()
    weights.sum_duplicates()
    weights.data.fill(1.0)
    weights.setdiag(0.0)
    weights.eliminate_zeros()
    return weights


def _weighted_stress(
    uv: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    desired: np.ndarray,
) -> float:
    projected = np.linalg.norm(uv[pair_i] - uv[pair_j], axis=1)
    return float(np.dot(projected - desired, projected - desired))


def weighted_smacof(
    geodesic: np.ndarray,
    weights,
    initial_uv: np.ndarray,
    iterations: int = 300,
    tolerance: float = 1.0e-7,
    cg_tolerance: float = 1.0e-7,
    cg_max_iterations: int = 200,
) -> tuple[np.ndarray, int, float]:
    """Minimize binary weighted LS-MDS stress using the paper's SMACOF step."""
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    upper_weights = triu(weights, k=1).tocoo()
    pair_i = upper_weights.row.astype(np.int64, copy=False)
    pair_j = upper_weights.col.astype(np.int64, copy=False)
    desired = geodesic[pair_i, pair_j]
    if not len(desired):
        raise ValueError("TCIE retained no pairwise distances")

    degree = np.asarray(weights.sum(axis=1)).ravel()
    v_matrix = diags(degree, format="csr") - weights
    reduced_v = v_matrix[1:, 1:].tocsr()
    uv = np.asarray(initial_uv, dtype=float).copy()
    uv -= uv.mean(axis=0, keepdims=True)
    normalizer = max(float(np.dot(desired, desired)), np.finfo(float).eps)
    previous_stress = _weighted_stress(uv, pair_i, pair_j, desired)
    report_every = max(1, iterations // 10)
    completed = 0

    for iteration in range(iterations):
        difference = uv[pair_i] - uv[pair_j]
        projected = np.linalg.norm(difference, axis=1)
        ratio = np.zeros_like(projected)
        nonzero = projected > 1.0e-12
        ratio[nonzero] = desired[nonzero] / projected[nonzero]
        pair_rhs = ratio[:, None] * difference
        rhs = np.zeros_like(uv)
        np.add.at(rhs, pair_i, pair_rhs)
        np.add.at(rhs, pair_j, -pair_rhs)

        candidate = np.zeros_like(uv)
        for dimension in range(2):
            initial_guess = uv[1:, dimension] - uv[0, dimension]
            solution, info = cg(
                reduced_v,
                rhs[1:, dimension],
                x0=initial_guess,
                rtol=cg_tolerance,
                atol=0.0,
                maxiter=cg_max_iterations,
            )
            if info < 0:
                raise RuntimeError(f"SMACOF linear solve failed in dimension {dimension}")
            if info > 0 and iteration == 0:
                print(
                    f"Warning: CG reached its {info}-iteration limit; "
                    "consider increasing --cg-max-iterations"
                )
            candidate[1:, dimension] = solution
        candidate -= candidate.mean(axis=0, keepdims=True)

        candidate_stress = _weighted_stress(candidate, pair_i, pair_j, desired)
        # An inexact CG solve can very slightly violate SMACOF's monotonicity.
        # Backtracking preserves the defining descent property.
        if candidate_stress > previous_stress:
            step_fraction = 0.5
            accepted = False
            for _ in range(12):
                trial = uv + step_fraction * (candidate - uv)
                trial_stress = _weighted_stress(trial, pair_i, pair_j, desired)
                if trial_stress <= previous_stress:
                    candidate, candidate_stress = trial, trial_stress
                    accepted = True
                    break
                step_fraction *= 0.5
            if not accepted:
                candidate, candidate_stress = uv, previous_stress

        relative_decrease = (previous_stress - candidate_stress) / max(
            previous_stress, np.finfo(float).eps
        )
        uv = candidate
        previous_stress = candidate_stress
        completed = iteration + 1
        if iteration == 0 or completed % report_every == 0 or completed == iterations:
            print(
                f"  SMACOF {completed:4d}/{iterations}: "
                f"normalized stress={candidate_stress / normalizer:.7g}, "
                f"relative decrease={relative_decrease:.3g}"
            )
        if relative_decrease >= 0.0 and relative_decrease < tolerance:
            print(f"  SMACOF converged after {completed} iterations")
            break

    return uv, completed, previous_stress / normalizer


def run_tcie(
    points: np.ndarray,
    boundary_mask: np.ndarray,
    neighbors: int = 12,
    criterion: int = 2,
    iterations: int = 300,
    tolerance: float = 1.0e-7,
    cg_tolerance: float = 1.0e-7,
    cg_max_iterations: int = 200,
    prefer_gpu: bool = True,
) -> tuple[np.ndarray, str]:
    """Run the four TCIE stages from Sect. 2 of the reference paper."""
    point_count = len(points)
    if point_count < 3:
        raise ValueError("At least three sampled points are required")
    neighbors = min(max(2, int(neighbors)), point_count - 1)
    available, device_name = isomap.gpu_is_available()
    use_gpu = bool(prefer_gpu and available)
    print(f"kNN backend: {'GPU' if use_gpu else 'CPU'} (k={neighbors})")
    graph = isomap.build_knn_graph(points, neighbors, use_gpu=use_gpu)
    component_count, _ = connected_components(graph, directed=False)
    if component_count != 1:
        raise ValueError(
            f"The kNN graph has {component_count} components; increase --neighbors"
        )

    print("TCIE step 1/4: computing all-pairs graph geodesics on CPU ...")
    if criterion == 1:
        geodesic, predecessors = dijkstra(
            graph, directed=False, return_predecessors=True
        )
    else:
        geodesic = dijkstra(graph, directed=False)
        predecessors = None
    if not np.all(np.isfinite(geodesic)):
        raise ValueError("The geodesic distance matrix contains disconnected pairs")

    boundary_count = int(np.count_nonzero(boundary_mask))
    print(f"TCIE step 2/4: using {boundary_count} detected boundary samples")
    print(f"TCIE step 3/4: selecting consistent geodesics with criterion {criterion} ...")
    if criterion == 1:
        consistent_i, consistent_j = criterion1_consistent_pairs(
            geodesic, predecessors, boundary_mask
        )
    elif criterion == 2:
        consistent_i, consistent_j, boundary_distance = criterion2_consistent_pairs(
            geodesic, boundary_mask
        )
        finite_boundary_distance = boundary_distance[np.isfinite(boundary_distance)]
        if len(finite_boundary_distance):
            print(
                "Distance to detected boundary: "
                f"median={np.median(finite_boundary_distance):.6g}, "
                f"max={np.max(finite_boundary_distance):.6g}"
            )
    else:
        raise ValueError("criterion must be 1 or 2")

    local_i, local_j = _local_graph_pairs(graph)
    weights = build_weight_matrix(
        point_count, consistent_i, consistent_j, local_i, local_j
    )
    weight_components, _ = connected_components(weights, directed=False)
    if weight_components != 1:
        raise ValueError(
            f"The retained TCIE weight graph has {weight_components} components"
        )
    retained_pairs = weights.nnz // 2
    total_pairs = point_count * (point_count - 1) // 2
    print(
        f"Retained weighted distances: {retained_pairs}/{total_pairs} "
        f"({100.0 * retained_pairs / max(1, total_pairs):.2f}%), "
        f"including {len(local_i)} short kNN edges"
    )

    print(f"Classical-MDS initialization: {'GPU' if use_gpu else 'CPU'}")
    initial_uv = (
        isomap.classical_mds_gpu(geodesic)
        if use_gpu
        else isomap.classical_mds_cpu(geodesic)
    )
    print("TCIE step 4/4: minimizing binary weighted LS-MDS stress with SMACOF ...")
    uv, completed, normalized_stress = weighted_smacof(
        geodesic,
        weights,
        initial_uv,
        iterations=iterations,
        tolerance=tolerance,
        cg_tolerance=cg_tolerance,
        cg_max_iterations=cg_max_iterations,
    )
    compute = (
        f"GPU kNN/MDS + CPU Dijkstra/SMACOF, {device_name}" if use_gpu else "CPU"
    )
    backend = (
        f"TCIE criterion {criterion} ({compute}, {completed} SMACOF iterations, "
        f"normalized stress {normalized_stress:.6g})"
    )
    return uv, backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stl", nargs="?", default="",
        help="Input STL file; when omitted, the same file dialog as Isomap.py opens",
    )
    parser.add_argument(
        "--density", type=float, default=1.0,
        help="Samples per square model unit (default: 1.0)",
    )
    parser.add_argument("--neighbors", type=int, default=12, help="k-neighbor count")
    parser.add_argument(
        "--criterion", type=int, choices=(1, 2), default=2,
        help="Paper consistency criterion 1 (path) or 2 (boundary distance)",
    )
    parser.add_argument(
        "--max-points", type=int, default=4000,
        help="Maximum samples; TCIE has quadratic memory/time cost (default: 4000)",
    )
    parser.add_argument("--iterations", type=int, default=300, help="SMACOF iterations")
    parser.add_argument(
        "--tolerance", type=float, default=1.0e-7,
        help="Relative SMACOF stress convergence threshold",
    )
    parser.add_argument(
        "--cg-tolerance", type=float, default=1.0e-7,
        help="Tolerance for each sparse SMACOF linear solve",
    )
    parser.add_argument(
        "--cg-max-iterations", type=int, default=200,
        help="Maximum conjugate-gradient iterations per coordinate",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU kNN and MDS")
    parser.add_argument(
        "--gradient-neighbors", type=int, default=24,
        help="Neighbors for point-cloud boundary/tangent estimation (default: 24)",
    )
    parser.add_argument(
        "--boundary-angle", type=float, default=135.0,
        help="Minimum empty tangent-plane angle for a boundary point, degrees",
    )
    parser.add_argument(
        "--gradient-arrows", type=int, default=150,
        help="Maximum boundary direction arrows to draw; 0 draws all",
    )
    parser.add_argument(
        "--gradient-out", default="tcie_boundary_directions.csv",
        help="Output CSV for point-cloud boundary directions",
    )
    parser.add_argument("--seed", type=int, default=0, help="Surface-sampling seed")
    parser.add_argument("--out", default="tcie_points.csv", help="Output CSV")
    parser.add_argument("--save-figure", default="", help="Optional output PNG")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stl_path = Path(args.stl) if args.stl else isomap.select_stl_file()
    if stl_path is None:
        raise SystemExit("No STL model selected")
    print(f"Input STL: {stl_path.resolve()}")

    # Keep STL -> point-cloud conversion exactly aligned with Isomap.py/CDA.py.
    mesh = isomap.load_stl(stl_path)
    mesh_boundary_count = len(
        isomap.ordered_boundary_loops(np.asarray(mesh.faces, dtype=np.int64))
    )
    print(f"Boundary loops in input mesh: {mesh_boundary_count}")
    print("TCIE: no explicit seam cut; sampling the original STL surface.")
    rng = np.random.default_rng(args.seed)
    requested_count = max(3, int(round(float(mesh.area) * args.density)))
    points, face_ids = isomap.sample_surface_uniform_density(
        mesh,
        args.density,
        rng,
        max_points=args.max_points,
        boundary_multiplier=0.0,
    )
    print(f"Mesh area: {mesh.area:.6g}")
    print(f"Interior target: {requested_count} ({args.density:g} per square model unit)")
    print(f"Total sampled points: {len(points)}")

    print("Detecting TCIE boundary samples from point-cloud neighborhoods ...")
    cloud_boundary_mask = isomap.point_cloud_boundary_mask(
        points,
        neighbors=args.gradient_neighbors,
        angle_threshold_degrees=args.boundary_angle,
    )
    normals, minimum_directions, height_gradient_norms, minimum_directional_gradients = (
        isomap.estimate_boundary_minimum_gradient_directions(
            points, cloud_boundary_mask, neighbors=args.gradient_neighbors
        )
    )
    print(
        f"Point-cloud boundary samples: {np.count_nonzero(cloud_boundary_mask)} "
        f"(empty-angle threshold {args.boundary_angle:g} degrees)"
    )
    isomap.save_boundary_directions(
        args.gradient_out,
        points,
        cloud_boundary_mask,
        normals,
        minimum_directions,
        height_gradient_norms,
        minimum_directional_gradients,
    )
    print(f"Boundary directions saved: {Path(args.gradient_out).resolve()}")

    available, gpu_status = isomap.gpu_is_available()
    print(f"CUDA status: {'available' if available else 'unavailable'} ({gpu_status})")
    uv, backend = run_tcie(
        points,
        cloud_boundary_mask,
        neighbors=args.neighbors,
        criterion=args.criterion,
        iterations=args.iterations,
        tolerance=args.tolerance,
        cg_tolerance=args.cg_tolerance,
        cg_max_iterations=args.cg_max_iterations,
        prefer_gpu=not args.cpu,
    )
    print(f"2D backend: {backend}")
    isomap.save_csv(args.out, points, uv, face_ids)
    print(f"Saved: {Path(args.out).resolve()}")

    if not args.no_show or args.save_figure:
        if args.no_show:
            isomap.plt.switch_backend("Agg")
        isomap.plot_result(
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
