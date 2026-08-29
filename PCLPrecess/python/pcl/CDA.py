#!/usr/bin/env python3
"""Sample an STL surface and unfold its point cloud with CDA.

This script deliberately reuses the STL loading, surface sampling, point-cloud
boundary detection, CSV output, and plotting code from ``Isomap.py``.  Only the
2-D embedding is replaced by Curvilinear Distance Analysis (CDA).

Dependencies are the same as Isomap.py:
    py -m pip install numpy scipy trimesh matplotlib cupy-cuda11x
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra

import Isomap as isomap


def _random_pairs(
    count: int, point_count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ordered, non-self point pairs without constructing an NxN index."""
    first = rng.integers(0, point_count, size=count, dtype=np.int64)
    second = rng.integers(0, point_count - 1, size=count, dtype=np.int64)
    second += second >= first
    return first, second


def _representative_distance_scale(
    geodesic: np.ndarray, rng: np.random.Generator
) -> float:
    """Estimate a robust global scale without materialising the upper triangle."""
    sample_count = min(200_000, max(20_000, 50 * len(geodesic)))
    first, second = _random_pairs(sample_count, len(geodesic), rng)
    values = geodesic[first, second]
    values = values[np.isfinite(values) & (values > 0.0)]
    if not len(values):
        raise ValueError("The geodesic distance matrix has no positive finite values")
    return float(np.median(values))


def run_cda(
    points: np.ndarray,
    neighbors: int = 12,
    iterations: int = 500,
    batch_size: int = 0,
    learning_rate: float = 0.03,
    lambda_start: float = 0.0,
    lambda_end: float = 0.0,
    seed: int = 0,
    prefer_gpu: bool = True,
    initialization: str = "pca",
) -> tuple[np.ndarray, str]:
    """Embed a point cloud in 2-D using annealed curvilinear-distance stress.

    For a target graph-geodesic distance D_ij and current projected distance
    d_ij, the frozen-weight stress used in each optimisation step is

        exp(-(d_ij / lambda)^2) * (d_ij - D_ij)^2 / D_ij.

    Consequently, a link which stretches across an emerging tear rapidly loses
    influence.  ``lambda`` is annealed from a nearly global value to a local
    value.  Pair mini-batches keep the optimisation practical after Isomap's
    necessarily dense all-pairs shortest-path calculation.
    """
    point_count = len(points)
    if point_count < 3:
        raise ValueError("At least three sampled points are required")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("learning-rate must be greater than zero")

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

    print("Computing all-pairs graph geodesics on CPU ...")
    geodesic = dijkstra(graph, directed=False)
    if not np.all(np.isfinite(geodesic)):
        raise ValueError("The geodesic distance matrix contains disconnected pairs")

    rng = np.random.default_rng(seed)
    global_scale = _representative_distance_scale(geodesic, rng)
    target = geodesic / global_scale
    if initialization == "mds":
        print(f"CDA initialization (classical MDS): {'GPU' if use_gpu else 'CPU'}")
        uv = (
            isomap.classical_mds_gpu(geodesic)
            if use_gpu
            else isomap.classical_mds_cpu(geodesic)
        )
        uv /= global_scale
    elif initialization == "pca":
        print("CDA initialization: PCA projection")
        centered_points = (points - points.mean(axis=0, keepdims=True)) / global_scale
        _, _, axes = np.linalg.svd(centered_points, full_matrices=False)
        uv = centered_points @ axes[:2].T
        scale_i, scale_j = _random_pairs(min(100_000, 30 * point_count), point_count, rng)
        projected_scale = float(np.median(np.linalg.norm(uv[scale_i] - uv[scale_j], axis=1)))
        if projected_scale <= np.finfo(float).eps:
            raise ValueError("PCA initialization collapsed to a point")
        # The median normalized graph-geodesic distance is approximately one.
        uv /= projected_scale
    elif initialization == "random":
        print("CDA initialization: random 2-D coordinates")
        uv = rng.normal(scale=0.5, size=(point_count, 2))
    else:
        raise ValueError(f"Unknown CDA initialization: {initialization}")
    uv -= uv.mean(axis=0, keepdims=True)
    # Perfectly symmetric cylinders otherwise have no numerical preference for
    # where a tear should nucleate.
    uv += rng.normal(scale=1.0e-3, size=uv.shape)

    upper_graph = graph.tocoo()
    local_mask = upper_graph.row < upper_graph.col
    local_i = upper_graph.row[local_mask].astype(np.int64, copy=False)
    local_j = upper_graph.col[local_mask].astype(np.int64, copy=False)
    if not len(local_i):
        raise ValueError("The kNN graph contains no edges")
    local_target = target[local_i, local_j]
    local_scale = float(np.median(local_target[local_target > 0.0]))

    if lambda_start <= 0.0:
        initial_i, initial_j = _random_pairs(min(100_000, 30 * point_count), point_count, rng)
        initial_distances = np.linalg.norm(uv[initial_i] - uv[initial_j], axis=1)
        lambda_start = max(float(np.percentile(initial_distances, 90.0)), 8.0 * local_scale)
    if lambda_end <= 0.0:
        lambda_end = 3.0 * local_scale
    if lambda_start < lambda_end:
        raise ValueError("lambda-start must be greater than or equal to lambda-end")

    if batch_size <= 0:
        batch_size = min(200_000, max(20_000, 30 * point_count))
    batch_size = max(512, int(batch_size))
    local_batch_size = batch_size // 2
    global_batch_size = batch_size - local_batch_size

    print(
        f"CDA optimization: {iterations} iterations, batch={batch_size}, "
        f"lambda={lambda_start:.4g}->{lambda_end:.4g} (normalized units)"
    )
    adam_m = np.zeros_like(uv)
    adam_v = np.zeros_like(uv)
    beta1, beta2 = 0.9, 0.999
    epsilon = 1.0e-8
    report_every = max(1, iterations // 10)

    for iteration in range(iterations):
        progress = iteration / max(1, iterations - 1)
        cur_lambda = lambda_start * (lambda_end / lambda_start) ** progress

        local_choice = rng.integers(0, len(local_i), size=local_batch_size)
        pair_i = local_i[local_choice]
        pair_j = local_j[local_choice]
        global_i, global_j = _random_pairs(global_batch_size, point_count, rng)
        pair_i = np.concatenate((pair_i, global_i))
        pair_j = np.concatenate((pair_j, global_j))

        desired = target[pair_i, pair_j]
        difference = uv[pair_i] - uv[pair_j]
        projected = np.linalg.norm(difference, axis=1)
        safe_projected = np.maximum(projected, 1.0e-8)
        safe_desired = np.maximum(desired, 0.1 * local_scale)
        ratio_squared = np.square(projected / cur_lambda)
        weight = np.exp(-np.minimum(ratio_squared, 50.0))

        # The CDA neighbourhood weights are held fixed within an iteration
        # (majorization/IRLS view); differentiating through them would reward
        # artificial stretching solely to reduce a pair's weight.
        coefficient = (
            2.0 * weight * (projected - desired) / (safe_desired * safe_projected)
        )
        pair_gradient = coefficient[:, None] * difference
        gradient = np.zeros_like(uv)
        np.add.at(gradient, pair_i, pair_gradient)
        np.add.at(gradient, pair_j, -pair_gradient)
        degree = np.bincount(
            np.concatenate((pair_i, pair_j)), minlength=point_count
        ).astype(float)
        gradient /= np.maximum(degree[:, None], 1.0)
        gradient_norm = np.linalg.norm(gradient, axis=1)
        clip = np.maximum(1.0, gradient_norm / 10.0)
        gradient /= clip[:, None]

        step = iteration + 1
        adam_m = beta1 * adam_m + (1.0 - beta1) * gradient
        adam_v = beta2 * adam_v + (1.0 - beta2) * np.square(gradient)
        corrected_m = adam_m / (1.0 - beta1**step)
        corrected_v = adam_v / (1.0 - beta2**step)
        uv -= learning_rate * corrected_m / (np.sqrt(corrected_v) + epsilon)
        uv -= uv.mean(axis=0, keepdims=True)

        if iteration == 0 or (iteration + 1) % report_every == 0 or iteration + 1 == iterations:
            stress = float(
                np.sum(weight * np.square(projected - desired) / safe_desired)
                / max(float(np.sum(weight)), epsilon)
            )
            active = float(np.mean(weight > np.exp(-1.0)))
            print(
                f"  iteration {iteration + 1:4d}/{iterations}: "
                f"lambda={cur_lambda:.4g}, stress={stress:.5g}, "
                f"active pairs={100.0 * active:.1f}%"
            )

    uv *= global_scale
    if use_gpu:
        gpu_stages = "GPU kNN/MDS" if initialization == "mds" else "GPU kNN"
        compute_description = f"{gpu_stages} + CPU Dijkstra/optimization, {device_name}"
    else:
        compute_description = "CPU"
    backend = f"CDA ({compute_description}, {iterations} iterations)"
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
        "--max-points", type=int, default=4000,
        help="Maximum total samples (default: 4000)",
    )
    parser.add_argument("--iterations", type=int, default=500, help="CDA iterations")
    parser.add_argument(
        "--init", choices=("pca", "mds", "random"), default="pca",
        help="2-D initialization; PCA is most likely to nucleate a cylindrical tear",
    )
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="Pairs per CDA iteration; 0 selects automatically",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.03,
        help="Adam learning rate in normalized coordinates (default: 0.03)",
    )
    parser.add_argument(
        "--lambda-start", type=float, default=0.0,
        help="Initial normalized CDA radius; 0 selects automatically",
    )
    parser.add_argument(
        "--lambda-end", type=float, default=0.0,
        help="Final normalized CDA radius; 0 selects automatically",
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
        "--gradient-out", default="cda_boundary_directions.csv",
        help="Output CSV for point-cloud boundary directions",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out", default="cda_points.csv", help="Output CSV")
    parser.add_argument("--save-figure", default="", help="Optional output PNG")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stl_path = Path(args.stl) if args.stl else isomap.select_stl_file()
    if stl_path is None:
        raise SystemExit("No STL model selected")
    print(f"Input STL: {stl_path.resolve()}")

    # Keep STL -> point-cloud conversion exactly aligned with Isomap.py.
    mesh = isomap.load_stl(stl_path)
    boundary_count = len(
        isomap.ordered_boundary_loops(np.asarray(mesh.faces, dtype=np.int64))
    )
    print(f"Boundary loops in input mesh: {boundary_count}")
    print("CDA: no explicit seam cut; sampling the original STL surface.")
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

    print("Detecting boundary points from point-cloud neighborhoods only ...")
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
    uv, backend = run_cda(
        points,
        neighbors=args.neighbors,
        iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_start=args.lambda_start,
        lambda_end=args.lambda_end,
        seed=args.seed,
        prefer_gpu=not args.cpu,
        initialization=args.init,
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
