#!/usr/bin/env python3
"""Predict accumulated coating thickness on a 3-D point cloud.

The supplied model is evaluated on the Isomap UV plane.  Its ``x`` coordinate
is the 2-D cross-track distance to the trajectory centerline, spray incidence
is perpendicular, and spray height is a fixed input.  Computed values map
back to the 3-D cloud through the unchanged point indices.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

# Matplotlib otherwise tries to create a font-cache lock in the user's home
# directory.  Keep all generated runtime state inside the project workspace.
RUNTIME_CACHE = Path(__file__).resolve().parents[1] / ".runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, colors
from scipy.spatial import cKDTree

try:
    from . import trajplanpcl
    from .trajectorygenerator import (
        TrajectoryParameters,
        TrajectoryResult,
        load_input_points,
        orient_normals_consistently,
    )
except ImportError:  # Allow direct script execution.
    import trajplanpcl
    from trajectorygenerator import (
        TrajectoryParameters,
        TrajectoryResult,
        load_input_points,
        orient_normals_consistently,
    )


@dataclass
class ThicknessParameters:
    """Parameters in the coating-thickness formula."""

    gamma_max: float = 1.0
    spray_height: float = 50.0
    h_opt: float = 50.0
    sigma_h: float = 10.0
    angle_power_k: float = 1.0
    constant_c: float = 1.0
    cone_half_angle_degrees: float = 30.0
    beta: float = 1.0
    height_sigma_cutoff: float = 4.0
    include_transitions: bool = False

    def validate(self) -> None:
        if self.gamma_max < 0.0:
            raise ValueError("gamma_max must not be negative")
        if self.spray_height < 0.0:
            raise ValueError("spray_height must not be negative")
        if self.h_opt <= 0.0:
            raise ValueError("h_opt must be positive")
        if self.sigma_h <= 0.0:
            raise ValueError("sigma_h must be positive")
        if self.angle_power_k < 0.0:
            raise ValueError("angle_power_k must not be negative")
        if self.constant_c < 0.0:
            raise ValueError("constant_c must not be negative")
        if not 0.0 < self.cone_half_angle_degrees < 89.0:
            raise ValueError("cone_half_angle_degrees must be between 0 and 89")
        if self.beta <= -1.0:
            raise ValueError("beta must be greater than -1")
        if self.height_sigma_cutoff <= 0.0:
            raise ValueError("height_sigma_cutoff must be positive")


@dataclass
class ThicknessResult:
    cloud_points: np.ndarray
    cloud_normals: np.ndarray
    thickness: np.ndarray
    contribution_counts: np.ndarray
    parameters: ThicknessParameters

    @property
    def minimum(self) -> float:
        return float(np.min(self.thickness)) if len(self.thickness) else 0.0

    @property
    def maximum(self) -> float:
        return float(np.max(self.thickness)) if len(self.thickness) else 0.0

    @property
    def mean(self) -> float:
        return float(np.mean(self.thickness)) if len(self.thickness) else 0.0

    @property
    def standard_deviation(self) -> float:
        return float(np.std(self.thickness)) if len(self.thickness) else 0.0

    @property
    def uniformity(self) -> float:
        """Return 1 - coefficient of variation, clamped to [0, 1]."""
        if self.mean <= np.finfo(float).eps:
            return 0.0
        return float(np.clip(1.0 - self.standard_deviation / self.mean, 0.0, 1.0))


@dataclass
class TrajectoryArrays:
    """Small trajectory representation accepted by the standalone CLI."""

    nozzle_points: np.ndarray
    surface_uv_points: Optional[np.ndarray]
    spray_directions: np.ndarray
    speeds: np.ndarray
    trajectory_ids: np.ndarray
    trajectory_types: np.ndarray
    spray_on: np.ndarray


def _trajectory_arrays(
    trajectory: TrajectoryResult | TrajectoryArrays,
) -> TrajectoryArrays:
    if isinstance(trajectory, TrajectoryResult):
        surface_uv_points = np.asarray(trajectory.surface_uv_points, dtype=float)
    else:
        surface_uv_points = (
            None if trajectory.surface_uv_points is None
            else np.asarray(trajectory.surface_uv_points, dtype=float)
        )
    return TrajectoryArrays(
        nozzle_points=np.asarray(trajectory.nozzle_points, dtype=float),
        surface_uv_points=surface_uv_points,
        spray_directions=np.asarray(trajectory.spray_directions, dtype=float),
        speeds=np.asarray(trajectory.speeds, dtype=float),
        trajectory_ids=np.asarray(trajectory.trajectory_ids, dtype=np.int64),
        trajectory_types=np.asarray(trajectory.trajectory_types),
        spray_on=np.asarray(trajectory.spray_on, dtype=np.int64),
    )


def path_quadrature_weights(trajectory: TrajectoryArrays) -> np.ndarray:
    """Assign trapezoidal path-length weights independently to every segment."""
    count = len(trajectory.nozzle_points)
    weights = np.zeros(count, dtype=float)
    for trajectory_id in np.unique(trajectory.trajectory_ids):
        indices = np.flatnonzero(trajectory.trajectory_ids == trajectory_id)
        if len(indices) < 2:
            continue
        distances = np.linalg.norm(
            np.diff(trajectory.nozzle_points[indices], axis=0), axis=1
        )
        weights[indices[0]] = 0.5 * distances[0]
        weights[indices[-1]] = 0.5 * distances[-1]
        if len(indices) > 2:
            weights[indices[1:-1]] = 0.5 * (distances[:-1] + distances[1:])
    return weights


def spray_model_values(
    radial_distance: np.ndarray,
    axial_height: np.ndarray,
    incidence_cosine: np.ndarray,
    speed: float | np.ndarray,
    parameters: ThicknessParameters,
) -> np.ndarray:
    """Evaluate H(x, h, v, theta) for already-valid cone candidates."""
    alpha = np.deg2rad(parameters.cone_half_angle_degrees)
    cone_radius = axial_height * np.tan(alpha)
    profile_base = np.clip(
        1.0 - np.square(radial_distance / np.maximum(cone_radius, 1.0e-12)),
        0.0,
        1.0,
    )
    height_factor = np.exp(
        -np.square(axial_height - parameters.h_opt)
        / (2.0 * parameters.sigma_h ** 2)
    )
    angle_factor = np.power(np.clip(incidence_cosine, 0.0, 1.0),
                            parameters.angle_power_k)
    radial_factor = np.power(profile_base, 0.5 * (parameters.beta + 1.0))
    return (
        parameters.gamma_max
        * height_factor
        * angle_factor
        * parameters.constant_c
        / (np.maximum(axial_height, 1.0e-12) * speed)
        * radial_factor
    )


def predict_thickness(
    cloud_points: np.ndarray,
    cloud_normals: np.ndarray,
    trajectory: TrajectoryResult | TrajectoryArrays,
    parameters: Optional[ThicknessParameters] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cloud_uv_points: Optional[np.ndarray] = None,
) -> ThicknessResult:
    """Accumulate coating bands in UV and map values back by point index.

    The deposition model treats spraying as perpendicular to the UV plane,
    so incidence cosine is always one and axial height is the fixed
    ``parameters.spray_height``.  No 3-D height or angle is inferred here.
    """
    parameters = parameters or ThicknessParameters()
    parameters.validate()
    points = np.asarray(cloud_points, dtype=float)
    normals = np.asarray(cloud_normals, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("cloud_points must have shape (N, 3)")
    if normals.shape != points.shape:
        raise ValueError("cloud_normals must match cloud_points")
    data = _trajectory_arrays(trajectory)
    count = len(data.nozzle_points)
    if not all(len(array) == count for array in (
        data.spray_directions,
        data.speeds,
        data.trajectory_ids,
        data.trajectory_types,
        data.spray_on,
    )):
        raise ValueError("Trajectory arrays have inconsistent lengths")
    if np.any(data.speeds <= 0.0):
        raise ValueError("Trajectory speed must be positive")
    if cloud_uv_points is None and isinstance(trajectory, TrajectoryResult):
        cloud_uv_points = trajectory.uv_points
    if cloud_uv_points is None:
        raise ValueError("cloud_uv_points are required for UV thickness prediction")
    point_uv = np.asarray(cloud_uv_points, dtype=float)
    path_uv = data.surface_uv_points
    if point_uv.shape != (len(points), 2):
        raise ValueError("cloud_uv_points must match cloud_points with shape (N, 2)")
    if path_uv is None or path_uv.shape != (count, 2):
        raise ValueError("surface_uv_points must match the trajectory with shape (M, 2)")
    thickness = np.zeros(len(points), dtype=float)
    contribution_counts = np.zeros(len(points), dtype=np.int64)
    if not len(points):
        return ThicknessResult(points, normals, thickness, contribution_counts, parameters)

    alpha = np.deg2rad(parameters.cone_half_angle_degrees)
    fixed_height = parameters.spray_height
    trajectory_groups = []
    for trajectory_id in np.unique(data.trajectory_ids):
        indices = np.flatnonzero(data.trajectory_ids == trajectory_id)
        if len(indices) < 2:
            continue
        is_spray = bool(np.any(data.spray_on[indices] != 0))
        if is_spray or parameters.include_transitions:
            trajectory_groups.append(indices)
    if not trajectory_groups:
        return ThicknessResult(points, normals, thickness, contribution_counts, parameters)

    total = len(trajectory_groups)
    update_stride = max(1, total // 100)
    for progress, indices in enumerate(trajectory_groups, start=1):
        path = path_uv[indices]
        path_speeds = data.speeds[indices]

        # Find each cloud sample's closest point on the complete 2-D UV path.
        # The remaining UV distance is the deposition model's cross-track x.
        best_distance_squared = np.full(len(points), np.inf, dtype=float)
        best_cross_track = np.zeros(len(points), dtype=float)
        best_speed = np.zeros(len(points), dtype=float)

        for local_index in range(len(path) - 1):
            start = path[local_index]
            end = path[local_index + 1]
            segment = end - start
            segment_squared = float(np.dot(segment, segment))
            if segment_squared <= 1.0e-20:
                continue
            fraction = np.clip(
                ((point_uv - start) @ segment) / segment_squared,
                0.0,
                1.0,
            )
            closest = start + fraction[:, None] * segment
            relative = point_uv - closest
            distance_squared = np.einsum("ij,ij->i", relative, relative)
            speed = (
                (1.0 - fraction) * path_speeds[local_index]
                + fraction * path_speeds[local_index + 1]
            )
            update = distance_squared < best_distance_squared
            if np.any(update):
                best_distance_squared[update] = distance_squared[update]
                best_cross_track[update] = np.sqrt(distance_squared[update])
                best_speed[update] = speed[update]

        cone_radius = fixed_height * np.tan(alpha)
        valid = (
            np.isfinite(best_distance_squared)
            & (best_cross_track < cone_radius)
            & (best_speed > 0.0)
        )
        if np.any(valid):
            selected = np.flatnonzero(valid)
            values = spray_model_values(
                best_cross_track[valid],
                np.full(np.count_nonzero(valid), fixed_height),
                np.ones(np.count_nonzero(valid)),
                best_speed[valid],
                parameters,
            )
            thickness[selected] += values
            contribution_counts[selected] += 1
        if progress_callback and (progress == total or progress % update_stride == 0):
            progress_callback(progress, total)

    return ThicknessResult(points, normals, thickness, contribution_counts, parameters)


def load_trajectory_csv(path: str | Path) -> TrajectoryArrays:
    """Load a CSV produced by ``trajectorygenerator.save_trajectory_csv``."""
    fields = {
        "nozzle_x": [], "nozzle_y": [], "nozzle_z": [],
        "surface_u": [], "surface_v": [],
        "spray_dir_x": [], "spray_dir_y": [], "spray_dir_z": [],
        "speed": [], "trajectory_id": [], "trajectory_type": [], "spray_on": [],
    }
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(fields).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Trajectory CSV is missing columns: {sorted(missing)}")
        for row in reader:
            for name in fields:
                fields[name].append(row[name])
    return TrajectoryArrays(
        nozzle_points=np.column_stack([
            np.asarray(fields[name], dtype=float)
            for name in ("nozzle_x", "nozzle_y", "nozzle_z")
        ]),
        surface_uv_points=np.column_stack([
            np.asarray(fields[name], dtype=float)
            for name in ("surface_u", "surface_v")
        ]),
        spray_directions=np.column_stack([
            np.asarray(fields[name], dtype=float)
            for name in ("spray_dir_x", "spray_dir_y", "spray_dir_z")
        ]),
        speeds=np.asarray(fields["speed"], dtype=float),
        trajectory_ids=np.asarray(fields["trajectory_id"], dtype=np.int64),
        trajectory_types=np.asarray(fields["trajectory_type"], dtype="U10"),
        spray_on=np.asarray(fields["spray_on"], dtype=np.int64),
    )


def estimate_cloud_normals(points: np.ndarray, neighbors: int = 24) -> np.ndarray:
    """Estimate and consistently orient normals for standalone simulation."""
    normals, _directions, _gradient, _minimum, _global, _backend = (
        trajplanpcl.estimate_global_minimum_gradient(
            points, neighbors=neighbors, prefer_gpu=False
        )
    )
    return orient_normals_consistently(points, normals, neighbors=min(neighbors, 10))


def save_thickness_csv(path: str | Path, result: ThicknessResult) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "point_id", "x", "y", "z", "normal_x", "normal_y", "normal_z",
            "thickness", "contribution_count",
        ])
        for index in range(len(result.cloud_points)):
            writer.writerow([
                index,
                *result.cloud_points[index],
                *result.cloud_normals[index],
                result.thickness[index],
                int(result.contribution_counts[index]),
            ])


def save_colored_ply(path: str | Path, result: ThicknessResult) -> None:
    """Write an ASCII PLY whose RGB values use the Turbo thickness map."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lower, upper = result.minimum, result.maximum
    scale = colors.Normalize(vmin=lower, vmax=upper if upper > lower else lower + 1.0)
    rgb = (colormaps["turbo"](scale(result.thickness))[:, :3] * 255.0).astype(np.uint8)
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(result.cloud_points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("property float thickness\nend_header\n")
        for point, color, value in zip(result.cloud_points, rgb, result.thickness):
            stream.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {value:.9g}\n"
            )


def plot_thickness(
    result: ThicknessResult,
    trajectory: Optional[TrajectoryResult | TrajectoryArrays] = None,
    output: str | Path = "",
    show: bool = True,
) -> None:
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        *result.cloud_points.T,
        c=result.thickness,
        cmap="turbo",
        s=6,
        depthshade=False,
    )
    if trajectory is not None:
        data = _trajectory_arrays(trajectory)
        axis.plot(*data.nozzle_points.T, color="black", linewidth=0.8,
                  label="nozzle trajectory")
        axis.legend(loc="best")
    figure.colorbar(scatter, ax=axis, shrink=0.72, pad=0.08, label="Thickness")
    axis.set_title("3-D coating thickness distribution")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_box_aspect(np.maximum(np.ptp(result.cloud_points, axis=0), 1.0e-12))
    figure.tight_layout()
    if output:
        figure.savefig(output, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cloud", help="Original PLY or STL")
    parser.add_argument("trajectory_csv", help="CSV from trajectorygenerator.py")
    parser.add_argument("--gamma-max", type=float, default=1.0)
    parser.add_argument("--spray-height", type=float, default=50.0)
    parser.add_argument("--h-opt", type=float, default=50.0)
    parser.add_argument("--sigma-h", type=float, default=10.0)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--constant-c", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=30.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--include-transitions", action="store_true")
    parser.add_argument("--normal-neighbors", type=int, default=24)
    parser.add_argument("--isomap-neighbors", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--out", default="thickness_result.csv")
    parser.add_argument("--ply", default="thickness_result.ply")
    parser.add_argument("--figure", default="thickness_heatmap.png")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_parameters = TrajectoryParameters(
        max_points=args.max_points,
        normal_neighbors=args.normal_neighbors,
    )
    points, _ = load_input_points(args.cloud, load_parameters)
    normals = estimate_cloud_normals(points, args.normal_neighbors)
    uv, _isomap_backend = trajplanpcl.run_isomap(
        points, args.isomap_neighbors, prefer_gpu=False
    )
    trajectory = load_trajectory_csv(args.trajectory_csv)
    parameters = ThicknessParameters(
        gamma_max=args.gamma_max,
        spray_height=args.spray_height,
        h_opt=args.h_opt,
        sigma_h=args.sigma_h,
        angle_power_k=args.k,
        constant_c=args.constant_c,
        cone_half_angle_degrees=args.alpha,
        beta=args.beta,
        include_transitions=args.include_transitions,
    )
    result = predict_thickness(
        points, normals, trajectory, parameters, cloud_uv_points=uv
    )
    save_thickness_csv(args.out, result)
    save_colored_ply(args.ply, result)
    plot_thickness(result, trajectory, output=args.figure, show=not args.no_show)
    print(
        f"Thickness min={result.minimum:.6g}, max={result.maximum:.6g}, "
        f"mean={result.mean:.6g}, std={result.standard_deviation:.6g}, "
        f"uniformity={result.uniformity:.4f}"
    )
    print(f"CSV: {Path(args.out).resolve()}")
    print(f"PLY: {Path(args.ply).resolve()}")
    print(f"Figure: {Path(args.figure).resolve()}")


if __name__ == "__main__":
    main()
