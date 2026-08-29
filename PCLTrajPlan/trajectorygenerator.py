#!/usr/bin/env python3
"""Generate segmented, equally sampled 3-D spray trajectories.

This module is a post-processing front end for :mod:`trajplanpcl`.  It keeps
coverage passes and their connecting transitions as separate trajectory
segments, estimates consistently oriented point-cloud normals, offsets the
surface path by the stand-off distance, and exports both a descriptive CSV
and a 12-column PAQ-style text file.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    from . import trajplanpcl
except ImportError:  # Allow ``py trajectorygenerator.py``.
    import trajplanpcl


@dataclass
class TrajectoryParameters:
    """Parameters controlling surface planning and spray-path generation."""

    density: float = 1.0
    max_points: int = 4000
    isomap_neighbors: int = 12
    normal_neighbors: int = 24
    boundary_angle_degrees: float = 135.0
    trajectory_spacing: float = 0.0
    point_spacing: float = 10.0
    inverse_neighbors: int = 12
    spray_distance: float = 50.0
    speed: float = 50.0
    seed: int = 0
    prefer_gpu: bool = True

    def validate(self) -> None:
        if self.density <= 0.0:
            raise ValueError("density must be positive")
        if self.max_points != 0 and self.max_points < 4:
            raise ValueError("max_points must be 0 or at least 4")
        if self.isomap_neighbors < 2:
            raise ValueError("isomap_neighbors must be at least 2")
        if self.normal_neighbors < 3:
            raise ValueError("normal_neighbors must be at least 3")
        if self.point_spacing <= 0.0:
            raise ValueError("point_spacing must be positive")
        if self.spray_distance < 0.0:
            raise ValueError("spray_distance must not be negative")
        if self.speed <= 0.0:
            raise ValueError("speed must be positive")


@dataclass
class TrajectoryResult:
    """All geometry and metadata needed by the UI and thickness simulator."""

    source_path: Path
    cloud_points: np.ndarray
    cloud_normals: np.ndarray
    surface_points: np.ndarray
    surface_uv_points: np.ndarray
    surface_normals: np.ndarray
    nozzle_points: np.ndarray
    spray_directions: np.ndarray
    quaternions_xyzw: np.ndarray
    trajectory_ids: np.ndarray
    point_ids: np.ndarray
    trajectory_types: np.ndarray
    spray_on: np.ndarray
    speeds: np.ndarray
    cumulative_time: np.ndarray
    segment_slices: List[slice]
    uv_points: np.ndarray
    actual_trajectory_spacing: float
    backend: str

    @property
    def segment_count(self) -> int:
        return len(self.segment_slices)

    @property
    def coverage_count(self) -> int:
        return int(np.count_nonzero([
            self.trajectory_types[item.start] == "spray"
            for item in self.segment_slices
        ]))

    def flipped_normals(self) -> "TrajectoryResult":
        """Return a copy with all normals reversed and spray poses rebuilt."""
        normals = -self.surface_normals
        cloud_normals = -self.cloud_normals
        nozzle_points = self.surface_points + (
            np.linalg.norm(self.nozzle_points - self.surface_points, axis=1)[:, None]
            * normals
        )
        directions = spray_directions_from_offset(self.surface_points, nozzle_points)
        quaternions = build_tool_quaternions(
            nozzle_points, directions, self.segment_slices
        )
        times = cumulative_travel_time(nozzle_points, self.speeds)
        return TrajectoryResult(
            source_path=self.source_path,
            cloud_points=self.cloud_points,
            cloud_normals=cloud_normals,
            surface_points=self.surface_points,
            surface_uv_points=self.surface_uv_points,
            surface_normals=normals,
            nozzle_points=nozzle_points,
            spray_directions=directions,
            quaternions_xyzw=quaternions,
            trajectory_ids=self.trajectory_ids,
            point_ids=self.point_ids,
            trajectory_types=self.trajectory_types,
            spray_on=self.spray_on,
            speeds=self.speeds,
            cumulative_time=times,
            segment_slices=self.segment_slices,
            uv_points=self.uv_points,
            actual_trajectory_spacing=self.actual_trajectory_spacing,
            backend=self.backend,
        )


def load_input_points(
    path: str | Path, parameters: TrajectoryParameters
) -> Tuple[np.ndarray, np.ndarray]:
    """Load PLY points or sample an STL using the existing planner helpers."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".ply":
        points = trajplanpcl.load_ply_points(input_path)
        points = trajplanpcl.limit_points_uniformly(
            points, parameters.max_points, parameters.seed
        )
        face_ids = np.full(len(points), -1, dtype=np.int64)
    elif suffix == ".stl":
        mesh = trajplanpcl.load_stl(input_path)
        rng = np.random.default_rng(parameters.seed)
        points, face_ids = trajplanpcl.sample_surface_uniform_density(
            mesh,
            parameters.density,
            rng,
            max_points=parameters.max_points,
            boundary_multiplier=0.0,
        )
    else:
        raise ValueError("Input must be a PLY point cloud or STL model")
    if len(points) < 4:
        raise ValueError("At least four input points are required")
    return np.asarray(points, dtype=float), np.asarray(face_ids, dtype=np.int64)


def _normalise_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, np.finfo(float).eps)


def spray_directions_from_offset(
    surface_points: np.ndarray,
    nozzle_points: np.ndarray,
) -> np.ndarray:
    """Return unit spray axes opposite to the surface-to-nozzle offset.

    The nozzle trajectory is produced by offsetting the surface trajectory.
    Consequently the physical spray direction must point back from every
    nozzle pose to its corresponding surface point; it is not a fixed world
    direction such as ``[0, 0, -1]``.
    """
    surface = np.asarray(surface_points, dtype=float)
    nozzle = np.asarray(nozzle_points, dtype=float)
    if surface.shape != nozzle.shape or surface.ndim != 2 or surface.shape[1] != 3:
        raise ValueError("surface_points and nozzle_points must have shape (N, 3)")
    offsets = nozzle - surface
    lengths = np.linalg.norm(offsets, axis=1, keepdims=True)
    if np.any(lengths <= np.finfo(float).eps):
        raise ValueError("Every nozzle point must have a non-zero surface offset")
    return -offsets / lengths


def orient_normals_consistently(
    points: np.ndarray,
    normals: np.ndarray,
    neighbors: int = 10,
) -> np.ndarray:
    """Propagate normal signs over a kNN graph.

    PCA determines a normal axis but not its sign.  Each connected component
    is seeded at its highest-Z point, then breadth-first propagation makes
    neighboring normals agree.  The UI can reverse the complete field later.
    """
    points = np.asarray(points, dtype=float)
    result = _normalise_rows(np.asarray(normals, dtype=float).copy())
    if len(points) != len(result):
        raise ValueError("points and normals must have equal row counts")
    if len(points) < 2:
        return result
    k = min(max(2, int(neighbors)) + 1, len(points))
    _, indices = cKDTree(points).query(points, k=k)
    indices = np.asarray(indices)
    if indices.ndim == 1:
        indices = indices[:, None]

    adjacency: List[List[int]] = [[] for _ in range(len(points))]
    for index, row in enumerate(indices):
        for neighbor in row[1:]:
            neighbor = int(neighbor)
            adjacency[index].append(neighbor)
            adjacency[neighbor].append(index)

    visited = np.zeros(len(points), dtype=bool)
    remaining = set(range(len(points)))
    while remaining:
        # A high-Z seed makes the initial sign deterministic for ordinary
        # height-like workpieces without claiming to infer physical outside.
        seed = max(remaining, key=lambda item: points[item, 2])
        if result[seed, 2] < 0.0:
            result[seed] *= -1.0
        queue = [seed]
        visited[seed] = True
        remaining.remove(seed)
        position = 0
        while position < len(queue):
            current = queue[position]
            position += 1
            for neighbor in adjacency[current]:
                if visited[neighbor]:
                    continue
                if float(np.dot(result[current], result[neighbor])) < 0.0:
                    result[neighbor] *= -1.0
                visited[neighbor] = True
                remaining.discard(neighbor)
                queue.append(neighbor)
    return result


def estimate_query_normals(
    cloud_points: np.ndarray,
    query_points: np.ndarray,
    neighbors: int = 24,
) -> np.ndarray:
    """Estimate PCA normals at arbitrary surface-path positions."""
    if len(cloud_points) < 4:
        raise ValueError("At least four cloud points are required")
    k = min(max(3, int(neighbors)), len(cloud_points))
    _, neighborhoods = cKDTree(cloud_points).query(query_points, k=k)
    neighborhoods = np.asarray(neighborhoods)
    if neighborhoods.ndim == 1:
        neighborhoods = neighborhoods[:, None]
    local = cloud_points[neighborhoods]
    centered = local - local.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    return _normalise_rows(eigenvectors[:, :, 0])


def _map_segment_to_xyz(
    start_uv: np.ndarray,
    end_uv: np.ndarray,
    sample_uv: np.ndarray,
    sample_xyz: np.ndarray,
    point_spacing: float,
    inverse_neighbors: int,
    trajectory_spacing: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map a UV line to 3-D and retain UV coordinates after 3-D resampling."""
    uv_length = float(np.linalg.norm(end_uv - start_uv))
    fine_step = min(point_spacing / 3.0, trajectory_spacing / 8.0)
    fine_step = max(fine_step, uv_length / 2000.0, 1.0e-8)
    count = max(2, int(np.ceil(uv_length / fine_step)) + 1)
    fractions = np.linspace(0.0, 1.0, count)
    dense_uv = start_uv + fractions[:, None] * (end_uv - start_uv)
    dense_xyz = trajplanpcl.inverse_map_uv_to_xyz(
        dense_uv, sample_uv, sample_xyz, neighbors=inverse_neighbors
    )
    distances = np.linalg.norm(np.diff(dense_xyz, axis=0), axis=1)
    keep = np.concatenate(([True], distances > 1.0e-12))
    dense_xyz = dense_xyz[keep]
    dense_uv = dense_uv[keep]
    if len(dense_xyz) < 2:
        return dense_xyz.copy(), dense_uv.copy()
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(
        np.diff(dense_xyz, axis=0), axis=1
    ))))
    length = float(cumulative[-1])
    if length <= 1.0e-12:
        return dense_xyz[:1].copy(), dense_uv[:1].copy()
    interval_count = max(1, int(np.ceil(length / point_spacing)))
    targets = np.linspace(0.0, length, interval_count + 1)
    xyz = np.column_stack([
        np.interp(targets, cumulative, dense_xyz[:, axis]) for axis in range(3)
    ])
    mapped_uv = np.column_stack([
        np.interp(targets, cumulative, dense_uv[:, axis]) for axis in range(2)
    ])
    return xyz, mapped_uv


def build_tool_quaternions(
    nozzle_points: np.ndarray,
    spray_directions: np.ndarray,
    segment_slices: Sequence[slice],
) -> np.ndarray:
    """Build XYZW quaternions with local +Z along the spray direction."""
    matrices = np.empty((len(nozzle_points), 3, 3), dtype=float)
    fallback_x = np.array([1.0, 0.0, 0.0])
    for segment in segment_slices:
        segment_points = nozzle_points[segment]
        if len(segment_points) == 1:
            tangents = fallback_x[None, :]
        else:
            tangents = np.gradient(segment_points, axis=0)
        for local_index, tangent in enumerate(tangents):
            index = segment.start + local_index
            z_axis = spray_directions[index]
            x_axis = tangent - np.dot(tangent, z_axis) * z_axis
            if np.linalg.norm(x_axis) <= 1.0e-10:
                reference = np.array([1.0, 0.0, 0.0])
                if abs(float(np.dot(reference, z_axis))) > 0.9:
                    reference = np.array([0.0, 1.0, 0.0])
                x_axis = reference - np.dot(reference, z_axis) * z_axis
            x_axis /= max(float(np.linalg.norm(x_axis)), np.finfo(float).eps)
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= max(float(np.linalg.norm(y_axis)), np.finfo(float).eps)
            x_axis = np.cross(y_axis, z_axis)
            matrices[index] = np.column_stack((x_axis, y_axis, z_axis))
    quaternions = Rotation.from_matrix(matrices).as_quat()
    for index in range(1, len(quaternions)):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
            quaternions[index] *= -1.0
    return quaternions


def cumulative_travel_time(points: np.ndarray, speeds: np.ndarray) -> np.ndarray:
    """Return monotonically increasing time using trapezoidal segment speed."""
    if len(points) == 0:
        return np.empty(0, dtype=float)
    if np.any(speeds <= 0.0):
        raise ValueError("All speeds must be positive")
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    segment_speeds = 0.5 * (speeds[:-1] + speeds[1:])
    return np.concatenate(([0.0], np.cumsum(distances / segment_speeds)))


def generate_trajectory(
    path: str | Path,
    parameters: Optional[TrajectoryParameters] = None,
) -> TrajectoryResult:
    """Run the complete segmented surface-to-nozzle trajectory workflow."""
    parameters = parameters or TrajectoryParameters()
    parameters.validate()
    input_path = Path(path)
    points, _face_ids = load_input_points(input_path, parameters)

    boundary_mask = trajplanpcl.point_cloud_boundary_mask(
        points,
        neighbors=parameters.normal_neighbors,
        angle_threshold_degrees=parameters.boundary_angle_degrees,
    )
    (
        raw_cloud_normals,
        _minimum_directions,
        _gradient_norms,
        _minimum_values,
        global_direction,
        normal_backend,
    ) = trajplanpcl.estimate_global_minimum_gradient(
        points,
        neighbors=parameters.normal_neighbors,
        prefer_gpu=parameters.prefer_gpu,
    )
    cloud_normals = orient_normals_consistently(
        points, raw_cloud_normals, neighbors=min(10, parameters.normal_neighbors)
    )
    uv, isomap_backend = trajplanpcl.run_isomap(
        points,
        parameters.isomap_neighbors,
        prefer_gpu=parameters.prefer_gpu,
    )
    travel_direction = trajplanpcl.project_global_direction_to_uv(
        points,
        uv,
        boundary_mask,
        global_direction,
        neighbors=parameters.normal_neighbors,
    )
    endpoints_uv, actual_spacing = trajplanpcl.generate_coverage_trajectory(
        uv, travel_direction, spacing=parameters.trajectory_spacing
    )

    segments: List[np.ndarray] = []
    uv_segments: List[np.ndarray] = []
    segment_types: List[str] = []
    for segment_index in range(len(endpoints_uv) - 1):
        segment, segment_uv = _map_segment_to_xyz(
            endpoints_uv[segment_index],
            endpoints_uv[segment_index + 1],
            uv,
            points,
            parameters.point_spacing,
            parameters.inverse_neighbors,
            actual_spacing,
        )
        if len(segment) == 0:
            continue
        segments.append(segment)
        uv_segments.append(segment_uv)
        segment_types.append("spray" if segment_index % 2 == 0 else "transition")
    if not segments:
        raise ValueError("Trajectory planning produced no non-empty 3-D segments")

    surface_points = np.vstack(segments)
    surface_uv_points = np.vstack(uv_segments)
    segment_slices: List[slice] = []
    trajectory_ids: List[int] = []
    point_ids: List[int] = []
    trajectory_types: List[str] = []
    cursor = 0
    for trajectory_id, (segment, segment_type) in enumerate(
        zip(segments, segment_types), start=1
    ):
        stop = cursor + len(segment)
        segment_slices.append(slice(cursor, stop))
        trajectory_ids.extend([trajectory_id] * len(segment))
        point_ids.extend(range(len(segment)))
        trajectory_types.extend([segment_type] * len(segment))
        cursor = stop

    raw_surface_normals = estimate_query_normals(
        points, surface_points, neighbors=parameters.normal_neighbors
    )
    # Orient query normals to their closest already oriented cloud normal.
    nearest = cKDTree(points).query(surface_points, k=1)[1]
    surface_normals = raw_surface_normals.copy()
    disagree = np.einsum(
        "ij,ij->i", surface_normals, cloud_normals[np.asarray(nearest)]
    ) < 0.0
    surface_normals[disagree] *= -1.0
    for index in range(1, len(surface_normals)):
        if np.dot(surface_normals[index - 1], surface_normals[index]) < 0.0:
            surface_normals[index] *= -1.0

    nozzle_points = surface_points + parameters.spray_distance * surface_normals
    spray_directions = spray_directions_from_offset(surface_points, nozzle_points)
    quaternions = build_tool_quaternions(
        nozzle_points, spray_directions, segment_slices
    )
    type_array = np.asarray(trajectory_types, dtype="U10")
    spray_on = (type_array == "spray").astype(np.int64)
    speeds = np.full(len(surface_points), parameters.speed, dtype=float)
    cumulative_time = cumulative_travel_time(nozzle_points, speeds)

    return TrajectoryResult(
        source_path=input_path.resolve(),
        cloud_points=points,
        cloud_normals=cloud_normals,
        surface_points=surface_points,
        surface_uv_points=surface_uv_points,
        surface_normals=surface_normals,
        nozzle_points=nozzle_points,
        spray_directions=spray_directions,
        quaternions_xyzw=quaternions,
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
        point_ids=np.asarray(point_ids, dtype=np.int64),
        trajectory_types=type_array,
        spray_on=spray_on,
        speeds=speeds,
        cumulative_time=cumulative_time,
        segment_slices=segment_slices,
        uv_points=uv,
        actual_trajectory_spacing=float(actual_spacing),
        backend=f"normals: {normal_backend}; Isomap: {isomap_backend}",
    )


def save_trajectory_csv(path: str | Path, result: TrajectoryResult) -> None:
    """Save full surface/nozzle geometry and segment metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "global_point_id", "trajectory_id", "trajectory_type", "point_id",
            "surface_x", "surface_y", "surface_z",
            "surface_u", "surface_v",
            "normal_x", "normal_y", "normal_z",
            "nozzle_x", "nozzle_y", "nozzle_z",
            "spray_dir_x", "spray_dir_y", "spray_dir_z",
            "qx", "qy", "qz", "qw", "speed", "cumulative_time", "spray_on",
        ])
        for index in range(len(result.surface_points)):
            writer.writerow([
                index,
                int(result.trajectory_ids[index]),
                result.trajectory_types[index],
                int(result.point_ids[index]),
                *result.surface_points[index],
                *result.surface_uv_points[index],
                *result.surface_normals[index],
                *result.nozzle_points[index],
                *result.spray_directions[index],
                *result.quaternions_xyzw[index],
                result.speeds[index],
                result.cumulative_time[index],
                int(result.spray_on[index]),
            ])


def save_paq(path: str | Path, result: TrajectoryResult) -> None:
    """Save the 12-column PAQ layout inferred from ``new_PAQ.txt``.

    Columns are XYZ, quaternion XYZW, speed, reserved zero, cumulative time,
    spray flag, and reserved zero.  The spray flag is 1 on coverage passes and
    0 on transitions.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for index in range(len(result.nozzle_points)):
            values = [
                *result.nozzle_points[index],
                *result.quaternions_xyzw[index],
                result.speeds[index],
                0.0,
                result.cumulative_time[index],
                int(result.spray_on[index]),
                0.0,
            ]
            stream.write(
                f"{values[0]:.3f} {values[1]:.3f} {values[2]:.3f} "
                f"{values[3]:.6f} {values[4]:.6f} {values[5]:.6f} {values[6]:.6f} "
                f"{values[7]:.3f} {values[8]:.3f} {values[9]:.6f} "
                f"{int(values[10])} {values[11]:.3f}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="")
    parser.add_argument("--trajectory-spacing", type=float, default=0.0)
    parser.add_argument("--point-spacing", type=float, default=10.0)
    parser.add_argument("--spray-distance", type=float, default=50.0)
    parser.add_argument("--speed", type=float, default=50.0)
    parser.add_argument("--normal-neighbors", type=int, default=24)
    parser.add_argument("--isomap-neighbors", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=4000)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--flip-normals", action="store_true")
    parser.add_argument("--csv", default="segmented_spray_trajectory.csv")
    parser.add_argument("--paq", default="segmented_spray_trajectory.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = Path(args.input) if args.input else trajplanpcl.select_input_file()
    if selected is None:
        raise SystemExit("No input file selected")
    parameters = TrajectoryParameters(
        density=args.density,
        max_points=args.max_points,
        isomap_neighbors=args.isomap_neighbors,
        normal_neighbors=args.normal_neighbors,
        trajectory_spacing=args.trajectory_spacing,
        point_spacing=args.point_spacing,
        spray_distance=args.spray_distance,
        speed=args.speed,
        prefer_gpu=not args.cpu,
    )
    print(f"Generating trajectory from: {selected.resolve()}")
    result = generate_trajectory(selected, parameters)
    if args.flip_normals:
        result = result.flipped_normals()
    save_trajectory_csv(args.csv, result)
    save_paq(args.paq, result)
    print(f"Segments: {result.segment_count} ({result.coverage_count} spray passes)")
    print(f"Points: {len(result.surface_points)}")
    print(f"Actual coverage spacing: {result.actual_trajectory_spacing:.6g}")
    print(f"Backend: {result.backend}")
    print(f"CSV: {Path(args.csv).resolve()}")
    print(f"PAQ: {Path(args.paq).resolve()}")


if __name__ == "__main__":
    main()
