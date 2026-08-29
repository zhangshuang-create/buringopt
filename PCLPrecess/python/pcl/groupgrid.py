#!/usr/bin/env python3
"""STL -> point cloud -> global minimum-height-gradient direction field.

For each sampled point, a local PCA estimates the surface normal.  The
projection of the world-up vector onto that tangent plane is the local height
gradient.  The tangent direction perpendicular to it is therefore the
minimum-gradient direction.  A covariance of all direction lines supplies one
global representative direction (sign-independent).

CuPy is used for the batched 3x3 eigendecomposition and vector arithmetic when
available; the KD-tree neighborhood search remains on CPU.  The global vector
is the eigenvector that minimizes the summed squared directional gradients.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree

try:
    import cupy as cp
except Exception:
    cp = None


def load_stl(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{path} is not a non-empty triangle mesh")
    mesh = loaded.copy()
    mesh.merge_vertices()
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface_even(mesh, count, seed=seed)
    if len(points) < max(3, count // 2):
        # sample_surface_even can reject points on very large or slender faces;
        # retain its even samples and fill the requested count area-uniformly.
        extra, _ = trimesh.sample.sample_surface(mesh, count - len(points), seed=seed)
        points = np.vstack((points, extra))
    if len(points) < 3:
        raise ValueError("Could not sample at least three surface points")
    return np.asarray(points, dtype=np.float64)


def neighborhood_covariances(points: np.ndarray, neighbors: int) -> np.ndarray:
    """Return one centered 3x3 covariance matrix per point."""
    k = min(max(3, int(neighbors)), len(points))
    _, indices = cKDTree(points).query(points, k=k)
    local = points[indices]
    centered = local - local.mean(axis=1, keepdims=True)
    return np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)


def minimum_gradient_directions(
    points: np.ndarray, neighbors: int = 24, use_gpu: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Estimate normals, height gradients, and tangent minimum-gradient lines."""
    covariance = neighborhood_covariances(points, neighbors)
    xp = cp if (use_gpu and cp is not None) else np
    device = "CuPy" if xp is cp else "NumPy"
    c = xp.asarray(covariance)
    # eigh returns eigenvalues ascending: eigenvector 0 is the PCA normal,
    # eigenvector 2 is a stable tangent fallback on nearly horizontal patches.
    _, eigenvectors = xp.linalg.eigh(c)
    normals = eigenvectors[:, :, 0]
    tangent_fallback = eigenvectors[:, :, 2]
    up = xp.asarray([0.0, 0.0, 1.0])
    height_gradient = up - xp.sum(normals * up, axis=1, keepdims=True) * normals
    gradient_norm = xp.linalg.norm(height_gradient, axis=1)

    # cross(n, up) is tangent and orthogonal to the height gradient.
    directions = xp.cross(normals, up)
    direction_norm = xp.linalg.norm(directions, axis=1)
    valid = direction_norm > 1e-10
    directions = xp.where(valid[:, None], directions / xp.maximum(direction_norm[:, None], 1e-12),
                          tangent_fallback)
    directions = directions / xp.maximum(xp.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    min_directional_gradient = xp.abs(xp.sum(height_gradient * directions, axis=1))

    values = tuple(np.asarray(xp.asnumpy(value) if xp is cp else value) for value in
                   (normals, directions, height_gradient, gradient_norm,
                    min_directional_gradient))
    return (*values, device)


def global_direction(height_gradients: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Minimize the summed squared height gradient for one global direction.

    If G = mean(g_i g_i^T), then d^T G d is the mean squared directional
    gradient.  Its smallest-eigenvalue eigenvector is the exact global optimum.
    """
    scatter = height_gradients.T @ height_gradients / len(height_gradients)
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    direction = eigenvectors[:, 0]
    if direction[2] < 0:
        direction = -direction
    mean_squared_gradient = float(eigenvalues[0])
    separation = float(eigenvalues[1] / max(eigenvalues[0], 1e-12))
    return direction, mean_squared_gradient, separation


def save_csv(path: Path, points: np.ndarray, normals: np.ndarray,
             directions: np.ndarray, gradient_norm: np.ndarray,
             min_gradient: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "index", "x", "y", "z", "nx", "ny", "nz", "dx", "dy", "dz",
            "height_gradient_norm", "minimum_directional_gradient",
        ])
        for i in range(len(points)):
            writer.writerow([i, *points[i], *normals[i], *directions[i],
                             gradient_norm[i], min_gradient[i]])


def plot_result(points: np.ndarray, directions: np.ndarray, global_dir: np.ndarray,
                gradient_norm: np.ndarray, output: Path, show: bool,
                title: str, arrows: int) -> None:
    figure = plt.figure(figsize=(11, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    cloud = axis.scatter(*points.T, c=gradient_norm, cmap="viridis", s=4,
                         alpha=0.45, depthshade=False)
    selected = np.linspace(0, len(points) - 1, min(arrows, len(points)), dtype=int)
    length = 0.05 * max(float(np.ptp(points, axis=0).max()), 1.0)
    axis.quiver(points[selected, 0], points[selected, 1], points[selected, 2],
                directions[selected, 0], directions[selected, 1], directions[selected, 2],
                length=length, normalize=True, color="tab:orange", linewidth=0.8,
                alpha=0.8, label="local minimum-gradient direction")
    center = points.mean(axis=0)
    axis.quiver(*center, *global_dir, length=3.0 * length, normalize=True,
                color="red", linewidth=3.0, arrow_length_ratio=0.2,
                label="global representative direction")
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-12))
    axis.legend(loc="best")
    figure.colorbar(cloud, ax=axis, shrink=0.65, label="|surface height gradient|")
    figure.savefig(output, dpi=200)
    if show:
        plt.show()
    plt.close(figure)


def choose_stl() -> Path:
    """When no CLI path is supplied, let the user select an STL interactively."""
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="Select an STL model",
            filetypes=(("STL mesh", "*.stl"), ("All files", "*.*")),
        )
        root.destroy()
    except Exception as exc:
        raise ValueError("请在命令行传入 STL 路径，例如 groupgrid.py model.stl") from exc
    if not selected:
        raise SystemExit("没有选择 STL 文件")
    return Path(selected)


def gpu_status() -> str:
    """Return a user-facing CUDA/CuPy availability message."""
    if cp is None:
        return "GPU acceleration: unavailable (CuPy is not installed; using NumPy)"
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count <= 0:
            return "GPU acceleration: unavailable (CuPy found, but no CUDA device)"
        device = cp.cuda.Device(0)
        raw_name = cp.cuda.runtime.getDeviceProperties(device.id)["name"]
        name = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
        return f"GPU acceleration: available (CuPy/CUDA, {device_count} device(s), {name})"
    except Exception as exc:
        return f"GPU acceleration: unavailable (CUDA check failed: {exc})"


def main() -> None:
    parser = argparse.ArgumentParser(description="STL point cloud minimum-gradient direction")
    parser.add_argument("stl", nargs="?", type=Path,
                        help="input STL; omitted opens a file-selection dialog")
    parser.add_argument("--points", type=int, default=10000, help="surface samples")
    parser.add_argument("--neighbors", type=int, default=24, help="local PCA neighbors")
    parser.add_argument("--arrows", type=int, default=300, help="number of displayed arrows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="output prefix")
    parser.add_argument("--cpu", action="store_true", help="disable CuPy")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    # Print this before opening the file dialog so an interactive run gives
    # immediate feedback instead of appearing to have stalled.
    print(gpu_status(), flush=True)
    if args.points < 3:
        raise ValueError("--points must be at least 3")
    stl_path = args.stl or choose_stl()
    mesh = load_stl(stl_path)
    points = sample_surface(mesh, args.points, args.seed)
    normals, directions, height_gradients, gradient_norm, min_gradient, backend = minimum_gradient_directions(
        points, args.neighbors, use_gpu=not args.cpu
    )
    representative, global_error, separation = global_direction(height_gradients)
    prefix = args.output or stl_path.with_name(stl_path.stem + "_minimum_gradient")
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_name(prefix.name + ".csv")
    image_path = prefix.with_name(prefix.name + ".png")
    save_csv(csv_path, points, normals, directions, gradient_norm, min_gradient)
    plot_result(points, directions, representative, gradient_norm, image_path,
                not args.no_show, stl_path.name, args.arrows)
    print(f"Backend: {backend}")
    print(f"Points: {len(points)}; neighbors: {min(args.neighbors, len(points))}")
    print(f"Global minimum-gradient direction: {representative.tolist()}")
    print(f"Global mean squared directional gradient: {global_error:.6g}")
    print(f"Smallest/second eigenvalue ratio: {separation:.4f}")
    print(f"Mean height-gradient norm: {gradient_norm.mean():.6g}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Figure: {image_path.resolve()}")


if __name__ == "__main__":
    main()
