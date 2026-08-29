#!/usr/bin/env python3
"""用质心距离剖面（CDP）识别并着色 STL 的闭合边界。"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

RUNTIME_CACHE = Path(__file__).resolve().parents[2] / ".runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class CDPResult:
    boundary: np.ndarray
    centroid: np.ndarray
    distance: np.ndarray
    smooth: np.ndarray
    peaks: np.ndarray
    valleys: np.ndarray
    edge_ids: np.ndarray
    space: str


def load_stl(path: str | Path) -> trimesh.Trimesh:
    """读取 STL，并合并 STL 中重复存储的三角形顶点。"""
    loaded = trimesh.load_mesh(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"在 {path} 中没有找到三角网格")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{path} 不是非空 STL 三角网格")
    mesh = loaded.copy()
    mesh.merge_vertices()
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def ordered_boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """从只属于一个三角面的边提取有序闭合边界。"""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) == 0:
        return []

    adjacency: dict[int, list[int]] = {}
    unused: set[tuple[int, int]] = set()
    for a, b in boundary_edges:
        a, b = int(a), int(b)
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
        unused.add((a, b))
    bad = [v for v, neighbors in adjacency.items() if len(neighbors) != 2]
    if bad:
        raise ValueError(
            "网格边界存在分叉或断口；CDP 要求闭合回路。"
            f"有 {len(bad)} 个边界顶点的邻接数不等于 2。"
        )

    loops: list[np.ndarray] = []
    while unused:
        start, current = next(iter(unused))
        unused.remove((start, current))
        previous, indices = start, [start]
        while current != start:
            indices.append(current)
            following = next(v for v in adjacency[current] if v != previous)
            edge = (min(current, following), max(current, following))
            if following != start and edge not in unused:
                raise ValueError("无法将 STL 边界排列成闭合回路")
            unused.discard(edge)
            previous, current = current, following
        loops.append(vertices[np.asarray(indices, dtype=np.int64)])
    return loops


def loop_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())


def resample_closed_loop(points: np.ndarray, count: int) -> np.ndarray:
    """沿闭合边界按相等弧长重采样，保证 CDP 的横轴均匀。"""
    closed = np.vstack((points, points[0]))
    segment_length = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    valid = segment_length > np.finfo(float).eps
    starts, ends, segment_length = closed[:-1][valid], closed[1:][valid], segment_length[valid]
    if len(segment_length) < 3:
        raise ValueError("边界至少需要三个非零线段")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_length)))
    position = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    segment = np.searchsorted(cumulative, position, side="right") - 1
    segment = np.minimum(segment, len(segment_length) - 1)
    t = (position - cumulative[segment]) / segment_length[segment]
    return starts[segment] + t[:, None] * (ends[segment] - starts[segment])


def circular_peaks(signal: np.ndarray, prominence: float, separation: int) -> np.ndarray:
    """在三份首尾拼接信号的中段找峰，消除序号 0 处的人为接缝。"""
    count = len(signal)
    peaks, _ = find_peaks(
        np.tile(signal, 3), prominence=prominence, distance=separation
    )
    return np.asarray(peaks[(peaks >= count) & (peaks < 2 * count)] - count, dtype=int)


def edge_labels(count: int, peaks: np.ndarray) -> np.ndarray:
    peaks = np.sort(np.unique(peaks))
    labels = np.empty(count, dtype=np.int64)
    for edge_id, start in enumerate(peaks):
        end = int(peaks[(edge_id + 1) % len(peaks)])
        if end > start:
            labels[start:end] = edge_id
        else:
            labels[start:] = edge_id
            labels[:end] = edge_id
    return labels


def projected_centroid_distance(boundary: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """计算边界在其自然轮廓平面内到质心的距离。

    非平面边界的向量面积给出其整体轮廓平面的法向。先移除沿该
    法向的起伏，再计算 CDP，可避免马鞍面边中部的高度极值被误判
    为角点。向量面积退化时使用 PCA 的最小方差方向作为法向。
    """
    centered = boundary - centroid
    area_vector = np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0)
    area_norm = float(np.linalg.norm(area_vector))
    scale = max(float(np.linalg.norm(centered, axis=1).max()), 1.0)
    if area_norm > np.finfo(float).eps * scale**2 * len(boundary):
        normal = area_vector / area_norm
    else:
        covariance = centered.T @ centered / len(centered)
        _, axes = np.linalg.eigh(covariance)
        normal = axes[:, 0]
    planar = centered - (centered @ normal)[:, None] * normal
    return np.linalg.norm(planar, axis=1)


def analyze_cdp(boundary: np.ndarray, space: str = "projected") -> CDPResult:
    """计算 CDP；显著波峰视为角点，相邻波峰间视为一条边。

    projected（默认）在边界整体轮廓平面中测距，适合马鞍面等非平面
    曲面；3d 保留原始三维欧氏距离，主要用于对比。
    """
    centroid = boundary.mean(axis=0)
    if space == "projected":
        distance = projected_centroid_distance(boundary, centroid)
    elif space == "3d":
        distance = np.linalg.norm(boundary - centroid, axis=1)
    else:
        raise ValueError("space must be 'projected' or '3d'")
    count = len(distance)
    # 小尺度循环平滑只压制 STL 离散噪声，尺度随采样数量自动变化。
    sigma = max(1.0, count / 500.0)
    smooth = gaussian_filter1d(distance, sigma=sigma, mode="wrap")
    amplitude = float(np.ptp(smooth))
    numerical_scale = max(float(np.max(np.abs(smooth))), 1.0)
    if amplitude <= 100.0 * np.finfo(float).eps * numerical_scale:
        raise ValueError("CDP 近似常数（例如圆形），不存在可区分的角点/边。")

    # 唯一内部判据是相对全幅值 3% 的峰显著度，与模型单位无关。
    prominence = 0.03 * amplitude
    separation = max(1, int(round(4.0 * sigma)))
    peaks = np.sort(circular_peaks(smooth, prominence, separation))
    valleys = np.sort(circular_peaks(-smooth, prominence, separation))
    if len(peaks) < 2:
        raise ValueError(f"CDP 只找到 {len(peaks)} 个显著波峰，无法划分不同边。")
    return CDPResult(
        boundary, centroid, distance, smooth, peaks, valleys,
        edge_labels(count, peaks), space,
    )


def palette(count: int) -> np.ndarray:
    """返回离散高对比颜色，避免少量边时抽样到与背景相近的灰色。"""
    if count <= 10:
        base = np.asarray(plt.colormaps["tab10"].colors)
        return base[:count, :3]
    if count <= 20:
        base = np.asarray(plt.colormaps["tab20"].colors)
        return base[:count, :3]
    return np.asarray(plt.colormaps["hsv"](np.arange(count) / count))[:, :3]


def circular_segment_indices(count: int, start: int, end: int) -> np.ndarray:
    """返回从 start 到 end（含两端）的循环有序序号。"""
    if end > start:
        return np.arange(start, end + 1, dtype=np.int64)
    return np.concatenate(
        (np.arange(start, count, dtype=np.int64), np.arange(0, end + 1, dtype=np.int64))
    )


def save_csv(path: Path, result: CDPResult, colors: np.ndarray) -> None:
    peak_set, valley_set = set(result.peaks), set(result.valleys)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "order", "x", "y", "z", "distance", "smooth_distance", "edge_id",
            "is_peak", "is_valley", "red", "green", "blue",
        ])
        for i, point in enumerate(result.boundary):
            rgb = np.round(255 * colors[result.edge_ids[i]]).astype(int)
            writer.writerow([
                i, *point, result.distance[i], result.smooth[i], result.edge_ids[i],
                int(i in peak_set), int(i in valley_set), *rgb,
            ])


def save_ply(
    path: Path, surface: np.ndarray, result: CDPResult, colors: np.ndarray
) -> None:
    gray = np.tile(np.array([185, 185, 185, 90], np.uint8), (len(surface), 1))
    boundary_rgb = np.round(255 * colors[result.edge_ids]).astype(np.uint8)
    boundary_rgba = np.column_stack(
        (boundary_rgb, np.full(len(boundary_rgb), 255, dtype=np.uint8))
    )
    cloud = trimesh.points.PointCloud(
        np.vstack((surface, result.boundary)), colors=np.vstack((gray, boundary_rgba))
    )
    cloud.export(path)


def plot_result(
    surface: np.ndarray,
    result: CDPResult,
    colors: np.ndarray,
    title: str,
    output: Path,
    show: bool,
) -> None:
    figure = plt.figure(figsize=(14, 6), constrained_layout=True)
    axis = figure.add_subplot(1, 2, 1, projection="3d")
    axis.scatter(*surface.T, s=2, c="0.72", alpha=0.22, depthshade=False)
    # 分别按循环顺序画连续线，而不是一次性散点着色；这样跨越序号 0 的
    # 那条边也不会在视觉上被拆开。相邻边共享的角点由黑色星号覆盖。
    for edge_id, start in enumerate(result.peaks):
        end = int(result.peaks[(edge_id + 1) % len(result.peaks)])
        indices = circular_segment_indices(len(result.boundary), int(start), end)
        segment = result.boundary[indices]
        axis.plot(
            *segment.T, color=colors[edge_id], linewidth=4.0,
            solid_capstyle="round", label=f"E{edge_id}",
        )
        middle = segment[len(segment) // 2]
        axis.text(*middle, f" E{edge_id}", color=colors[edge_id],
                  fontsize=10, fontweight="bold")
    axis.scatter(
        *result.boundary[result.peaks].T, s=90, c="black", marker="*",
        depthshade=False, label="CDP peak / corner",
    )
    axis.scatter(*result.centroid, s=80, c="red", marker="x", label="centroid")
    axis.set_title(f"{title}\nCDP detected {len(result.peaks)} edges")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_box_aspect(np.maximum(np.ptp(surface, axis=0), 1e-12))
    axis.legend(ncol=2)

    profile = figure.add_subplot(1, 2, 2)
    order = np.arange(len(result.distance))
    profile.plot(order, result.distance, c="0.75", lw=0.8, label="raw distance")
    for edge_id in range(len(result.peaks)):
        mask = result.edge_ids == edge_id
        profile.scatter(order[mask], result.smooth[mask], s=8, c=[colors[edge_id]])
    profile.scatter(
        result.peaks, result.smooth[result.peaks], marker="^", s=65, c="black",
        label="peaks / corners",
    )
    if len(result.valleys):
        profile.scatter(
            result.valleys, result.smooth[result.valleys], marker="v", s=50,
            facecolors="none", edgecolors="black", label="valleys",
        )
    profile.set_title(f"Centroid Distance Profile (CDP, {result.space})")
    profile.set_xlabel("Ordered boundary sample")
    profile.set_ylabel("Distance to centroid")
    profile.grid(alpha=0.2)
    profile.legend()
    figure.savefig(output, dpi=200)
    if show:
        plt.show()
    plt.close(figure)


def choose_stl() -> Path:
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="Select an STL for CDP",
            filetypes=(("STL mesh", "*.stl"), ("All files", "*.*")),
        )
        root.destroy()
    except Exception as exc:
        raise ValueError("无法打开文件选择框，请在命令行传入 STL 路径。") from exc
    if not selected:
        raise SystemExit("没有选择 STL")
    return Path(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 CDP 给 STL 闭合边界分边并着色")
    parser.add_argument("stl", nargs="?", type=Path, help="输入 STL；省略则打开文件框")
    parser.add_argument("--points", type=int, default=10000, help="均匀表面点数")
    parser.add_argument("--boundary-points", type=int, default=1000, help="等弧长边界点数")
    parser.add_argument("--loop", type=int, default=0, help="按周长排序的边界序号")
    parser.add_argument("--seed", type=int, default=0, help="表面采样随机种子")
    parser.add_argument(
        "--cdp-space", choices=("projected", "3d"), default="projected",
        help="CDP 测距空间：轮廓平面 projected（默认）或原始 3d",
    )
    parser.add_argument("--output", type=Path, help="输出前缀，默认 <STL名>_cdp")
    parser.add_argument("--no-show", action="store_true", help="仅保存，不显示窗口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stl = args.stl or choose_stl()
    if args.points < 100 or args.boundary_points < 16:
        raise ValueError("--points 至少为 100，--boundary-points 至少为 16")
    mesh = load_stl(stl)
    loops = ordered_boundary_loops(mesh)
    if not loops:
        raise ValueError("没有找到拓扑边界；完全封闭（watertight）的 STL 没有边界回路。")
    loops.sort(key=loop_length, reverse=True)
    if not 0 <= args.loop < len(loops):
        raise ValueError(f"--loop 应在 0 到 {len(loops) - 1} 之间")

    boundary = resample_closed_loop(loops[args.loop], args.boundary_points)
    result = analyze_cdp(boundary, space=args.cdp_space)
    surface, _ = trimesh.sample.sample_surface_even(mesh, args.points, seed=args.seed)
    if not len(surface):
        raise ValueError("STL 表面采样失败")

    output = args.output or stl.with_name(stl.stem + "_cdp")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = output.with_name(output.name + "_result.png")
    ply = output.with_name(output.name + "_colored.ply")
    table = output.with_name(output.name + "_boundary.csv")
    colors = palette(len(result.peaks))
    save_csv(table, result, colors)
    save_ply(ply, surface, result, colors)
    plot_result(surface, result, colors, stl.name, image, not args.no_show)

    print(f"均匀表面点: {len(surface)}")
    print(f"闭合边界数: {len(loops)}；使用边界: {args.loop}")
    print(f"CDP 波峰/彩色边数: {len(result.peaks)}")
    print(f"CDP 测距空间: {result.space}")
    print(f"波峰序号: {result.peaks.tolist()}")
    print(f"结果图: {image.resolve()}")
    print(f"彩色点云: {ply.resolve()}")
    print(f"边界数据: {table.resolve()}")


if __name__ == "__main__":
    main()
