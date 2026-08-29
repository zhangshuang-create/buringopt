#!/usr/bin/env python3
"""均匀采样 STL 表面，并使用积分不变量曲率 (IIC) 自动识别并分段着色不同的边缘。

通过在每个边界点放置自适应半径的球，统计其内部包含的均匀表面点数，计算面积重叠率（Area Ratio），
配合强力低通平滑与自适应峰值检测，实现高抗噪、免调参的角点与边缘识别。

依赖库:
    pip install numpy scipy trimesh matplotlib
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

# 将 matplotlib 缓存路径设在本地项目目录中，以防在受限环境中运行出错
RUNTIME_CACHE = Path(__file__).resolve().parent / ".runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


@dataclass
class IICResult:
    boundary: np.ndarray
    ratio: np.ndarray
    smoothed_ratio: np.ndarray
    curvature: np.ndarray
    peaks: np.ndarray
    edge_ids: np.ndarray
    r: float


def load_stl(path: str | Path) -> trimesh.Trimesh:
    """加载并轻度清理 STL 模型，合并顶点以确保拓扑正确性。"""
    loaded = trimesh.load_mesh(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"在 {path} 中未找到任何三角网格。")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{path} 不是一个有效的非空三角网格模型。")

    mesh = loaded.copy()
    mesh.merge_vertices()
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def ordered_boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """提取三维网格中的封闭物理边界，并重构成有序的 3D 折线点集。"""
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
        ia, ib = int(a), int(b)
        adjacency.setdefault(ia, []).append(ib)
        adjacency.setdefault(ib, []).append(ia)
        unused.add((ia, ib))

    bad = [vertex for vertex, neighbors in adjacency.items() if len(neighbors) != 2]
    if bad:
        raise ValueError(
            f"STL 的边界存在分支或未完全闭合。IIC 算法需要一个完整的流形闭合环路。"
            f"（检测到有 {len(bad)} 个边界顶点的度数不为 2）。"
        )

    loops: list[np.ndarray] = []
    while unused:
        first_edge = next(iter(unused))
        start, current = first_edge
        unused.remove(first_edge)
        indices = [start]
        previous = start

        while current != start:
            indices.append(current)
            candidates = [v for v in adjacency[current] if v != previous]
            if len(candidates) != 1:
                raise ValueError("无法将边界点有序化为封闭环路。")
            following = candidates[0]
            edge = (min(current, following), max(current, following))
            if following != start and edge not in unused:
                raise ValueError("边界环路中包含重复或断开的边。")
            unused.discard(edge)
            previous, current = current, following

        loops.append(vertices[np.asarray(indices, dtype=np.int64)])
    return loops


def closed_polyline_length(points: np.ndarray) -> float:
    """计算闭合折线的总弧长。"""
    return float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())


def resample_closed_boundary(points: np.ndarray, count: int) -> np.ndarray:
    """按照等弧长（等距离间隔）对闭合边界进行重采样。"""
    if count < 16:
        raise ValueError("边界采样点数不能少于 16。")
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    valid = lengths > np.finfo(float).eps
    starts = closed[:-1][valid]
    ends = closed[1:][valid]
    lengths = lengths[valid]
    if len(lengths) < 3:
        raise ValueError("所选边界包含的非零段数少于 3 段。")

    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    positions = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    segment = np.searchsorted(cumulative, positions, side="right") - 1
    segment = np.minimum(segment, len(lengths) - 1)
    fraction = (positions - cumulative[segment]) / lengths[segment]
    return starts[segment] + fraction[:, None] * (ends[segment] - starts[segment])


def circular_peaks(signal: np.ndarray, prominence: float, distance: int) -> np.ndarray:
    """在循环/闭合信号中寻找峰值，避免在索引 0 处产生虚假截断。"""
    count = len(signal)
    repeated = np.tile(signal, 3)
    peaks, _ = find_peaks(repeated, prominence=prominence, distance=distance)
    return np.asarray(peaks[(peaks >= count) & (peaks < 2 * count)] - count, dtype=int)


def label_circular_edges(count: int, peaks: np.ndarray) -> np.ndarray:
    """根据检测到的角点峰值索引，将相邻两个角点之间的边界样本划分到同一个边编号。"""
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


def analyze_iic(
    boundary: np.ndarray,
    surface_points: np.ndarray,
    mesh_area: float,
    r_multiplier: float,
) -> IICResult:
    """积分不变量曲率计算与拐角分段。"""
    # 1. 计算边界相邻采样点平均间距
    closed_boundary = np.vstack([boundary, boundary[0]])
    segment_lengths = np.linalg.norm(np.diff(closed_boundary, axis=0), axis=1)
    d_avg = float(np.mean(segment_lengths))

    # 2. 设定自适应积分圆半径 r (默认为平均距离的 8.0 倍，能大幅提高抗噪能力)
    r = r_multiplier * d_avg

    # 3. 构建表面均匀点云的 KDTree，用于快速邻域搜索
    tree = cKDTree(surface_points)

    # 4. 对每个边界点，统计其半径 r 范围内包含的表面点数量
    counts = np.array(
        tree.query_ball_point(boundary, r, return_length=True), dtype=float
    )

    # 5. 根据均匀采样密度，计算一个完整平坦圆（无边界遮挡）内理应包含的点数 N_full
    density = len(surface_points) / mesh_area
    n_full = np.pi * (r**2) * density

    if n_full <= np.finfo(float).eps:
        raise ValueError(
            "表面点云密度太低，无法进行邻域积分面积重叠计算。请调大 --points 的数值。"
        )

    # 6. 计算重叠率（Area Ratio）
    ratio = counts / n_full

    # 7. 强力低通平滑滤波：对于 600 个点，平滑窗口大小（sigma）扩大至约 15，彻底滤除离散波动
    count = len(ratio)
    sigma = max(5.0, count / 40.0)  # 升级点：大幅增强高斯平滑尺度
    smoothed_ratio = gaussian_filter1d(ratio, sigma=sigma, mode="wrap")

    # 8. 设定直边基准线（取平滑占比曲线的中位数，通常极度贴近 0.5）
    baseline = np.median(smoothed_ratio)

    # 9. 统一特征度量：不管凸角还是凹角，偏离绝对值在角点处都会取得极大值
    curvature = np.abs(smoothed_ratio - baseline)

    # 10. 在一维特征信号上进行自适应峰值检测
    dynamic_range = float(np.ptp(curvature))
    numerical_scale = max(float(np.max(np.abs(curvature))), 1.0)
    if dynamic_range <= 100.0 * np.finfo(float).eps * numerical_scale:
        raise ValueError(
            "积分不变量曲线几乎为一条直线，未检测到明显的物理转折角（如完美的圆）。"
        )

    # 升级点：将检测显著门槛从 5% 提高到 20%，彻底杜绝局部微小起伏产生的伪峰值
    prominence = 0.20 * dynamic_range
    separation = max(1, int(round(3.0 * sigma)))
    peaks = circular_peaks(curvature, prominence, separation)

    if len(peaks) < 2:
        raise ValueError(
            f"IIC 算法仅检测到 {len(peaks)} 个显著拐角。如果您的形状确实有拐角但未能识别，"
            f"请增大面采样数 --points，或适当调整 --radius-multiplier。"
        )

    return IICResult(
        boundary=boundary,
        ratio=ratio,
        smoothed_ratio=smoothed_ratio,
        curvature=curvature,
        peaks=np.sort(peaks),
        edge_ids=label_circular_edges(count, peaks),
        r=r,
    )


def edge_palette(edge_count: int) -> np.ndarray:
    """根据边数自适应生成色卡。"""
    # 只要边数不超过 20，就使用高对比度的离散颜色卡，拒绝连续渐变色
    cmap = plt.colormaps["tab20" if edge_count <= 20 else "hsv"]
    positions = np.arange(edge_count) / max(edge_count, 1)
    return np.asarray(cmap(positions))[:, :3]


def save_boundary_csv(path: Path, result: IICResult, colors: np.ndarray) -> None:
    """保存分段好的边界点数据到 CSV。"""
    peak_set = set(result.peaks.tolist())
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "order",
                "x",
                "y",
                "z",
                "area_ratio",
                "smoothed_ratio",
                "curvature",
                "edge_id",
                "is_corner_peak",
                "red",
                "green",
                "blue",
            ]
        )
        for index, point in enumerate(result.boundary):
            color = colors[result.edge_ids[index]]
            writer.writerow(
                [
                    index,
                    *point,
                    result.ratio[index],
                    result.smoothed_ratio[index],
                    result.curvature[index],
                    int(result.edge_ids[index]),
                    int(index in peak_set),
                    *np.round(255 * color).astype(int),
                ]
            )


def save_colored_cloud(
    path: Path, surface_points: np.ndarray, result: IICResult, colors: np.ndarray
) -> None:
    """保存灰色表面点与彩色边缘点合并后的 PLY 点云。"""
    surface_rgba = np.tile(
        np.array([185, 185, 185, 90], dtype=np.uint8), (len(surface_points), 1)
    )
    boundary_rgb = np.round(255 * colors[result.edge_ids]).astype(np.uint8)
    boundary_rgba = np.column_stack(
        (boundary_rgb, np.full(len(boundary_rgb), 255, dtype=np.uint8))
    )
    cloud = trimesh.points.PointCloud(
        np.vstack((surface_points, result.boundary)),
        colors=np.vstack((surface_rgba, boundary_rgba)),
    )
    cloud.export(path)


def plot_result(
    surface_points: np.ndarray,
    result: IICResult,
    colors: np.ndarray,
    title: str,
    output: Path,
    show: bool,
) -> None:
    """绘制 3D 边界边缘渲染点云与 1D 积分面积占比波形图。"""
    figure = plt.figure(figsize=(14, 6), constrained_layout=True)

    # 左图：3D 均匀点云与分段边可视化
    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis_3d.scatter(*surface_points.T, s=2, c="0.72", alpha=0.22, depthshade=False)
    axis_3d.scatter(
        *result.boundary.T,
        s=15,
        c=colors[result.edge_ids],
        depthshade=False,
        label="IIC Segmented Edge",
    )
    axis_3d.scatter(
        *result.boundary[result.peaks].T,
        s=110,
        c="black",
        marker="*",
        depthshade=False,
        label="IIC Corner Point",
    )
    axis_3d.set_title(
        f"Model: {title}\nAuto-detected: {len(result.peaks)} edges (r = {result.r:.3g})"
    )
    axis_3d.set_xlabel("X")
    axis_3d.set_ylabel("Y")
    axis_3d.set_zlabel("Z")
    axis_3d.set_box_aspect(np.maximum(np.ptp(surface_points, axis=0), 1e-12))
    axis_3d.legend(loc="best")

    # 右图：IIC 面积重叠率（Area Ratio）波形
    axis_profile = figure.add_subplot(1, 2, 2)
    order = np.arange(len(result.ratio))

    # 绘制原始波动极其剧烈的波形（淡灰色细线）
    axis_profile.plot(
        order, result.ratio, color="0.85", linewidth=0.5, label="Raw Ratio (Noisy)"
    )

    # 绘制被强力低通滤波后的干净波形（以各段提取色显示）
    for edge_id in range(len(result.peaks)):
        mask = result.edge_ids == edge_id
        axis_profile.scatter(
            order[mask], result.smoothed_ratio[mask], s=8, color=colors[edge_id]
        )

    # 标记检测到的物理大拐角（黑色三角）
    axis_profile.scatter(
        result.peaks,
        result.smoothed_ratio[result.peaks],
        marker="^",
        s=75,
        c="black",
        label="Corners (Extrema)",
    )

    # 绘制理论参考基准线：0.5 (直边)、0.25 (凸角)、0.75 (凹角)
    axis_profile.axhline(
        0.5, color="green", linestyle="--", alpha=0.6, label="Straight Edge (50%)"
    )
    axis_profile.axhline(
        0.25, color="red", linestyle="--", alpha=0.6, label="Convex Corner (25%)"
    )
    if np.any(result.smoothed_ratio > 0.6):
        axis_profile.axhline(
            0.75, color="blue", linestyle="--", alpha=0.6, label="Concave Corner (75%)"
        )

    axis_profile.set_title("Boundary Area Overlap Ratio (IIC)")
    axis_profile.set_xlabel("Ordered boundary sample index")
    axis_profile.set_ylabel("Inside Area Ratio (ratio_i)")
    axis_profile.set_ylim(-0.05, 1.05)
    axis_profile.grid(alpha=0.2)
    axis_profile.legend(loc="best")

    figure.savefig(output, dpi=200)
    if show:
        plt.show()
    plt.close(figure)


def choose_stl() -> Path:
    """若命令行未指定 STL 文件路径，弹窗提示用户选择一个 stl 文件。"""
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="Select an STL for IIC edge detection",
            filetypes=(("STL mesh", "*.stl"), ("All files", "*.*")),
        )
        root.destroy()
    except Exception as exc:
        raise ValueError("请在命令行中传递 STL 文件路径，或安装 tkinter 库。") from exc
    if not selected:
        raise SystemExit("未选择任何 STL 文件。")
    return Path(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="利用自适应积分不变量曲率 (IIC) 自动识别并分段着色单条 STL 闭合边界上的边。"
    )
    parser.add_argument("stl", nargs="?", type=Path, help="输入的 STL 路径")
    parser.add_argument(
        "--points", type=int, default=10000, help="表面均匀采样点的总数 (默认: 10000)"
    )
    parser.add_argument(
        "--boundary-points",
        type=int,
        default=600,
        help="物理闭合边界重采样后的点数 (默认: 600)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="按周长从长到短排序要处理的边界索引，0 为最长外轮廓 (默认: 0)",
    )
    parser.add_argument(
        "--radius-multiplier",
        type=float,
        default=8.0,
        help="积分球自适应半径的乘数，即 r = multiplier * d_avg (默认: 8.0)",
    )
    parser.add_argument("--seed", type=int, default=0, help="表面均匀采样的随机种子")
    parser.add_argument(
        "--output",
        type=Path,
        help="输出文件的前缀 (默认在 STL 同级目录下生成带有 _iic 后缀的文件)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="仅保存结果图表，而不弹出 matplotlib 交互式绘图窗口",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stl_path = args.stl if args.stl is not None else choose_stl()
    if args.points < 100:
        raise ValueError("--points 不能少于 100")
    if args.boundary_points < 16:
        raise ValueError("--boundary-points 不能少于 16")

    # 1. 加载 STL
    mesh = load_stl(stl_path)

    # 2. 提取闭合边界
    loops = ordered_boundary_loops(mesh)
    if not loops:
        raise ValueError(
            "在网格中未检测到物理开放边界。IIC 算法需要一个封闭的边界环路，而封闭水密网格没有拓扑边界。"
        )
    loops.sort(key=closed_polyline_length, reverse=True)
    if not 0 <= args.loop < len(loops):
        raise ValueError(f"--loop 索引必须在 0 到 {len(loops) - 1} 之间。")

    # 3. 对目标边界进行均匀等弧长重采样
    boundary = resample_closed_boundary(loops[args.loop], args.boundary_points)

    # 4. 均匀采样 3D 表面作为内部面积微元点云
    surface_points, _ = trimesh.sample.sample_surface_even(
        mesh, args.points, seed=args.seed
    )
    if len(surface_points) == 0:
        raise ValueError("无法在 STL 表面采样到任何点。")

    # 5. 运行 IIC 算法识别边缘与角点
    result = analyze_iic(
        boundary, surface_points, float(mesh.area), args.radius_multiplier
    )

    # 6. 配置输出路径并保存结果
    output = args.output or stl_path.with_name(stl_path.stem + "_iic")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure_path = output.with_name(output.name + "_result.png")
    csv_path = output.with_name(output.name + "_boundary.csv")
    ply_path = output.with_name(output.name + "_colored.ply")
    colors = edge_palette(len(result.peaks))

    # 保存 CSV、PLY 彩色点云以及可视化图片
    save_boundary_csv(csv_path, result, colors)
    save_colored_cloud(ply_path, surface_points, result, colors)
    plot_result(
        surface_points,
        result,
        colors,
        stl_path.name,
        figure_path,
        show=not args.no_show,
    )

    # 打印输出报告
    print(f"STL 表面均匀采样数: {len(surface_points)}")
    print(f"网格检测到的封闭边界环路数: {len(loops)}; 当前分析的边界索引: {args.loop}")
    print(f"IIC 积分圆半径 (r): {result.r:.4g}")
    print(f"IIC 成功识别的边界段(边)数: {len(result.peaks)}")
    print(f"识别到的角点/峰值索引位置: {result.peaks.tolist()}")
    print(f"【可视化结果保存至】: {figure_path.resolve()}")
    print(f"【彩色边缘 PLY 点云保存至】: {ply_path.resolve()}")
    print(f"【各段边界 CSV 点集数据保存至】: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
