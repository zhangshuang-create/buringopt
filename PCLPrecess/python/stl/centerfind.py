from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import trimesh
import vtk


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load STL with standard vertex normalization."""
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [item for item in loaded.geometry.values()
                      if isinstance(item, trimesh.Trimesh)]
        if not geometries:
            raise ValueError("The selected file contains no triangular mesh.")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.vertices) or not len(loaded.faces):
        raise ValueError("The selected file contains no triangular mesh.")
    mesh = loaded.copy()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    if np.asarray(mesh.faces).ndim != 2 or np.asarray(mesh.faces).shape[1] != 3:
        raise ValueError("Only triangular meshes are supported.")
    return mesh


def extract_boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """Trace ordered loops from edges occurring exactly once."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    all_edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    all_edges = np.sort(all_edges.astype(np.int64, copy=False), axis=1)
    unique_edges, counts = np.unique(all_edges, axis=0, return_counts=True)
    boundary_edges = [tuple(map(int, edge)) for edge, count in zip(unique_edges, counts)
                      if int(count) == 1]
    if not boundary_edges:
        raise ValueError("No open boundaries found. The STL may be closed.")

    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    unused = {tuple(edge) for edge in boundary_edges}
    loops: list[np.ndarray] = []

    def key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    while unused:
        start, current = next(iter(unused))
        unused.remove((start, current))
        ordered = [start, current]
        previous = start
        while current != start:
            candidates = [
                candidate for candidate in adjacency[current]
                if candidate != previous and key(current, candidate) in unused
            ]
            if not candidates:
                if key(current, start) in unused:
                    unused.remove(key(current, start))
                    break
                raise ValueError("Boundary is open, branched, or non-manifold.")
            next_vertex = candidates[0]
            unused.remove(key(current, next_vertex))
            previous, current = current, next_vertex
            if current != start:
                ordered.append(current)
            if len(ordered) > len(boundary_edges) + 2:
                raise ValueError("Boundary tracing exceeded the expected edge count.")
        if len(ordered) >= 3:
            loops.append(vertices[np.asarray(ordered, dtype=np.int64)])
    return loops


def classify_boundaries(loops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(loops) < 2:
        raise ValueError(f"Expected at least two boundaries, found {len(loops)}.")
    ordered = sorted(loops, key=lambda points: float(points[:, 2].mean()))
    return ordered[-1], ordered[0]


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute centroid and orthogonal PCA axes for one point set."""
    samples = np.asarray(points, dtype=float)
    center = samples.mean(axis=0)
    centered = samples - center
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    return center, vectors[:, order].T, values[order]


def find_line_boundary_intersections(origin: np.ndarray, direction: np.ndarray, loop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the two opposite continuous intersection points between the 2nd PCA line and the boundary loop."""
    dir_u = direction / max(np.linalg.norm(direction), 1.0e-12)
    n_pts = len(loop)

    best_pos_t = float("inf")
    best_pos_pt = None
    best_neg_t = float("inf")
    best_neg_pt = None

    for i in range(n_pts):
        p1 = loop[i]
        p2 = loop[(i + 1) % n_pts]
        v = p2 - p1
        denom_mat = np.column_stack([dir_u, -v])
        diff = p1 - origin
        normal_eq = denom_mat.T @ denom_mat
        if abs(np.linalg.det(normal_eq)) < 1.0e-12:
            continue
        t_s = np.linalg.solve(normal_eq, denom_mat.T @ diff)
        t, s = float(t_s[0]), float(t_s[1])

        if -1.0e-4 <= s <= 1.0001:
            s_clamped = float(np.clip(s, 0.0, 1.0))
            candidate_pt = p1 + s_clamped * v
            dist_to_axis = np.linalg.norm(candidate_pt - (origin + t * dir_u))
            if dist_to_axis < 5.0:
                if t > 1.0e-4 and t < best_pos_t:
                    best_pos_t = t
                    best_pos_pt = candidate_pt.copy()
                elif t < -1.0e-4 and abs(t) < best_neg_t:
                    best_neg_t = abs(t)
                    best_neg_pt = candidate_pt.copy()

    if best_pos_pt is None:
        projs = [np.dot(p - origin, dir_u) for p in loop]
        best_pos_pt = loop[int(np.argmax(projs))].copy()
    if best_neg_pt is None:
        projs = [np.dot(p - origin, dir_u) for p in loop]
        best_neg_pt = loop[int(np.argmin(projs))].copy()

    return best_pos_pt, best_neg_pt


def find_closest_interpolated_point(target: np.ndarray, loop: np.ndarray) -> tuple[np.ndarray, float]:
    """Find continuous interpolated closest point and Euclidean distance on a polyline loop."""
    n_pts = len(loop)
    best_dist = float("inf")
    best_pt = None

    for i in range(n_pts):
        p1 = loop[i]
        p2 = loop[(i + 1) % n_pts]
        edge = p2 - p1
        denom = float(np.dot(edge, edge))
        if denom <= 1.0e-12:
            candidate = p1
        else:
            ratio = float(np.clip(np.dot(target - p1, edge) / denom, 0.0, 1.0))
            candidate = p1 + ratio * edge
        dist = float(np.linalg.norm(candidate - target))
        if dist < best_dist:
            best_dist = dist
            best_pt = candidate.copy()

    final_pt = best_pt if best_pt is not None else loop[0].copy()
    return final_pt, best_dist


def plane_normal_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Compute normalized plane normal vector from three points."""
    v1 = p2 - p1
    v2 = p3 - p1
    norm = np.cross(v1, v2)
    norm_len = np.linalg.norm(norm)
    if norm_len < 1.0e-12:
        return np.array([0.0, 0.0, 1.0])
    return norm / norm_len


def segments_to_continuous_polylines(segments: np.ndarray, tol: float = 1e-4) -> list[np.ndarray]:
    """Stitch disconnected line segments into ordered continuous polyline arrays."""
    if len(segments) == 0:
        return []

    lines_2d = segments.reshape(-1, 2, 3)
    nodes = []
    edges = []

    def get_node_id(pt: np.ndarray) -> int:
        for idx, n in enumerate(nodes):
            if np.linalg.norm(n - pt) < tol:
                return idx
        nodes.append(pt)
        return len(nodes) - 1

    for seg in lines_2d:
        id1 = get_node_id(seg[0])
        id2 = get_node_id(seg[1])
        if id1 != id2:
            edges.append((id1, id2))

    adj: dict[int, list[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    visited_edges = set()
    polylines = []

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    endpoints = [n for n, neighbors in adj.items() if len(neighbors) == 1]
    all_starts = endpoints + [n for n in adj.keys() if n not in endpoints]

    for start in all_starts:
        for nbr in adj[start]:
            ek = edge_key(start, nbr)
            if ek in visited_edges:
                continue
            path = [start, nbr]
            visited_edges.add(ek)
            curr = nbr
            prev = start
            while True:
                nexts = [x for x in adj[curr] if x != prev and edge_key(curr, x) not in visited_edges]
                if not nexts:
                    break
                nx = nexts[0]
                visited_edges.add(edge_key(curr, nx))
                path.append(nx)
                prev, curr = curr, nx
            if len(path) >= 2:
                polylines.append(np.array([nodes[idx] for idx in path]))

    return polylines


def polyline_data(points_array: np.ndarray, closed: bool = True) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    for point in points_array:
        points.InsertNextPoint(*map(float, point))
    lines = vtk.vtkCellArray()
    n = len(points_array)
    lines.InsertNextCell(n + 1 if closed else n)
    for index in range(n):
        lines.InsertCellPoint(index)
    if closed:
        lines.InsertCellPoint(0)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    return polydata


def mesh_data(mesh: trimesh.Trimesh) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    for point in np.asarray(mesh.vertices, dtype=float):
        points.InsertNextPoint(*map(float, point))
    cells = vtk.vtkCellArray()
    for face in np.asarray(mesh.faces, dtype=np.int64):
        cells.InsertNextCell(3)
        for index in face:
            cells.InsertCellPoint(int(index))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    return polydata


def actor(polydata: vtk.vtkPolyData, color: tuple[float, float, float],
          width: float = 1.0, opacity: float = 1.0) -> vtk.vtkActor:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    result = vtk.vtkActor()
    result.SetMapper(mapper)
    result.GetProperty().SetColor(*color)
    result.GetProperty().SetLineWidth(width)
    result.GetProperty().SetOpacity(opacity)
    return result


def sphere_actor(center: np.ndarray, radius: float, color: tuple[float, float, float]) -> vtk.vtkActor:
    source = vtk.vtkSphereSource()
    source.SetCenter(*map(float, center))
    source.SetRadius(float(radius))
    source.SetThetaResolution(24)
    source.SetPhiResolution(16)
    source.Update()
    return actor(source.GetOutput(), color, opacity=0.95)


def axis_actor(center: np.ndarray, direction: np.ndarray, half_length: float,
               color: tuple[float, float, float]) -> vtk.vtkActor:
    direction = np.asarray(direction, dtype=float)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    endpoints = np.vstack([center - half_length * direction, center + half_length * direction])
    return actor(polyline_data(endpoints, closed=False), color, width=4.0)


def select_stl() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(
        title="选择 STL 模型",
        filetypes=[("STL files", "*.stl *.STL"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(selected) if selected else None


def show_model(path: Path) -> None:
    mesh = load_mesh(path)
    loops = extract_boundary_loops(mesh)
    top, bottom = classify_boundaries(loops)

    # 1. 顶部 PCA 坐标系
    center, axes, eigenvalues = pca_frame(top)
    second_axis = axes[1]
    second_axis /= max(np.linalg.norm(second_axis), 1.0e-12)
    bottom_center = np.asarray(bottom, dtype=float).mean(axis=0)

    # 2. 红色切平面参数（原方案：PCA 原点、第二主轴方向延展点、底边界中心）
    red_p1 = center
    red_p2 = center + second_axis
    red_p3 = bottom_center
    red_normal = plane_normal_from_points(red_p1, red_p2, red_p3)

    # 3. 求解第二主方向直线上与上边界的【两个交点】
    top_pos_pt, top_neg_pt = find_line_boundary_intersections(center, second_axis, top)

    # 4. 对两个交点分别在底部边界寻找最近插值点并比较距离
    bot_pos_pt, dist_pos = find_closest_interpolated_point(top_pos_pt, bottom)
    bot_neg_pt, dist_neg = find_closest_interpolated_point(top_neg_pt, bottom)

    if dist_pos <= dist_neg:
        best_top_pt = top_pos_pt
        best_bot_pt = bot_pos_pt
        selected_side = "Positive side (+second_axis)"
        best_dist = dist_pos
    else:
        best_top_pt = top_neg_pt
        best_bot_pt = bot_neg_pt
        selected_side = "Negative side (-second_axis)"
        best_dist = dist_neg

    green_p1 = center
    green_p2 = best_top_pt
    green_p3 = best_bot_pt
    green_normal = plane_normal_from_points(green_p1, green_p2, green_p3)

    span = max(float(np.linalg.norm(np.ptp(np.asarray(mesh.vertices), axis=0))), 1.0)
    half_length = 0.35 * span
    marker_radius = max(0.016 * span, 1.0e-3)

    print(f"Loaded: {path}")
    print(f"Selected shorter side: {selected_side} with length = {best_dist:.4f}")

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.96, 0.97, 0.98)

    # 模型线框显示并进一步降低透明度至 0.08
    model = actor(mesh_data(mesh), (0.65, 0.68, 0.72), opacity=0.08)
    model.GetProperty().SetRepresentationToWireframe()
    renderer.AddActor(model)

    # 上下边界环
    renderer.AddActor(actor(polyline_data(top, closed=True), (0.95, 0.05, 0.05), width=4.0))
    renderer.AddActor(actor(polyline_data(bottom, closed=True), (0.05, 0.20, 0.95), width=4.0))

    # 中心与 PCA 轴
    renderer.AddActor(sphere_actor(bottom_center, marker_radius * 1.05, (0.55, 0.05, 0.80)))
    renderer.AddActor(sphere_actor(center, marker_radius, (1.0, 0.65, 0.0)))
    renderer.AddActor(axis_actor(center, axes[0], half_length, (0.9, 0.1, 0.1)))
    renderer.AddActor(axis_actor(center, axes[1], half_length, (0.1, 0.75, 0.1)))
    renderer.AddActor(axis_actor(center, axes[2], half_length, (0.1, 0.25, 0.95)))

    # 显示绿色方案选定的两端特征点
    renderer.AddActor(sphere_actor(green_p2, marker_radius * 1.1, (0.1, 0.9, 0.1)))
    renderer.AddActor(sphere_actor(green_p3, marker_radius * 1.1, (0.05, 0.95, 0.3)))

    # 5. 绘制【红色切平面】与模型的交线
    try:
        red_segments = trimesh.intersections.mesh_plane(mesh, plane_normal=red_normal, plane_origin=red_p1)
        red_curves = segments_to_continuous_polylines(red_segments)
        for rc in red_curves:
            rc_actor = actor(polyline_data(rc, closed=False), (0.95, 0.15, 0.15), width=4.0)
            rc_actor.GetProperty().SetRenderLinesAsTubes(True)
            renderer.AddActor(rc_actor)
        print(f"Red plane intersection curves: {len(red_curves)}")
    except Exception as e:
        print(f"红色平面交线计算失败: {e}")

    # 6. 绘制【绿色切平面】与模型的交线（高亮主切缝）
    try:
        green_segments = trimesh.intersections.mesh_plane(mesh, plane_normal=green_normal, plane_origin=green_p1)
        green_curves = segments_to_continuous_polylines(green_segments)

        target_curve = None
        other_curves = []
        min_cost = float("inf")

        for c in green_curves:
            cost1 = np.linalg.norm(c[0] - green_p2) + np.linalg.norm(c[-1] - green_p3)
            cost2 = np.linalg.norm(c[-1] - green_p2) + np.linalg.norm(c[0] - green_p3)
            cost = min(cost1, cost2)
            if cost < min_cost:
                min_cost = cost
                if target_curve is not None:
                    other_curves.append(target_curve)
                target_curve = c
            else:
                other_curves.append(c)

        # 绿色主目标表面交线（锁死两端点）
        if target_curve is not None:
            if np.linalg.norm(target_curve[0] - green_p2) > np.linalg.norm(target_curve[-1] - green_p2):
                target_curve = target_curve[::-1]
            target_curve[0] = green_p2
            target_curve[-1] = green_p3

            target_actor = actor(polyline_data(target_curve, closed=False), (0.05, 0.98, 0.15), width=5.5)
            target_actor.GetProperty().SetRenderLinesAsTubes(True)
            renderer.AddActor(target_actor)

        # 绿色平面对侧的次级交线
        for oc in other_curves:
            oc_actor = actor(polyline_data(oc, closed=False), (0.0, 0.55, 0.15), width=2.5)
            oc_actor.GetProperty().SetRenderLinesAsTubes(True)
            renderer.AddActor(oc_actor)

        print(f"Green plane intersection curves: {len(green_curves)}")
    except Exception as e:
        print(f"绿色平面交线计算失败: {e}")

    window = vtk.vtkRenderWindow()
    window.SetWindowName("Mesh Intersection Seam Curves Viewer (Red: Old Seam | Bright Green: New Seam)")
    window.SetSize(1300, 850)
    window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    window.Render()
    renderer.ResetCamera()
    window.Render()
    interactor.Initialize()
    interactor.Start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render STL intersection seam curves without plane geometry.")
    parser.add_argument("mesh", nargs="?", help="STL path; opens a file dialog when omitted")
    args = parser.parse_args()
    path = Path(args.mesh).resolve() if args.mesh else select_stl()
    if path is None:
        return
    try:
        show_model(path)
    except Exception as exc:
        try:
            messagebox.showerror("运行失败", str(exc))
        except tk.TclError:
            pass
        raise


if __name__ == "__main__":
    main()