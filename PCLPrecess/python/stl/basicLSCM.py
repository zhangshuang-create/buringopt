"""论文第 2.2 节（式 1-5、算法 1）的最小二乘共形映射实现。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr


def _boundary_vertices(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    """算法 1 第 4 步：出现一次的三角形边为网格边界。"""
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) == 0:
        raise ValueError("LSCM requires a surface mesh with a boundary.")
    boundary = np.unique(boundary_edges)
    if boundary[0] < 0 or boundary[-1] >= vertex_count:
        raise ValueError("Face indices are outside the vertex array.")
    return boundary


def _locked_boundary_pair(vertices: np.ndarray, boundary: np.ndarray) -> Tuple[int, int]:
    """从边界顶点中选取欧氏距离最远的一对作为论文算法 1 的两个锁定点。"""
    points = vertices[boundary]
    best_i, best_j, best_distance = 0, 1, -1.0
    for i in range(len(points) - 1):
        distances = np.einsum("ij,ij->i", points[i + 1:] - points[i], points[i + 1:] - points[i])
        j_local = int(np.argmax(distances))
        if float(distances[j_local]) > best_distance:
            best_i, best_j = i, i + 1 + j_local
            best_distance = float(distances[j_local])
    if best_distance <= 0.0:
        raise ValueError("Cannot select two distinct locked boundary vertices.")
    return int(boundary[best_i]), int(boundary[best_j])


def _triangle_matrix(points: np.ndarray) -> Tuple[float, np.ndarray]:
    """严格按照论文式 (1) 建立局部坐标，并按照式 (5) 计算 M_Tm。"""
    p1, p2, p3 = points
    edge12 = p2 - p1
    length12 = float(np.linalg.norm(edge12))
    if length12 <= np.finfo(float).eps:
        raise ValueError("Degenerate triangle: p_m1 and p_m2 coincide.")

    x_axis = edge12 / length12
    cross = np.cross(x_axis, p3 - p1)
    cross_length = float(np.linalg.norm(cross))
    if cross_length <= np.finfo(float).eps:
        raise ValueError("Degenerate triangle with zero area.")
    normal = cross / cross_length
    y_axis = np.cross(normal, x_axis)

    local_x = np.array([0.0, length12, float(np.dot(p3 - p1, x_axis))])
    local_y = np.array([0.0, 0.0, float(np.dot(p3 - p1, y_axis))])
    area = 0.5 * abs(
        (local_x[1] - local_x[0]) * (local_y[2] - local_y[0])
        - (local_x[2] - local_x[0]) * (local_y[1] - local_y[0])
    )
    if area <= np.finfo(float).eps:
        raise ValueError("Degenerate triangle with zero local area.")

    x1, x2, x3 = local_x
    y1, y2, y3 = local_y
    matrix = np.array(
        [[y2 - y3, y3 - y1, y1 - y2],
         [x3 - x2, x1 - x3, x2 - x1]],
        dtype=float,
    ) / (2.0 * area)
    return area, matrix


def lscm(
    vertices: np.ndarray,
    faces: np.ndarray,
    locked_vertices: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    按论文式 (4) 最小化 E_LSCM，返回与顶点一一对应的 (u, v)。

    locked_vertices 对应算法 1 第 6 步；未给定时选取最远的两个边界顶点。
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3).")
    if len(vertices) < 3 or len(faces) == 0:
        raise ValueError("The triangular mesh is empty.")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("Face indices are outside the vertex array.")

    vertex_count = len(vertices)
    boundary = _boundary_vertices(faces, vertex_count)
    if locked_vertices is None:
        lock0, lock1 = _locked_boundary_pair(vertices, boundary)
    else:
        lock0, lock1 = map(int, locked_vertices)
        if lock0 == lock1 or lock0 not in boundary or lock1 not in boundary:
            raise ValueError("The two locked vertices must be distinct boundary vertices.")

    rows = []
    columns = []
    values = []
    for triangle_index, face in enumerate(faces):
        area, matrix = _triangle_matrix(vertices[face])
        scale = np.sqrt(area)
        a, b = matrix[0], matrix[1]
        row0, row1 = 2 * triangle_index, 2 * triangle_index + 1
        for local_index, vertex_index in enumerate(face):
            # 式 (4): Mv - [[0,-1],[1,0]] Mu。
            rows.extend((row0, row0, row1, row1))
            columns.extend((vertex_index, vertex_count + vertex_index,
                            vertex_index, vertex_count + vertex_index))
            values.extend((scale * b[local_index], scale * a[local_index],
                           -scale * a[local_index], scale * b[local_index]))

    system = coo_matrix(
        (values, (rows, columns)),
        shape=(2 * len(faces), 2 * vertex_count),
    ).tocsr()

    locked_columns = np.array([lock0, lock1, vertex_count + lock0, vertex_count + lock1])
    locked_values = np.array([
        0.0,
        float(np.linalg.norm(vertices[lock1] - vertices[lock0])),
        0.0,
        0.0,
    ])
    free_mask = np.ones(2 * vertex_count, dtype=bool)
    free_mask[locked_columns] = False
    free_columns = np.flatnonzero(free_mask)
    right_hand_side = -(system[:, locked_columns] @ locked_values)
    free_solution = lsqr(system[:, free_columns], right_hand_side)[0]

    solution = np.zeros(2 * vertex_count, dtype=float)
    solution[locked_columns] = locked_values
    solution[free_columns] = free_solution
    return np.column_stack((solution[:vertex_count], solution[vertex_count:]))


def _select_mesh_file() -> str:
    """打开原生文件选择窗口，选择待参数化的三角网格。"""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select triangular mesh for LSCM",
        filetypes=[
            ("Mesh files", "*.stl *.obj *.ply"),
            ("STL files", "*.stl"),
            ("OBJ files", "*.obj"),
            ("PLY files", "*.ply"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path


def _load_triangle_mesh(path: str):
    """读取三角网格，并恢复 STL 中被重复存储顶点的原有邻接关系。"""
    import trimesh

    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [geometry for geometry in mesh.geometry.values()
                      if isinstance(geometry, trimesh.Trimesh)]
        if not geometries:
            raise ValueError("The selected file contains no triangular mesh.")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("The selected file contains no triangular mesh.")
    # STL 按三角形重复保存顶点；LSCM 必须在共享顶点的连通拓扑上组装能量。
    # 这里只合并完全重合的顶点，不做平滑、补洞或曲面重建。
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("The selected mesh must contain triangular faces.")
    return mesh, np.asarray(mesh.vertices, dtype=float), faces


def _show_parameterized_mesh(vertices: np.ndarray, uv: np.ndarray, faces: np.ndarray) -> None:
    """左侧显示原始三维网格，右侧显示论文 LSCM 的二维 UV 网格。"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(15, 7))
    axes_3d = figure.add_subplot(121, projection="3d")
    triangles = vertices[faces]
    surface = Poly3DCollection(
        triangles,
        facecolor=(0.68, 0.78, 0.88, 0.65),
        edgecolor=(0.2, 0.2, 0.2, 0.35),
        linewidth=0.25,
    )
    axes_3d.add_collection3d(surface)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.5 * max(float(np.max(maximum - minimum)), np.finfo(float).eps)
    axes_3d.set_xlim(center[0] - radius, center[0] + radius)
    axes_3d.set_ylim(center[1] - radius, center[1] + radius)
    axes_3d.set_zlim(center[2] - radius, center[2] + radius)
    axes_3d.set_box_aspect((1, 1, 1))
    axes_3d.set_xlabel("x")
    axes_3d.set_ylabel("y")
    axes_3d.set_zlabel("z")
    axes_3d.set_title("Original 3D mesh")

    axes_uv = figure.add_subplot(122)
    axes_uv.triplot(uv[:, 0], uv[:, 1], faces, color="black", linewidth=0.45)
    axes_uv.set_aspect("equal", adjustable="box")
    axes_uv.set_xlabel("u")
    axes_uv.set_ylabel("v")
    axes_uv.set_title("LSCM UV mesh")
    figure.tight_layout()
    plt.show()


def main() -> None:
    mesh_path = _select_mesh_file()
    if not mesh_path:
        return
    _, vertices, faces = _load_triangle_mesh(mesh_path)
    uv = lscm(vertices, faces)
    _show_parameterized_mesh(vertices, uv, faces)


if __name__ == "__main__":
    main()
