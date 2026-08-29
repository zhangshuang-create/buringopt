"""二维局部-全局 As-Rigid-As-Possible (ARAP) 三角网格参数化。"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def _boundary_vertices(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if not len(boundary_edges):
        raise ValueError("ARAP parameterization requires a mesh boundary.")
    return np.unique(boundary_edges)


def _farthest_pair(vertices: np.ndarray, boundary: np.ndarray) -> Tuple[int, int]:
    points = vertices[boundary]
    best = (0, 1)
    best_squared_distance = -1.0
    for i in range(len(points) - 1):
        squared_distances = np.einsum(
            "ij,ij->i", points[i + 1:] - points[i], points[i + 1:] - points[i]
        )
        j = int(np.argmax(squared_distances))
        if float(squared_distances[j]) > best_squared_distance:
            best = (i, i + 1 + j)
            best_squared_distance = float(squared_distances[j])
    if best_squared_distance <= 0.0:
        raise ValueError("Cannot select two distinct boundary vertices.")
    return int(boundary[best[0]]), int(boundary[best[1]])


def _triangle_local_coordinates(points: np.ndarray) -> np.ndarray:
    """将一个空间三角形刚性放置到自己的二维局部坐标系。"""
    p0, p1, p2 = points
    edge01 = p1 - p0
    length01 = float(np.linalg.norm(edge01))
    if length01 <= np.finfo(float).eps:
        raise ValueError("Degenerate triangle.")
    x_axis = edge01 / length01
    normal_vector = np.cross(x_axis, p2 - p0)
    normal_length = float(np.linalg.norm(normal_vector))
    if normal_length <= np.finfo(float).eps:
        raise ValueError("Degenerate triangle.")
    normal = normal_vector / normal_length
    y_axis = np.cross(normal, x_axis)
    return np.array([
        [0.0, 0.0],
        [length01, 0.0],
        [float(np.dot(p2 - p0, x_axis)), float(np.dot(p2 - p0, y_axis))],
    ])


def _cotangent(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(abs(a[0] * b[1] - a[1] * b[0]))
    if denominator <= np.finfo(float).eps:
        raise ValueError("Degenerate triangle.")
    return float(np.dot(a, b) / denominator)


def _reference_data(vertices: np.ndarray, faces: np.ndarray):
    local_triangles = np.empty((len(faces), 3, 2), dtype=float)
    triangle_weights = np.empty((len(faces), 3), dtype=float)
    # 每项对应边 (0,1)、(1,2)、(2,0)，权重为其对角的 cot(alpha)/2。
    for triangle_index, face in enumerate(faces):
        local = _triangle_local_coordinates(vertices[face])
        local_triangles[triangle_index] = local
        triangle_weights[triangle_index] = 0.5 * np.array([
            _cotangent(local[0] - local[2], local[1] - local[2]),
            _cotangent(local[1] - local[0], local[2] - local[0]),
            _cotangent(local[2] - local[1], local[0] - local[1]),
        ])
    return local_triangles, triangle_weights


def _initial_projection(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    return centered @ right_vectors[:2].T


def arap_parameterize(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 20,
    tolerance: float = 1.0e-8,
) -> np.ndarray:
    """使用局部旋转/全局位置交替优化计算二维 ARAP 参数坐标。"""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("faces must have shape (M, 3).")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("Face indices are outside the vertex array.")
    if iterations < 1:
        raise ValueError("iterations must be positive.")

    boundary = _boundary_vertices(faces)
    locked = np.array(_farthest_pair(vertices, boundary), dtype=np.int64)
    local_triangles, triangle_weights = _reference_data(vertices, faces)
    uv = _initial_projection(vertices)
    locked_values = uv[locked].copy()

    edge_pairs = ((0, 1), (1, 2), (2, 0))
    row_indices = []
    column_indices = []
    matrix_values = []
    for triangle_index, face in enumerate(faces):
        for edge_index, (local_i, local_j) in enumerate(edge_pairs):
            i, j = int(face[local_i]), int(face[local_j])
            weight = float(triangle_weights[triangle_index, edge_index])
            row_indices.extend((i, i, j, j))
            column_indices.extend((i, j, j, i))
            matrix_values.extend((weight, -weight, weight, -weight))
    laplacian = coo_matrix(
        (matrix_values, (row_indices, column_indices)),
        shape=(len(vertices), len(vertices)),
    ).tocsr()

    free_mask = np.ones(len(vertices), dtype=bool)
    free_mask[locked] = False
    free = np.flatnonzero(free_mask)
    free_system = laplacian[free][:, free]
    locked_system = laplacian[free][:, locked]

    for _ in range(iterations):
        rotations = np.empty((len(faces), 2, 2), dtype=float)
        # 局部步骤：逐三角形求使当前 UV 边最接近局部参考边的最佳旋转。
        for triangle_index, face in enumerate(faces):
            local = local_triangles[triangle_index]
            covariance = np.zeros((2, 2), dtype=float)
            for edge_index, (local_i, local_j) in enumerate(edge_pairs):
                source_edge = local[local_i] - local[local_j]
                uv_edge = uv[face[local_i]] - uv[face[local_j]]
                covariance += triangle_weights[triangle_index, edge_index] * np.outer(uv_edge, source_edge)
            left, _, right_t = np.linalg.svd(covariance)
            rotation = left @ right_t
            if np.linalg.det(rotation) < 0.0:
                left[:, -1] *= -1.0
                rotation = left @ right_t
            rotations[triangle_index] = rotation

        # 全局步骤：固定两个边界点，解最小二乘能量对应的稀疏线性系统。
        right_hand_side = np.zeros((len(vertices), 2), dtype=float)
        for triangle_index, face in enumerate(faces):
            local = local_triangles[triangle_index]
            rotation = rotations[triangle_index]
            for edge_index, (local_i, local_j) in enumerate(edge_pairs):
                i, j = int(face[local_i]), int(face[local_j])
                weight = float(triangle_weights[triangle_index, edge_index])
                target_edge = rotation @ (local[local_i] - local[local_j])
                right_hand_side[i] += weight * target_edge
                right_hand_side[j] -= weight * target_edge

        new_uv = uv.copy()
        rhs_free = right_hand_side[free] - locked_system @ locked_values
        new_uv[free, 0] = spsolve(free_system, rhs_free[:, 0])
        new_uv[free, 1] = spsolve(free_system, rhs_free[:, 1])
        new_uv[locked] = locked_values
        relative_change = float(np.linalg.norm(new_uv - uv) / max(np.linalg.norm(uv), 1.0))
        uv = new_uv
        if relative_change <= tolerance:
            break
    return uv


def _select_mesh_file() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select triangular mesh for ARAP",
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
    import trimesh

    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [geometry for geometry in mesh.geometry.values()
                      if isinstance(geometry, trimesh.Trimesh)]
        if not geometries:
            raise ValueError("The selected file contains no triangular mesh.")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("The selected file contains no triangular mesh.")
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("The selected mesh must contain triangular faces.")
    return np.asarray(mesh.vertices, dtype=float), faces


def _show_meshes(vertices: np.ndarray, uv: np.ndarray, faces: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(15, 7))
    axes_3d = figure.add_subplot(121, projection="3d")
    collection = Poly3DCollection(
        vertices[faces], facecolor=(0.68, 0.78, 0.88, 0.65),
        edgecolor=(0.2, 0.2, 0.2, 0.35), linewidth=0.25,
    )
    axes_3d.add_collection3d(collection)
    minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
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
    axes_uv.set_title("ARAP UV mesh")
    figure.tight_layout()
    plt.show()


def main() -> None:
    mesh_path = _select_mesh_file()
    if not mesh_path:
        return
    vertices, faces = _load_triangle_mesh(mesh_path)
    uv = arap_parameterize(vertices, faces)
    _show_meshes(vertices, uv, faces)


if __name__ == "__main__":
    main()
