from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def _ordered_boundary(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if not len(boundary_edges):
        raise ValueError("SLIM parameterization requires an open mesh boundary.")
    adjacency = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("The mesh boundary must be a manifold loop.")
    start = min(adjacency)
    ordered = [start]
    previous, current = -1, start
    while True:
        candidates = [item for item in adjacency[current] if item != previous]
        following = candidates[0]
        if following == start:
            break
        ordered.append(following)
        previous, current = current, following
        if len(ordered) > len(adjacency):
            raise ValueError("Invalid mesh boundary.")
    if len(ordered) != len(adjacency):
        raise ValueError("SLIM currently requires one connected boundary loop.")
    return np.asarray(ordered, dtype=np.int64)


def _triangle_reference(points: np.ndarray):
    p0, p1, p2 = points
    edge = p1 - p0
    length = float(np.linalg.norm(edge))
    if length <= np.finfo(float).eps:
        raise ValueError("Degenerate triangle.")
    x_axis = edge / length
    cross = np.cross(x_axis, p2 - p0)
    cross_length = float(np.linalg.norm(cross))
    if cross_length <= np.finfo(float).eps:
        raise ValueError("Degenerate triangle.")
    y_axis = np.cross(cross / cross_length, x_axis)
    local = np.array([
        [0.0, 0.0], [length, 0.0],
        [float(np.dot(p2 - p0, x_axis)), float(np.dot(p2 - p0, y_axis))],
    ])
    source_edges = np.column_stack((local[1] - local[0], local[2] - local[0]))
    determinant = float(np.linalg.det(source_edges))
    if determinant <= np.finfo(float).eps:
        raise ValueError("Degenerate triangle.")
    return np.linalg.inv(source_edges), 0.5 * determinant


def _injective_initialization(vertices: np.ndarray, faces: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """将边界按弧长放到圆周，再用 Tutte 调和方程获得无翻转初值。"""
    boundary_points = vertices[boundary]
    lengths = np.linalg.norm(np.roll(boundary_points, -1, axis=0) - boundary_points, axis=1)
    perimeter = float(np.sum(lengths))
    if perimeter <= np.finfo(float).eps:
        raise ValueError("The boundary has zero length.")
    angles = 2.0 * np.pi * np.concatenate(([0.0], np.cumsum(lengths[:-1]))) / perimeter
    uv = np.zeros((len(vertices), 2), dtype=float)
    uv[boundary] = np.column_stack((np.cos(angles), np.sin(angles)))

    mesh_edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    mesh_edges = np.unique(np.sort(mesh_edges, axis=1), axis=0)
    
    # 向量化构建拉普拉斯矩阵
    num_vertices = len(vertices)
    num_edges = len(mesh_edges)
    rows = np.concatenate([mesh_edges[:, 0], mesh_edges[:, 0], mesh_edges[:, 1], mesh_edges[:, 1]])
    columns = np.concatenate([mesh_edges[:, 0], mesh_edges[:, 1], mesh_edges[:, 1], mesh_edges[:, 0]])
    values = np.concatenate([np.ones(num_edges), -np.ones(num_edges), np.ones(num_edges), -np.ones(num_edges)])
    
    laplacian = coo_matrix((values, (rows, columns)), shape=(num_vertices, num_vertices)).tocsr()
    free_mask = np.ones(len(vertices), dtype=bool)
    free_mask[boundary] = False
    free = np.flatnonzero(free_mask)
    if len(free):
        right = -(laplacian[free][:, boundary] @ uv[boundary])
        uv[free, 0] = spsolve(laplacian[free][:, free], right[:, 0])
        uv[free, 1] = spsolve(laplacian[free][:, free], right[:, 1])
    return uv


def slim_parameterize(vertices: np.ndarray, faces: np.ndarray, iterations: int = 100) -> np.ndarray:
    """优化对称 Dirichlet 能量并保持所有三角形局部可逆（高度向量化加速版）。"""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("faces must have shape (M, 3).")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("Face indices are outside the vertex array.")

    boundary = _ordered_boundary(faces)
    inverse_sources = np.empty((len(faces), 2, 2), dtype=float)
    areas = np.empty(len(faces), dtype=float)
    for index, face in enumerate(faces):
        inverse_sources[index], areas[index] = _triangle_reference(vertices[face])
    uv = _injective_initialization(vertices, faces, boundary)

    free_mask = np.ones(len(vertices), dtype=bool)
    free_mask[boundary] = False
    free = np.flatnonzero(free_mask)
    if not len(free):
        return uv

    def energy_and_gradient(flat_coordinates):
        current = uv.copy()
        current[free] = flat_coordinates.reshape((-1, 2))
        
        # 1. 批量提取三角形顶点坐标 (M, 3, 2)
        triangles = current[faces]
        
        # 2. 批量构建目标边向量 (M, 2)
        E1 = triangles[:, 1] - triangles[:, 0]
        E2 = triangles[:, 2] - triangles[:, 0]
        
        # 3. 批量构建目标边矩阵 (M, 2, 2) 并计算雅氏矩阵 (M, 2, 2)
        target_edges = np.stack((E1, E2), axis=-1)
        jacobian = target_edges @ inverse_sources
        
        # 4. 批量计算行列式 (M,)
        det = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
        
        # 阻碍条件处理（若有三角形翻转，返回大值）
        if np.any(det <= 1.0e-10):
            return 1.0e30, np.zeros_like(flat_coordinates)
        
        # 5. 批量求解 2x2 矩阵的逆 (M, 2, 2)
        inv_jacobian = np.empty_like(jacobian)
        inv_jacobian[:, 0, 0] = jacobian[:, 1, 1] / det
        inv_jacobian[:, 1, 1] = jacobian[:, 0, 0] / det
        inv_jacobian[:, 0, 1] = -jacobian[:, 0, 1] / det
        inv_jacobian[:, 1, 0] = -jacobian[:, 1, 0] / det
        
        # 6. 计算 Symmetric Dirichlet 能量 (标量)
        f2_jac = np.sum(jacobian ** 2, axis=(1, 2))
        f2_inv = np.sum(inv_jacobian ** 2, axis=(1, 2))
        energy = np.sum(areas * (f2_jac + f2_inv))
        
        # 7. 批量计算关于雅氏矩阵的梯度 (M, 2, 2)
        # 对应数学公式: dE/dJ = 2*J - 2*(J^-1)^T * J^-1 * (J^-1)^T
        inv_j_t = inv_jacobian.transpose(0, 2, 1)
        inv_term = inv_j_t @ inv_jacobian @ inv_j_t
        jacobian_gradient = areas[:, np.newaxis, np.newaxis] * (2.0 * jacobian - 2.0 * inv_term)
        
        # 8. 批量计算关于边向量的梯度 (M, 2, 2)
        edge_gradient = jacobian_gradient @ inverse_sources.transpose(0, 2, 1)
        
        # 9. 批量累加梯度到对应顶点上 (N, 2)
        gradient = np.zeros_like(current)
        np.add.at(gradient, faces[:, 1], edge_gradient[:, :, 0])
        np.add.at(gradient, faces[:, 2], edge_gradient[:, :, 1])
        np.add.at(gradient, faces[:, 0], -(edge_gradient[:, :, 0] + edge_gradient[:, :, 1]))
        
        return energy, gradient[free].reshape(-1)

    result = minimize(
        energy_and_gradient,
        uv[free].reshape(-1),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(iterations), "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    candidate = uv.copy()
    candidate[free] = result.x.reshape((-1, 2))
    
    # 终检面翻转情况
    triangles = candidate[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    signed_double_area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    if np.any(signed_double_area <= 0.0):
        raise RuntimeError("SLIM optimization produced a flipped triangle.")
    return candidate


def _select_mesh_file() -> str:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select triangular mesh for SLIM",
        filetypes=[("Mesh files", "*.stl *.obj *.ply"), ("STL files", "*.stl"),
                   ("OBJ files", "*.obj"), ("PLY files", "*.ply"), ("All files", "*.*")],
    )
    root.destroy()
    return path


def _load_triangle_mesh(path: str):
    import trimesh
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [item for item in mesh.geometry.values() if isinstance(item, trimesh.Trimesh)]
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
    axes_3d.add_collection3d(Poly3DCollection(
        vertices[faces], facecolor=(0.68, 0.78, 0.88, 0.65),
        edgecolor=(0.2, 0.2, 0.2, 0.35), linewidth=0.25))
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
    axes_uv.set_title("SLIM UV mesh")
    figure.tight_layout()
    plt.show()


def main() -> None:
    path = _select_mesh_file()
    if not path:
        return
    vertices, faces = _load_triangle_mesh(path)
    uv = slim_parameterize(vertices, faces)
    _show_meshes(vertices, uv, faces)


if __name__ == "__main__":
    main()