#!/usr/bin/env python3
"""CycleCut graph preprocessing followed by an Isomap embedding.

This implementation follows the CycleCut procedure described by Mike Gashler
and Tony Martinez (2012): long atomic cycles in a neighborhood graph are found
with nested breadth-first searches, squeezed according to edge capacities,
and zero-capacity edges are removed.  A minimal-restoration pass then returns
every removed edge that does not recreate a long atomic cycle.  Classical
Isomap (shortest paths + MDS) is run on the resulting cut graph.

When no input is supplied, an STL file-selection dialog is opened.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

RUNTIME_CACHE = Path(__file__).resolve().parents[2] / ".runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

import Isomap as isomap_reference


Edge = Tuple[int, int]


@dataclass
class CycleCutResult:
    graph: object
    original_edges: np.ndarray
    edge_lengths: np.ndarray
    initial_capacities: np.ndarray
    residual_capacities: np.ndarray
    removed_edge_ids: np.ndarray
    restored_edge_ids: np.ndarray
    squeeze_iterations: int
    detected_cycle_lengths: np.ndarray

    @property
    def cut_edges(self) -> np.ndarray:
        return self.original_edges[self.removed_edge_ids]


def _edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def graph_edge_arrays(graph) -> tuple[np.ndarray, np.ndarray]:
    """Extract each undirected weighted edge exactly once."""
    matrix = coo_matrix(graph)
    mask = (matrix.row < matrix.col) & (matrix.data > 0.0)
    rows = np.asarray(matrix.row[mask], dtype=np.int64)
    columns = np.asarray(matrix.col[mask], dtype=np.int64)
    edges = np.column_stack((rows, columns))
    lengths = np.asarray(matrix.data[mask], dtype=float)
    return edges, lengths


def make_adjacency(vertex_count: int, edges: np.ndarray, active: np.ndarray) -> List[Set[int]]:
    adjacency: List[Set[int]] = [set() for _ in range(vertex_count)]
    for edge_id in np.flatnonzero(active):
        a, b = map(int, edges[edge_id])
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def _tree_cycle(
    a: int,
    b: int,
    parent: np.ndarray,
    depth: np.ndarray,
) -> List[int]:
    """Return the fundamental cycle formed by a BFS-tree non-tree edge."""
    left: List[int] = []
    right: List[int] = []
    u, v = int(a), int(b)
    while depth[u] > depth[v]:
        left.append(u)
        u = int(parent[u])
    while depth[v] > depth[u]:
        right.append(v)
        v = int(parent[v])
    while u != v:
        left.append(u)
        right.append(v)
        u = int(parent[u])
        v = int(parent[v])
    left.append(u)
    return left + right[::-1]


def _cycle_arcs(cycle: Sequence[int], start: int, end: int) -> Tuple[List[int], List[int]]:
    count = len(cycle)
    forward = [int(cycle[start])]
    index = start
    while index != end:
        index = (index + 1) % count
        forward.append(int(cycle[index]))
    backward = [int(cycle[start])]
    index = start
    while index != end:
        index = (index - 1) % count
        backward.append(int(cycle[index]))
    return forward, backward


def find_n_chord(
    cycle: Sequence[int],
    adjacency: Sequence[Set[int]],
) -> Optional[Tuple[List[int], int, int]]:
    """Nested BFS: find a path shorter than both arcs between cycle vertices.

    Cycle edges themselves are excluded from the inner BFS.  Reaching another
    cycle vertex in fewer hops than the shorter on-cycle arc identifies an
    n-chord.  The returned path includes both cycle endpoints.
    """
    cycle = list(map(int, cycle))
    count = len(cycle)
    if count < 4:
        return None
    positions = {vertex: index for index, vertex in enumerate(cycle)}
    cycle_edges = {
        _edge(cycle[index], cycle[(index + 1) % count]) for index in range(count)
    }
    maximum_depth = (count - 1) // 2

    for start_index, start in enumerate(cycle):
        queue = deque([start])
        parent: Dict[int, int] = {start: -1}
        distance: Dict[int, int] = {start: 0}
        while queue:
            current = queue.popleft()
            current_distance = distance[current]
            if current_distance >= maximum_depth:
                continue
            for neighbor in adjacency[current]:
                if _edge(current, neighbor) in cycle_edges:
                    continue
                if neighbor in parent:
                    continue
                new_distance = current_distance + 1
                if neighbor in positions and neighbor != start:
                    end_index = positions[neighbor]
                    separation = abs(end_index - start_index)
                    shorter_arc = min(separation, count - separation)
                    if new_distance < shorter_arc:
                        path = [neighbor]
                        cursor = current
                        while cursor != -1:
                            path.append(cursor)
                            cursor = parent[cursor]
                        path.reverse()
                        return path, start_index, end_index
                    # Do not pass through a cycle vertex that failed the
                    # shortcut test; an n-chord may only touch at its ends.
                    continue
                parent[neighbor] = current
                distance[neighbor] = new_distance
                queue.append(neighbor)
    return None


def _split_cycle_with_chord(
    cycle: Sequence[int],
    chord_path: Sequence[int],
    start_index: int,
    end_index: int,
) -> Tuple[List[int], List[int]]:
    forward, backward = _cycle_arcs(cycle, start_index, end_index)
    # chord_path runs start -> end. Append the reverse interior of each arc to
    # return from end to start without duplicating either endpoint.
    first = list(map(int, chord_path)) + forward[-2:0:-1]
    second = list(map(int, chord_path)) + backward[-2:0:-1]
    return first, second


def reduce_to_long_atomic_cycle(
    cycle: Sequence[int],
    adjacency: Sequence[Set[int]],
    cycle_threshold: int,
    recursion_budget: int = 128,
) -> Optional[List[int]]:
    """Recursively split on n-chords until a long atomic descendant remains."""
    pending: List[List[int]] = [list(map(int, cycle))]
    inspected = 0
    while pending and inspected < recursion_budget:
        candidate = pending.pop()
        inspected += 1
        if len(candidate) < cycle_threshold or len(set(candidate)) != len(candidate):
            continue
        chord = find_n_chord(candidate, adjacency)
        if chord is None:
            return candidate
        path, start, end = chord
        descendants = _split_cycle_with_chord(candidate, path, start, end)
        descendants = sorted(descendants, key=len)
        for descendant in descendants:
            if len(descendant) >= cycle_threshold:
                pending.append(descendant)
    return None


def find_long_atomic_cycle(
    adjacency: Sequence[Set[int]],
    cycle_threshold: int,
    search_roots: int = 32,
) -> Optional[List[int]]:
    """Outer BFS cycle discovery followed by inner-BFS n-chord reduction."""
    vertex_count = len(adjacency)
    degrees = np.asarray([len(items) for items in adjacency], dtype=np.int64)
    candidates = np.flatnonzero(degrees > 1)
    if not len(candidates):
        return None
    if search_roots <= 0 or search_roots >= len(candidates):
        roots = candidates
    else:
        order = np.argsort(degrees[candidates])[::-1]
        high_degree = candidates[order[: max(1, search_roots // 2)]]
        spread = candidates[
            np.linspace(0, len(candidates) - 1, search_roots - len(high_degree), dtype=int)
        ]
        roots = np.unique(np.concatenate((high_degree, spread)))

    tested_edges: Set[Edge] = set()
    for root in roots:
        parent = np.full(vertex_count, -1, dtype=np.int64)
        depth = np.full(vertex_count, -1, dtype=np.int64)
        depth[root] = 0
        queue = deque([int(root)])
        non_tree_edges: List[Tuple[int, int]] = []
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if depth[neighbor] < 0:
                    depth[neighbor] = depth[current] + 1
                    parent[neighbor] = current
                    queue.append(neighbor)
                elif parent[current] != neighbor and parent[neighbor] != current:
                    edge = _edge(current, neighbor)
                    if edge not in tested_edges:
                        tested_edges.add(edge)
                        non_tree_edges.append(edge)
        non_tree_edges.sort(
            key=lambda item: depth[item[0]] + depth[item[1]], reverse=True
        )
        for a, b in non_tree_edges:
            fundamental = _tree_cycle(a, b, parent, depth)
            if len(fundamental) < cycle_threshold:
                continue
            atomic = reduce_to_long_atomic_cycle(
                fundamental, adjacency, cycle_threshold
            )
            if atomic is not None:
                return atomic
    return None


def _shortest_hop_distance(
    adjacency: Sequence[Set[int]],
    start: int,
    end: int,
) -> Optional[int]:
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current, distance = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor == end:
                return distance + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def cycle_cut(
    graph,
    cycle_threshold: int = 20,
    capacity_mode: str = "uniform",
    search_roots: int = 32,
    max_squeeze_iterations: int = 10000,
) -> CycleCutResult:
    """Cut all discovered atomic cycles with length >= cycle_threshold."""
    if cycle_threshold < 4:
        raise ValueError("cycle_threshold must be at least 4")
    if capacity_mode not in {"uniform", "inverse-distance"}:
        raise ValueError("capacity_mode must be uniform or inverse-distance")
    edges, lengths = graph_edge_arrays(graph)
    if not len(edges):
        raise ValueError("The neighborhood graph has no edges")
    if capacity_mode == "uniform":
        initial = np.ones(len(edges), dtype=float)
    else:
        median = max(float(np.median(lengths)), np.finfo(float).eps)
        initial = np.clip(median / np.maximum(lengths, 1.0e-12), 0.25, 4.0)
    residual = initial.copy()
    active = np.ones(len(edges), dtype=bool)
    edge_ids = {_edge(int(a), int(b)): index for index, (a, b) in enumerate(edges)}
    adjacency = make_adjacency(graph.shape[0], edges, active)
    removed_order: List[int] = []
    cycle_lengths: List[int] = []

    for iteration in range(max_squeeze_iterations):
        cycle = find_long_atomic_cycle(adjacency, cycle_threshold, search_roots)
        if cycle is None:
            break
        cycle_edge_ids = np.asarray([
            edge_ids[_edge(cycle[index], cycle[(index + 1) % len(cycle)])]
            for index in range(len(cycle))
        ], dtype=np.int64)
        height = float(np.min(residual[cycle_edge_ids]))
        residual[cycle_edge_ids] -= height
        saturated = cycle_edge_ids[residual[cycle_edge_ids] <= 1.0e-12]
        if not len(saturated):
            raise RuntimeError("Cycle squeezing made no progress")
        for edge_id in np.unique(saturated):
            edge_id = int(edge_id)
            if not active[edge_id]:
                continue
            active[edge_id] = False
            a, b = map(int, edges[edge_id])
            adjacency[a].discard(b)
            adjacency[b].discard(a)
            removed_order.append(edge_id)
        cycle_lengths.append(len(cycle))
    else:
        raise RuntimeError(
            "CycleCut reached max_squeeze_iterations; increase the limit or lambda"
        )

    # Minimal restoration. Try every saturated edge in reverse removal order.
    # Edges that reconnect components are always safe because they create no
    # cycle. Otherwise perform the paper's direct test: tentatively restore the
    # edge and retain it only when no long atomic cycle reappears.
    restored: List[int] = []
    for edge_id in reversed(removed_order):
        a, b = map(int, edges[edge_id])
        distance = _shortest_hop_distance(adjacency, a, b)
        active[edge_id] = True
        adjacency[a].add(b)
        adjacency[b].add(a)
        if distance is not None:
            recreated = find_long_atomic_cycle(
                adjacency, cycle_threshold, search_roots
            )
            if recreated is not None:
                active[edge_id] = False
                adjacency[a].discard(b)
                adjacency[b].discard(a)
                continue
        restored.append(edge_id)

    final_removed = np.flatnonzero(~active).astype(np.int64)
    rows = edges[active, 0]
    columns = edges[active, 1]
    weights = lengths[active]
    cut_graph = coo_matrix(
        (
            np.concatenate((weights, weights)),
            (
                np.concatenate((rows, columns)),
                np.concatenate((columns, rows)),
            ),
        ),
        shape=graph.shape,
    ).tocsr()
    component_count, labels = connected_components(cut_graph, directed=False)
    if component_count != 1:
        sizes = np.bincount(labels)
        raise RuntimeError(
            "Minimal restoration left the graph disconnected: "
            f"{component_count} components, sizes={sizes.tolist()}"
        )
    return CycleCutResult(
        graph=cut_graph,
        original_edges=edges,
        edge_lengths=lengths,
        initial_capacities=initial,
        residual_capacities=residual,
        removed_edge_ids=final_removed,
        restored_edge_ids=np.asarray(restored, dtype=np.int64),
        squeeze_iterations=len(cycle_lengths),
        detected_cycle_lengths=np.asarray(cycle_lengths, dtype=np.int64),
    )


def run_cyclecut_isomap(
    points: np.ndarray,
    neighbors: int = 12,
    cycle_threshold: int = 20,
    capacity_mode: str = "uniform",
    search_roots: int = 32,
) -> tuple[np.ndarray, CycleCutResult, str]:
    """Build kNN graph, apply CycleCut, then run shortest paths and MDS."""
    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        raise ValueError("At least four points are required")
    neighbors = min(max(2, int(neighbors)), len(points) - 1)
    graph = isomap_reference.build_knn_graph(points, neighbors, use_gpu=False)
    component_count, _ = connected_components(graph, directed=False)
    if component_count != 1:
        raise ValueError(
            f"The original {neighbors}-NN graph has {component_count} components; "
            "increase --neighbors"
        )
    result = cycle_cut(
        graph,
        cycle_threshold=cycle_threshold,
        capacity_mode=capacity_mode,
        search_roots=search_roots,
    )
    geodesic = dijkstra(result.graph, directed=False)
    if not np.isfinite(geodesic).all():
        raise RuntimeError("The restored CycleCut graph is disconnected")
    uv = isomap_reference.classical_mds_cpu(geodesic)
    return uv, result, "CPU CycleCut + Dijkstra + classical MDS"


def save_cut_edges_csv(
    path: str | Path,
    points: np.ndarray,
    result: CycleCutResult,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "cut_index", "node_a", "node_b", "length", "initial_capacity",
            "residual_capacity", "ax", "ay", "az", "bx", "by", "bz",
        ])
        for cut_index, edge_id in enumerate(result.removed_edge_ids):
            a, b = map(int, result.original_edges[edge_id])
            writer.writerow([
                cut_index, a, b, result.edge_lengths[edge_id],
                result.initial_capacities[edge_id], result.residual_capacities[edge_id],
                *points[a], *points[b],
            ])


def plot_cyclecut_result(
    points: np.ndarray,
    uv: np.ndarray,
    result: CycleCutResult,
    output: str | Path = "",
    show: bool = True,
) -> None:
    figure = plt.figure(figsize=(13, 5.5))
    ax3d = figure.add_subplot(1, 2, 1, projection="3d")
    ax3d.scatter(*points.T, c=uv[:, 0], cmap="viridis", s=4, alpha=0.75)
    for a, b in result.cut_edges:
        segment = points[[a, b]]
        ax3d.plot(*segment.T, color="crimson", linewidth=1.6, alpha=0.9)
    ax3d.set_title(f"CycleCut removed edges ({len(result.cut_edges)})")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    ax2d = figure.add_subplot(1, 2, 2)
    ax2d.scatter(uv[:, 0], uv[:, 1], c=uv[:, 0], cmap="viridis", s=5)
    ax2d.set_title("CycleCut + Isomap 2-D embedding")
    ax2d.set_xlabel("u")
    ax2d.set_ylabel("v")
    ax2d.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    if output:
        figure.savefig(output, dpi=190)
    if show:
        plt.show()
    else:
        plt.close(figure)


def select_stl_file() -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("No STL path was provided and tkinter is unavailable") from exc
    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title="Select an STL model for CycleCut",
        filetypes=[("STL models", "*.stl"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(selected) if selected else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl", nargs="?", default="")
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument(
        "--cycle-threshold", "--lambda", dest="cycle_threshold",
        type=int, default=20,
        help="Minimum atomic-cycle length lambda (default: 20)",
    )
    parser.add_argument(
        "--capacity", choices=("uniform", "inverse-distance"), default="uniform",
    )
    parser.add_argument("--search-roots", type=int, default=32)
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="cyclecut_points.csv")
    parser.add_argument("--cut-edges-out", default="cyclecut_removed_edges.csv")
    parser.add_argument("--save-figure", default="")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stl_path = Path(args.stl) if args.stl else select_stl_file()
    if stl_path is None:
        raise SystemExit("No STL model selected")
    if stl_path.suffix.lower() != ".stl":
        raise SystemExit("The selected input must be an STL model")
    print(f"Input STL: {stl_path.resolve()}")
    mesh = isomap_reference.load_stl(stl_path)
    rng = np.random.default_rng(args.seed)
    points, face_ids = isomap_reference.sample_surface_uniform_density(
        mesh, args.density, rng,
        max_points=args.max_points,
        boundary_multiplier=0.0,
    )
    print(f"Sampled points: {len(points)}")
    print(
        f"Building {args.neighbors}-NN graph and cutting atomic cycles with "
        f"lambda={args.cycle_threshold} ..."
    )
    uv, result, backend = run_cyclecut_isomap(
        points,
        neighbors=args.neighbors,
        cycle_threshold=args.cycle_threshold,
        capacity_mode=args.capacity,
        search_roots=args.search_roots,
    )
    print(f"Squeeze iterations: {result.squeeze_iterations}")
    print(f"Temporarily removed edges: {len(result.removed_edge_ids) + len(result.restored_edge_ids)}")
    print(f"Restored edges: {len(result.restored_edge_ids)}")
    print(f"Final cut edges: {len(result.removed_edge_ids)}")
    if len(result.detected_cycle_lengths):
        print(
            "Detected atomic-cycle lengths: "
            f"min={result.detected_cycle_lengths.min()}, "
            f"max={result.detected_cycle_lengths.max()}, "
            f"count={len(result.detected_cycle_lengths)}"
        )
    print(f"Backend: {backend}")
    isomap_reference.save_csv(args.out, points, uv, face_ids)
    save_cut_edges_csv(args.cut_edges_out, points, result)
    print(f"Embedding CSV: {Path(args.out).resolve()}")
    print(f"Cut-edge CSV: {Path(args.cut_edges_out).resolve()}")
    if not args.no_show or args.save_figure:
        if args.no_show:
            plt.switch_backend("Agg")
        plot_cyclecut_result(
            points, uv, result,
            output=args.save_figure,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
