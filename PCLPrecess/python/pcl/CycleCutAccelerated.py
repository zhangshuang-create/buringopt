#!/usr/bin/env python3
"""Accelerated hybrid CPU/GPU CycleCut followed by Isomap.

Acceleration strategy:
* CuPy performs k-nearest-neighbor search and classical MDS on CUDA.
* Independent outer-BFS roots are searched in multiple CPU processes.
* Removed edges use a fast incremental restoration test instead of a complete
  atomic-cycle scan after every edge.
* Fast restoration uses local n-chord/connectivity tests by default. Optional
  ``--exact-restoration`` performs slower global batch verification.

The original CycleCut.py is left unchanged for reference.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

RUNTIME_CACHE = Path(__file__).resolve().parents[2] / ".runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("CUPY_CACHE_DIR", str(RUNTIME_CACHE / "cupy"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

import CycleCut as cyclecut_reference
import Isomap as isomap_reference


class DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int64)
        self.rank = np.zeros(count, dtype=np.int8)

    def find(self, value: int) -> int:
        root = int(value)
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            next_value = int(self.parent[value])
            self.parent[value] = root
            value = next_value
        return root

    def union(self, first: int, second: int) -> bool:
        a, b = self.find(first), self.find(second)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        return True


def _choose_roots(adjacency: Sequence[Set[int]], search_roots: int) -> np.ndarray:
    degrees = np.asarray([len(items) for items in adjacency], dtype=np.int64)
    candidates = np.flatnonzero(degrees > 1)
    if not len(candidates):
        return np.empty(0, dtype=np.int64)
    if search_roots <= 0 or search_roots >= len(candidates):
        return candidates
    high_count = max(1, search_roots // 2)
    order = np.argsort(degrees[candidates])[::-1]
    high = candidates[order[:high_count]]
    spread_count = max(0, search_roots - len(high))
    spread = (
        candidates[np.linspace(0, len(candidates) - 1, spread_count, dtype=int)]
        if spread_count else np.empty(0, dtype=np.int64)
    )
    return np.unique(np.concatenate((high, spread)))


def _search_root_chunk(
    adjacency: Sequence[Set[int]],
    roots: Sequence[int],
    cycle_threshold: int,
) -> Optional[List[int]]:
    """Worker-safe atomic-cycle search over an explicit subset of BFS roots."""
    vertex_count = len(adjacency)
    tested_edges = set()
    for root_value in roots:
        root = int(root_value)
        parent = np.full(vertex_count, -1, dtype=np.int64)
        depth = np.full(vertex_count, -1, dtype=np.int64)
        depth[root] = 0
        queue = deque([root])
        non_tree_edges: List[Tuple[int, int]] = []
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if depth[neighbor] < 0:
                    depth[neighbor] = depth[current] + 1
                    parent[neighbor] = current
                    queue.append(neighbor)
                elif parent[current] != neighbor and parent[neighbor] != current:
                    edge = cyclecut_reference._edge(current, neighbor)
                    if edge not in tested_edges:
                        tested_edges.add(edge)
                        non_tree_edges.append(edge)
        non_tree_edges.sort(
            key=lambda item: depth[item[0]] + depth[item[1]], reverse=True
        )
        for a, b in non_tree_edges:
            fundamental = cyclecut_reference._tree_cycle(a, b, parent, depth)
            if len(fundamental) < cycle_threshold:
                continue
            atomic = cyclecut_reference.reduce_to_long_atomic_cycle(
                fundamental,
                adjacency,
                cycle_threshold,
            )
            if atomic is not None:
                return atomic
    return None


def find_long_atomic_cycle_parallel(
    adjacency: Sequence[Set[int]],
    cycle_threshold: int,
    search_roots: int,
    workers: int,
    executor: Optional[ProcessPoolExecutor],
) -> Optional[List[int]]:
    roots = _choose_roots(adjacency, search_roots)
    if not len(roots):
        return None
    if workers <= 1 or executor is None or len(roots) == 1:
        return _search_root_chunk(adjacency, roots, cycle_threshold)
    chunks = [chunk for chunk in np.array_split(roots, min(workers, len(roots))) if len(chunk)]
    futures = [
        executor.submit(_search_root_chunk, adjacency, chunk.tolist(), cycle_threshold)
        for chunk in chunks
    ]
    # Inspect results in root-chunk order, not completion order, so the same
    # seed and parameters select the same cycle on every run.
    found = None
    for future in futures:
        candidate = future.result()
        if candidate is not None:
            found = candidate
            for pending in futures:
                if pending is not future:
                    pending.cancel()
            break
    # Running tasks cannot always be cancelled; consume them before the next
    # graph version is submitted to the persistent process pool.
    for future in futures:
        if not future.cancelled():
            try:
                future.result()
            except Exception:
                if found is None:
                    raise
    return found


def _limited_bidirectional_distance(
    adjacency: Sequence[Set[int]],
    start: int,
    end: int,
    maximum: int,
) -> Optional[int]:
    """Find a short path using two frontiers; None means no path within limit."""
    if start == end:
        return 0
    front_a = {int(start)}
    front_b = {int(end)}
    distance_a = {int(start): 0}
    distance_b = {int(end): 0}
    for _ in range(maximum):
        if not front_a or not front_b:
            return None
        if len(front_a) <= len(front_b):
            current_front = front_a
            front_a = set()
            for current in current_front:
                depth = distance_a[current]
                for neighbor in adjacency[current]:
                    if neighbor in distance_a:
                        continue
                    new_depth = depth + 1
                    if neighbor in distance_b:
                        total = new_depth + distance_b[neighbor]
                        return total if total <= maximum else None
                    distance_a[neighbor] = new_depth
                    front_a.add(neighbor)
        else:
            current_front = front_b
            front_b = set()
            for current in current_front:
                depth = distance_b[current]
                for neighbor in adjacency[current]:
                    if neighbor in distance_b:
                        continue
                    new_depth = depth + 1
                    if neighbor in distance_a:
                        total = new_depth + distance_a[neighbor]
                        return total if total <= maximum else None
                    distance_b[neighbor] = new_depth
                    front_b.add(neighbor)
    return None


def _reachable_without_edge(
    adjacency: Sequence[Set[int]], start: int, end: int
) -> bool:
    queue = deque([int(start)])
    visited = {int(start)}
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor == end:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def accelerated_cycle_cut(
    graph,
    cycle_threshold: int = 20,
    capacity_mode: str = "uniform",
    search_roots: int = 16,
    workers: int = 4,
    exact_restoration: bool = False,
    progress: bool = True,
) -> cyclecut_reference.CycleCutResult:
    if cycle_threshold < 4:
        raise ValueError("cycle_threshold must be at least 4")
    edges, lengths = cyclecut_reference.graph_edge_arrays(graph)
    if capacity_mode == "uniform":
        initial = np.ones(len(edges), dtype=float)
    elif capacity_mode == "inverse-distance":
        median = max(float(np.median(lengths)), np.finfo(float).eps)
        initial = np.clip(median / np.maximum(lengths, 1.0e-12), 0.25, 4.0)
    else:
        raise ValueError("capacity_mode must be uniform or inverse-distance")

    residual = initial.copy()
    active = np.ones(len(edges), dtype=bool)
    edge_ids = {
        cyclecut_reference._edge(int(a), int(b)): index
        for index, (a, b) in enumerate(edges)
    }
    adjacency = cyclecut_reference.make_adjacency(graph.shape[0], edges, active)
    removed_order: List[int] = []
    cycle_lengths: List[int] = []
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    started = time.perf_counter()
    try:
        iteration = 0
        while True:
            iteration += 1
            search_start = time.perf_counter()
            if progress:
                print(
                    f"  [squeeze {iteration}] searching {search_roots} BFS roots "
                    f"with {workers} worker(s) ...",
                    flush=True,
                )
            cycle = find_long_atomic_cycle_parallel(
                adjacency,
                cycle_threshold,
                search_roots,
                workers,
                executor,
            )
            if cycle is None:
                if progress:
                    print(
                        f"  no more atomic cycles (search {time.perf_counter()-search_start:.2f}s)",
                        flush=True,
                    )
                break
            cycle_edge_ids = np.asarray([
                edge_ids[cyclecut_reference._edge(
                    cycle[index], cycle[(index + 1) % len(cycle)]
                )]
                for index in range(len(cycle))
            ], dtype=np.int64)
            height = float(np.min(residual[cycle_edge_ids]))
            residual[cycle_edge_ids] -= height
            saturated = np.unique(
                cycle_edge_ids[residual[cycle_edge_ids] <= 1.0e-12]
            )
            removed_now = 0
            for edge_id in saturated:
                edge_id = int(edge_id)
                if not active[edge_id]:
                    continue
                active[edge_id] = False
                a, b = map(int, edges[edge_id])
                adjacency[a].discard(b)
                adjacency[b].discard(a)
                removed_order.append(edge_id)
                removed_now += 1
            cycle_lengths.append(len(cycle))
            if progress:
                print(
                    f"  found length={len(cycle)}, saturated={removed_now}, "
                    f"active_edges={int(active.sum())}, "
                    f"search={time.perf_counter()-search_start:.2f}s",
                    flush=True,
                )

        # Build component state once. Edges joining different components are
        # always safe because they create no cycle. Remaining edges are tested
        # in batches: accept a whole batch when it creates no long atomic
        # cycle; otherwise split it recursively. This preserves the direct
        # restoration criterion while requiring far fewer global searches than
        # one complete scan per removed edge.
        disjoint = DisjointSet(graph.shape[0])
        for edge_id in np.flatnonzero(active):
            a, b = map(int, edges[edge_id])
            disjoint.union(a, b)
        restored: List[int] = []
        total_removed = len(removed_order)
        restore_start = time.perf_counter()
        if progress:
            mode = "exact batch" if exact_restoration else "fast local"
            print(f"  {mode} restoration: testing {total_removed} edges ...", flush=True)

        cyclic_candidates: List[int] = []
        for edge_id in reversed(removed_order):
            a, b = map(int, edges[edge_id])
            if disjoint.find(a) != disjoint.find(b):
                active[edge_id] = True
                adjacency[a].add(b)
                adjacency[b].add(a)
                disjoint.union(a, b)
                restored.append(edge_id)
            elif not exact_restoration:
                distance = _limited_bidirectional_distance(
                    adjacency, a, b, maximum=cycle_threshold - 2
                )
                if distance is not None and distance + 1 < cycle_threshold:
                    active[edge_id] = True
                    adjacency[a].add(b)
                    adjacency[b].add(a)
                    restored.append(edge_id)
            else:
                cyclic_candidates.append(edge_id)

        batch_searches = 0

        def restore_batch(batch: Sequence[int]) -> None:
            nonlocal batch_searches
            if not batch:
                return
            for edge_id in batch:
                a, b = map(int, edges[edge_id])
                adjacency[a].add(b)
                adjacency[b].add(a)
                active[edge_id] = True
            batch_searches += 1
            # Restoration performs many relatively small searches. Running
            # them locally is substantially faster on Windows than repeatedly
            # serializing the changing adjacency graph to worker processes.
            recreated = find_long_atomic_cycle_parallel(
                adjacency, cycle_threshold, search_roots, 1, None
            )
            if recreated is None:
                restored.extend(batch)
                if progress and (len(batch) > 1 or batch_searches % 10 == 0):
                    print(
                        f"    accepted batch={len(batch)}, restored={len(restored)}, "
                        f"searches={batch_searches}",
                        flush=True,
                    )
                return
            for edge_id in batch:
                a, b = map(int, edges[edge_id])
                adjacency[a].discard(b)
                adjacency[b].discard(a)
                active[edge_id] = False
            if len(batch) == 1:
                return
            midpoint = len(batch) // 2
            restore_batch(batch[:midpoint])
            restore_batch(batch[midpoint:])

        if exact_restoration:
            restore_batch(cyclic_candidates)
            final_check = find_long_atomic_cycle_parallel(
                adjacency, cycle_threshold, search_roots, 1, None
            )
            if final_check is not None:
                raise RuntimeError(
                    "Batch restoration invariant failed: a long atomic cycle remains"
                )
            if progress:
                print(
                    f"  exact batch restoration finished in "
                    f"{time.perf_counter()-restore_start:.2f}s, "
                    f"searches={batch_searches}, restored={len(restored)}",
                    flush=True,
                )
        elif progress:
            print(
                f"  fast local restoration finished in "
                f"{time.perf_counter()-restore_start:.2f}s, "
                f"restored={len(restored)}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

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
        raise RuntimeError(
            f"Accelerated CycleCut produced {component_count} components: "
            f"{np.bincount(labels).tolist()}"
        )
    if progress:
        print(
            f"  CycleCut total={time.perf_counter()-started:.2f}s, "
            f"squeezes={len(cycle_lengths)}, final_cuts={len(final_removed)}",
            flush=True,
        )
    return cyclecut_reference.CycleCutResult(
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


def run_accelerated_cyclecut_isomap(
    points: np.ndarray,
    neighbors: int,
    cycle_threshold: int,
    capacity_mode: str,
    search_roots: int,
    workers: int,
    use_gpu: bool,
    exact_restoration: bool = False,
) -> tuple[np.ndarray, cyclecut_reference.CycleCutResult, str]:
    points = np.asarray(points, dtype=float)
    started = time.perf_counter()
    print(
        f"KNN: {'GPU' if use_gpu else 'CPU'} backend, k={neighbors} ...",
        flush=True,
    )
    graph = isomap_reference.build_knn_graph(points, neighbors, use_gpu=use_gpu)
    print(f"KNN finished in {time.perf_counter()-started:.2f}s", flush=True)
    component_count, _ = connected_components(graph, directed=False)
    if component_count != 1:
        raise ValueError(
            f"The original kNN graph has {component_count} components; increase --neighbors"
        )
    result = accelerated_cycle_cut(
        graph,
        cycle_threshold=cycle_threshold,
        capacity_mode=capacity_mode,
        search_roots=search_roots,
        workers=workers,
        exact_restoration=exact_restoration,
        progress=True,
    )
    shortest_start = time.perf_counter()
    print("Computing all-pairs shortest paths on CPU ...", flush=True)
    geodesic = dijkstra(result.graph, directed=False)
    print(f"Shortest paths finished in {time.perf_counter()-shortest_start:.2f}s", flush=True)
    if not np.isfinite(geodesic).all():
        raise RuntimeError("The final CycleCut graph is disconnected")
    mds_start = time.perf_counter()
    print(f"MDS: {'GPU' if use_gpu else 'CPU'} backend ...", flush=True)
    uv = (
        isomap_reference.classical_mds_gpu(geodesic)
        if use_gpu else isomap_reference.classical_mds_cpu(geodesic)
    )
    print(f"MDS finished in {time.perf_counter()-mds_start:.2f}s", flush=True)
    backend = (
        "CUDA kNN + parallel CPU CycleCut + CPU Dijkstra + CUDA MDS"
        if use_gpu else
        "CPU kNN + parallel CPU CycleCut + CPU Dijkstra + CPU MDS"
    )
    return uv, result, backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl", nargs="?", default="")
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--cycle-threshold", "--lambda", dest="cycle_threshold",
                        type=int, default=16)
    parser.add_argument("--capacity", choices=("uniform", "inverse-distance"),
                        default="uniform")
    parser.add_argument("--search-roots", type=int, default=12)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-points", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--exact-restoration", action="store_true",
        help="Use slow global validation during minimal restoration",
    )
    parser.add_argument("--out", default="cyclecut_accelerated_points.csv")
    parser.add_argument("--cut-edges-out", default="cyclecut_accelerated_removed_edges.csv")
    parser.add_argument("--save-figure", default="")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stl_path = Path(args.stl) if args.stl else cyclecut_reference.select_stl_file()
    if stl_path is None:
        raise SystemExit("No STL model selected")
    if stl_path.suffix.lower() != ".stl":
        raise SystemExit("The selected input must be an STL model")
    available, gpu_status = isomap_reference.gpu_is_available()
    use_gpu = bool(not args.cpu and available)
    print(f"Input STL: {stl_path.resolve()}")
    print(f"CUDA: {'enabled' if use_gpu else 'disabled'} ({gpu_status})")
    mesh = isomap_reference.load_stl(stl_path)
    points, face_ids = isomap_reference.sample_surface_uniform_density(
        mesh,
        args.density,
        np.random.default_rng(args.seed),
        max_points=args.max_points,
        boundary_multiplier=0.0,
    )
    print(f"Sampled points: {len(points)}")
    uv, result, backend = run_accelerated_cyclecut_isomap(
        points,
        neighbors=min(max(2, args.neighbors), len(points) - 1),
        cycle_threshold=args.cycle_threshold,
        capacity_mode=args.capacity,
        search_roots=args.search_roots,
        workers=max(1, args.workers),
        use_gpu=use_gpu,
        exact_restoration=args.exact_restoration,
    )
    isomap_reference.save_csv(args.out, points, uv, face_ids)
    cyclecut_reference.save_cut_edges_csv(args.cut_edges_out, points, result)
    print(f"Backend: {backend}")
    print(f"Final cut edges: {len(result.removed_edge_ids)}")
    print(f"Embedding CSV: {Path(args.out).resolve()}")
    print(f"Cut-edge CSV: {Path(args.cut_edges_out).resolve()}")
    if not args.no_show or args.save_figure:
        if args.no_show:
            plt.switch_backend("Agg")
        cyclecut_reference.plot_cyclecut_result(
            points,
            uv,
            result,
            output=args.save_figure,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
