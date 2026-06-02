"""
Task 2 - Centrality Analysis
----------------------------
(a) Compute and compare degree, betweenness, closeness centrality and PageRank
    for all nodes.
(b) Identify the top ten influential nodes and justify the choice.

Notes on scalability
--------------------
Exact betweenness centrality is O(V*E) and is infeasible on ~22k nodes / ~171k
edges in reasonable time. We therefore use NetworkX's sampled estimator
(`betweenness_centrality(G, k=...)`), which approximates betweenness from k pivot
sources. Closeness is computed on the largest connected component (closeness is
only well defined within a connected component). All choices are reported so the
methodology is reproducible and defensible.

Run:  python src/task2_centrality.py [--betweenness-k 500]
"""

import os
import json
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import networkx as nx

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm is not installed
    def tqdm(it, **kwargs):
        return it

from utils import load_graph, load_targets, label_lookup, OUT_DIR


# ----------------------------------------------------------------------------
# Parallel helpers
# ----------------------------------------------------------------------------
# These run in worker processes, so they must be module-level (picklable) and
# rely on a per-worker global graph set by _init_worker to avoid re-pickling the
# whole graph on every task.

_WORKER_G = None


def _init_worker(graph):
    """Initializer: stash the graph in each worker process once."""
    global _WORKER_G
    _WORKER_G = graph


def _betweenness_from_sources(G, sources):
    """
    Accumulate betweenness contributions from a given list of source nodes.
    Mirrors networkx's Brandes accumulation but restricted to `sources`, so the
    sum across all chunks equals the k-sampled estimate.
    """
    from networkx.algorithms.centrality.betweenness import (
        _single_source_shortest_path_basic as sssp,
        _accumulate_endpoints,
        _accumulate_basic,
    )
    betweenness = dict.fromkeys(G, 0.0)
    for s in sources:
        S, P, sigma, _ = sssp(G, s)
        betweenness, _ = _accumulate_basic(betweenness, S, P, sigma, s)
    return betweenness


def _closeness_chunk(nodes):
    """Compute closeness for a chunk of nodes in a worker process."""
    G = _WORKER_G
    return {n: nx.closeness_centrality(G, u=n) for n in nodes}


def _chunk(seq, n_chunks):
    """Split a list into roughly n_chunks contiguous chunks."""
    seq = list(seq)
    k, m = divmod(len(seq), n_chunks)
    return [seq[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


def _rescale_like_networkx(betweenness, n, sampled_nodes):
    """
    Apply networkx's exact rescaling for k-sampled, endpoint-excluded,
    normalized, undirected betweenness. Reusing the library function guarantees
    our parallel sum is normalized identically to nx.betweenness_centrality.
    """
    from networkx.algorithms.centrality.betweenness import _rescale
    return _rescale(
        betweenness, n,
        normalized=True, directed=False,
        endpoints=False, sampled_nodes=sampled_nodes,
    )


def parallel_betweenness(G, k, n_jobs, seed=42):
    """k-sampled betweenness, with pivot sources split across worker processes."""
    import random
    rng = random.Random(seed)
    nodes = list(G.nodes())
    k = min(k, len(nodes))
    pivots = rng.sample(nodes, k)

    if n_jobs == 1:
        # Single-process path with a progress bar over pivots.
        from networkx.algorithms.centrality.betweenness import (
            _single_source_shortest_path_basic as sssp,
            _accumulate_basic,
        )
        bt = dict.fromkeys(G, 0.0)
        for s in tqdm(pivots, desc="betweenness", unit="pivot"):
            S, P, sigma, _ = sssp(G, s)
            bt, _ = _accumulate_basic(bt, S, P, sigma, s)
        return _rescale_like_networkx(bt, len(nodes), pivots)

    chunks = _chunk(pivots, n_jobs)
    combined = dict.fromkeys(G, 0.0)
    with Pool(processes=n_jobs, initializer=_init_worker, initargs=(G,)) as pool:
        for partial_bt in tqdm(pool.imap_unordered(_betweenness_from_sources_star, chunks),
                               total=len(chunks), desc="betweenness", unit="chunk"):
            for v, val in partial_bt.items():
                combined[v] += val
    return _rescale_like_networkx(combined, len(nodes), pivots)


def _betweenness_from_sources_star(sources):
    """Pool wrapper using the per-worker global graph."""
    return _betweenness_from_sources(_WORKER_G, sources)


def _closeness_bfs_from_source(source):
    """
    Single-source shortest-path lengths from `source` in the per-worker graph.
    Returns (reached_count, {node: distance}) for accumulating sampled closeness.
    """
    G = _WORKER_G
    lengths = nx.single_source_shortest_path_length(G, source)
    return lengths


def parallel_closeness(G, n_jobs, sample_k=None, seed=42):
    """
    Closeness on the largest connected component.

    If sample_k is set (and smaller than the component), uses the Eppstein-Wang
    sampled estimator: run BFS from k random pivot sources and estimate each
    node v's closeness from its mean distance to those pivots. This is O(k*E)
    instead of O(V*E) and gives a ranking essentially identical to exact
    closeness for identifying central nodes.

    If sample_k is None, computes exact closeness (slow on large graphs).
    """
    import random
    lcc_nodes = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(lcc_nodes).copy()
    n_c = Gc.number_of_nodes()
    all_nodes = list(Gc.nodes())

    # ---- Exact path (small graphs only) ----
    if sample_k is None or sample_k >= n_c:
        if n_jobs == 1:
            clo_partial = {n: nx.closeness_centrality(Gc, u=n)
                           for n in tqdm(all_nodes, desc="closeness", unit="node")}
        else:
            chunks = _chunk(all_nodes, n_jobs * 4)
            clo_partial = {}
            with Pool(processes=n_jobs, initializer=_init_worker, initargs=(Gc,)) as pool:
                for d in tqdm(pool.imap_unordered(_closeness_chunk, chunks),
                              total=len(chunks), desc="closeness", unit="chunk"):
                    clo_partial.update(d)
        return {n: clo_partial.get(n, 0.0) for n in G.nodes()}

    # ---- Sampled path (Eppstein-Wang) ----
    rng = random.Random(seed)
    pivots = rng.sample(all_nodes, sample_k)

    # Accumulate, for every node, the sum of distances to pivots and the count
    # of pivots that reached it.
    dist_sum = dict.fromkeys(all_nodes, 0.0)
    reach_cnt = dict.fromkeys(all_nodes, 0)

    if n_jobs == 1:
        results = (_bfs_lengths(Gc, p) for p in tqdm(pivots, desc="closeness", unit="pivot"))
        for lengths in results:
            for node, d in lengths.items():
                dist_sum[node] += d
                reach_cnt[node] += 1
    else:
        with Pool(processes=n_jobs, initializer=_init_worker, initargs=(Gc,)) as pool:
            for lengths in tqdm(pool.imap_unordered(_closeness_bfs_from_source, pivots),
                                total=len(pivots), desc="closeness", unit="pivot"):
                for node, d in lengths.items():
                    dist_sum[node] += d
                    reach_cnt[node] += 1

    # Eppstein-Wang estimate of closeness centrality:
    #   C(v) ≈ (reached-1) / (n_c-1) * (reached-1) / (sum of sampled distances * n_c/k)
    # We use the standard scaled form so values are comparable to exact closeness.
    # Eppstein-Wang sampled estimate. Closeness is 1 / (average distance to all
    # other nodes); we estimate the average distance from the k sampled pivots.
    clo = {}
    for v in all_nodes:
        s = dist_sum[v]
        r = reach_cnt[v]
        if s > 0 and r > 1:
            est_avg_dist = s / r          # estimated mean distance to all nodes
            clo[v] = 1.0 / est_avg_dist   # closeness centrality estimate
        else:
            clo[v] = 0.0

    return {n: clo.get(n, 0.0) for n in G.nodes()}


def _bfs_lengths(G, source):
    """Serial-path BFS lengths (used when n_jobs == 1)."""
    return nx.single_source_shortest_path_length(G, source)


def compute_centralities(G, betweenness_k=500, closeness_k=1000, n_jobs=None, seed=42):
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    n_jobs = max(1, n_jobs)
    print(f"[task2] using {n_jobs} worker process(es)")

    print("[task2] degree centrality ...")
    deg = nx.degree_centrality(G)

    print("[task2] PageRank ...")
    pr = nx.pagerank(G, alpha=0.85)

    print(f"[task2] approximate betweenness (k={betweenness_k} pivots) ...")
    btw = parallel_betweenness(G, betweenness_k, n_jobs, seed=seed)

    if closeness_k and closeness_k > 0:
        print(f"[task2] sampled closeness (k={closeness_k} pivots) ...")
    else:
        print("[task2] exact closeness (largest connected component) ...")
    clo = parallel_closeness(G, n_jobs, sample_k=closeness_k, seed=seed)

    return deg, btw, clo, pr


def build_table(G, deg, btw, clo, pr, name_map, type_map):
    rows = []
    for n in G.nodes():
        rows.append({
            "node": n,
            "page_name": name_map.get(n, ""),
            "page_type": type_map.get(n, ""),
            "degree": G.degree(n),
            "degree_centrality": deg[n],
            "betweenness": btw[n],
            "closeness": clo[n],
            "pagerank": pr[n],
        })
    return pd.DataFrame(rows)


def top_n(df, column, n=10):
    cols = ["node", "page_name", "page_type", column]
    return df.sort_values(column, ascending=False).head(n)[cols].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--betweenness-k", type=int, default=500,
                        help="Number of pivot sources for approximate betweenness.")
    parser.add_argument("--closeness-k", type=int, default=1000,
                        help="Pivot sources for sampled closeness (0 = exact, slow).")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Worker processes (default: all cores minus one).")
    args = parser.parse_args()

    G = load_graph()
    targets = load_targets()
    name_map, type_map = label_lookup(targets)

    deg, btw, clo, pr = compute_centralities(
        G, betweenness_k=args.betweenness_k, closeness_k=args.closeness_k,
        n_jobs=args.n_jobs)
    df = build_table(G, deg, btw, clo, pr, name_map, type_map)

    # Save the full per-node table
    csv_path = os.path.join(OUT_DIR, "task2_centralities.csv")
    df.to_csv(csv_path, index=False)

    # Correlation between measures - shows where they agree / disagree
    corr = df[["degree_centrality", "betweenness", "closeness", "pagerank"]].corr(method="spearman")

    print("\n=== Spearman rank correlation between centrality measures ===")
    print(corr.round(3).to_string())

    measures = ["degree_centrality", "betweenness", "closeness", "pagerank"]
    summary = {}
    for m in measures:
        print(f"\n=== Top 10 by {m} ===")
        t = top_n(df, m, 10)
        print(t.to_string(index=False))
        summary[m] = t.to_dict(orient="records")

    # Composite ranking: average of the per-measure rank across all four measures.
    # This identifies nodes that are consistently influential, not just by one lens.
    ranked = df.copy()
    for m in measures:
        ranked[m + "_rank"] = ranked[m].rank(ascending=False)
    ranked["mean_rank"] = ranked[[m + "_rank" for m in measures]].mean(axis=1)
    top_overall = ranked.sort_values("mean_rank").head(10)[
        ["node", "page_name", "page_type", "mean_rank"] + measures
    ].reset_index(drop=True)

    print("\n=== Top 10 overall influential nodes (mean rank across all measures) ===")
    print(top_overall.to_string(index=False))

    with open(os.path.join(OUT_DIR, "task2_top_nodes.json"), "w") as fh:
        json.dump({
            "spearman_correlation": corr.round(4).to_dict(),
            "top_by_measure": summary,
            "top_overall_mean_rank": top_overall.to_dict(orient="records"),
        }, fh, indent=2, default=str)

    corr.to_csv(os.path.join(OUT_DIR, "task2_centrality_correlation.csv"))

    print(f"\nSaved per-node centralities -> {csv_path}")
    print(f"Saved top-node summary      -> {os.path.join(OUT_DIR, 'task2_top_nodes.json')}")


if __name__ == "__main__":
    main()
