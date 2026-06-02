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
import numpy as np
import pandas as pd
import networkx as nx

from utils import load_graph, load_targets, label_lookup, OUT_DIR


def compute_centralities(G, betweenness_k=500, seed=42):
    print("[task2] degree centrality ...")
    deg = nx.degree_centrality(G)

    print("[task2] PageRank ...")
    pr = nx.pagerank(G, alpha=0.85)

    print(f"[task2] approximate betweenness (k={betweenness_k} pivots) ...")
    btw = nx.betweenness_centrality(G, k=min(betweenness_k, G.number_of_nodes()),
                                    seed=seed, normalized=True)

    print("[task2] closeness centrality (on largest connected component) ...")
    lcc_nodes = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(lcc_nodes)
    clo_partial = nx.closeness_centrality(Gc)
    # Nodes outside the LCC get closeness 0 (unreachable from most of the graph)
    clo = {n: clo_partial.get(n, 0.0) for n in G.nodes()}

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
    args = parser.parse_args()

    G = load_graph()
    targets = load_targets()
    name_map, type_map = label_lookup(targets)

    deg, btw, clo, pr = compute_centralities(G, betweenness_k=args.betweenness_k)
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
