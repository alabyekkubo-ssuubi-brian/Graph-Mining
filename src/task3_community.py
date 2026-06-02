"""
Task 3 - Community Detection
----------------------------
(a) Use Louvain and Girvan-Newman to identify and visualize communities.
(b) Report number of communities, size distribution, and modularity for each;
    discuss which partition is more meaningful.

Scalability note
----------------
Louvain is near-linear and runs on the full graph. Girvan-Newman repeatedly
recomputes edge betweenness and is O(E^2 * V) in the worst case, which is
infeasible on ~171k edges. We therefore run Girvan-Newman on a connected
subgraph (the largest connected component restricted to the highest-degree
nodes, capped by --gn-nodes) and clearly report this restriction. Louvain is
still run on the entire network so the headline partition is global.

Run:  python src/task3_community.py [--gn-nodes 400]
"""

import os
import json
import argparse
import itertools
from collections import Counter

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import community as community_louvain  # python-louvain

from utils import load_graph, OUT_DIR


def run_louvain(G, seed=42):
    print("[task3] Louvain on the full graph ...")
    partition = community_louvain.best_partition(G, random_state=seed)
    modularity = community_louvain.modularity(partition, G)
    comm_sizes = Counter(partition.values())
    return partition, modularity, comm_sizes


def build_gn_subgraph(G, max_nodes, mode="ego", seed=42):
    """
    Build a subgraph for Girvan-Newman.

    mode="ego"  : a BFS region grown from a moderately-connected seed node.
                  This preserves natural community structure, so GN can find
                  meaningful communities and the Louvain-vs-GN comparison is fair.
    mode="hubs" : the highest-degree nodes. This is the densely interconnected
                  "elite core"; GN struggles here, which is itself a finding.

    Returns the largest connected component of the chosen subgraph.
    """
    import random
    rng = random.Random(seed)
    lcc = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(lcc)
    if Gc.number_of_nodes() <= max_nodes:
        return Gc.copy()

    if mode == "hubs":
        nodes = [n for n, _ in sorted(Gc.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]]
    else:  # ego / BFS region
        # Seed from a node of moderate degree (around the 75th percentile) so the
        # region is connected but not dominated by a single mega-hub.
        degs = sorted(Gc.degree(), key=lambda x: x[1])
        seed_node = degs[int(len(degs) * 0.75)][0]
        # Grow a BFS frontier until we reach max_nodes.
        visited = {seed_node}
        frontier = [seed_node]
        while frontier and len(visited) < max_nodes:
            nxt = []
            for u in frontier:
                for w in Gc.neighbors(u):
                    if w not in visited:
                        visited.add(w)
                        nxt.append(w)
                        if len(visited) >= max_nodes:
                            break
                if len(visited) >= max_nodes:
                    break
            frontier = nxt
        nodes = list(visited)

    H = Gc.subgraph(nodes).copy()
    H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
    return H


def run_girvan_newman(H, max_communities=10):
    """
    Run Girvan-Newman on subgraph H, scanning successive partitions and keeping
    the one with the highest modularity (up to max_communities splits).
    """
    print(f"[task3] Girvan-Newman on subgraph "
          f"({H.number_of_nodes()} nodes, {H.number_of_edges()} edges) ...")
    gn_gen = nx.community.girvan_newman(H)
    best = {"modularity": -1, "communities": None, "k": None}

    for communities in itertools.islice(gn_gen, max_communities):
        comms = [set(c) for c in communities]
        q = nx.community.modularity(H, comms)
        if q > best["modularity"]:
            best = {"modularity": q, "communities": comms, "k": len(comms)}
    return best


def visualize_partition(G, partition, out_path, title, max_nodes=1500, seed=42):
    """Spring-layout visualization, sampling nodes if the graph is large."""
    if G.number_of_nodes() > max_nodes:
        # Sample the highest-degree nodes for a readable plot.
        nodes = [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]]
        H = G.subgraph(nodes)
    else:
        H = G

    pos = nx.spring_layout(H, seed=seed, k=0.15, iterations=40)
    colors = [partition.get(n, 0) for n in H.nodes()]

    fig, ax = plt.subplots(figsize=(13, 11))
    nx.draw_networkx_nodes(H, pos, node_size=18, node_color=colors,
                           cmap=plt.cm.tab20, ax=ax)
    nx.draw_networkx_edges(H, pos, alpha=0.08, width=0.3, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_size_distribution(sizes, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    s = sorted(sizes, reverse=True)
    ax.bar(range(len(s)), s, color="#55A868")
    ax.set_title(title)
    ax.set_xlabel("Community (rank by size)")
    ax.set_ylabel("Number of nodes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gn-nodes", type=int, default=400,
                        help="Max nodes for the Girvan-Newman subgraph.")
    parser.add_argument("--gn-mode", choices=["ego", "hubs"], default="ego",
                        help="ego = natural BFS region (clean communities); "
                             "hubs = highest-degree elite core (a finding in itself).")
    args = parser.parse_args()

    G = load_graph()

    # ---- Louvain (full graph) ----
    partition, mod_louvain, sizes_louvain = run_louvain(G)
    n_comm_louvain = len(sizes_louvain)
    size_list_louvain = sorted(sizes_louvain.values(), reverse=True)

    print("\n=== Louvain (full graph) ===")
    print(f"Communities ........ {n_comm_louvain}")
    print(f"Modularity ......... {mod_louvain:.4f}")
    print(f"Largest 5 sizes .... {size_list_louvain[:5]}")

    visualize_partition(
        G, partition,
        os.path.join(OUT_DIR, "task3_louvain_communities.png"),
        f"Louvain communities (Q={mod_louvain:.3f}, {n_comm_louvain} communities)",
    )
    plot_size_distribution(
        size_list_louvain,
        os.path.join(OUT_DIR, "task3_louvain_sizes.png"),
        "Louvain community size distribution",
    )

    # ---- Girvan-Newman (subgraph) ----
    H = build_gn_subgraph(G, args.gn_nodes, mode=args.gn_mode)
    gn = run_girvan_newman(H, max_communities=12)
    gn_partition = {}
    for cid, comm in enumerate(gn["communities"]):
        for node in comm:
            gn_partition[node] = cid
    gn_sizes = sorted((len(c) for c in gn["communities"]), reverse=True)

    print("\n=== Girvan-Newman (subgraph) ===")
    print(f"Subgraph mode ...... {args.gn_mode}")
    print(f"Subgraph size ...... {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    print(f"Communities ........ {gn['k']}")
    print(f"Modularity ......... {gn['modularity']:.4f}")
    print(f"Sizes .............. {gn_sizes}")

    # Fair comparison: run Louvain on the SAME subgraph so the two algorithms are
    # compared on identical input, not full-graph vs subgraph.
    sub_part = community_louvain.best_partition(H, random_state=42)
    sub_mod_louvain = community_louvain.modularity(sub_part, H)
    sub_n_louvain = len(set(sub_part.values()))
    print(f"\n=== Louvain on the SAME subgraph (for fair comparison) ===")
    print(f"Communities ........ {sub_n_louvain}")
    print(f"Modularity ......... {sub_mod_louvain:.4f}")

    visualize_partition(
        H, gn_partition,
        os.path.join(OUT_DIR, "task3_girvan_newman_communities.png"),
        f"Girvan-Newman ({args.gn_mode} subgraph) Q={gn['modularity']:.3f}, {gn['k']} communities",
        max_nodes=args.gn_nodes,
    )
    visualize_partition(
        H, sub_part,
        os.path.join(OUT_DIR, "task3_louvain_subgraph_communities.png"),
        f"Louvain (same {args.gn_mode} subgraph) Q={sub_mod_louvain:.3f}, {sub_n_louvain} communities",
        max_nodes=args.gn_nodes,
    )

    summary = {
        "louvain_full_graph": {
            "num_communities": n_comm_louvain,
            "modularity": mod_louvain,
            "size_distribution_top20": size_list_louvain[:20],
            "scope": "full graph (the headline partition)",
        },
        "girvan_newman_subgraph": {
            "num_communities": gn["k"],
            "modularity": gn["modularity"],
            "size_distribution": gn_sizes,
            "scope": f"{args.gn_mode} subgraph of {H.number_of_nodes()} nodes / {H.number_of_edges()} edges",
        },
        "louvain_same_subgraph": {
            "num_communities": sub_n_louvain,
            "modularity": sub_mod_louvain,
            "scope": "Louvain re-run on the identical GN subgraph for a fair head-to-head",
        },
    }
    with open(os.path.join(OUT_DIR, "task3_community_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSaved community summary -> {os.path.join(OUT_DIR, 'task3_community_summary.json')}")
    print("Saved visualizations    -> outputs/task3_*.png")


if __name__ == "__main__":
    main()
