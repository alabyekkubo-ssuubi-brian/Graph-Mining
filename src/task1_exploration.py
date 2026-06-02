"""
Task 1 - Graph Construction and Exploration
-------------------------------------------
(a) Load the dataset, construct the graph, and report:
      number of nodes, number of edges, graph density,
      size of the largest connected component.
(b) Calculate and plot the degree distribution.

Run:  python src/task1_exploration.py
Outputs are written to outputs/.
"""

import os
import json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_graph, OUT_DIR


def basic_stats(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)

    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    lcc_size = len(largest)

    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees))

    stats = {
        "num_nodes": n,
        "num_edges": m,
        "density": density,
        "num_connected_components": len(components),
        "largest_cc_size": lcc_size,
        "largest_cc_fraction": lcc_size / n,
        "avg_degree": avg_degree,
        "max_degree": int(np.max(degrees)),
        "min_degree": int(np.min(degrees)),
    }
    return stats, degrees


def plot_degree_distribution(degrees, out_path):
    degrees = np.asarray(degrees)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear-scale histogram
    axes[0].hist(degrees, bins=60, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Degree distribution (linear scale)")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Number of nodes")

    # Log-log degree distribution (reveals heavy tail / power-law shape)
    values, counts = np.unique(degrees, return_counts=True)
    axes[1].loglog(values, counts, marker=".", linestyle="None", color="#C44E52")
    axes[1].set_title("Degree distribution (log-log)")
    axes[1].set_xlabel("Degree (log)")
    axes[1].set_ylabel("Count (log)")

    fig.suptitle("Facebook Page-Page Network - Degree Distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    G = load_graph()

    stats, degrees = basic_stats(G)

    print("\n=== Task 1: Graph statistics ===")
    print(f"Nodes ....................... {stats['num_nodes']:,}")
    print(f"Edges ....................... {stats['num_edges']:,}")
    print(f"Density ..................... {stats['density']:.6f}")
    print(f"Connected components ........ {stats['num_connected_components']:,}")
    print(f"Largest component (nodes) ... {stats['largest_cc_size']:,} "
          f"({stats['largest_cc_fraction']*100:.2f}% of graph)")
    print(f"Average degree .............. {stats['avg_degree']:.2f}")
    print(f"Degree range ................ [{stats['min_degree']}, {stats['max_degree']}]")

    # Persist stats
    with open(os.path.join(OUT_DIR, "task1_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    # Plot
    plot_path = os.path.join(OUT_DIR, "task1_degree_distribution.png")
    plot_degree_distribution(degrees, plot_path)
    print(f"\nSaved degree-distribution plot -> {plot_path}")
    print(f"Saved statistics             -> {os.path.join(OUT_DIR, 'task1_stats.json')}")


if __name__ == "__main__":
    main()
