# Project 2 — Graph Mining: Facebook Large Page-Page Network

ALABYEKKUBO SSUUBI BRIAN
AYEBALE LINDA KELLEN


Graph-mining analysis of the **Facebook Large Page-Page Network** (SNAP / MUSAE):
an undirected graph of ~22,470 official Facebook pages (nodes) connected by mutual
likes (edges), labelled into four categories — politicians, governmental
organisations, TV shows, and companies.

This repository implements all four required tasks for the data mining project.:

| Task | File | What it does |
|------|------|--------------|
| 1. Construction & Exploration | `src/task1_exploration.py` | Nodes, edges, density, largest connected component; degree-distribution plots (linear + log-log). |
| 2. Centrality Analysis | `src/task2_centrality.py` | Degree, betweenness, closeness, PageRank; rank-correlation between measures; top-10 influential nodes by each measure and by a composite mean-rank. |
| 3. Community Detection | `src/task3_community.py` | Louvain (full graph) and Girvan-Newman (subgraph); community counts, size distributions, modularity; visualizations. |
| 4. Link Prediction | `src/task4_link_prediction.py` | Topological-feature classifier (Jaccard, Adamic-Adar, resource allocation, preferential attachment) with ROC-AUC, PR-AUC, accuracy, and per-heuristic baselines. |

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the data

Download the dataset from
<https://snap.stanford.edu/data/facebook-large-page-page-network.html>
and place these files in the `data/` directory:

```
data/
├── musae_facebook_edges.csv      # columns: id_1, id_2
└── musae_facebook_target.csv     # columns: id, facebook_id, page_name, page_type  (optional but recommended)
```

The loader (`src/utils.py`) auto-detects comma- vs whitespace-separated files
and the presence of a header, so a plain SNAP edge list also works. The target
file is optional — without it, nodes are reported by id instead of page name.

> To verify the pipeline before downloading the real data, run
> `python make_test_data.py` to generate a small synthetic graph in the same
> format, then run any task below.

## 3. Run

Run everything:

```bash
python run_all.py
```

Or run tasks individually:

```bash
python src/task1_exploration.py
python src/task2_centrality.py  --betweenness-k 500
python src/task3_community.py   --gn-nodes 400
python src/task4_link_prediction.py            # add --sample-nodes 3000 for a faster run
```

### Speeding up Task 2 (centrality)

Betweenness and closeness are the slowest computations. Both are **sampled** and
**parallelized** across CPU cores, with live progress bars (via `tqdm`):

```bash
python src/task2_centrality.py --n-jobs 8 --betweenness-k 300 --closeness-k 500
```

- `--n-jobs N` — worker processes (default: all cores minus one).
- `--betweenness-k K` — pivot sources for sampled betweenness. **Lower = faster.**
  k=500 ranks ~0.99 correlated with exact betweenness; 200-300 is fine.
- `--closeness-k K` — pivot sources for sampled closeness (Eppstein-Wang
  estimator). **This is the main bottleneck on large graphs.** Exhaustive
  closeness on 22k nodes is very slow; sampling from k≈500-1000 pivots makes it
  fast while keeping a ranking ~0.99 correlated with exact closeness and ~9/10
  of the top nodes identical. Set `--closeness-k 0` to force exact (slow).

On the full 22,470-node Facebook graph this brings Task 2 down to roughly a
minute or two on a multi-core machine, versus tens of minutes for exhaustive
closeness. The sampled betweenness result is numerically identical to the
single-process result (validated against `networkx.betweenness_centrality`); the
sampled closeness is an unbiased estimator validated to ~0.99 rank correlation
with the exact values.

> **Windows note:** multiprocessing uses "spawn" on Windows, so each worker
> re-imports the module and receives a copy of the graph. This adds a few
> seconds of startup but is harmless. All multiprocessing code is guarded by
> `if __name__ == "__main__":` as Windows requires.

All results (plots, CSVs, JSON summaries) are written to `outputs/`.

## 4. Methodology notes & scalability decisions

These are the defensible engineering choices to highlight in the report/video:

- **Approximate betweenness.** Exact betweenness is O(V·E) — infeasible on
  22k nodes. We use NetworkX's pivot-sampled estimator
  (`betweenness_centrality(G, k=500)`), trading a small amount of accuracy for
  tractability. Increase `--betweenness-k` for higher fidelity.
- **Closeness on the largest component.** Closeness is only well defined inside
  a connected component, so it is computed on the LCC; nodes outside it are
  assigned 0.
- **Composite influence ranking.** Beyond reporting the top 10 per measure, we
  rank nodes by their *mean rank* across all four measures, surfacing nodes that
  are consistently central rather than central under a single lens.
- **Girvan-Newman on a subgraph.** Girvan-Newman recomputes edge betweenness at
  every split (≈O(E²V)) and cannot run on the full graph. We apply it to the
  highest-degree nodes of the largest component (`--gn-nodes`, default 400) and
  report this restriction explicitly. **Louvain is run on the full graph**, so
  the global partition is unaffected by this limitation.
- **Link-prediction without leakage.** Edges are split into train/test; all
  topological features are computed on the *training* graph only, so test edges
  never influence their own features. Negatives are sampled non-edges, balanced
  against positives. We report both the combined-model metrics and each single
  heuristic's ROC-AUC as a baseline.

## 5. Output files

```
outputs/
├── task1_stats.json
├── task1_degree_distribution.png
├── task2_centralities.csv               # per-node, all four measures
├── task2_centrality_correlation.csv
├── task2_top_nodes.json                 # top-10 per measure + composite
├── task3_community_summary.json
├── task3_louvain_communities.png
├── task3_louvain_sizes.png
├── task3_girvan_newman_communities.png
└── task4_link_prediction.json
```

## 6. Repository structure

```
graph-mining/
├── README.md
├── requirements.txt
├── run_all.py              # runs all four tasks
├── make_test_data.py       # synthetic data for a quick smoke test
├── data/                   # place dataset CSVs here
├── outputs/                # generated results
└── src/
    ├── utils.py            # shared loader + helpers
    ├── task1_exploration.py
    ├── task2_centrality.py
    ├── task3_community.py
    └── task4_link_prediction.py
```
