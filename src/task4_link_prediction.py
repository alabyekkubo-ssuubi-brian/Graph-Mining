"""
Task 4 - Link Prediction
------------------------
Predict missing/future edges and evaluate with relevant metrics.

Method
------
1. Split observed edges into train (85%) and test (15%) positive sets.
2. Sample an equal number of non-edges as negatives for each set.
3. Build the "observed" graph from training edges only (no leakage).
4. For every candidate pair compute four classic topological scores on the
   observed graph:
       - Jaccard coefficient
       - Adamic-Adar index
       - Preferential attachment
       - Resource allocation index
5. Train a logistic-regression classifier on these features and evaluate on the
   held-out test pairs.
6. Report ROC-AUC, average precision (PR-AUC), and accuracy, plus the AUC of
   each individual heuristic as a baseline.

Run:  python src/task4_link_prediction.py [--sample-nodes 0]
      (--sample-nodes N restricts to the N highest-degree nodes for a faster run;
       0 = use the full graph.)
"""

import os
import json
import random
import argparse

import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from utils import load_graph, OUT_DIR

RNG = random.Random(42)
np.random.seed(42)


def split_edges(G, test_frac=0.15):
    """Split edges into train/test while keeping the train graph connected-ish."""
    edges = list(G.edges())
    RNG.shuffle(edges)
    n_test = int(len(edges) * test_frac)

    # Use a spanning structure to avoid removing bridges where possible:
    # add test edges only if both endpoints still have degree > 1 in train graph.
    train_G = G.copy()
    test_pos = []
    for u, v in edges:
        if len(test_pos) >= n_test:
            break
        if train_G.degree(u) > 1 and train_G.degree(v) > 1:
            train_G.remove_edge(u, v)
            test_pos.append((u, v))
    train_pos = list(train_G.edges())
    return train_G, train_pos, test_pos


def sample_negatives(G, n_samples, exclude):
    """Sample node pairs that are NOT edges in G (and not in `exclude`)."""
    nodes = list(G.nodes())
    negatives = set()
    exclude_set = set(frozenset(e) for e in exclude)
    attempts = 0
    max_attempts = n_samples * 50
    while len(negatives) < n_samples and attempts < max_attempts:
        u, v = RNG.sample(nodes, 2)
        key = frozenset((u, v))
        if not G.has_edge(u, v) and key not in exclude_set and key not in negatives:
            negatives.add(key)
        attempts += 1
    return [tuple(p) for p in negatives]


def heuristic_features(G, pairs):
    """Compute the four topological scores for each (u, v) pair on graph G."""
    pairs = list(pairs)

    def to_dict(gen):
        d = {}
        for u, v, s in gen:
            d[frozenset((u, v))] = s
        return d

    jac = to_dict(nx.jaccard_coefficient(G, pairs))
    aa = to_dict(nx.adamic_adar_index(G, pairs))
    ra = to_dict(nx.resource_allocation_index(G, pairs))
    pa = to_dict(nx.preferential_attachment(G, pairs))

    feats = []
    for u, v in pairs:
        k = frozenset((u, v))
        feats.append([
            jac.get(k, 0.0),
            aa.get(k, 0.0),
            ra.get(k, 0.0),
            pa.get(k, 0.0),
        ])
    return np.asarray(feats, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-nodes", type=int, default=0,
                        help="Restrict to N highest-degree nodes (0 = full graph).")
    args = parser.parse_args()

    G = load_graph()

    if args.sample_nodes and G.number_of_nodes() > args.sample_nodes:
        top = [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:args.sample_nodes]]
        G = G.subgraph(top).copy()
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        print(f"[task4] restricted to subgraph: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")

    feature_names = ["jaccard", "adamic_adar", "resource_allocation", "pref_attachment"]

    # 1-3. Split and build observed (training) graph
    train_G, train_pos, test_pos = split_edges(G, test_frac=0.15)
    print(f"[task4] train edges: {len(train_pos):,}  test edges: {len(test_pos):,}")

    # 2. Negatives (equal to positives in each split)
    train_neg = sample_negatives(G, len(train_pos), exclude=test_pos)
    test_neg = sample_negatives(G, len(test_pos), exclude=train_pos + test_pos)

    # 4. Features computed on the OBSERVED (training) graph only
    X_train = heuristic_features(train_G, train_pos + train_neg)
    y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))
    X_test = heuristic_features(train_G, test_pos + test_neg)
    y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))

    # 5. Classifier
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(scaler.transform(X_train), y_train)

    proba = clf.predict_proba(scaler.transform(X_test))[:, 1]
    preds = (proba >= 0.5).astype(int)

    results = {
        "model": "LogisticRegression on [Jaccard, Adamic-Adar, ResourceAlloc, PrefAttachment]",
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "n_train_pos": len(train_pos),
        "n_test_pos": len(test_pos),
        "feature_coefficients": dict(zip(feature_names, clf.coef_[0].round(4).tolist())),
    }

    # 6. Individual-heuristic baselines (single-feature ROC-AUC on the test set)
    baselines = {}
    for i, name in enumerate(feature_names):
        baselines[name] = float(roc_auc_score(y_test, X_test[:, i]))
    results["single_feature_auc"] = baselines

    print("\n=== Task 4: Link Prediction results ===")
    print(f"Combined model ROC-AUC ......... {results['roc_auc']:.4f}")
    print(f"Combined model PR-AUC (AP) ..... {results['average_precision']:.4f}")
    print(f"Combined model accuracy ........ {results['accuracy']:.4f}")
    print("\nSingle-heuristic ROC-AUC baselines:")
    for name, auc in baselines.items():
        print(f"  {name:<22} {auc:.4f}")
    print("\nLogistic-regression coefficients:")
    for name, c in results["feature_coefficients"].items():
        print(f"  {name:<22} {c:+.4f}")

    with open(os.path.join(OUT_DIR, "task4_link_prediction.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved results -> {os.path.join(OUT_DIR, 'task4_link_prediction.json')}")


if __name__ == "__main__":
    main()
