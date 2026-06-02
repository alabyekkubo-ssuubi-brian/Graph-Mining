"""
Shared utilities for the Facebook Large Page-Page Network graph-mining project.

Handles dataset loading robustly across the common file layouts:
  - SNAP / MUSAE download:  musae_facebook_edges.csv  (header: id_1,id_2)
                            musae_facebook_target.csv (header: id,facebook_id,page_name,page_type)
  - Generic edge list:      whitespace- or comma-separated, with or without header.

All other task modules import `load_graph` and `load_targets` from here so the
loading logic lives in exactly one place.
"""

import os
import csv
import networkx as nx
import pandas as pd

# Resolve paths relative to the repo root regardless of where scripts are run from.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def _find_file(candidates):
    """Return the first existing path among candidate filenames inside DATA_DIR."""
    for name in candidates:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path
    return None


def _sniff_delimiter(path):
    """Detect whether a text file is comma- or whitespace-separated."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        first = fh.readline()
    if "," in first:
        return ","
    if "\t" in first:
        return "\t"
    return None  # whitespace


def _has_header(path, delimiter):
    """Heuristic: a header row has non-numeric tokens in its first line."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        first = fh.readline().strip()
    tokens = first.split(delimiter) if delimiter else first.split()
    for tok in tokens[:2]:
        try:
            int(tok)
        except ValueError:
            return True
    return False


def load_graph(verbose=True):
    """
    Load the Facebook page-page network as an undirected NetworkX graph.

    Returns
    -------
    G : networkx.Graph
        Undirected, self-loops removed, nodes stored as integers.
    """
    edge_path = _find_file([
        "musae_facebook_edges.csv",
        "facebook_edges.csv",
        "edges.csv",
        "facebook_large_edges.csv",
        "facebook_edges.txt",
        "edges.txt",
        "facebook.edges",
        "facebook.edgelist",
        "facebook_combined.txt",
    ])
    if edge_path is None:
        raise FileNotFoundError(
            f"No edge file found in {DATA_DIR}. "
            "Place 'musae_facebook_edges.csv' there (see README)."
        )

    delimiter = _sniff_delimiter(edge_path)
    skiprows = 1 if _has_header(edge_path, delimiter) else 0

    if delimiter == ",":
        df = pd.read_csv(edge_path, header=0 if skiprows else None,
                         usecols=[0, 1], names=None if skiprows else ["id_1", "id_2"])
        df = df.iloc[:, :2]
    else:
        df = pd.read_csv(edge_path, sep=r"\s+", header=0 if skiprows else None,
                         usecols=[0, 1], names=None if skiprows else ["id_1", "id_2"],
                         engine="python")
        df = df.iloc[:, :2]

    df.columns = ["src", "dst"]
    df = df.dropna().astype({"src": int, "dst": int})

    G = nx.from_pandas_edgelist(df, "src", "dst")
    G.remove_edges_from(nx.selfloop_edges(G))

    if verbose:
        print(f"[load_graph] file: {os.path.basename(edge_path)}")
        print(f"[load_graph] nodes: {G.number_of_nodes():,}  edges: {G.number_of_edges():,}")
    return G


def load_targets(verbose=True):
    """
    Load node metadata (page name and page type) if the target file is present.

    Returns
    -------
    df : pandas.DataFrame or None
        Columns: id, page_name, page_type  (None if file absent).
    """
    target_path = _find_file([
        "musae_facebook_target.csv",
        "facebook_target.csv",
        "target.csv",
    ])
    if target_path is None:
        if verbose:
            print("[load_targets] target file not found; proceeding without metadata.")
        return None

    df = pd.read_csv(target_path)
    # Normalise expected column names
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "id":
            rename[c] = "id"
        elif "name" in lc:
            rename[c] = "page_name"
        elif "type" in lc or "category" in lc:
            rename[c] = "page_type"
    df = df.rename(columns=rename)
    if verbose:
        print(f"[load_targets] loaded {len(df):,} node labels.")
    return df


def label_lookup(targets):
    """Return dicts mapping node id -> page_name and id -> page_type."""
    if targets is None:
        return {}, {}
    name = dict(zip(targets["id"], targets.get("page_name", targets["id"])))
    ptype = dict(zip(targets["id"], targets.get("page_type", ["unknown"] * len(targets))))
    return name, ptype


def describe_node(node, name_map, type_map):
    """Human-readable label for a node, falling back to its id."""
    nm = name_map.get(node, str(node))
    tp = type_map.get(node, "?")
    return f"{node} ({nm} | {tp})"
