"""
Run the full Project 2 pipeline end-to-end.

Usage:
    python run_all.py
    python run_all.py --betweenness-k 500 --gn-nodes 400 --lp-sample-nodes 0
"""
import argparse
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")


def run(script, *args):
    cmd = [sys.executable, os.path.join(SRC, script), *map(str, args)]
    print("\n" + "=" * 70)
    print("RUNNING:", " ".join(cmd))
    print("=" * 70)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--betweenness-k", type=int, default=500)
    p.add_argument("--closeness-k", type=int, default=1000)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--gn-nodes", type=int, default=400)
    p.add_argument("--gn-mode", choices=["ego", "hubs"], default="ego")
    p.add_argument("--lp-sample-nodes", type=int, default=0)
    a = p.parse_args()

    run("task1_exploration.py")
    bt_args = ["--betweenness-k", a.betweenness_k, "--closeness-k", a.closeness_k]
    if a.n_jobs is not None:
        bt_args += ["--n-jobs", a.n_jobs]
    run("task2_centrality.py", *bt_args)
    run("task3_community.py", "--gn-nodes", a.gn_nodes, "--gn-mode", a.gn_mode)
    run("task4_link_prediction.py", "--sample-nodes", a.lp_sample_nodes)

    print("\nAll tasks complete. See the outputs/ directory.")


if __name__ == "__main__":
    main()
