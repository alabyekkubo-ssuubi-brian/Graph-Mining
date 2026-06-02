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
    p.add_argument("--gn-nodes", type=int, default=400)
    p.add_argument("--lp-sample-nodes", type=int, default=0)
    a = p.parse_args()

    run("task1_exploration.py")
    run("task2_centrality.py", "--betweenness-k", a.betweenness_k)
    run("task3_community.py", "--gn-nodes", a.gn_nodes)
    run("task4_link_prediction.py", "--sample-nodes", a.lp_sample_nodes)

    print("\nAll tasks complete. See the outputs/ directory.")


if __name__ == "__main__":
    main()
