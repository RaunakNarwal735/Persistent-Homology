import argparse
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import floyd_warshall
from ripser import ripser
from persim import plot_diagrams, bottleneck, wasserstein
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------

def ensure_outdir(path: str = "outputs") -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_graph_from_csv(csv_path: str, directed: bool = True) -> nx.Graph:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    if not {"source", "target"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: source,target,(optional)weight")
    has_w = "weight" in df.columns

    G = nx.DiGraph() if directed else nx.Graph()
    if has_w:
        for _, r in df.iterrows():
            G.add_edge(r["source"], r["target"], weight=float(r["weight"]))
    else:
        for _, r in df.iterrows():
            G.add_edge(r["source"], r["target"], weight=1.0)

    UG = nx.Graph()
    for u, v, w in G.edges(data=True):
        wt = w.get("weight", 1.0)
        if UG.has_edge(u, v):
            UG[u][v]["weight"] = min(UG[u][v]["weight"], wt)
        else:
            UG.add_edge(u, v, weight=wt)
    return UG


def shortest_path_distance_matrix(G: nx.Graph, disconnected_value: float = None) -> (np.ndarray, list):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        i, j = idx[u], idx[v]
        D[i, j] = min(D[i, j], w)
        D[j, i] = min(D[j, i], w)
    D = floyd_warshall(D)
    if disconnected_value is None:
        finite = D[np.isfinite(D)]
        disconnected_value = 2.0 * np.max(finite) if finite.size > 0 else 1.0
    D[~np.isfinite(D)] = disconnected_value
    return np.asarray(D), nodes


def persistence_from_distance(D: np.ndarray, maxdim: int = 1, thresh: float = None):
    finite = D[np.isfinite(D)]
    if thresh is None:
        thresh = np.percentile(finite, 95) if finite.size else 1.0
    res = ripser(D, maxdim=maxdim, thresh=thresh, distance_matrix=True)
    return res["dgms"], res


def plot_and_save(diagrams, title: str, outdir: Path, tag: str):
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5,4))
    plot_diagrams(diagrams, show=False)
    plt.title(f"Persistence Diagram: {title}")
    fp = outdir / f"{tag}_diagram.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=200)
    plt.close()
    return fp


def plot_graph(G: nx.Graph, outdir: Path, name: str):
    plt.figure(figsize=(6,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=400, font_size=8)
    fp = outdir / f"{name}_graph.png"
    plt.savefig(fp, dpi=200)
    plt.close()
    return fp


def compare_diagrams(diagA, diagB):
    out = {}
    for dim in [0, 1]:
        A = diagA[dim] if len(diagA) > dim else np.empty((0,2))
        B = diagB[dim] if len(diagB) > dim else np.empty((0,2))
        try:
            bn = bottleneck(A, B)
        except Exception:
            bn = np.nan
        try:
            ws = wasserstein(A, B, matching=False, p=2.)
        except Exception:
            ws = np.nan
        out[f"H{dim}_bottleneck"] = bn
        out[f"H{dim}_wasserstein"] = ws
    return out

# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(graphA: nx.Graph, graphB: nx.Graph, nameA: str, nameB: str, outdir: str = "outputs"):
    out = ensure_outdir(outdir)

    DA, nodesA = shortest_path_distance_matrix(graphA)
    DB, nodesB = shortest_path_distance_matrix(graphB)

    dgmA, resA = persistence_from_distance(DA, maxdim=1)
    dgmB, resB = persistence_from_distance(DB, maxdim=1)

    np.save(out / f"{nameA}_dgms.npy", np.array(dgmA, dtype=object), allow_pickle=True)
    np.save(out / f"{nameB}_dgms.npy", np.array(dgmB, dtype=object), allow_pickle=True)

    figA = plot_and_save(dgmA, title=nameA, outdir=out, tag=f"{nameA}_tda")
    figB = plot_and_save(dgmB, title=nameB, outdir=out, tag=f"{nameB}_tda")
    gfigA = plot_graph(graphA, out, nameA)
    gfigB = plot_graph(graphB, out, nameB)

    dist = compare_diagrams(dgmA, dgmB)

    summary_path = out / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Graph A: {nameA}\nNodes: {len(nodesA)} Edges: {graphA.number_of_edges()}\n")
        f.write(f"Graph B: {nameB}\nNodes: {len(nodesB)} Edges: {graphB.number_of_edges()}\n\n")
        for k, v in dist.items():
            f.write(f"{k}: {v}\n")

    # Save JSON summary too
    json_summary = {
        "GraphA": {"name": nameA, "nodes": len(nodesA), "edges": graphA.number_of_edges(), "graph_plot": str(gfigA), "diagram_plot": str(figA)},
        "GraphB": {"name": nameB, "nodes": len(nodesB), "edges": graphB.number_of_edges(), "graph_plot": str(gfigB), "diagram_plot": str(figB)},
        "Metrics": dist
    }
    json_path = out / "summary.json"
    with open(json_path, "w") as jf:
        json.dump(json_summary, jf, indent=2)

    return {
        "figA": str(figA),
        "figB": str(figB),
        "graphA": str(gfigA),
        "graphB": str(gfigB),
        "metrics": dist,
        "summary_txt": str(summary_path),
        "summary_json": str(json_path)
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Persistent homology comparison of two chemical networks")
    ap.add_argument("--graphA", type=str, default=None, help="CSV edge list for graph A")
    ap.add_argument("--graphB", type=str, default=None, help="CSV edge list for graph B")
    ap.add_argument("--nameA", type=str, default="A", help="Name label for graph A")
    ap.add_argument("--nameB", type=str, default="B", help="Name label for graph B")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    ap.add_argument("--demo", action="store_true", help="Run with demo toy graphs instead of CSV input")
    return ap.parse_args()


def make_demo_graphs():
    G1 = nx.cycle_graph(5)
    G2 = nx.path_graph(5)
    return G1, G2


def main():
    args = parse_args()

    if args.demo:
        GA, GB = make_demo_graphs()
        nameA, nameB = "Cycle5", "Path5"
    else:
        if not args.graphA or not args.graphB:
            raise SystemExit("Provide --graphA and --graphB CSVs (or use --demo)")
        GA = load_graph_from_csv(args.graphA)
        GB = load_graph_from_csv(args.graphB)
        nameA, nameB = args.nameA, args.nameB

    results = run_pipeline(GA, GB, nameA, nameB, outdir=args.outdir)

    print("=== Persistent Homology Comparison ===")
    print(f"A diagram image: {results['figA']}")
    print(f"B diagram image: {results['figB']}")
    print(f"A graph image: {results['graphA']}")
    print(f"B graph image: {results['graphB']}")
    print("Distances:")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v}")
    print(f"Summary saved to: {results['summary_txt']}")
    print(f"JSON summary saved to: {results['summary_json']}")


if __name__ == "__main__":
    main()
