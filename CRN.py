
import argparse
import os
from pathlib import Path
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
    """Load an edge list CSV with columns source,target,(optional) weight.
    Symmetrize for undirected persistent homology processing."""
    df = pd.read_csv(csv_path)
    cols = [c.lower() for c in df.columns]
    colmap = {c: c.lower() for c in df.columns}
    df.rename(columns=colmap, inplace=True)
    if not {"source", "target"}.issubset(df.columns.str.lower()):
        raise ValueError("CSV must contain columns: source,target,(optional)weight")
    has_w = "weight" in df.columns

    G = nx.DiGraph() if directed else nx.Graph()
    if has_w:
        for _, r in df.iterrows():
            G.add_edge(r["source"], r["target"], weight=float(r["weight"]))
    else:
        for _, r in df.iterrows():
            G.add_edge(r["source"], r["target"], weight=1.0)

    # Convert to undirected for topology (clique/Rips) on metric space
    UG = nx.Graph()
    for u, v, w in G.edges(data=True):
        wt = w.get("weight", 1.0)
        if UG.has_edge(u, v):
            UG[u][v]["weight"] = min(UG[u][v]["weight"], wt)
        else:
            UG.add_edge(u, v, weight=wt)
    return UG


def toy_kegg_like_graph(name: str) -> nx.Graph:
    """Provide two illustrative, non-trivial toy graphs loosely inspired by
    less-commonly juxtaposed KEGG pathways. These are *not* literal KEGG
    graphs; replace with your parsed edges for real analysis.
    """
    G = nx.Graph()
    if name.lower() in {"aromatic", "map00400", "phea_tyr_trp_biosyn"}:
        # Loosely mimics branched biosynthesis with feedback and cross-talk
        nodes = [
            "chorismate", "prephenate", "anthranilate", "tyr", "phe", "trp",
            "aroA", "aroB", "aroC", "tyrA", "pheA", "trpE", "trpD",
            "reg1", "reg2"
        ]
        G.add_nodes_from(nodes)
        edges = [
            ("chorismate", "prephenate"), ("chorismate", "anthranilate"),
            ("prephenate", "tyr"), ("prephenate", "phe"),
            ("anthranilate", "trp"),
            ("aroA", "chorismate"), ("aroB", "chorismate"), ("aroC", "chorismate"),
            ("tyrA", "tyr"), ("pheA", "phe"), ("trpE", "anthranilate"), ("trpD", "trp"),
            ("tyr", "reg1"), ("phe", "reg1"), ("trp", "reg2"), ("reg1", "reg2"),
            ("tyr", "phe"), ("phe", "trp"), ("trp", "tyr") # cross-links creating cycles
        ]
        for u, v in edges:
            G.add_edge(u, v, weight=1.0)
        return G

    if name.lower() in {"terpenoid", "map00900", "terp_backbone"}:
        # Loosely mimics linear-to-branching isoprenoid chain growth with cycles via exchange
        nodes = [
            "acetyl-coa", "acetoacetyl-coa", "hmg-coa", "mevalonate",
            "ipp", "dmapp", "gpp", "fpp", "ggpp",
            "idi", "mvd", "hmgr", "regx", "regy"
        ]
        G.add_nodes_from(nodes)
        edges = [
            ("acetyl-coa", "acetoacetyl-coa"), ("acetoacetyl-coa", "hmg-coa"),
            ("hmg-coa", "mevalonate"), ("mevalonate", "ipp"), ("ipp", "dmapp"),
            ("ipp", "gpp"), ("dmapp", "gpp"), ("gpp", "fpp"), ("fpp", "ggpp"),
            ("idi", "ipp"), ("idi", "dmapp"), ("mvd", "mevalonate"), ("hmgr", "hmg-coa"),
            ("fpp", "regx"), ("ggpp", "regy"), ("regx", "regy"),
            ("gpp", "regx")
        ]
        for u, v in edges:
            G.add_edge(u, v, weight=1.0)
        return G

    # Generic small graph fallback
    G.add_nodes_from(["A", "B", "C", "D", "E"])    
    for e in [("A","B"),("B","C"),("C","D"),("D","A"),("B","D"),("C","E")]:
        G.add_edge(*e, weight=1.0)
    return G


def shortest_path_distance_matrix(G: nx.Graph, disconnected_value: float = None) -> (np.ndarray, list):
    """Return all-pairs shortest-path distances as a dense matrix and a node order."""
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    # Init with large values
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    # Use edge weights as lengths
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        i, j = idx[u], idx[v]
        D[i, j] = min(D[i, j], w)
        D[j, i] = min(D[j, i], w)
    # Floyd–Warshall
    D = floyd_warshall(D)
    # Replace inf with a large finite value to keep Rips bounded
    if disconnected_value is None:
        finite = D[np.isfinite(D)]
        if finite.size == 0:
            disconnected_value = 1.0
        else:
            disconnected_value = 2.0 * np.max(finite) if finite.size > 0 else 1.0
    D[~np.isfinite(D)] = disconnected_value
    return np.asarray(D), nodes


def persistence_from_distance(D: np.ndarray, maxdim: int = 1, thresh: float = None):
    """Compute Vietoris–Rips persistent homology from a precomputed distance matrix."""
    if thresh is None:
        finite = D[np.isfinite(D)]
        thresh = np.percentile(finite, 95) if finite.size else 1.0
    res = ripser(D, maxdim=maxdim, thresh=thresh, metric="precomputed")
    return res["dgms"], res


def plot_and_save(diagrams, title: str, outdir: Path, tag: str):
    outdir.mkdir(parents=True, exist_ok=True)
    # Barcode-like diagram (using persim's scatter); then persistence diagram
    # 1) Persistence diagram
    plt.figure(figsize=(5,4))
    plot_diagrams(diagrams, show=False)
    plt.title(f"Persistence Diagram: {title}")
    fp = outdir / f"{tag}_diagram.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=200)
    plt.close()
    return fp


def compare_diagrams(diagA, diagB):
    # Compute distances for H0 and H1 separately, then aggregate
    out = {}
    for dim in [0, 1]:
        A = diagA[dim] if len(diagA) > dim else np.empty((0,2))
        B = diagB[dim] if len(diagB) > dim else np.empty((0,2))
        # Bottleneck
        try:
            bn = bottleneck(A, B)
        except Exception:
            bn = np.nan
        # Wasserstein (p=2)
        try:
            ws = wasserstein(A, B, matching=False, p=2.)
        except Exception:
            ws = np.nan
        out[f"H{dim}_bottleneck"] = bn
        out[f"H{dim}_wasserstein"] = ws
    return out

# -----------------------------
# Main CLI
# -----------------------------

def run_pipeline(graphA: nx.Graph, graphB: nx.Graph, nameA: str, nameB: str, outdir: str = "outputs"):
    out = ensure_outdir(outdir)

    # Distances
    DA, nodesA = shortest_path_distance_matrix(graphA)
    DB, nodesB = shortest_path_distance_matrix(graphB)

    # Persistent homology (H0/H1 by default)
    dgmA, resA = persistence_from_distance(DA, maxdim=1)
    dgmB, resB = persistence_from_distance(DB, maxdim=1)

    # Save diagrams as .npy for reuse
    np.save(out / f"{nameA}_dgms.npy", dgmA, allow_pickle=True)
    np.save(out / f"{nameB}_dgms.npy", dgmB, allow_pickle=True)

    # Plots
    figA = plot_and_save(dgmA, title=nameA, outdir=out, tag=f"{nameA}_tda")
    figB = plot_and_save(dgmB, title=nameB, outdir=out, tag=f"{nameB}_tda")

    # Quantitative comparison
    dist = compare_diagrams(dgmA, dgmB)

    # Save summary
    summary_path = out / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Graph A: {nameA}\nNodes: {len(nodesA)} Edges: {graphA.number_of_edges()}\n")
        f.write(f"Graph B: {nameB}\nNodes: {len(nodesB)} Edges: {graphB.number_of_edges()}\n\n")
        for k, v in dist.items():
            f.write(f"{k}: {v}\n")
    return {
        "figA": str(figA),
        "figB": str(figB),
        "metrics": dist,
        "summary": str(summary_path)
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Persistent homology comparison of two chemical networks")
    ap.add_argument("--demo", action="store_true", help="Run toy comparison (aromatic vs terpenoid)")
    ap.add_argument("--graphA", type=str, default=None, help="CSV edge list for graph A")
    ap.add_argument("--graphB", type=str, default=None, help="CSV edge list for graph B")
    ap.add_argument("--nameA", type=str, default="A", help="Name label for graph A")
    ap.add_argument("--nameB", type=str, default="B", help="Name label for graph B")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.demo:
        GA = toy_kegg_like_graph("aromatic")
        GB = toy_kegg_like_graph("terpenoid")
        args.nameA = "Aromatic_like"
        args.nameB = "Terpenoid_like"
    else:
        if not args.graphA or not args.graphB:
            raise SystemExit("Provide --graphA and --graphB CSVs, or pass --demo")
        GA = load_graph_from_csv(args.graphA)
        GB = load_graph_from_csv(args.graphB)

    results = run_pipeline(GA, GB, args.nameA, args.nameB, outdir=args.outdir)

    print("=== Persistent Homology Comparison ===")
    print(f"A diagram image: {results['figA']}")
    print(f"B diagram image: {results['figB']}")
    print("Distances:")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v}")
    print(f"Summary saved to: {results['summary']}")


if __name__ == "__main__":
    main()
