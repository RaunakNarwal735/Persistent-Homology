import argparse # For command-line argument parsing
import os # Operating system interfaces
from pathlib import Path # Object-oriented filesystem paths
import json 
import math 
import numpy as np
import pandas as pd
import networkx as nx # NetworkX for graph handling
from scipy.sparse.csgraph import floyd_warshall # For shortest path computations
from ripser import ripser # Persistent homology computations
from persim import plot_diagrams, bottleneck, wasserstein # Diagram plotting and distance metrics
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET # XML parsing
# Set seaborn style (for better aesthetics)
sns.set(style="whitegrid")
# sns.set_context("notebook")
def ensure_outdir(path: str = "outputs") -> Path:
    """
    
    Parameters
    ----------
    path: str :
         (Default value = "outputs")

    Returns
    -------
     Ensures output directory exists; creates it if missing.
     Returns a pathlib object to the directory.
  
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# Converts a KEGG KGML pathway file into a simple CSV list.
# Extracts all entries (nodes) and relations/reactions (edges).
# KGML stores reaction edges differently; this unifies them into a single format.

def kgml_to_csv(kgml_path: str, out_csv: str):
    """

    Parameters
    ----------
    kgml_path: str :
        
    out_csv: str :
        

    Returns
    -------

    """
    tree = ET.parse(kgml_path) 
    root = tree.getroot() # Get root element of the XML tree

    entries = {} # Map of entry IDs to names
    for entry in root.findall("entry"): 
        eid = entry.attrib.get("id") 
        name = entry.attrib.get("name", eid)
        entries[eid] = name

    edges = [] # List of edges (source, target, type)
    for relation in root.findall("relation"):
        e1 = relation.attrib.get("entry1")
        e2 = relation.attrib.get("entry2")
        rtype = relation.attrib.get("type", "relation")
        if e1 in entries and e2 in entries:
            edges.append((entries[e1], entries[e2], rtype))

    for reaction in root.findall("reaction"): 
        substrates = [s.attrib.get("id") for s in reaction.findall("substrate")]
        products = [p.attrib.get("id") for p in reaction.findall("product")]
        for s in substrates:
            for p in products:
                if s in entries and p in entries:
                    edges.append((entries[s], entries[p], "reaction"))

    df = pd.DataFrame(edges, columns=["source", "target", "type"])
    df.to_csv(out_csv, index=False)

    summary = {"nodes": len(entries), "edges": len(df)}
    print(f"Converted {kgml_path} -> {out_csv} (nodes={summary['nodes']}, edges={summary['edges']})")
    return summary

# Load an edge-list CSV into a NetworkX graph.
# Handles both weighted and unweighted edges.

def load_graph_from_csv(csv_path: str, directed: bool = True) -> nx.Graph:
    """

    Parameters
    ----------
    csv_path: str :
        
    directed: bool :
         (Default value = True)

    Returns
    -------

    """
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    if not {"source", "target"}.issubset(df.columns): 
        raise ValueError("CSV must contain columns: source,target,(optional)weight or type")
    has_w = "weight" in df.columns

    G = nx.DiGraph() if directed else nx.Graph() # Create directed or undirected graph
    if has_w:
        for _, r in df.iterrows():
            G.add_edge(str(r["source"]), str(r["target"]), weight=float(r["weight"]))
    else:
        for _, r in df.iterrows():
            G.add_edge(str(r["source"]), str(r["target"]), weight=1.0)

    UG = nx.Graph() # Convert to undirected graph with minimum weights
    for u, v, w in G.edges(data=True):
        wt = w.get("weight", 1.0)
        if UG.has_edge(u, v):
            UG[u][v]["weight"] = min(UG[u][v]["weight"], wt)
        else:
            UG.add_edge(u, v, weight=wt)
    print(f"Loaded CSV graph {csv_path}: nodes={UG.number_of_nodes()}, edges={UG.number_of_edges()}")
    return UG

# Load a KGML file as a graph (NetworkX undirected).
# Extract nodes and edges, similar to kgml_to_csv but directly into graph form.

def load_graph_from_kgml(kgml_path: str) -> nx.Graph:
    """

    Parameters
    ----------
    kgml_path: str :
        

    Returns
    -------

    """
    tree = ET.parse(kgml_path)
    root = tree.getroot()
    entries = {}
    for entry in root.findall("entry"):
        eid = entry.attrib.get("id")
        name = entry.attrib.get("name", eid)
        entries[eid] = name

    G = nx.Graph()
    for relation in root.findall("relation"):
        e1 = relation.attrib.get("entry1")
        e2 = relation.attrib.get("entry2")
        if e1 in entries and e2 in entries:
            G.add_edge(entries[e1], entries[e2], weight=1.0)
    for reaction in root.findall("reaction"):
        subs = [s.attrib.get("id") for s in reaction.findall("substrate")]
        prods = [p.attrib.get("id") for p in reaction.findall("product")]
        for s in subs:
            for p in prods:
                if s in entries and p in entries:
                    G.add_edge(entries[s], entries[p], weight=1.0)
    print(f"Loaded KGML graph {kgml_path}: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    return G

# Compute graph diameter safely, even for:
# empty graphs
# single node graphs
# disconnected graphs
# Falls back to Floyd–Warshall if NetworkX cannot find diameter.

def graph_diameter(G: nx.Graph) -> int:
    """

    Parameters
    ----------
    G: nx.Graph :
        

    Returns
    -------

    """
    
    if G.number_of_nodes() == 0:
        return 0
    # For graphs with single node or no edges, diameter is 0
    if G.number_of_nodes() == 1:
        return 0
    # For connected graphs
    if nx.is_connected(G):
        try:
            return int(nx.diameter(G))
        except Exception:
            # fallback to longest shortest path via Floyd–Warshall
            D = dict(nx.all_pairs_shortest_path_length(G))
            maxd = max(d for u in D for d in D[u].values())
            return int(maxd)
    # For disconnected, compute per component
    maxd = 0
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        if sub.number_of_nodes() <= 1:
            d = 0
        else:
            try:
                d = nx.diameter(sub)
            except Exception:
                D = dict(nx.all_pairs_shortest_path_length(sub))
                d = max(dv for u in D for dv in D[u].values())
        if d > maxd:
            maxd = d
    return int(maxd)

# Convert the graph into a full distance matrix.
# Uses Floyd–Warshall for all-pairs shortest paths.
# Infinite distances (disconnected nodes) are replaced with a large constant.

def shortest_path_distance_matrix(G: nx.Graph, disconnected_value: float = None) -> (np.ndarray, list):
    """

    Parameters
    ----------
    G: nx.Graph :
        
    disconnected_value: float :
         (Default value = None) -> (np.ndarray)
    list :
        

    Returns
    -------

    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return np.array([[]]), []
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

# Run Ripser on the distance matrix to compute persistence diagrams.
# maxdim controls homology dimension (H0, H1,...).
# thresh limits the filtration to keep runtime reasonable.

# Returns diagrams and full Ripser result.
def persistence_from_distance(D: np.ndarray, maxdim: int = 1, thresh: float = None):
    
    """

    Parameters
    ----------
    D: np.ndarray :
        
    maxdim: int :
         (Default value = 1)
    thresh: float :
         (Default value = None)

    Returns
    -------

    """
    if D.size == 0:
        return [np.empty((0,2)) for _ in range(maxdim+1)], None
    finite = D[np.isfinite(D)]
    if thresh is None:
        thresh = np.percentile(finite, 95) if finite.size else 1.0
    res = ripser(D, maxdim=maxdim, thresh=thresh, distance_matrix=True)
    return res["dgms"], res

# Convert persistence diagrams to a dictionary format for JSON serialization.
def persistence_to_dict(dgms):
    """

    Parameters
    ----------
    dgms :
        

    Returns
    -------

    """
    return {f"H{dim}": dgms[dim].tolist() if len(dgms) > dim else [] for dim in range(len(dgms))}

# Plot and save persistence diagrams.
def plot_and_save(diagrams, title: str, outdir: Path, tag: str, show: bool = False):
    """

    Parameters
    ----------
    diagrams :
        
    title: str :
        
    outdir: Path :
        
    tag: str :
        
    show: bool :
         (Default value = False)

    Returns
    -------

    """
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,5))
    plot_diagrams(diagrams, show=False)
    plt.title(f"Persistence Diagram: {title}")
    fp = outdir / f"{tag}_diagram.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=200)
    if show:
        plt.show()
    plt.close()
    return fp

# Plot and save the graph structure.
def plot_graph(G: nx.Graph, outdir: Path, name: str, show: bool = False):
    """

    Parameters
    ----------
    G: nx.Graph :
        
    outdir: Path :
        
    name: str :
        
    show: bool :
         (Default value = False)

    Returns
    -------

    """
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,7))
    pos = nx.spring_layout(G, seed=42)

    degrees = dict(G.degree())
    if len(degrees) == 0:
        node_colors = []
        node_sizes = []
    else:
        deg_vals = np.array(list(degrees.values()), dtype=float)
        min_sz, max_sz = 200, 900
        if deg_vals.max() == deg_vals.min():
            node_sizes = np.full(len(deg_vals), (min_sz + max_sz) / 2.0)
        else:
            node_sizes = min_sz + (deg_vals - deg_vals.min()) / (deg_vals.max() - deg_vals.min()) * (max_sz - min_sz)

        cmap = plt.get_cmap('viridis')
        if deg_vals.max() == deg_vals.min():
            node_colors = [cmap(0.5) for _ in deg_vals]
        else:
            norm = plt.Normalize(vmin=deg_vals.min(), vmax=deg_vals.max())
            node_colors = [cmap(norm(v)) for v in deg_vals]

    nx.draw_networkx_edges(G, pos, alpha=0.6, width=1.2)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes if len(G.nodes())>0 else 100,
                           node_color=node_colors if len(G.nodes())>0 else '#4c72b0', linewidths=0.5, edgecolors='k')
    plt.axis('off')

    fp = outdir / f"{name}_graph.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight', pad_inches=0.1)
    if show:
        plt.show()
    plt.close()
    return fp

# Compute bottleneck and Wasserstein distances between two diagrams.
def compute_distance_metrics(dgA, dgB, p: int = 2):
    """

    Parameters
    ----------
    dgA :
        
    dgB :
        
    p: int :
         (Default value = 2)

    Returns
    -------

    """
    A = dgA if dgA is not None else np.empty((0,2))
    B = dgB if dgB is not None else np.empty((0,2))
    A = np.asarray(A)
    B = np.asarray(B)

    if A.size == 0 and B.size == 0:
        bn = 0.0
    elif A.size == 0:
        pers = B[:,1] - B[:,0]
        bn = float(np.max(pers) / 2.0) if pers.size > 0 else 0.0
    elif B.size == 0:
        pers = A[:,1] - A[:,0]
        bn = float(np.max(pers) / 2.0) if pers.size > 0 else 0.0
    else:
        try:
            bn = float(bottleneck(A, B))
        except Exception:
            bn = max(float(np.max(A[:,1]-A[:,0]) / 2.0), float(np.max(B[:,1]-B[:,0]) / 2.0))

    if A.size == 0 and B.size == 0:
        ws = 0.0
    elif A.size == 0:
        pers = B[:,1] - B[:,0]
        ws = float((np.sum((pers / 2.0) ** p)) ** (1.0 / p))
    elif B.size == 0:
        pers = A[:,1] - A[:,0]
        ws = float((np.sum((pers / 2.0) ** p)) ** (1.0 / p))
    else:
        try:
            ws = float(wasserstein(A, B, matching=False, p=float(p)))
        except Exception:
            persA = A[:,1] - A[:,0]
            persB = B[:,1] - B[:,0]
            ws = float((np.sum((persA / 2.0) ** p) + np.sum((persB / 2.0) ** p)) ** (1.0 / p))

    return bn, ws

# Compute distance metrics for all homology dimensions in the diagrams.
def compare_diagrams_full(diagA, diagB, p: int = 2):
    """

    Parameters
    ----------
    diagA :
        
    diagB :
        
    p: int :
         (Default value = 2)

    Returns
    -------

    """
    metrics = {}
    maxdim = max(len(diagA), len(diagB))
    for dim in range(maxdim):
        A = diagA[dim] if len(diagA) > dim else np.empty((0,2))
        B = diagB[dim] if len(diagB) > dim else np.empty((0,2))
        bn, ws = compute_distance_metrics(A, B, p=p)
        metrics[f"H{dim}_bottleneck"] = bn
        metrics[f"H{dim}_wasserstein"] = ws
    return metrics

# Main pipeline to run the full analysis.
def run_pipeline(graphA: nx.Graph, graphB: nx.Graph, nameA: str, nameB: str, outdir: str = "outputs", maxdim: int = 1, p: int = 2):
    """

    Parameters
    ----------
    graphA: nx.Graph :
        
    graphB: nx.Graph :
        
    nameA: str :
        
    nameB: str :
        
    outdir: str :
         (Default value = "outputs")
    maxdim: int :
         (Default value = 1)
    p: int :
         (Default value = 2)

    Returns
    -------

    """
    out = ensure_outdir(outdir)

    DA, nodesA = shortest_path_distance_matrix(graphA)
    DB, nodesB = shortest_path_distance_matrix(graphB)

    dgmA, resA = persistence_from_distance(DA, maxdim=maxdim)
    dgmB, resB = persistence_from_distance(DB, maxdim=maxdim)

    np.save(out / f"{nameA}_dgms.npy", np.array(dgmA, dtype=object), allow_pickle=True)
    np.save(out / f"{nameB}_dgms.npy", np.array(dgmB, dtype=object), allow_pickle=True)

    with open(out / f"{nameA}_dgms.json", "w") as f:
        json.dump(persistence_to_dict(dgmA), f, indent=2)
    with open(out / f"{nameB}_dgms.json", "w") as f:
        json.dump(persistence_to_dict(dgmB), f, indent=2)

    figA = plot_and_save(dgmA, title=nameA, outdir=out, tag=f"{nameA}_tda")
    figB = plot_and_save(dgmB, title=nameB, outdir=out, tag=f"{nameB}_tda")
    gfigA = plot_graph(graphA, out, nameA)
    gfigB = plot_graph(graphB, out, nameB)

    metrics = compare_diagrams_full(dgmA, dgmB, p=p)

    diamA = graph_diameter(graphA)
    diamB = graph_diameter(graphB)
    diam_scale = max(diamA, diamB)

    norm_metrics = {}
    for k, v in metrics.items():
        norm_metrics[k] = v
        
        if diam_scale > 0 and (isinstance(v, (int, float)) and not math.isinf(v)):
            norm_metrics[f"{k}_norm"] = float(v) / float(diam_scale)
        elif diam_scale > 0 and math.isinf(v):
            norm_metrics[f"{k}_norm"] = float('inf')
        else:
            norm_metrics[f"{k}_norm"] = 0.0

    summary = {
        "GraphA": {"name": nameA, "nodes": len(nodesA), "edges": graphA.number_of_edges(), "diameter": diamA, "graph_plot": str(gfigA), "diagram_plot": str(figA)},
        "GraphB": {"name": nameB, "nodes": len(nodesB), "edges": graphB.number_of_edges(), "diameter": diamB, "graph_plot": str(gfigB), "diagram_plot": str(figB)},
        "Metrics": norm_metrics
    }

    summary_path = out / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(json.dumps(summary, indent=2))
    json_path = out / "summary.json"
    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=2)

    print(f"Saved results to {out}")
    return {
        "figA": str(figA),
        "figB": str(figB),
        "graphA": str(gfigA),
        "graphB": str(gfigB),
        "metrics": norm_metrics,
        "summary_txt": str(summary_path),
        "summary_json": str(json_path)
    }


def parse_args():
    """ """
    ap = argparse.ArgumentParser(description="Persistent homology comparison of two chemical networks")
    ap.add_argument("--graphA", type=str, default=None, help="CSV or KGML file for graph A")
    ap.add_argument("--graphB", type=str, default=None, help="CSV or KGML file for graph B")
    ap.add_argument("--nameA", type=str, default="A", help="Name label for graph A")
    ap.add_argument("--nameB", type=str, default="B", help="Name label for graph B")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    ap.add_argument("--demo", action="store_true", help="Run with demo toy graphs instead of input files")
    ap.add_argument("--maxdim", type=int, default=1, help="Max homology dimension to compute (default 1)")
    ap.add_argument("--p", type=int, default=2, help="Wasserstein p-norm (default 2)")
    ap.add_argument("--convert-kgml", action="store_true", help="When given KGML inputs, also create CSV conversions")
    return ap.parse_args()


def make_demo_graphs():
    """ """
    G1 = nx.cycle_graph(5)
    G2 = nx.path_graph(5)
    G1 = nx.relabel_nodes(G1, lambda x: f"n{x}")
    G2 = nx.relabel_nodes(G2, lambda x: f"n{x}")
    return G1, G2


def validate_conversion(kgml_path: str, csv_path: str) -> dict:
    """

    Parameters
    ----------
    kgml_path: str :
        
    csv_path: str :
        

    Returns
    -------

    """
    Gkg = load_graph_from_kgml(kgml_path)
    Gcsv = load_graph_from_csv(csv_path)
    report = {
        "kgml_nodes": Gkg.number_of_nodes(),
        "kgml_edges": Gkg.number_of_edges(),
        "csv_nodes": Gcsv.number_of_nodes(),
        "csv_edges": Gcsv.number_of_edges(),
        "node_overlap": len(set(Gkg.nodes()) & set(Gcsv.nodes())),
        "edge_overlap": len(set(Gkg.edges()) & set(Gcsv.edges()))
    }
    return report


def main():
    """ """
    args = parse_args()

    if args.demo:
        GA, GB = make_demo_graphs()
        nameA, nameB = "Cycle5", "Path5"
    else:
        if not args.graphA or not args.graphB:
            raise SystemExit("Provide --graphA and --graphB files (CSV or KGML) (or use --demo)")

        def load_graph(path):
            """

            Parameters
            ----------
            path :
                

            Returns
            -------

            """
            if path.lower().endswith(".kgml"):
                return load_graph_from_kgml(path)
            elif path.lower().endswith(".csv"):
                return load_graph_from_csv(path)
            else:
                raise ValueError(f"Unrecognized file extension for {path}. Expected .csv or .kgml")

        GA = load_graph(args.graphA)
        GB = load_graph(args.graphB)
        nameA, nameB = args.nameA, args.nameB

        if args.convert_kgml:
            if args.graphA.lower().endswith('.kgml'):
                csvA = Path(args.graphA).with_suffix('.csv')
                kgml_to_csv(args.graphA, str(csvA))
            if args.graphB.lower().endswith('.kgml'):
                csvB = Path(args.graphB).with_suffix('.csv')
                kgml_to_csv(args.graphB, str(csvB))

    results = run_pipeline(GA, GB, nameA, nameB, outdir=args.outdir, maxdim=args.maxdim, p=args.p)

    print("=== Persistent Homology Comparison ===")
    print(f"A diagram image: {results['figA']}")
    print(f"B diagram image: {results['figB']}")
    print(f"A graph image: {results['graphA']}")
    print(f"B graph image: {results['graphB']}")
    print("Distances (raw and normalized):")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v}")

    if not args.demo and args.convert_kgml:
        if args.graphA.lower().endswith('.kgml'):
            csvA = str(Path(args.graphA).with_suffix('.csv'))
            reportA = validate_conversion(args.graphA, csvA)
            print("Conversion report A:", reportA)
        if args.graphB.lower().endswith('.kgml'):
            csvB = str(Path(args.graphB).with_suffix('.csv'))
            reportB = validate_conversion(args.graphB, csvB)
            print("Conversion report B:", reportB)


if __name__ == "__main__":
    main()
