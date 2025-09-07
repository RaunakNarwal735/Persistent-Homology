import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from dtaidistance import dtw
import argparse
import sys
import os
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, ConstantInputWarning
import warnings

# Global seaborn style
sns.set_theme(style="white", context="notebook")

# -------------------------------
# 1. Parse KGML into reactions
# -------------------------------
def parse_kgml(kgml_file):
    tree = ET.parse(kgml_file)
    root = tree.getroot()
    
    species = {}
    reactions = []
    
    for entry in root.findall("entry"):
        if entry.get("type") == "compound":
            species[entry.get("id")] = entry.get("name")
    
    for reaction in root.findall("reaction"):
        substrates = [s.get("id") for s in reaction.findall("substrate")]
        products = [p.get("id") for p in reaction.findall("product")]
        for s in substrates:
            for p in products:
                reactions.append((s, p))
    
    return list(species.keys()), reactions, species

# -------------------------------
# 2. Build ODEs
# -------------------------------
def build_ode_system(species, reactions, rate=1.0):
    n = len(species)
    species_index = {s: i for i, s in enumerate(species)}
    
    S = np.zeros((n, len(reactions)))
    for j, (s, p) in enumerate(reactions):
        if s in species_index and p in species_index:
            S[species_index[s], j] -= 1
            S[species_index[p], j] += 1
    
    def odes(t, x):
        v = np.zeros(len(reactions))
        for j, (s, p) in enumerate(reactions):
            if s in species_index:
                v[j] = rate * x[species_index[s]]
        return S @ v
    
    return odes, species_index

# -------------------------------
# 3. Simulate network
# -------------------------------
def simulate_network(kgml_file, t_span=(0, 20), n_points=200):
    species, reactions, mapping = parse_kgml(kgml_file)
    odes, idx = build_ode_system(species, reactions)
    
    x0 = np.ones(len(species))
    sol = solve_ivp(odes, t_span, x0, t_eval=np.linspace(*t_span, n_points))
    return sol.t, sol.y, list(mapping.values())

# -------------------------------
# 4. Similarity scoring
# -------------------------------
def compute_similarity(y1, y2):
    y1 = y1 / np.max(y1, axis=1, keepdims=True)
    y2 = y2 / np.max(y2, axis=1, keepdims=True)
    
    dtw_scores, mse_scores, corr_scores = [], [], []
    
    for i in range(min(y1.shape[0], y2.shape[0])):
        # DTW
        dtw_scores.append(dtw.distance(y1[i], y2[i]))
        # MSE
        mse_scores.append(mean_squared_error(y1[i], y2[i]))
        # Correlation (handle constant signals)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            try:
                corr, _ = pearsonr(y1[i], y2[i])
            except Exception:
                corr = np.nan
        corr_scores.append(corr)
    
    return {
        "dtw_avg": np.nanmean(dtw_scores),
        "mse_avg": np.nanmean(mse_scores),
        "corr_avg": np.nanmean(corr_scores),
        "dtw": dtw_scores,
        "mse": mse_scores,
        "corr": corr_scores
    }

# -------------------------------
# 5. Plot functions
# -------------------------------
def plot_trajectories(t, y, species, title, save_path):
    colors = sns.color_palette("tab20", len(species))
    plt.figure(figsize=(12, 7))
    for i in range(y.shape[0]):
        plt.plot(t, y[i], label=species[i], color=colors[i % len(colors)], lw=1.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_overlay(t1, y1, t2, y2, species, save_path, n_show=10):
    colors = sns.color_palette("tab10", n_show)
    plt.figure(figsize=(12, 7))
    for i in range(min(n_show, y1.shape[0], y2.shape[0])):
        plt.plot(t1, y1[i], color=colors[i], linestyle="-", lw=2, label=f"{species[i]} (Net1)")
        plt.plot(t2, y2[i], color=colors[i], linestyle="--", lw=2, label=f"{species[i]} (Net2)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Overlay of species dynamics (first {} species)".format(n_show))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_similarity_bar(scores, species, metric, save_path):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=species[:len(scores)], y=scores, palette="crest", hue=None, legend=False)
    plt.xticks(rotation=90)
    plt.ylabel(metric)
    plt.title(f"Per-species dynamic similarity ({metric})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_heatmap(y, species, title, save_path):
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        y, cmap="coolwarm", xticklabels=False, yticklabels=species,
        cbar_kws={'label': 'Concentration'}
    )
    plt.xlabel("Time index")
    plt.ylabel("Species")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic similarity of two KGML networks")
    parser.add_argument("file1", help="First KGML file")
    parser.add_argument("file2", help="Second KGML file")
    parser.add_argument("--outdir", default=r"C:\Users\rishu narwal\Desktop\MPI-CBG\outputs", help="Output base directory")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    try:
        t1, y1, sp1 = simulate_network(args.file1)
        t2, y2, sp2 = simulate_network(args.file2)
    except Exception as e:
        sys.exit(f"Error parsing or simulating: {e}")

    # Save plots
    plot_trajectories(t1, y1, sp1, f"Dynamics of {args.file1}", os.path.join(run_dir, "dynamics_net1.png"))
    plot_trajectories(t2, y2, sp2, f"Dynamics of {args.file2}", os.path.join(run_dir, "dynamics_net2.png"))
    plot_overlay(t1, y1, t2, y2, sp1, os.path.join(run_dir, "overlay.png"))
    plot_heatmap(y1, sp1, f"Heatmap Network 1", os.path.join(run_dir, "heatmap_net1.png"))
    plot_heatmap(y2, sp2, f"Heatmap Network 2", os.path.join(run_dir, "heatmap_net2.png"))

    # Compute similarity
    results = compute_similarity(y1, y2)
    plot_similarity_bar(results["dtw"], sp1, "DTW", os.path.join(run_dir, "similarity_dtw.png"))
    plot_similarity_bar(results["mse"], sp1, "MSE", os.path.join(run_dir, "similarity_mse.png"))
    plot_similarity_bar(results["corr"], sp1, "Correlation", os.path.join(run_dir, "similarity_corr.png"))

    # Save scores to text
    with open(os.path.join(run_dir, "similarity.txt"), "w") as f:
        f.write(f"Dynamic similarity scores:\n")
        f.write(f"DTW average: {results['dtw_avg']:.3f}\n")
        f.write(f"MSE average: {results['mse_avg']:.3f}\n")
        f.write(f"Correlation average: {results['corr_avg']:.3f}\n\n")
        f.write("Per-species scores:\n")
        for s, d, m, c in zip(sp1, results["dtw"], results["mse"], results["corr"]):
            f.write(f"{s}: DTW={d:.3f}, MSE={m:.3f}, Corr={c:.3f}\n")

    print(f"Results saved in: {run_dir}")
    print(f"DTW avg: {results['dtw_avg']:.3f}, MSE avg: {results['mse_avg']:.3f}, Corr avg: {results['corr_avg']:.3f}")
