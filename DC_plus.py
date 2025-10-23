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
import json
import logging
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, ConstantInputWarning
import warnings
import pandas as pd
from copy import deepcopy

sns.set_theme(style="white", context="notebook")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def parse_kgml(kgml_file):
    
    tree = ET.parse(kgml_file)
    root = tree.getroot()

    species_map = {}
    for entry in root.findall('entry'):
        if entry.get('type') == 'compound':
            kid = entry.get('id')
            name = entry.get('name')
            species_map[kid] = name

    reactions = []
    for reaction in root.findall('reaction'):
        rid = reaction.get('id', None)
        # Attempt to read substrates/products and possible stoichiometry attributes
        subs = []
        for s in reaction.findall('substrate'):
            sid = s.get('id')
            # KGML doesn't always include stoichiometry; default to 1
            sto = float(s.get('stoichiometry', 1.0)) if s.get('stoichiometry') else 1.0
            subs.append((sid, sto))
        prods = []
        for p in reaction.findall('product'):
            pid = p.get('id')
            sto = float(p.get('stoichiometry', 1.0)) if p.get('stoichiometry') else 1.0
            prods.append((pid, sto))
        reactions.append({'id': rid, 'substrates': subs, 'products': prods, 'element': reaction})

    species_list = sorted(list(species_map.keys()))
    return species_list, reactions, species_map

# ---------------------------- Parameter handling ----------------------------

def load_params(params_path):
    
    if params_path is None:
        return {}
    if params_path.endswith('.json'):
        with open(params_path) as f:
            return json.load(f)
    else:
        # Try CSV
        df = pd.read_csv(params_path)
        params = {'reactions': {}, 'default_k': None}
        for _, row in df.iterrows():
            params['reactions'][str(row['reaction_id'])] = {'k': float(row['k']), 'type': row.get('type', 'massaction')}
        return params

# ---------------------------- ODE system builder ----------------------------

def build_ode_system(species_list, reactions, params=None, default_k=1.0):
   
    n = len(species_list)
    species_index = {s: i for i, s in enumerate(species_list)}
    m = len(reactions)

    # Stoichiometric matrix (n x m)
    S = np.zeros((n, m))
    for j, rx in enumerate(reactions):
        for sid, sto in rx['substrates']:
            if sid in species_index:
                S[species_index[sid], j] -= sto
        for pid, sto in rx['products']:
            if pid in species_index:
                S[species_index[pid], j] += sto

    # Reaction parameter list (k, type, substrates list (indices), products list)
    reaction_info = []
    params = params or {}
    reaction_params = params.get('reactions', {}) if isinstance(params, dict) else {}
    global_default_k = params.get('default_k', default_k) if isinstance(params, dict) else default_k

    for j, rx in enumerate(reactions):
        rid = rx.get('id') or str(j)
        p = reaction_params.get(rid, {})
        k = p.get('k', global_default_k)
        rtype = p.get('type', 'massaction')
        substrate_idxs = [species_index[sid] for sid, _ in rx['substrates'] if sid in species_index]
        product_idxs = [species_index[pid] for pid, _ in rx['products'] if pid in species_index]
        sto_subs = [sto for sid, sto in rx['substrates'] if sid in species_index]
        reaction_info.append({'id': rid, 'k': float(k), 'type': rtype, 'substrate_idxs': substrate_idxs, 'product_idxs': product_idxs, 'sto_subs': sto_subs})

    def odes(t, x):
        # Compute reaction velocities v (length m)
        v = np.zeros(m)
        for j, info in enumerate(reaction_info):
            if info['type'] == 'massaction':
                if len(info['substrate_idxs']) == 0:
                    # zero-order or constant rate
                    v[j] = info['k']
                else:
                    # mass-action: k * prod([x_i ** sto_i])
                    prod = 1.0
                    for idx, sto in zip(info['substrate_idxs'], info.get('sto_subs', [1]*len(info['substrate_idxs']))):
                        # protect against negative/zero
                        prod *= max(x[idx], 0.0) ** sto
                    v[j] = info['k'] * prod
            else:
                # placeholder for other kinetics (e.g., michaelis-menten)
                # implementers can extend by adding cases here
                v[j] = info['k']
        dxdt = S.dot(v)
        return dxdt

    return odes, species_index, reaction_info

# ---------------------------- Simulation utilities ----------------------------

def simulate(species_list, reactions, params=None, t_span=(0, 20), n_points=200, method='RK45', atol=1e-6, rtol=1e-3, steady_threshold=None):
    
    odes, species_index, reaction_info = build_ode_system(species_list, reactions, params)
    x0 = np.ones(len(species_list))

    if steady_threshold is None:
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(odes, t_span, x0, method=method, atol=atol, rtol=rtol, t_eval=t_eval)
        return sol.t, sol.y

    # Adaptive: integrate in chunks until steady
    t0, tmax = t_span
    t_current = t0
    x_current = x0.copy()
    t_all = [t0]
    y_all = [x0.copy()]
    dt_chunk = max((tmax - t0) / 10.0, 1.0)
    while t_current < tmax:
        t_next = min(t_current + dt_chunk, tmax)
        sol = solve_ivp(odes, (t_current, t_next), x_current, method=method, atol=atol, rtol=rtol, t_eval=np.linspace(t_current, t_next, max(2, int(n_points * (t_next - t_current) / (tmax - t0)))))
        # append excluding first point to avoid duplicates
        for tt, xx in zip(sol.t[1:], sol.y.T[1:]):
            t_all.append(tt)
            y_all.append(xx.copy())
        # check steady
        if len(y_all) > 1:
            dx = np.linalg.norm(y_all[-1] - y_all[-2])
            if dx < steady_threshold:
                break
        t_current = t_next
        x_current = y_all[-1].copy()
    return np.array(t_all), np.vstack(y_all).T

# Monte Carlo wrapper
def simulate_mc(species_list, reactions, params=None, t_span=(0, 20), n_points=200, method='RK45', mc_runs=1, init_scale=0.1, **kwargs):
    
    runs = []
    for i in range(mc_runs):
        t, y = simulate(species_list, reactions, params=params, t_span=t_span, n_points=n_points, method=method, **kwargs)
        runs.append((t, y))
    # align time grids by choosing the smallest common set (simple strategy: take first run's grid)
    t_ref = runs[0][0]
    Ys = np.stack([np.interp(t_ref, r[0], r[1]) if (len(r[0]) != len(t_ref) or not np.allclose(r[0], t_ref)) else r[1] for r in runs], axis=2)
    y_mean = np.mean(Ys, axis=2)
    return t_ref, y_mean, runs
#
def normalize_rows(y):
    # avoid division by zero
    mx = np.max(y, axis=1, keepdims=True)
    mx[mx == 0] = 1.0
    return y / mx


def compute_similarity(y1, y2):
    
    # align number of species
    n = min(y1.shape[0], y2.shape[0])
    y1n = normalize_rows(y1[:n])
    y2n = normalize_rows(y2[:n])

    dtw_scores, mse_scores, corr_scores = [], [], []
    for i in range(n):
        # DTW (on normalized)
        try:
            dtw_scores.append(dtw.distance(y1n[i], y2n[i]))
        except Exception:
            dtw_scores.append(np.nan)
        # MSE (on raw scaled to same max)
        try:
            mse_scores.append(mean_squared_error(y1[i], y2[i]))
        except Exception:
            mse_scores.append(np.nan)
        # Correlation
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=ConstantInputWarning)
            try:
                corr, _ = pearsonr(y1n[i], y2n[i])
            except Exception:
                corr = np.nan
        corr_scores.append(corr)

    return {
        'dtw_avg': float(np.nanmean(dtw_scores)),
        'mse_avg': float(np.nanmean(mse_scores)),
        'corr_avg': float(np.nanmean(corr_scores)),
        'dtw': dtw_scores,
        'mse': mse_scores,
        'corr': corr_scores
    }

# ---------------------------- Plotting ----------------------------

def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_trajectories(t, y, species, title, save_path):
    fig = plt.figure(figsize=(12, 7))
    colors = sns.color_palette('tab20', max(4, len(species)))
    for i in range(y.shape[0]):
        plt.plot(t, y[i], label=species[i], lw=1.5, color=colors[i % len(colors)])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title(title)
    save_plot(fig, save_path)


def plot_overlay(t1, y1, t2, y2, species, save_path):
    fig = plt.figure(figsize=(14, 8))
    n = min(y1.shape[0], y2.shape[0])
    colors = sns.color_palette('tab20', max(4, n))
    
    for i in range(n):
        plt.plot(t1, y1[i], linestyle='-', lw=1.8, color=colors[i % len(colors)], label=f"{species[i]} (Net1)")
        plt.plot(t2, y2[i], linestyle='--', lw=1.8, color=colors[i % len(colors)], label=f"{species[i]} (Net2)")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title(f'Overlay of all {n} common species')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)



def plot_heatmap(y, species, title, save_path):
    fig = plt.figure(figsize=(12, 8))
    # ensure y is species x time
    sns.heatmap(y, cmap='coolwarm', xticklabels=False, yticklabels=species, cbar_kws={'label': 'Concentration'})
    plt.xlabel('Time index')
    plt.ylabel('Species')
    plt.title(title)
    save_plot(fig, save_path)


def plot_bar(scores, species, metric, save_path):
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=species[:len(scores)], y=scores, palette='crest')
    plt.xticks(rotation=90)
    plt.ylabel(metric)
    plt.title(f'Per-species dynamic similarity ({metric})')
    save_plot(fig, save_path)

# ---------------------------- CLI / main ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Improved dynamic KGML network comparison')
    parser.add_argument('file1')
    parser.add_argument('file2')
    parser.add_argument('--params', default=None, help='JSON/CSV with reaction parameters')
    parser.add_argument('--outdir', default='./outputs')
    parser.add_argument('--tmax', type=float, default=20.0)
    parser.add_argument('--npoints', type=int, default=200)
    parser.add_argument('--method', default='RK45')
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--steady_threshold', type=float, default=None, help='If set, adaptively stop when change < threshold')
    parser.add_argument('--mc', type=int, default=1, help='Monte Carlo runs')
    args = parser.parse_args()

    params = load_params(args.params)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.outdir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    try:
        s1, r1, map1 = parse_kgml(args.file1)
        s2, r2, map2 = parse_kgml(args.file2)
    except Exception as e:
        logging.error('Failed to parse KGML files: %s', e)
        sys.exit(1)

    # match species by KEGG id (intersection)
    common = sorted(list(set(s1) & set(s2)))
    if len(common) == 0:
        logging.warning('No species in common by KEGG id; attempting to match by name')
        # attempt name-based match
        names1 = {v: k for k, v in map1.items()}
        names2 = {v: k for k, v in map2.items()}
        common_names = set(names1.keys()) & set(names2.keys())
        common = [names1[n] for n in common_names]

    if len(common) == 0:
        logging.error('No common species found between the two networks. Aborting comparison.')
        sys.exit(1)

    # simulate each network (Monte Carlo optional)
    t1, y1_mean, runs1 = simulate_mc(s1, r1, params=params, t_span=(0, args.tmax), n_points=args.npoints, method=args.method, mc_runs=args.mc, steady_threshold=args.steady_threshold, atol=args.atol, rtol=args.rtol)
    t2, y2_mean, runs2 = simulate_mc(s2, r2, params=params, t_span=(0, args.tmax), n_points=args.npoints, method=args.method, mc_runs=args.mc, steady_threshold=args.steady_threshold, atol=args.atol, rtol=args.rtol)

    # Extract only common species in the order of 'common'
    idx1 = [s1.index(cid) for cid in common]
    idx2 = [s2.index(cid) for cid in common]
    y1_common = y1_mean[idx1]
    y2_common = y2_mean[idx2]
    species_names = [map1[cid] for cid in common]

    # Save plots
    plot_trajectories(t1, y1_common, species_names, f'Dynamics {os.path.basename(args.file1)}', os.path.join(run_dir, 'dynamics_net1.png'))
    plot_trajectories(t2, y2_common, species_names, f'Dynamics {os.path.basename(args.file2)}', os.path.join(run_dir, 'dynamics_net2.png'))
    plot_overlay(t1, y1_common, t2, y2_common, species_names, os.path.join(run_dir, 'overlay.png'))
    plot_heatmap(y1_common, species_names, 'Heatmap Net1', os.path.join(run_dir, 'heatmap_net1.png'))
    plot_heatmap(y2_common, species_names, 'Heatmap Net2', os.path.join(run_dir, 'heatmap_net2.png'))

    results = compute_similarity(y1_common, y2_common)

    plot_bar(results['dtw'], species_names, 'DTW', os.path.join(run_dir, 'similarity_dtw.png'))
    plot_bar(results['mse'], species_names, 'MSE', os.path.join(run_dir, 'similarity_mse.png'))
    plot_bar(results['corr'], species_names, 'Correlation', os.path.join(run_dir, 'similarity_corr.png'))

    # Save structured JSON
    summary = {
        'file1': os.path.abspath(args.file1),
        'file2': os.path.abspath(args.file2),
        'timestamp': timestamp,
        'common_species': {cid: map1[cid] for cid in common},
        'metrics': {
            'dtw_avg': results['dtw_avg'],
            'mse_avg': results['mse_avg'],
            'corr_avg': results['corr_avg'],
            'per_species': [{
                'id': cid,
                'name': map1[cid],
                'dtw': float(d),
                'mse': float(m),
                'corr': float(c) if not pd.isna(c) else None
            } for cid, d, m, c in zip(common, results['dtw'], results['mse'], results['corr'])]
        }
    }
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info('Results saved in: %s', run_dir)
    logging.info('DTW avg: %.3f, MSE avg: %.3f, Corr avg: %.3f', results['dtw_avg'], results['mse_avg'], results['corr_avg'])


if __name__ == '__main__':
    main()
