import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from dtaidistance import dtw
import argparse
import sys
import os
import networkx as nx
from datetime import datetime
import json
import logging
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, ConstantInputWarning
import warnings
import pandas as pd
from copy import deepcopy
from matplotlib.animation import FuncAnimation
sns.set_theme(style="whitegrid", context="talk")
import random
import math
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def parse_kgml(kgml_file):
    """

    Parameters
    ----------
    kgml_file :
        

    Returns
    -------

    """
    
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
        
        subs = []
        for s in reaction.findall('substrate'):
            sid = s.get('id')
            
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
#break little 30 mintues
def parse_sbml(sbml_file):
    """

    Parameters
    ----------
    sbml_file :
        

    Returns
    -------

    """
    tree = ET.parse(sbml_file)
    root = tree.getroot()
    ns = {'sbml': 'http://www.sbml.org/sbml/level2/version4'}
    species_elems = root.findall('.//sbml:listOfSpecies/sbml:species', ns)
    species_list = [s.get('id') for s in species_elems]
    reactions_elems = root.findall('.//sbml:listOfReactions/sbml:reaction', ns)
    reactions = []
    for r in reactions_elems:
        rid = r.get('id')
        subs = []
        for s in r.findall('.//sbml:listOfReactants/sbml:speciesReference', ns):
            sid = s.get('species')
            sto = float(s.get('stoichiometry', '1'))
            subs.append((sid, sto))
        prods = []
        for p in r.findall('.//sbml:listOfProducts/sbml:speciesReference', ns):
            pid = p.get('species')
            sto = float(p.get('stoichiometry', '1'))
            prods.append((pid, sto))
        reactions.append({'id': rid, 'substrates': subs, 'products': prods})
    return species_list, reactions

def parse_auto(file_path):
    """

    Parameters
    ----------
    file_path :
        

    Returns
    -------

    """
    with open(file_path, 'r', encoding='utf-8') as f:
        first_kb = f.read(2048)
    if '<kgml' in first_kb.lower() or 'pathway' in first_kb.lower():
        return parse_kgml(file_path)
    elif '<sbml' in first_kb.lower():
        return parse_sbml(file_path)
    else:
        raise ValueError("Unrecognized file type: not KGML or SBML")

def load_params(params_path):
    """

    Parameters
    ----------
    params_path :
        

    Returns
    -------

    """
    
    if params_path is None:
        return {}
    if params_path.endswith('.json'):
        with open(params_path) as f:
            return json.load(f)
    else:
        
        df = pd.read_csv(params_path)
        params = {'reactions': {}, 'default_k': None}
        for _, row in df.iterrows():
            params['reactions'][str(row['reaction_id'])] = {'k': float(row['k']), 'type': row.get('type', 'massaction')}
        return params
#3

def build_ode_system(species_list, reactions, params=None, default_k=1.0):
    """

    Parameters
    ----------
    species_list :
        
    reactions :
        
    params :
         (Default value = None)
    default_k :
         (Default value = 1.0)

    Returns
    -------

    """
   
    n = len(species_list)
    species_index = {s: i for i, s in enumerate(species_list)}
    m = len(reactions)

    
    S = np.zeros((n, m))
    for j, rx in enumerate(reactions):
        for sid, sto in rx['substrates']:
            if sid in species_index:
                S[species_index[sid], j] -= sto
        for pid, sto in rx['products']:
            if pid in species_index:
                S[species_index[pid], j] += sto

    
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
        """

        Parameters
        ----------
        t :
            
        x :
            

        Returns
        -------

        """
        
        v = np.zeros(m)
        for j, info in enumerate(reaction_info):
            if info['type'] == 'massaction':
                if len(info['substrate_idxs']) == 0:
                    
                    v[j] = info['k']
                else:
                    
                    prod = 1.0
                    for idx, sto in zip(info['substrate_idxs'], info.get('sto_subs', [1]*len(info['substrate_idxs']))):
                        
                        prod *= max(x[idx], 0.0) ** sto
                    v[j] = info['k'] * prod
            else:
                # placeholder for other kinetics (e.g., michaelis-menten)
                
                v[j] = info['k']
        dxdt = S.dot(v)
        return dxdt

    return odes, species_index, reaction_info



def simulate(species_list, reactions, params=None, t_span=(0, 20), n_points=200, method='RK45', atol=1e-6, rtol=1e-3, steady_threshold=None):
    """

    Parameters
    ----------
    species_list :
        
    reactions :
        
    params :
         (Default value = None)
    t_span :
         (Default value = (0)
    20) :
        
    n_points :
         (Default value = 200)
    method :
         (Default value = 'RK45')
    atol :
         (Default value = 1e-6)
    rtol :
         (Default value = 1e-3)
    steady_threshold :
         (Default value = None)

    Returns
    -------

    """
    
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

def simulate_gillespie(species_list, reactions, params=None, t_max=50.0, max_events=100000, init_counts=None, rng_seed=None):
    """

    Parameters
    ----------
    species_list :
        
    reactions :
        
    params :
         (Default value = None)
    t_max :
         (Default value = 50.0)
    max_events :
         (Default value = 100000)
    init_counts :
         (Default value = None)
    rng_seed :
         (Default value = None)

    Returns
    -------

    """
    n = len(species_list)
    species_index = {sid: i for i, sid in enumerate(species_list)}
    if init_counts is None:
        X = np.full(n, 50, dtype=int)
    else:
        X = np.array(init_counts, dtype=int)
    rate_dict = {r['id']: params.get(r['id'], 1.0) if params else 1.0 for r in reactions}
    updates = []
    for r in reactions:
        dx = np.zeros(n, dtype=int)
        for sid, sto in r['substrates']:
            dx[species_index[sid]] -= int(sto)
        for pid, sto in r['products']:
            dx[species_index[pid]] += int(sto)
        updates.append(dx)
    if rng_seed is not None:
        random.seed(rng_seed)
        np.random.seed(rng_seed)
    t = 0.0
    times = [t]
    states = [X.copy()]
    events = 0
    while t < t_max and events < max_events:
        props = []
        for r in reactions:
            k = rate_dict[r['id']]
            a = k
            for sid, sto in r['substrates']:
                idx = species_index[sid]
                count = X[idx]
                if count < sto:
                    a = 0.0
                    break
                if sto == 1:
                    a *= count
                else:
                    ff = 1
                    for s in range(int(sto)):
                        ff *= (count - s)
                    a *= ff
            props.append(a)
        a0 = sum(props)
        if a0 <= 0:
            break
        r1, r2 = random.random(), random.random()
        tau = -math.log(r1) / a0
        cumsum = np.cumsum(props)
        j = np.searchsorted(cumsum, r2 * a0)
        X = X + updates[j]
        t += tau
        times.append(t)
        states.append(X.copy())
        events += 1
    return np.array(times), np.vstack(states).T


NA_CONST = 6.02214076e23

def concs_to_counts(x_conc, volume_l):
    """

    Parameters
    ----------
    x_conc :
        
    volume_l :
        

    Returns
    -------

    """
    scale = NA_CONST * volume_l
    return np.maximum(0, np.round(x_conc * scale).astype(int))

def counts_to_concs(x_counts, volume_l):
    """

    Parameters
    ----------
    x_counts :
        
    volume_l :
        

    Returns
    -------

    """
    return x_counts / (NA_CONST * volume_l)

def convert_k_ode_to_k_ssa(reaction_info, volume_l):
    """

    Parameters
    ----------
    reaction_info :
        
    volume_l :
        

    Returns
    -------

    """
    scale = NA_CONST * volume_l
    k_ssa = {}
    for r in reaction_info:
        order = sum(r.get('sto_subs', [1]*len(r.get('substrate_idxs', []))))
        if order <= 1:
            k_ssa[r['id']] = float(r['k'])
        else:
            k_ssa[r['id']] = float(r['k']) / (scale ** (order - 1))
    return k_ssa

NA_CONST = 6.02214076e23

def estimate_volume_for_target_events(species_list, reaction_info, params=None,
                                      x0_conc=None, t_span=(0.0, 20.0), target_events=2000,
                                      min_volume=1e-21, max_volume=1e-9):
    """

    Parameters
    ----------
    species_list :
        
    reaction_info :
        
    params :
         (Default value = None)
    x0_conc :
         (Default value = None)
    t_span :
         (Default value = (0.0)
    20.0) :
        
    target_events :
         (Default value = 2000)
    min_volume :
         (Default value = 1e-21)
    max_volume :
         (Default value = 1e-9)

    Returns
    -------

    """
    if x0_conc is None:
        x0_conc = np.ones(len(species_list), dtype=float)
    A_per_S = 0.0
    for r in reaction_info:
        k_ode = float(r.get('k', 1.0))
        sto_idxs = r.get('substrate_idxs', [])
        sto_vals = r.get('sto_subs', [1]*len(sto_idxs))
        prod_term = 1.0
        if len(sto_idxs) == 0:
            prod_term = 1.0
        else:
            for idx, sto in zip(sto_idxs, sto_vals):
                conc = float(x0_conc[idx]) if idx < len(x0_conc) else 0.0
                prod_term *= (conc ** float(sto))
        A_per_S += k_ode * prod_term
    if A_per_S <= 0:
        return 1e-15
    t0, t1 = float(t_span[0]), float(t_span[1])
    total_time = max(1e-12, t1 - t0)
    target_rate = float(target_events) / total_time
    S_required = target_rate / A_per_S
    V_required = float(S_required) / NA_CONST
    if V_required < min_volume:
        V_clamped = min_volume
    elif V_required > max_volume:
        V_clamped = max_volume
    else:
        V_clamped = V_required
    return V_clamped

def gillespie_setup_diagnostics(species_list, reactions, params, volume_l, n_sample=5, x0_conc=None):
    """

    Parameters
    ----------
    species_list :
        
    reactions :
        
    params :
        
    volume_l :
        
    n_sample :
         (Default value = 5)
    x0_conc :
         (Default value = None)

    Returns
    -------

    """
    odes, species_index, reaction_info = build_ode_system(species_list, reactions, params)
    if x0_conc is None:
        x0_conc = np.ones(len(species_list))
    init_counts = concs_to_counts(x0_conc, volume_l)
    rate_dict = convert_k_ode_to_k_ssa(reaction_info, volume_l)
    sample_rates = list(rate_dict.items())[:min(n_sample, len(rate_dict))]
    n_species = len(species_list)
    n_reactions = len(reactions)
    diagnostics = {
        'n_species': n_species,
        'n_reactions': n_reactions,
        'init_counts_sample': init_counts[:min(10, n_species)].tolist(),
        'rate_sample': [(rid, float(k)) for rid, k in sample_rates],
    }
    props_preview = []
    X = init_counts.copy()
    for r in reactions[:min(10, len(reactions))]:
        k = rate_dict.get(r['id'], 0.0)
        a = k
        for sid, sto in r['substrates']:
            idx = species_index[sid]
            count = X[idx]
            if count < sto:
                a = 0.0
                break
            if sto == 1:
                a *= count
            else:
                ff = 1
                for s in range(int(sto)):
                    ff *= (count - s)
                a *= ff
        props_preview.append(float(a))
    diagnostics['props_preview'] = props_preview
    diagnostics['sum_props_preview'] = float(sum(props_preview))
    return diagnostics
def simulate_gillespie_mc_auto(species_list, reactions, params=None, t_span=(0,50), n_points=200, mc_runs=20,
                               volume_l=None, x0_conc=None, target_events=2000, debug=True):
    """

    Parameters
    ----------
    species_list :
        
    reactions :
        
    params :
         (Default value = None)
    t_span :
         (Default value = (0)
    50) :
        
    n_points :
         (Default value = 200)
    mc_runs :
         (Default value = 20)
    volume_l :
         (Default value = None)
    x0_conc :
         (Default value = None)
    target_events :
         (Default value = 2000)
    debug :
         (Default value = True)

    Returns
    -------

    """
    odes, species_index, reaction_info = build_ode_system(species_list, reactions, params)
    if x0_conc is None:
        x0_conc = np.ones(len(species_list))
    if volume_l is None:
        vol = estimate_volume_for_target_events(species_list, reaction_info, params=params,
                                                x0_conc=x0_conc, t_span=t_span, target_events=target_events)
    else:
        vol = float(volume_l)
    diag = gillespie_setup_diagnostics(species_list, reactions, params, vol, x0_conc=x0_conc)
    if debug:
        logging.info("Gillespie setup diagnostics: n_species=%d n_reactions=%d", diag['n_species'], diag['n_reactions'])
        logging.info(" init_counts_sample: %s", diag['init_counts_sample'])
        logging.info(" rate_sample (first few): %s", diag['rate_sample'])
        logging.info(" props_preview (first few reactions): %s sum=%.4e", diag['props_preview'], diag['sum_props_preview'])
        logging.info(" chosen volume_l = %.4e L", vol)
    t_ref = np.linspace(t_span[0], t_span[1], n_points)
    Ys = []
    for i in range(mc_runs):
        t, X = simulate_gillespie(species_list, reactions, params=convert_k_ode_to_k_ssa(reaction_info, vol),
                                  t_max=t_span[1], init_counts=concs_to_counts(x0_conc, vol), rng_seed=i)
        X_conc = counts_to_concs(X, vol)
        if len(t) < 2:
            logging.warning("SSA run produced <2 time points; len(t)=%d. This indicates no events; consider increasing volume or rate constants.", len(t))
        Y_interp = np.array([np.interp(t_ref, t, X_conc[j].astype(float)) for j in range(X_conc.shape[0])])
        Ys.append(Y_interp)
    Ys = np.stack(Ys, axis=2)
    mean = Ys.mean(axis=2)
    std = Ys.std(axis=2)
    return t_ref, mean, std, Ys


def simulate_mc(species_list, reactions, params=None, t_span=(0, 20), n_points=200, method='RK45', mc_runs=1, init_scale=0.1, **kwargs):
    """

    Parameters
    ----------
    species_list :
        
    reactions :
        
    params :
         (Default value = None)
    t_span :
         (Default value = (0)
    20) :
        
    n_points :
         (Default value = 200)
    method :
         (Default value = 'RK45')
    mc_runs :
         (Default value = 1)
    init_scale :
         (Default value = 0.1)
    **kwargs :
        

    Returns
    -------

    """
    
    runs = []
    for i in range(mc_runs):
        t, y = simulate(species_list, reactions, params=params, t_span=t_span, n_points=n_points, method=method, **kwargs)
        runs.append((t, y))
    
    t_ref = runs[0][0]
    Ys = np.stack([np.interp(t_ref, r[0], r[1]) if (len(r[0]) != len(t_ref) or not np.allclose(r[0], t_ref)) else r[1] for r in runs], axis=2)
    y_mean = np.mean(Ys, axis=2)
    return t_ref, y_mean, runs
#
def normalize_rows(y):
    """

    Parameters
    ----------
    y :
        

    Returns
    -------

    """
   
    mx = np.max(y, axis=1, keepdims=True)
    mx[mx == 0] = 1.0
    return y / mx


def compute_similarity(y1, y2):
    """

    Parameters
    ----------
    y1 :
        
    y2 :
        

    Returns
    -------

    """
    
    # align number of species
    n = min(y1.shape[0], y2.shape[0])
    y1n = normalize_rows(y1[:n])
    y2n = normalize_rows(y2[:n])

    dtw_scores, mse_scores, corr_scores = [], [], []
    for i in range(n):
        
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
#

def save_plot(fig, path):
    """

    Parameters
    ----------
    fig :
        
    path :
        

    Returns
    -------

    """
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def build_network_graph(species_list, reaction_info, species_map=None):
    """Build directed network from reaction_info.
    Nodes = species, Edges = reactions.

    Parameters
    ----------
    species_list :
        
    reaction_info :
        
    species_map :
         (Default value = None)

    Returns
    -------

    """
    G = nx.DiGraph()
    # Add all species as nodes
    for sid in species_list:
        G.add_node(sid)
    
    
    for rx in reaction_info:
        for s_idx in rx['substrate_idxs']:
            for p_idx in rx['product_idxs']:
                s = species_list[s_idx]
                p = species_list[p_idx]
                G.add_edge(s, p, reaction_id=rx['id'])
    
    return G


def animate_network_polished_v3(G, species_list, reaction_info, t, y, pos, save_path,
                                duration_sec=10, slow_factor=0.25):
    """

    Parameters
    ----------
    G :
        
    species_list :
        
    reaction_info :
        
    t :
        
    y :
        
    pos :
        
    save_path :
        
    duration_sec :
         (Default value = 10)
    slow_factor :
         (Default value = 0.25)

    Returns
    -------

    """
    
    fig, ax = plt.subplots(figsize=(16, 12))  

    
    palette = sns.color_palette("rocket", n_colors=len(species_list))
    node_color_map = {sid: palette[i] for i, sid in enumerate(species_list)}

    
    base_node_size = 400

    
    total_frames = len(t)
    fps = 30
    skip = max(1, total_frames // (duration_sec * fps))
    frame_indices = np.arange(0, total_frames, skip)
    interval = 1000 / fps / slow_factor

    def update(frame_idx):
        """

        Parameters
        ----------
        frame_idx :
            

        Returns
        -------

        """
        ax.clear()
        frame = frame_indices[frame_idx]

        
        node_sizes = base_node_size + 5000 * y[:, frame]
        node_colors = [node_color_map[sid] for sid in species_list]

        
        edge_widths = []
        for u, v, d in G.edges(data=True):
            rx_id = d['reaction_id']
            rx_idx = next((i for i, r in enumerate(reaction_info) if r['id'] == rx_id), None)
            if rx_idx is not None:
                info = reaction_info[rx_idx]
                rate = 1.0
                if info['substrate_idxs']:
                    rate = info['k']
                    for idx, sto in zip(info['substrate_idxs'], info.get('sto_subs', [1]*len(info['substrate_idxs']))):
                        rate *= max(y[idx, frame], 0.0) ** sto
                edge_widths.append(1 + 5 * rate)  
            else:
                edge_widths.append(1.0)

        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                               node_color=node_colors, ax=ax)
        nx.draw_networkx_edges(G, pos, width=4, edge_color='Black', ax=ax)

        
        ax.set_title(f'Metabolic Network Dynamics  = {t[frame]:.2f}',
                     fontsize=18)
        ax.axis('off')

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=interval)
    anim.save(save_path, writer='ffmpeg', dpi=200)
    plt.close(fig)


def plot_trajectories(t, y, species, title, save_path):
    """

    Parameters
    ----------
    t :
        
    y :
        
    species :
        
    title :
        
    save_path :
        

    Returns
    -------

    """
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
    """

    Parameters
    ----------
    t1 :
        
    y1 :
        
    t2 :
        
    y2 :
        
    species :
        
    save_path :
        

    Returns
    -------

    """
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
    """

    Parameters
    ----------
    y :
        
    species :
        
    title :
        
    save_path :
        

    Returns
    -------

    """
    fig = plt.figure(figsize=(12, 8))
    # ensure y is species x time
    sns.heatmap(y, cmap='coolwarm', xticklabels=False, yticklabels=species, cbar_kws={'label': 'Concentration'})
    plt.xlabel('Time index')
    plt.ylabel('Species')
    plt.title(title)
    save_plot(fig, save_path)


def plot_bar(scores, species, metric, save_path):
    """

    Parameters
    ----------
    scores :
        
    species :
        
    metric :
        
    save_path :
        

    Returns
    -------

    """
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=species[:len(scores)], y=scores, palette='crest')
    plt.xticks(rotation=90)
    plt.ylabel(metric)
    plt.title(f'Per-species dynamic similarity ({metric})')
    save_plot(fig, save_path)



def main():
    """ """
    parser = argparse.ArgumentParser(description='Improved dynamic KGML network comparison')
    parser.add_argument('file1', default=None)
    parser.add_argument('file2', default=None)
    parser.add_argument('--volume', type=float, default=1e-15,
                    help='System volume in litres (used for Gillespie conversion)')
    parser.add_argument('--animate', default=None, help='Do you want animated video')
    parser.add_argument('--params', default=None, help='JSON/CSV with reaction parameters')
    parser.add_argument('--outdir', default='./outputs')
    parser.add_argument('--tmax', type=float, default=20.0)
    parser.add_argument('--npoints', type=int, default=200)
    parser.add_argument('--method', default='RK45')
    parser.add_argument('--dynamic', choices=['ode', 'gillespie'], default='ode',
                        help='Choose dynamic model: ode or gillespie')
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
            parsed1 = parse_auto(args.file1)
            parsed2 = parse_auto(args.file2)
            if len(parsed1) == 3:
               s1, r1, map1 = parsed1
            else:
               s1, r1 = parsed1
               map1 = {sid: sid for sid in s1}
            if len(parsed2) == 3:
               s2, r2, map2 = parsed2
            else:
               s2, r2 = parsed2
               map2 = {sid: sid for sid in s2}
    except Exception as e:
            logging.error('Failed to parse files: %s', e)
            sys.exit(1)


    
    common = sorted(list(set(s1) & set(s2)))
    if len(common) == 0:
        logging.warning('No species in common by KEGG id; attempting to match by name')
        
        names1 = {v: k for k, v in map1.items()}
        names2 = {v: k for k, v in map2.items()}
        common_names = set(names1.keys()) & set(names2.keys())
        common = [names1[n] for n in common_names]

    if len(common) == 0:
        logging.error('No common species found between the two networks. Aborting comparison.')
        sys.exit(1)

   
    logging.info(f"Running dynamic mode: {args.dynamic.upper()}")
    if args.dynamic == 'ode':
        t1, y1_mean, runs1 = simulate_mc(s1, r1, params=params, t_span=(0, args.tmax),
                                     n_points=args.npoints, method=args.method,
                                     mc_runs=args.mc, steady_threshold=args.steady_threshold,
                                     atol=args.atol, rtol=args.rtol)
        t2, y2_mean, runs2 = simulate_mc(s2, r2, params=params, t_span=(0, args.tmax),
                                     n_points=args.npoints, method=args.method,
                                     mc_runs=args.mc, steady_threshold=args.steady_threshold,
                                     atol=args.atol, rtol=args.rtol)
    elif args.dynamic == 'gillespie':
        odes1, idx1, rinfo1 = build_ode_system(s1, r1, params)
        odes2, idx2, rinfo2 = build_ode_system(s2, r2, params)
        V1 = estimate_volume_for_target_events(s1, rinfo1, params=params, x0_conc=np.ones(len(s1)), t_span=(0, args.tmax), target_events=2000)
        V2 = estimate_volume_for_target_events(s2, rinfo2, params=params, x0_conc=np.ones(len(s2)), t_span=(0, args.tmax), target_events=2000)
        V_shared = float((V1 + V2) / 2.0)
        logging.info("Estimated volumes V1=%.4e L V2=%.4e L using target_events=2000; using shared V=%.4e L", V1, V2, V_shared)
        t1, y1_mean, _, _ = simulate_gillespie_mc_auto(s1, r1, params=params, t_span=(0, args.tmax), n_points=args.npoints, mc_runs=max(1, args.mc), volume_l=V_shared, x0_conc=np.ones(len(s1)))
        t2, y2_mean, _, _ = simulate_gillespie_mc_auto(s2, r2, params=params, t_span=(0, args.tmax), n_points=args.npoints, mc_runs=max(1, args.mc), volume_l=V_shared, x0_conc=np.ones(len(s2)))

    else:
        logging.error(f"Unknown dynamic mode: {args.dynamic}")
        sys.exit(1)




    
    idx1 = [s1.index(cid) for cid in common]
    idx2 = [s2.index(cid) for cid in common]
    y1_common = y1_mean[idx1]
    y2_common = y2_mean[idx2]
    species_names = [map1[cid] for cid in common]

   
   
    _, _, reaction_info1 = build_ode_system(s1, r1, params=params)
    G1 = build_network_graph(s1, reaction_info1, map1)
    pos1 = nx.spring_layout(G1, seed=42) 
    if args.animate == "yes":
        animate_network_polished_v3(G1, s1, reaction_info1, t1, y1_mean, pos1,
                            save_path=os.path.join(run_dir, 'network1.mp4'))

    _, _, reaction_info2 = build_ode_system(s2, r2, params=params)
    G2 = build_network_graph(s2, reaction_info2, map2)
    pos2 = nx.spring_layout(G2, seed=42)  
    if args.animate == "yes":
        animate_network_polished_v3(G2, s2, reaction_info2, t2, y2_mean, pos2,
                            save_path=os.path.join(run_dir, 'network2.mp4'))



    plot_trajectories(t1, y1_common, species_names, f'Dynamics {os.path.basename(args.file1)}', os.path.join(run_dir, 'dynamics_net1.png'))
    plot_trajectories(t2, y2_common, species_names, f'Dynamics {os.path.basename(args.file2)}', os.path.join(run_dir, 'dynamics_net2.png'))
    plot_overlay(t1, y1_common, t2, y2_common, species_names, os.path.join(run_dir, 'overlay.png'))
    plot_heatmap(y1_common, species_names, 'Heatmap Net1', os.path.join(run_dir, 'heatmap_net1.png'))
    plot_heatmap(y2_common, species_names, 'Heatmap Net2', os.path.join(run_dir, 'heatmap_net2.png'))
    
    results = compute_similarity(y1_common, y2_common)

    plot_bar(results['dtw'], species_names, 'DTW', os.path.join(run_dir, 'similarity_dtw.png'))
    plot_bar(results['mse'], species_names, 'MSE', os.path.join(run_dir, 'similarity_mse.png'))
    plot_bar(results['corr'], species_names, 'Correlation', os.path.join(run_dir, 'similarity_corr.png'))

    
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
