import json
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import copy
import pickle

def build_zeta_model_2d(L, zeta, avg_k, weight_mean, weight_dev, alpha, seed=None):
    if seed is not None:
        np.random.seed(seed)
    N = L * L
    coords = {}
    nodes = []
    for i in range(L):
        for j in range(L):
            label = (i + 1, j + 1)
            coords[label] = (i, j)
            nodes.append(label)
    G = nx.Graph()
    for node_label in nodes:
        G.add_node(node_label, pos=coords[node_label])
    pair_weights = []
    sum_w = 0.0
    for idx, node_i in enumerate(nodes):
        x1, y1 = coords[node_i]
        for node_j in nodes[idx + 1:]:
            x2, y2 = coords[node_j]
            dx = min(abs(x1 - x2), L - abs(x1 - x2))
            dy = min(abs(y1 - y2), L - abs(y1 - y2))
            r_ij = np.sqrt(dx ** 2 + dy ** 2)
            w_ij = np.exp(-r_ij / zeta)
            pair_weights.append((node_i, node_j, w_ij))
            sum_w += w_ij
    A = (N * avg_k / 2.0) / sum_w
    print(f"\nNormalization constant A = {A:.6f}")
    for i, j, w_ij in pair_weights:
        p_ij = A * w_ij
        if np.random.rand() < p_ij:
            G.add_edge(i, j)
    return G, coords

def compute_loads(G):
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return {node: 0.0 for node in G.nodes()}
    try:
        loads = nx.betweenness_centrality(G, weight='weight', normalized=False)
    except:
        loads = {node: 0.0 for node in G.nodes()}
    return loads

def remove_random_links(G, p, seed=None, removed_edges=None):
    if seed is None:
        np.random.seed(seed)
    G_copy = G.copy()
    all_edges = list(G_copy.edges())
    total_edges = len(all_edges)
    if total_edges == 0:
        return G_copy, set()
    num_to_remove = int((1 - p) * total_edges)
    edges_to_remove = set()
    if removed_edges is not None:
        for edge in removed_edges:
            if G_copy.has_edge(*edge):
                edges_to_remove.add(edge)
    remaining_edges = [e for e in all_edges if e not in edges_to_remove and (e[1], e[0]) not in edges_to_remove]
    additional_to_remove = max(0, num_to_remove - len(edges_to_remove))
    if additional_to_remove > 0 and len(remaining_edges) > 0:
        additional_edges = np.random.choice(len(remaining_edges), size=min(additional_to_remove, len(remaining_edges)), replace=False)
        for idx in additional_edges:
            edges_to_remove.add(remaining_edges[idx])
    for edge in edges_to_remove:
        if G_copy.has_edge(*edge):
            G_copy.remove_edge(*edge)
    return G_copy, edges_to_remove

def cascade_step(G, capacities1, current_loads):
    G_new = G.copy()
    failed_nodes = []
    for node in G.nodes():
        if current_loads[node] > capacities1[node]:
            failed_nodes.append(node)
    for node in failed_nodes:
        if G_new.has_node(node):
            G_new.remove_node(node)
    new_loads = compute_loads(G_new)
    return G_new, new_loads, failed_nodes

def cascade_on_graph(G_t1, initial_loads, capacities1, coords):
    loads_t1 = compute_loads(G_t1)
    G_current = G_t1.copy()
    current_loads = loads_t1.copy()
    timestep = 1
    timesteps = [timestep]
    surviving_nodes = [G_current.number_of_nodes()]
    surviving_edges = [G_current.number_of_edges()]
    failed_nodes_per_step = []
    all_failed_nodes = set()
    cascade_states = [(G_current.copy(), current_loads.copy(), timestep)]
    while True:
        timestep += 1
        G_new, new_loads, failed_nodes = cascade_step(G_current, capacities1, current_loads)
        if not failed_nodes:
            break
        timesteps.append(timestep)
        surviving_nodes.append(G_new.number_of_nodes())
        surviving_edges.append(G_new.number_of_edges())
        failed_nodes_per_step.append(failed_nodes)
        all_failed_nodes.update(failed_nodes)
        cascade_states.append((G_new.copy(), new_loads.copy(), timestep))
        G_current, current_loads = G_new, new_loads
        if G_current.number_of_nodes() == 0:
            break
    if G_current.number_of_nodes() > 0:
        comps = list(nx.connected_components(G_current))
        giant_size = max(len(c) for c in comps)
    else:
        giant_size = 0
    return {
        'timesteps': timesteps,
        'surviving_nodes': surviving_nodes,
        'surviving_edges': surviving_edges,
        'failed_nodes_per_step': failed_nodes_per_step,
        'final_giant_component_size': giant_size,
        'final_graph': G_current.copy(),
        'cascade_states': cascade_states
    }

def plot_network_state(G, coords, loads, capacities1, title, filename, failed_nodes=None):
    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"{title}\n(Network Collapsed - No Nodes Remaining)")
        if coords:
            ax.set_xlim(-0.5, max(pos[0] for pos in coords.values()) + 0.5)
            ax.set_ylim(-0.5, max(pos[1] for pos in coords.values()) + 0.5)
        ax.axis('off')
        plt.tight_layout()
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = {node: coords[node] for node in G.nodes() if node in coords}
    current_loads = {node: loads.get(node, 0) for node in G.nodes()}
    failed_color = 'red'
    overloaded_color = 'orange'
    normal_low_color = 'lightblue'
    normal_high_color = 'darkblue'
    node_colors = []
    node_sizes = []
    load_vals = [current_loads[node] for node in G.nodes()]
    if load_vals:
        min_load = min(load_vals)
        max_load = max(load_vals)
        load_range = max_load - min_load
    else:
        min_load = max_load = load_range = 0
    for node in G.nodes():
        is_overloaded = current_loads[node] > capacities1.get(node, float('inf'))
        if failed_nodes and node in failed_nodes:
            node_colors.append(failed_color)
            node_sizes.append(100)
        elif is_overloaded:
            node_colors.append(overloaded_color)
            node_sizes.append(80)
        else:
            if load_range > 0:
                load_ratio = (current_loads[node] - min_load) / load_range
                node_colors.append(normal_high_color if load_ratio > 0.5 else normal_low_color)
            else:
                node_colors.append(normal_low_color)
            degree = G.degree(node) if G.has_node(node) else 1
            node_sizes.append((degree + 1) * 15)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights:
        min_w, max_w = min(edge_weights), max(edge_weights)
        if max_w > min_w:
            edge_widths = [1 + 2 * (w - min_w) / (max_w - min_w) for w in edge_weights]
        else:
            edge_widths = [1.5 for _ in edge_weights]
    else:
        edge_widths = []
    nx.draw(G, pos=pos, ax=ax, node_size=node_sizes, node_color=node_colors,
            with_labels=True, font_size=3, edge_color='gray',
            width=edge_widths, alpha=0.8)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=normal_high_color, markersize=10, label='High Load'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=normal_low_color, markersize=10, label='Low Load'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=overloaded_color, markersize=10, label='Overloaded'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=failed_color, markersize=10, label='Failed')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

def plot_cascade_results_theta_grouped(results, L, coords=None, theta_values=None):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    color_map = cm.get_cmap('tab10')
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    if theta_values is None:
        theta_values = sorted(results.keys())

    # Figure 1: Node Survival & Edge Survival
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, theta in enumerate(theta_values):
        color = color_map(idx % 10)
        ls = line_styles[idx % len(line_styles)]
        p_vals = sorted(results[theta].keys())
        for p in p_vals:
            data = results[theta][p]
            if data['timesteps']:
                ax1.plot(data['timesteps'], data['surviving_nodes'], marker='o', color=color, linestyle=ls, label=f'�={theta}, p={p:.3f}')
                ax2.plot(data['timesteps'], data['surviving_edges'], marker='s', color=color, linestyle=ls, label=f'�={theta}, p={p:.3f}')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Surviving nodes')
    ax1.set_title('Node Survival During Cascade')
    ax1.legend(fontsize=4, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Surviving edges')
    ax2.set_title('Edge Survival During Cascade')
    ax2.legend(fontsize=4, loc='upper right')
    ax2.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.suptitle(f"Cascade Simulation Results L{L}:.2fz{zeta}:.2fk{avg_k}:.2f:: Node & Edge Survival", fontsize=14, fontweight='bold', y=1.02)

    # Figure 2: Giant Component vs p & Cascade Duration vs p
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, theta in enumerate(theta_values):
        color = color_map(idx % 10)
        ls = line_styles[idx % len(line_styles)]
        p_vals = sorted(results[theta].keys())
        giant_sizes = [results[theta][p]['final_giant_component_size'] for p in p_vals]
        ax3.plot(p_vals, giant_sizes, marker='o', color=color, linestyle=ls, label=f'�={theta}')
        cascade_durations = [len(results[theta][p]['timesteps']) - 1 for p in p_vals]
        ax4.plot(p_vals, cascade_durations, marker='s', color=color, linestyle=ls, label=f'�={theta}')
    ax3.set_xlabel('p (link survival probability)')
    ax3.set_ylabel('Final giant component size')
    ax3.set_title('Percolation: Giant Component vs p')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax4.set_xlabel('p (link survival probability)')
    ax4.set_ylabel('Cascade duration (time steps)')
    ax4.set_title('Cascade Duration vs p')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.suptitle(f"Cascade Simulation Results L{L}:.2fz{zeta}:.2fk{avg_k}:.2f:: Giant Component & Duration", fontsize=14, fontweight='bold', y=1.02)

    # Figure 3: Evolution of Giant Component Size & Failures per Timestep
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, theta in enumerate(theta_values):
        color = color_map(idx % 10)
        ls = line_styles[idx % len(line_styles)]
        p_vals = sorted(results[theta].keys())
        for p in p_vals:
            data = results[theta][p]
            giant_sizes_t = []
            for state in data.get('cascade_states', []):
                G = state[0]
                if G.number_of_nodes() > 0:
                    comps = list(nx.connected_components(G))
                    giant_sizes_t.append(max(len(comp) for comp in comps))
                else:
                    giant_sizes_t.append(0)
            if giant_sizes_t:
                ax5.plot(range(len(giant_sizes_t)), giant_sizes_t, marker='o', color=color, linestyle=ls, label=f'�={theta}, p={p:.3f}')
            failures_per_step = [len(failed) for failed in data['failed_nodes_per_step']]
            if failures_per_step:
                ax6.plot(range(1, len(failures_per_step)+1), failures_per_step, marker='s', color=color, linestyle=ls, label=f'�={theta}, p={p:.3f}')
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Giant Component Size')
    ax5.set_title('Evolution of Giant Component Size')
    ax5.legend(fontsize=4, loc='upper right')
    ax5.grid(True)
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Number of Failures')
    ax6.set_title('Instant Failures per Timestep')
    ax6.legend(fontsize=4, loc='upper right')
    ax6.grid(True)
    fig3.tight_layout()
    fig3.suptitle(f"Cascade Simulation Results L{L}:.2fz{zeta}:.2fk{avg_k}:.2f: Giant Component Evolution & Failures", fontsize=14, fontweight='bold', y=1.02)
    plt.show()


def plot_cascade_results_thta_grouped(results, L, coords=None, theta_values=None):
    import matplotlib.pyplot as plt

    # VIBGYOR colors
    vibgyor = ['violet', 'indigo', 'blue', 'green', 'yellow', 'orange', 'red']
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    if theta_values is None:
        theta_values = sorted(results.keys())

    # Figure 1: Node Survival & Edge Survival
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, theta in enumerate(theta_values):
        color = vibgyor[idx % len(vibgyor)]
        ls = line_styles[idx % len(line_styles)]
        for p in sorted(results[theta]):
            data = results[theta][p]
            if data['timesteps']:
                ax1.plot(data['timesteps'],
                         data['surviving_nodes'],
                         marker='o', color=color, linestyle=ls)
                ax2.plot(data['timesteps'],
                         data['surviving_edges'],
                         marker='s', color=color, linestyle=ls)
    ax1.set(xlabel='Time step', ylabel='Surviving nodes',
            title='Node Survival During Cascade')
    ax1.grid(True, alpha=0.3)
    ax2.set(xlabel='Time step', ylabel='Surviving edges',
            title='Edge Survival During Cascade')
    ax2.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.suptitle(f"Cascade Results L={L}", fontsize=14, fontweight='bold', y=2.22)

    # Figure 2: Giant Component vs p & Cascade Duration vs p
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, theta in enumerate(theta_values):
        color = vibgyor[idx % len(vibgyor)]
        ls = line_styles[idx % len(line_styles)]
        p_vals = sorted(results[theta])
        giant_sizes = [results[theta][p]['final_giant_component_size'] for p in p_vals]
        durations  = [len(results[theta][p]['timesteps']) - 1       for p in p_vals]
        ax3.plot(p_vals, giant_sizes, marker='o', color=color, linestyle=ls)
        ax4.plot(p_vals, durations,   marker='s', color=color, linestyle=ls)
    ax3.set(xlabel='p', ylabel='Final giant component size',
            title='Giant Component vs p')
    ax3.grid(True, alpha=0.3)
    ax4.set(xlabel='p', ylabel='Cascade duration (steps)',
            title='Cascade Duration vs p')
    ax4.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.suptitle(f"Cascade Results L={L}", fontsize=14, fontweight='bold', y=2.22)

    # Figure 3: Evolution & Failures per Timestep
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, theta in enumerate(theta_values):
        color = vibgyor[idx % len(vibgyor)]
        ls = line_styles[idx % len(line_styles)]
        for p in sorted(results[theta]):
            data = results[theta][p]
            # evolution of giant component
            times = range(len(data['cascade_states']))
            sizes = [
                len(max(nx.connected_components(G_snap), key=len))
                if G_snap.number_of_nodes() else 0
                for (G_snap, _, _) in data['cascade_states']
            ]
            failures = [len(f) for f in data['failed_nodes_per_step']]
            ax5.plot(times,   sizes,    marker='o', color=color, linestyle=ls)
            ax6.plot(range(1, len(failures)+1), failures,
                     marker='s', color=color, linestyle=ls)
    ax5.set(xlabel='Timestep', ylabel='Giant component size',
            title='Evolution of Giant Component')
    ax5.grid(True, alpha=0.3)
    ax6.set(xlabel='Timestep', ylabel='Number of failures',
            title='Instant Failures per Timestep')
    ax6.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.suptitle(f"Cascade Results L={L}zeta{zeta}", fontsize=14, fontweight='bold', y=2.22)

    plt.show()

def plot_spatiotemporal_failure(results, coords, L, p=None):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    p_vals = sorted(results.keys())
    if p is None:
        p = p_vals[0]
    data = results[p]
    node_failure_time = {}
    for t, failed in enumerate(data['failed_nodes_per_step'], start=1):
        for node in failed:
            node_failure_time[node] = t
    for node in data['final_graph'].nodes():
        if node not in node_failure_time:
            node_failure_time[node] = 0
    timesteps = list(node_failure_time.values())
    min_timestep = min(timesteps)
    max_timestep = max(timesteps)
    cmap = cm.get_cmap('rainbow')
    norm = mcolors.Normalize(vmin=min_timestep, vmax=max_timestep)
    node_colors = [cmap(norm(node_failure_time[node])) for node in data['final_graph'].nodes()]
    plt.figure(figsize=(7, 7))
    pos = {node: coords[node] for node in data['final_graph'].nodes() if node in coords}
    xs = [pos[node][0] for node in data['final_graph'].nodes()]
    ys = [pos[node][1] for node in data['final_graph'].nodes()]
    plt.scatter(xs, ys, c=node_colors, s=30, marker='o', edgecolors='none', zorder=2)
    plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), L, L, fill=False, edgecolor='black', linewidth=2, zorder=1))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Failure Timestep')
    cbar.ax.set_ylabel('Failure Timestep', rotation=270, labelpad=15)
    plt.title(f'Spatiotemporal Distribution of Failures (p={p:.2f})')
    plt.xlim(-0.5, L-0.5)
    plt.ylim(-0.5, L-0.5)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_cascade_results_theta(results, L, coords=None, theta_values=None):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    color_map = cm.get_cmap('tab10')
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    if theta_values is None:
        theta_values = sorted(results.keys())
    for idx, theta in enumerate(theta_values):
        color = color_map(idx % 10)
        ls = line_styles[idx % len(line_styles)]
        p_vals = sorted(results[theta].keys())
        # 1. Node survival
        for p in p_vals:
            data = results[theta][p]
            if data['timesteps']:
                ax1.plot(data['timesteps'], data['surviving_nodes'], marker='o', color=color, linestyle=ls, label=f'�={theta}, p={p:.3f}')
        # 2. Edge survival
        for p in p_vals:
            data = results[theta][p]
            if data['timesteps']:
                ax2.plot(data['timesteps'], data['surviving_edges'], marker='s', color=color, linestyle=ls, label=f'�={theta}, p={p:.3f}')
        # 3. Giant component vs p
        giant_sizes = [results[theta][p]['final_giant_component_size'] for p in p_vals]
        ax3.plot(p_vals, giant_sizes, marker='o', color=color, linestyle=ls, label=f'�={theta}')
        # 4. Cascade duration vs p
        cascade_durations = [len(results[theta][p]['timesteps']) - 1 for p in p_vals]
        ax4.plot(p_vals, cascade_durations, marker='s', color=color, linestyle=ls, label=f'�={theta}')
        # 5. Evolution of giant component size
        for p in p_vals:
            data = results[theta][p]
            giant_sizes_t = []
            for state in data.get('cascade_states', []):
                G = state[0]
                if G.number_of_nodes() > 0:
                    comps = list(nx.connected_components(G))
                    giant_sizes_t.append(max(len(comp) for comp in comps))
                else:
                    giant_sizes_t.append(0)
            if giant_sizes_t:
                ax5.plot(range(len(giant_sizes_t)), giant_sizes_t, marker='o', color=color, linestyle=ls, label=f'�={theta}, p={p:.3f}')
        # 6. Failures per step
        for p in p_vals:
            data = results[theta][p]
            failures_per_step = [len(failed) for failed in data['failed_nodes_per_step']]
            if failures_per_step:
                ax6.plot(range(1, len(failures_per_step)+1), failures_per_step, marker='s', color=color, linestyle=ls, label=f'�={theta}, p={p:.3f}')
    # Set labels and legends with optimal locations and minimal font size
    ax1.set_xlabel('Time step', fontsize=8)
    ax1.set_ylabel('Surviving nodes', fontsize=8)
    ax1.set_title('Node Survival During Cascade', pad=10, fontsize=10)
    # ax1.legend(fontsize=4, loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Time step', fontsize=8)
    ax2.set_ylabel('Surviving edges', fontsize=8)
    ax2.set_title('Edge Survival During Cascade', pad=10, fontsize=10)
    # ax2.legend(fontsize=4, loc='upper right')
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel('p (link survival probability)', fontsize=8)
    ax3.set_ylabel('Final giant component size', fontsize=8)
    ax3.set_title('Percolation: Giant Component vs p', pad=10, fontsize=10)
    ax3.legend(fontsize=4, loc='best')
    ax3.grid(True, alpha=0.3)

    ax4.set_xlabel('p (link survival probability)', fontsize=8)
    ax4.set_ylabel('Cascade duration (time steps)', fontsize=8)
    ax4.set_title('Cascade Duration vs p', pad=10, fontsize=10)
    ax4.legend(fontsize=4, loc='best')
    ax4.grid(True, alpha=0.3)

    ax5.set_xlabel('Timestep', fontsize=8)
    ax5.set_ylabel('Giant Component Size', fontsize=8)
    ax5.set_title('Evolution of Giant Component Size', pad=10, fontsize=10)
    ax5.legend(fontsize=4, loc='upper right')
    ax5.grid(True)

    ax6.set_xlabel('Timestep', fontsize=8)
    ax6.set_ylabel('Number of Failures', fontsize=8)
    ax6.set_title('Instant Failures per Timestep', pad=10, fontsize=10)
    # ax6.legend(fontsize=4, loc='upper right')
    ax6.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0)
    plt.subplots_adjust(top=0.93)
    plt.suptitle(f"Cascade Simulation Results L{L}:.2f z{zeta}:.2f k{avg_k}:.2f : (varying � and p)", fontsize=12, fontweight='bold')
    plt.show()

def plot_cascade_results_theta_grouped(results, L, coords=None, theta_values=None):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    color_map = cm.get_cmap('tab10')
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    if theta_values is None:
        theta_values = sorted(results.keys())

    # Figure 1: Node Survival & Edge Survival
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, theta in enumerate(theta_values):
        color = color_map(idx % 10)
        ls = line_styles[idx % len(line_styles)]
        p_vals = sorted(results[theta].keys())
        for p in p_vals:
            data = results[theta][p]
            if data['timesteps']:
                ax1.plot(data['timesteps'], data['surviving_nodes'], marker='o', color=color, linestyle=ls, label=f'�={theta}, p={p:.3f}')
                ax2.plot(data['timesteps'], data['surviving_edges'], marker='s', color=color, linestyle=ls, label=f'�={theta}, p={p:.3f}')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Surviving nodes')
    ax1.set_title('Node Survival During Cascade')
    # ax1.legend(fontsize=4, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Surviving edges')
    ax2.set_title('Edge Survival During Cascade')
    # ax2.legend(fontsize=4, loc='upper right')
    ax2.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.suptitle(f"Cascade Simulation Results L{L}z{zeta}k{avg_k}: Node & Edge Survival", fontsize=14, fontweight='bold')

    # Figure 2: Giant Component vs p & Cascade Duration vs p
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, theta in enumerate(theta_values):
        color = color_map(idx % 10)
        ls = line_styles[idx % len(line_styles)]
        p_vals = sorted(results[theta].keys())
        giant_sizes = [results[theta][p]['final_giant_component_size'] for p in p_vals]
        ax3.plot(p_vals, giant_sizes, marker='o', color=color, linestyle=ls, label=f'�={theta}')
        cascade_durations = [len(results[theta][p]['timesteps']) - 1 for p in p_vals]
        ax4.plot(p_vals, cascade_durations, marker='s', color=color, linestyle=ls, label=f'�={theta}')
    ax3.set_xlabel('p (link survival probability)')
    ax3.set_ylabel('Final giant component size')
    ax3.set_title('Percolation: Giant Component vs p')
    #ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax4.set_xlabel('p (link survival probability)')
    ax4.set_ylabel('Cascade duration (time steps)')
    ax4.set_title('Cascade Duration vs p')
    #ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.suptitle(f"Cascade Simulation Results L{L}z{zeta}k{avg_k} : Giant Component & Duration", fontsize=14, fontweight='bold')

    # Figure 3: Evolution of Giant Component Size & Failures per Timestep
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, theta in enumerate(theta_values):
        color = color_map(idx % 10)
        ls = line_styles[idx % len(line_styles)]
        p_vals = sorted(results[theta].keys())
        for p in p_vals:
            data = results[theta][p]
            giant_sizes_t = []
            for state in data.get('cascade_states', []):
                G = state[0]
                if G.number_of_nodes() > 0:
                    comps = list(nx.connected_components(G))
                    giant_sizes_t.append(max(len(comp) for comp in comps))
                else:
                    giant_sizes_t.append(0)
            if giant_sizes_t:
                ax5.plot(range(len(giant_sizes_t)), giant_sizes_t, marker='o', color=color, linestyle=ls, label=f'�={theta}, p={p:.3f}')
            failures_per_step = [len(failed) for failed in data['failed_nodes_per_step']]
            if failures_per_step:
                ax6.plot(range(1, len(failures_per_step)+1), failures_per_step, marker='s', color=color, linestyle=ls, label=f'�={theta}, p={p:.3f}')
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Giant Component Size')
    ax5.set_title('Evolution of Giant Component Size')
    #ax5.legend(fontsize=4, loc='upper right')
    ax5.grid(True)
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Number of Failures')
    ax6.set_title('Instant Failures per Timestep')
    #ax6.legend(fontsize=4, loc='upper right')
    ax6.grid(True)
    fig3.tight_layout()
    fig3.suptitle(f"Cascade Simulation Results L{L}z{zeta}k{avg_k} : Giant Component Evolution & Failures", fontsize=14, fontweight='bold')
    plt.show()

# if __name__ == "__main__":
#     start = time.time()
#     L = 20
#     avg_k = 4   
#     alpha = 2
#     weight_mean = 1.0
#     weight_dev = 0.3
#     zeta_vals = [3, 15]
#     theta_values = [-1.0, -0.8, -0.4, 0, 1.0, 1.4]
#     # theta_values = [0.8, 1.0, 1.2]
#     all_results = {}
#     for zeta in zeta_vals:
#         print(f"\n=== Simulating � = {zeta} ===")
#         G_base, coords = build_zeta_model_2d(L=L, zeta=zeta, avg_k=avg_k, weight_mean=None, weight_dev=None, alpha=alpha, seed=None)
#         p_values = np.unique(np.round(np.concatenate([np.linspace(0.4, 0.98, 6), np.linspace(0.70, 0.90, 12)]), 4)).tolist()
#         p_values = sorted(p_values, reverse=True)
#         results_for_theta = {}
#         for theta in theta_values:
#             print(f"\n--- � = {theta} ---")
#             # Assign edge weights for this theta
#             G = G_base.copy()
#             for u, v in G.edges():
#                 k_i = G.degree(u)
#                 k_j = G.degree(v)
#                 weight = (k_i * k_j) ** theta
#                 G[u][v]['weight'] = weight
#             initial_loads = nx.betweenness_centrality(G, weight='weight', normalized=False)
#             capacities1 = {i: initial_loads[i] * (1 + alpha) for i in G.nodes()}
#             removed_edges_cumulative = set()
#             results_for_p = {}
#             for p in p_values:
#                 G_t1, new_removed = remove_random_links(G, p, seed=None, removed_edges=removed_edges_cumulative)
#                 removed_edges_cumulative |= new_removed
#                 print(f"  � Cascade for �={theta}, p={p:.4f}")
#                 result_p = cascade_on_graph(G_t1, initial_loads, capacities1, coords)
#                 results_for_p[p] = result_p
#                 dur = len(result_p['timesteps']) - 1
#                 giant = result_p['final_giant_component_size']
#                 print(f"     duration={dur}, giant={giant}")
#             results_for_theta[theta] = results_for_p
#         all_results[zeta] = results_for_theta
#     end1 = time.time()
#     print(f"\nAll simulations done in {(end1-start):.1f}s")
#     for zeta, theta_dict in all_results.items():
#         print(f"\n--- Plotting results for � = {zeta} ---")
#         plot_cascade_results_theta(theta_dict, L, coords, theta_values=theta_values)
#         # plot_spatiotemporal_failure(theta_dict, coords, L, p=None)
#     end = time.time()
#     print(f"\nTotal execution time: {end - start:.2f} seconds.")
#     filename = f"l={L}z={zeta_vals}.txt" 
#     try:
#         with open(filename, 'w') as file:
#             file.write(f"{(end1-start):.2f} seconds. Total time: {end - start:.2f} seconds.") # Corrected text_to_save
#         print(f"Text successfully saved to '{filename}'")
#     except IOError as e:
#         print(f"Error saving file: {e}")



if __name__ == "__main__":
    start = time.time()
    L = 30
    avg_k = 4   
    alpha = 2
    weight_mean = 1.0
    weight_dev = 0.3
    zeta_vals = [3, 15]
    theta_values = [-1.0, -0.8, -0.4, 0, 0.8, 1.0, 1.4]
    # theta_values = [0, 0.4, 1.0, 1.4]
    all_results = {}
    
    # Create results directory if not exists
    results_dir = f"results_L{L}zeta{zeta_vals}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save simulation parameters
    params = {
        'L': L,
        'avg_k': avg_k,
        'alpha': alpha,
        'weight_mean': weight_mean,
        'weight_dev': weight_dev,
        'zeta_vals': zeta_vals,
        'theta_values': theta_values
    }
    with open(f"{results_dir}/params.json", 'w') as f:
        json.dump(params, f, indent=2)

    for zeta in zeta_vals:
        zeta_start = time.time()
        print(f"\n=== Simulating zeta = {zeta} ===")
        G_base, coords = build_zeta_model_2d(L=L, zeta=zeta, avg_k=avg_k, 
                                            weight_mean=None, weight_dev=None, 
                                            alpha=alpha, seed=None)
        # Save base graph and coordinates
        with open(f"{results_dir}/base_graph_zeta{zeta}.pkl", 'wb') as f:
            pickle.dump((G_base, coords), f)
            
        p_values = np.unique(np.round(np.concatenate([
            np.linspace(0.4, 0.98, 6), 
            np.linspace(0.70, 0.90, 12)
        ]), 4)).tolist()
        p_values = sorted(p_values, reverse=True)
        
        # Save p-values
        with open(f"{results_dir}/p_values.json", 'w') as f:
            json.dump(p_values, f)
        
        results_for_theta = {}
        for theta in theta_values:
            print(f"\n--- theta = {theta} ---")
            G = G_base.copy()
            for u, v in G.edges():
                k_i = G.degree(u)
                k_j = G.degree(v)
                weight = (k_i * k_j) ** theta
                G[u][v]['weight'] = weight
                
            initial_loads = nx.betweenness_centrality(G, weight='weight', normalized=False)
            capacities1 = {i: initial_loads[i] * (1 + alpha) for i in G.nodes()}
            removed_edges_cumulative = set()
            results_for_p = {}
            
            for p in p_values:
                print(f"  Running cascade for theta={theta}, p={p:.4f}")
                G_t1, new_removed = remove_random_links(G, p, seed=None, 
                                                       removed_edges=removed_edges_cumulative)
                removed_edges_cumulative |= new_removed
                
                result_p = cascade_on_graph(G_t1, initial_loads, capacities1, coords)
                
                # Simplify results for storage
                simplified = {
                    'timesteps': result_p['timesteps'],
                    'surviving_nodes': result_p['surviving_nodes'],
                    'surviving_edges': result_p['surviving_edges'],
                    'failed_nodes_per_step': [
                        list(nodes) for nodes in result_p['failed_nodes_per_step']
                    ],
                    'final_giant_component_size': result_p['final_giant_component_size'],
                    'giant_sizes_over_time': result_p.get('giant_sizes_over_time', []),
                    'node_failure_time': result_p.get('node_failure_time', {})
                }
                results_for_p[p] = simplified
                
                # Save results incrementally
                with open(f"{results_dir}/results_zeta{zeta}_theta{theta}.json", 'w') as f:
                    json.dump(results_for_p, f, indent=2)
                    
            results_for_theta[theta] = results_for_p
            
        all_results[zeta] = results_for_theta
        zeta_time = time.time() - zeta_start
        print(f"Zeta {zeta} completed in {zeta_time:.2f} seconds")
        
    total_time = time.time() - start
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    # Save final aggregated results
    with open(f"{results_dir}/all_results.json", 'w') as f:
        json.dump(all_results, f, default=lambda x: list(x) if isinstance(x, set) else x, indent=2)
