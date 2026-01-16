import json
import pickle
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def load_results(results_dir):
    """Load saved simulation results from directory"""
    try:
        # Load parameters
        with open(f"{results_dir}/params.json") as f:
            params = json.load(f)
        
        # Load p-values
        with open(f"{results_dir}/p_values.json") as f:
            p_values = json.load(f)
        
        # Load base graphs and coordinates
        base_graphs = {}
        for zeta in params['zeta_vals']:
            with open(f"{results_dir}/base_graph_zeta{zeta}.pkl", 'rb') as f:
                base_graphs[zeta] = pickle.load(f)
                
        # Load aggregated results
        with open(f"{results_dir}/all_results.json") as f:
            all_results = json.load(f)
            
        return params, p_values, base_graphs, all_results
        
    except FileNotFoundError as e:
        print(f"Error loading results: {e}")
        return None, None, None, None
    

def plot_cascade_results_theta_grouped(results, L, zeta, avg_k, coords=None, theta_values=None):
    color_map = plt.colormaps['tab10']  # Fixed deprecated get_cmap
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    if theta_values is None:
        theta_values = sorted(results.keys())

    # Figure 1: Node Survival & Edge Survival
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, theta in enumerate(theta_values):
        color = color_map(idx % 10)
        ls = line_styles[idx % len(line_styles)]
        p_vals = sorted(results[theta].keys())
        for p_str in p_vals:  # p_str is a string from JSON keys
            p = float(p_str)  # Convert to float for formatting
            data = results[theta][p_str]  # Use string key to access data
            if data['timesteps']:
                ax1.plot(data['timesteps'], data['surviving_nodes'], 
                         marker='o', color=color, linestyle=ls, 
                         label=f'θ={theta}, p={p:.3f}')
                ax2.plot(data['timesteps'], data['surviving_edges'], 
                         marker='s', color=color, linestyle=ls, 
                         label=f'θ={theta}, p={p:.3f}')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Surviving nodes')
    ax1.set_title('Node Survival During Cascade')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Surviving edges')
    ax2.set_title('Edge Survival During Cascade')
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, alpha=0.3)
    
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.suptitle(f"Cascade Simulation Results L={L}, ζ={zeta}, k={avg_k}: Node & Edge Survival", 
                 fontsize=14, fontweight='bold', y=1.02)

    # Figure 2: Giant Component vs p & Cascade Duration vs p
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, theta in enumerate(theta_values):
        color = color_map(idx % 10)
        ls = line_styles[idx % len(line_styles)]
        p_vals = sorted(results[theta].keys())
        
        # Convert p strings to floats for plotting
        p_floats = [float(p) for p in p_vals]
        giant_sizes = [results[theta][p]['final_giant_component_size'] for p in p_vals]
        cascade_durations = [len(results[theta][p]['timesteps']) - 1 for p in p_vals]
        
        ax3.plot(p_floats, giant_sizes, marker='o', 
                color=color, linestyle=ls, label=f'θ={theta}')
        ax4.plot(p_floats, cascade_durations, marker='s', 
                color=color, linestyle=ls, label=f'θ={theta}')
    
    ax3.set_xlabel('p (link survival probability)')
    ax3.set_ylabel('Final giant component size')
    ax3.set_title('Giant Component vs p')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('p (link survival probability)')
    ax4.set_ylabel('Cascade duration (time steps)')
    ax4.set_title('Cascade Duration vs p')
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.suptitle(f"Cascade Simulation Results L={L}, ζ={zeta}, k={avg_k}: Giant Component & Duration", 
                 fontsize=14, fontweight='bold', y=1.02)

    # Figure 3: Evolution of Giant Component Size & Failures per Timestep
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, theta in enumerate(theta_values):
        color = color_map(idx % 10)
        ls = line_styles[idx % len(line_styles)]
        p_vals = sorted(results[theta].keys())
        for p_str in p_vals:
            p = float(p_str)  # Convert to float for formatting
            data = results[theta][p_str]  # Use string key to access data
            
            # Use precomputed giant sizes
            if 'giant_sizes_over_time' in data and data['giant_sizes_over_time']:
                ax5.plot(range(len(data['giant_sizes_over_time'])), 
                         data['giant_sizes_over_time'], 
                         marker='o', color=color, linestyle=ls, 
                         label=f'θ={theta}, p={p:.3f}')
            
            if 'failed_nodes_per_step' in data and data['failed_nodes_per_step']:
                failures_per_step = [len(failed) for failed in data['failed_nodes_per_step']]
                ax6.plot(range(1, len(failures_per_step)+1), 
                         failures_per_step, 
                         marker='s', color=color, linestyle=ls, 
                         label=f'θ={theta}, p={p:.3f}')
    
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Giant Component Size')
    ax5.set_title('Evolution of Giant Component Size')
    ax5.legend(fontsize=12, loc='best')
    ax5.grid(True)
    
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Number of Failures')
    ax6.set_title('Instant Failures per Timestep')
    ax6.legend(fontsize=12, loc='best')
    ax6.grid(True)
    
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.suptitle(f"Cascade Simulation Results L={L}, ζ={zeta}, k={avg_k}: Giant Evolution & Failures", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.show()

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

def plot_spatiotemporal_failure(results, coords, L, p=None):
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

def main():
    # Set the path to your results directory
    results_dir = "/home/ameen/Downloads/ameen/results_L30zeta[3, 15]"
    
    # Load the results
    params, p_values, base_graphs, all_results = load_results(results_dir)
    
    if not all_results:
        print("Failed to load results. Please check the directory path.")
        return
    
    print(f"Loaded results for L={params['L']}")
    
    # Plot results for each zeta value
    for zeta, theta_dict in all_results.items():
        print(f"\n--- Plotting results for ζ = {zeta} ---")
        G_base, coords = base_graphs[int(zeta)]  # Convert key to int
        
        # Convert theta keys to float
        float_theta_dict = {}
        for theta_str, p_dict in theta_dict.items():
            float_theta_dict[float(theta_str)] = p_dict
        
        # Plot the cascade results
        plot_cascade_results_theta_grouped(
            results=float_theta_dict,
            L=params['L'],
            zeta=zeta,
            avg_k=params['avg_k'],
            coords=coords,
            theta_values=params['theta_values']
        )
        
        # You can add other plotting functions here if needed
        # For example, to plot spatiotemporal failure for a specific p value:
        # plot_spatiotemporal_failure(float_theta_dict[0.8], coords, params['L'], p=0.8)

if __name__ == "__main__":
    main()