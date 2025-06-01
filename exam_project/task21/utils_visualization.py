import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from aggregative_tracking import compute_r_0, compute_agents_barycenter
matplotlib.use('TkAgg')

def visualize_graph(G):
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True)
    plt.show()
    
def plot_aggregative_tracking_results(cost, norm_grad_cost):
    max_iters = cost.shape[0]
    fig, axes = plt.subplots(figsize=(8, 6), nrows=1, ncols=2)
    
    ax1 = axes[0]
    ax1.semilogy(np.arange(max_iters-1), cost[:-1], color='cornflowerblue')
    ax1.set_title('Total Cost')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    
    ax2 = axes[1]
    ax2.semilogy(np.arange(max_iters-1), norm_grad_cost[:-1], color='indianred')
    ax2.set_title('Gradient Norm')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel(r'$||∇\ell||$')
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout()
    plt.show()

def visualize_world(agents, intruders, noise_radius, world_size, d):
    if d <= 3 and d > 1:
        fig = plt.figure(figsize=(10, 8))
        if d == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        ax.set_title('World visualization')
        padding = 0.2
        if d == 2:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_ylim(-padding * world_size, world_size * (1 + padding))
            ax.scatter(agents[:, 0], agents[:, 1], c='black', marker='o', s=50, label='Agent')
            ax.scatter(intruders[:, 0], intruders[:, 1], facecolors='none', edgecolors='cyan', marker='s', s=50, label='Intruder')
            sigma = compute_agents_barycenter(agents)
            ax.scatter(sigma[0], sigma[1], facecolors='none', edgecolors='mediumseagreen', marker='h', s=50, label='Sigma')
            target = compute_r_0(intruders, noise_radius, world_size, d)
            intruders_barycenter = compute_agents_barycenter(intruders)
            if not np.array_equal(intruders_barycenter, target):
                ax.scatter(intruders_barycenter[0], intruders_barycenter[1], c='purple', alpha=0.35, marker='x', s=50, label='Intruders\' CoG')
            ax.scatter(target[0], target[1], c='red', marker='x', s=50, label=r'$r_0$')
            ax.grid(True)
            ax.set_aspect('equal')
        elif d == 3:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_ylim(-padding * world_size, world_size * (1 + padding))
            ax.set_zlim(-padding * world_size, world_size * (1 + padding))
            ax.scatter(agents[:, 0], agents[:, 1], agents[:, 2], c='black', marker='o', s=50, label='Agent')
            ax.scatter(intruders[:, 0], intruders[:, 1], intruders[:, 2], facecolors='none', edgecolors='cyan', marker='s', s=50, label='Intruder')
            sigma = compute_agents_barycenter(agents)
            ax.scatter(sigma[0], sigma[1], sigma[2], facecolors='none', edgecolors='mediumseagreen', marker='h', s=50, label='Sigma')
            target = compute_r_0(intruders, noise_radius, world_size, d)
            intruders_barycenter = compute_agents_barycenter(intruders)
            if not np.allclose(intruders_barycenter, target): 
                ax.scatter(intruders_barycenter[0], intruders_barycenter[1], intruders_barycenter[2], 
                           c='purple', alpha=0.35, marker='x', s=50, label='Intruders\' CoG')
            ax.scatter(target[0], target[1], target[2], c='red', marker='x', s=50, label=r'$r_0$')
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()
            x_range = abs(x_limits[1] - x_limits[0])
            y_range = abs(y_limits[1] - y_limits[0])
            z_range = abs(z_limits[1] - z_limits[0])
            max_range = max(x_range, y_range, z_range)
            mid_x = np.mean(x_limits)
            mid_y = np.mean(y_limits)
            mid_z = np.mean(z_limits)            
            ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
            ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
            ax.set_zlim3d([mid_z - max_range/2, mid_z + max_range/2])
            ax.grid(True)            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"Visualization only supports dimensions 1-3. Current dimension: {d}")
        return None

def plot_graph_with_connections(G):
    pos = nx.spring_layout(G)  
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
    # Mostriamo il titolo
    plt.title("Graph")
    plt.axis('off')  
    plt.show()
    
def animate_world_evolution(intruders, z_history, r_0, world_size, d, speed=10):
    T, n_agents, *_ = z_history.shape
    frame_skip = int(speed) + 1
    positions = z_history[::frame_skip]
    sigma_positions = np.array([compute_agents_barycenter(pos) for pos in positions])
    pause_frames = int(3 * 20)
    positions = np.concatenate([positions, np.repeat(positions[-1:], pause_frames, axis=0)])
    sigma_positions = np.concatenate([sigma_positions, np.repeat(sigma_positions[-1:], pause_frames, axis=0)])
    num_steps = len(positions)
    fig = plt.figure(figsize=(10, 8))
    if d == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel('Z')
    else:
        ax = fig.add_subplot(111)
    ax.set_title('Agents to Intruders Animation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    padding = 0.2
    x_min, x_max = 0, world_size
    y_min, y_max = 0, world_size
    z_min, z_max = 0, world_size if d == 3 else 0
    ax.set_xlim(x_min - padding * x_max, x_max + padding * x_max)
    ax.set_ylim(y_min - padding * y_max, y_max + padding * y_max)
    if d == 3:
        ax.set_zlim(z_min - padding * z_max, z_max + padding * z_max)
    if d == 3:
        intruder_plot = ax.scatter(intruders[:, 0], intruders[:, 1], intruders[:, 2], facecolors='none', edgecolors='cyan', marker='s', s=50, label='Intruder')
        ref_point = ax.scatter(r_0[0], r_0[1], r_0[2], c='red', marker='x', s=50, label=r'$r_0$')
        sigma_scatter = ax.scatter(sigma_positions[0, 0], sigma_positions[0, 1], sigma_positions[0, 2],facecolors='none', edgecolors='mediumseagreen', marker='h', s=50, label='Sigma')
        agent_scatter = ax.scatter(positions[0, :, 0], positions[0, :, 1], positions[0, :, 2], c='black', marker='o', s=50, label='Agent')
        path_lines = []
        for i in range(n_agents):
            line = ax.plot([positions[0, i, 0]], [positions[0, i, 1]], [positions[0, i, 2]],'gray', linestyle='--', alpha=0.5)[0]
            path_lines.append(line)
        sigma_path = ax.plot([sigma_positions[0, 0]], [sigma_positions[0, 1]], [sigma_positions[0, 2]],'mediumseagreen', linestyle='--', alpha=0.5)[0]
        def update(frame):
            agent_scatter._offsets3d = (positions[frame, :, 0], positions[frame, :, 1], positions[frame, :, 2])
            sigma_scatter._offsets3d = ([sigma_positions[frame, 0]], [sigma_positions[frame, 1]], [sigma_positions[frame, 2]])
            for i, line in enumerate(path_lines):
                x_data = positions[:frame+1, i, 0]
                y_data = positions[:frame+1, i, 1]
                z_data = positions[:frame+1, i, 2]
                line.set_data(x_data, y_data)
                line.set_3d_properties(z_data)
            x_data = sigma_positions[:frame+1, 0]
            y_data = sigma_positions[:frame+1, 1]
            z_data = sigma_positions[:frame+1, 2]
            sigma_path.set_data(x_data, y_data)
            sigma_path.set_3d_properties(z_data)
            return [agent_scatter, sigma_scatter, sigma_path] + path_lines
    else:
        intruder_plot = ax.scatter(intruders[:, 0], intruders[:, 1], facecolors='none', edgecolors='cyan', marker='s', s=50, label='Intruder')
        ref_point = ax.scatter(r_0[0], r_0[1], c='red', marker='x', s=50, label=r'$r_0$')
        sigma_scatter = ax.scatter(sigma_positions[0, 0], sigma_positions[0, 1],facecolors='none', edgecolors='mediumseagreen', marker='h', s=50, label='Sigma')
        agent_scatter = ax.scatter(positions[0, :, 0], positions[0, :, 1], c='black', marker='o', s=50, label='Agent')
        path_lines = []
        plt.grid()
        for i in range(n_agents):
            line = ax.plot([positions[0, i, 0]], [positions[0, i, 1]],'gray', linestyle='--', alpha=0.5)[0]
            path_lines.append(line)
        sigma_path = ax.plot([sigma_positions[0, 0]], [sigma_positions[0, 1]],'mediumseagreen', linestyle='--', alpha=0.5)[0]
        def update(frame):
            agent_scatter.set_offsets(positions[frame])
            sigma_scatter.set_offsets([sigma_positions[frame]])
            for i, line in enumerate(path_lines):
                x_data = positions[:frame+1, i, 0]
                y_data = positions[:frame+1, i, 1]
                line.set_data(x_data, y_data)
            x_data = sigma_positions[:frame+1, 0]
            y_data = sigma_positions[:frame+1, 1]
            sigma_path.set_data(x_data, y_data)
            return [agent_scatter, sigma_scatter, sigma_path] + path_lines
    ax.legend()
    anim = FuncAnimation(fig, update, frames=num_steps, blit=False, interval=50, repeat=True)
    if d == 3:
        ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()
    return anim