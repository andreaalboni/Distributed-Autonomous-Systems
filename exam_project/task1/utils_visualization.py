import os
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')
    
def visualize_graph(G):
    """Visualizes the given NetworkX graph"""
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True)
    plt.show()
    
def plot_gradient_tracking_results(z, cost, norm_grad_cost, agents, targets, norm_error, task, save=False):
    """Plot the cost, gradient norm, and per-target error norms for gradient tracking results."""
    max_iters = len(cost)
    fig, axes = plt.subplots(figsize=(8, 6), nrows=1, ncols=2)
    
    ax = axes[0]
    ax.semilogy(np.arange(max_iters-1), cost[:-1], color='magenta')       
    ax.set_title('Total Cost')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    
    ax = axes[1]
    ax.semilogy(np.arange(max_iters-1), norm_grad_cost[:-1], color='cyan')
    ax.set_title('Gradient Norm')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$||∇\ell||$')
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout()
    plt.show()
    if save:
        task = task.replace('.', '')
        os.makedirs('./plots', exist_ok=True)
        fig.savefig(f'./plots/gradient_tracking_results_task{task}.png', format='png')
        print(f"Plots saved as 'gradient_tracking_results_task{task}.png")
    
    if task == '1.2':
        fig, axes = plt.subplots(figsize=(14, 6), nrows=1, ncols=len(targets))
        fig.suptitle('Norm of the error for each target', fontsize=14)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(agents)))
        for j in range(len(targets)):
            ax = axes[j]
            for i in range(len(agents)):
                line = ax.semilogy(np.arange(max_iters-1), norm_error[:-1, i, j], 
                        color=colors[i],
                        label=f'Agent {i}')
            ax.set_title(f'Target {j}')
            ax.set_xlabel('Iteration')
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        plt.tight_layout()
        plt.show()

def visualize_world(agents, targets, world_size, d):
    """Visualizes the positions of agents and targets in a 1D, 2D, or 3D world."""
    if d <= 3 and d > 0:
        agents = agents * world_size
        targets = targets * world_size
        fig = plt.figure(figsize=(8, 8))
        if d == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        ax.set_title('World visualization')
        padding = 0.2
        if d == 1:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_yticks([])
            ax.scatter(agents, np.zeros_like(agents), c='blue', marker='o', s=50, label='Agent')
            ax.scatter(targets, np.zeros_like(targets), c='red', marker='x', s=50, label='Target')
            ax.grid(False)
            ax.set_aspect('equal')
        elif d == 2:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_ylim(-padding * world_size, world_size * (1 + padding))
            ax.scatter(agents[:, 0], agents[:, 1], c='blue', marker='o', s=50, label='Agent')
            ax.scatter(targets[:, 0], targets[:, 1], c='red', marker='x', s=50, label='Target')
            ax.grid(True)
            ax.set_aspect('equal')
        elif d == 3:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_ylim(-padding * world_size, world_size * (1 + padding))
            ax.set_zlim(-padding * world_size, world_size * (1 + padding))
            ax.scatter(agents[:, 0], agents[:, 1], agents[:, 2], c='blue', marker='o', s=50, label='Agent')
            ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c='red', marker='x', s=50, label='Target')
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
        plt.show()
    else:
        print(f"Visualization only supports dimensions 1-3. Current dimension: {d}")
        return None

def animate_world_evolution(agents, targets, z_history, type, world_size, d, task, speed=100, save=False):
    """Animates the evolution of agent-target positions in 1D, 2D, or 3D worlds."""
    if d > 0 and d <= 3:
        agents = agents * world_size
        targets = targets * world_size

        z_history = z_history * world_size
        T, n_agents, n_targets, _ = z_history.shape
        num_frames = speed
        indices = np.unique(np.geomspace(1, T-1, num=num_frames).astype(int))
        positions = z_history[indices]
        pause_frames = int(3 * 20)
        positions = np.concatenate([positions, np.repeat(positions[-1:], pause_frames, axis=0)])
        num_steps = len(positions)
        fig = plt.figure(figsize=(8, 8))
        if d == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        ax.set_title(f'Agents to Targets Animation - {type} Graph')
        padding = 0.2
        if d == 1:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_yticks([])
            ax.scatter(agents, np.zeros_like(agents), c='blue', marker='o', s=50, label='Agent')
            ax.scatter(targets, np.zeros_like(targets), c='red', marker='x', s=50, label='Target')
            ax.grid(False)
        elif d == 2:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_ylim(-padding * world_size, world_size * (1 + padding))
            ax.scatter(agents[:, 0], agents[:, 1], c='blue', marker='o', s=50, label='Agent')
            ax.scatter(targets[:, 0], targets[:, 1], c='red', marker='x', s=50, label='Target')
            ax.grid(True)
        elif d == 3:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_ylim(-padding * world_size, world_size * (1 + padding))
            ax.set_zlim(-padding * world_size, world_size * (1 + padding))
            ax.scatter(agents[:, 0], agents[:, 1], agents[:, 2], c='blue', marker='o', s=50, label='Agent')
            ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c='red', marker='x', s=50, label='Target')
            ax.grid(True)
            ax.scatter([], [], [], c='green', s=20, label='Estimation')
        ax.legend()
        if d != 3:
            ax.set_aspect('equal')
        if d == 3:
            estimation_points = None
            def update(frame):
                nonlocal estimation_points
                if estimation_points is not None:
                    for point in estimation_points:
                        point.remove()
                
                pos = positions[frame]
                x_data = []
                y_data = []
                z_data = []
                for i in range(n_agents):
                    for j in range(n_targets):
                        x_data.append(pos[i, j, 0])
                        y_data.append(pos[i, j, 1])
                        z_data.append(pos[i, j, 2])
                
                estimation_points = [ax.scatter(x_data, y_data, z_data, c='green', s=20)]
                return estimation_points
            anim = FuncAnimation(
                fig, update,
                frames=num_steps,
                interval=50,
                repeat=True
            )
        else:
            scatters = []
            for _ in range(n_agents * n_targets):
                if d == 1:
                    scatter = ax.scatter([], [], c='green', s=20)
                elif d == 2:
                    scatter = ax.scatter([], [], c='green', s=20)
                scatters.append(scatter)
            def init():
                for scatter in scatters:
                    scatter.set_offsets(np.empty((0, 2)))
                return scatters
            def update(frame):
                pos = positions[frame]
                index = 0
                for i in range(n_agents):
                    for j in range(n_targets):
                        if d == 1:
                            scatters[index].set_offsets(np.column_stack((pos[i, j], 0)))
                        elif d == 2:
                            scatters[index].set_offsets(pos[i, j])
                        index += 1
                return scatters
            anim = FuncAnimation(
                fig, update,
                frames=num_steps,
                init_func=init,
                blit=True,
                interval=50,
                repeat=True
            )
        plt.show()
        if save:
            task = task.replace('.', '')
            os.makedirs('./videos', exist_ok=True)
            anim.save(f'./videos/world_evolution_task{task}.gif', writer='pillow', fps=15)
            print(f"Animation saved as 'world_evolution_task{task}.gif'")
        return anim
    else:
        print(f"Animation only supports dimensions 1-3. Current dimension: {d}")
        return None