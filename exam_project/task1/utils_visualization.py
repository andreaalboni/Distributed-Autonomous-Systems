import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
    
def visualize_graph(G):
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True)
    plt.show()
    
def visualize_world(agents, targets, world_size):
    # De-Normalization:
    agents = agents * world_size[0]
    targets = targets * world_size[0]
    
    plt.figure(figsize=(8, 8))    
    plt.scatter(agents[:, 0], agents[:, 1], c='blue', marker='o', label='Agent')
    plt.scatter(targets[:, 0], targets[:, 1], c='red', marker='x', label='Target')
    padding = 0.2 
    x_min, x_max = 0, world_size[0]
    y_min, y_max = 0, world_size[1]
    plt.xlim(x_min - padding * x_max, x_max + padding * x_max)
    plt.ylim(y_min - padding * y_max, y_max + padding * y_max)
    plt.title('World Visualization')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_gradient_traking_results(z, cost, norm_grad_cost, agent_grad_norm, agents, targets, graph_type=None):
    """
    Visualizes the results of the gradient tracking algorithm.
    """
    # Plot cost and gradient norm
    fig, axes = plt.subplots(figsize=(8, 6), nrows=1, ncols=2)
    
    ax = axes[0]
    ax.semilogy(np.arange(len(cost)-1), cost[:-1], color='violet')
    ax.set_title('Cost vs Iteration')
    ax.set_xlabel('Iteration')
    
    ax = axes[1]
    ax.semilogy(np.arange(len(norm_grad_cost)-1), norm_grad_cost[:-1], color='cyan')
    ax.semilogy(np.arange(len(agent_grad_norm)-1), agent_grad_norm[:-1], color='purple')
    ax.set_title('Gradient of the cost vs Iteration')
    ax.set_xlabel('Iteration')
    
    plt.show()

def animate_world_evolution(agents, targets, z_history, type, world_size, speed=4):
    # De-Normalization:
    agents = agents * world_size[0]
    targets = targets * world_size[0]
    z_history = z_history * world_size[0]

    T, n_agents, n_targets, _ = z_history.shape
    frame_skip = int(speed)
    frame_skip += 1
    num_steps = T // frame_skip
    positions = z_history[::frame_skip]  # Reduce data for animation
    # Add pause frames at the end
    pause_frames = int(3 * 20)  # 3 seconds at 20 fps
    positions = np.concatenate([positions, np.repeat(positions[-1:], pause_frames, axis=0)])
    num_steps = len(positions)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f'Agents to Targets Animation - {type} Graph')
    padding = 0.2
    x_min, x_max = 0, world_size[0]
    y_min, y_max = 0, world_size[1]
    ax.set_xlim(x_min - padding * x_max, x_max + padding * x_max)
    ax.set_ylim(y_min - padding * y_max, y_max + padding * y_max)
    ax.set_aspect('equal')
    ax.grid(True)
    # Plot static agents and targets
    ax.scatter(agents[:, 0], agents[:, 1], c='blue', marker='o', s=50, label='Agent')
    ax.scatter(targets[:, 0], targets[:, 1], c='red', marker='x', s=50, label='Target')
    ax.legend()
    # Create dynamic scatter objects for each agent-target pair
    scatters = []
    for _ in range(n_agents * n_targets):
        scatter = ax.scatter([], [], c='green', s=10)
        scatters.append(scatter)
    def init():
        for scatter in scatters:
            scatter.set_offsets(np.empty((0, 2)))
        return scatters
    def update(frame):
        pos = positions[frame]  # shape: (n_agents, n_targets, 2)
        index = 0
        for i in range(n_agents):
            for j in range(n_targets):
                scatters[index].set_offsets(pos[i, j])
                index += 1
        return scatters
    _ = FuncAnimation(
        fig, update,
        frames=num_steps,
        init_func=init,
        blit=True,
        interval=1,
        repeat=True
    )
    plt.show()