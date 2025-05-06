import numpy as np
import matplotlib.pyplot as plt
import warnings
import networkx as nx
from matplotlib.animation import FuncAnimation



def get_default_params():
    return {
        'num_targets': 2,
        'ratio_at': 5,
        'world_size': [5, 5],
        'radius_fov': np.inf,
        'noise_level': 0.1,
        'bias': 0.0
    }

def is_in_fov(agent_pos, target_pos, radius_fov):
    return np.linalg.norm(agent_pos - target_pos) <= radius_fov
        
def spawn_agent_near_target(target, world_size, radius_fov, existing_agents, existing_targets):
    while True:
        candidate = np.random.uniform(0, world_size[0], size=2)
        if (is_in_fov(candidate, target, radius_fov) and 
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_targets)):
            return candidate
        
def spawn_candidate(world_size, existing_agents, existing_targets):
    while True:
        candidate = np.random.uniform(0, world_size[0], size=2)
        if (not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and 
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_targets)):
            return candidate

def generate_agents_and_targets(num_targets, ratio_at, world_size, radius_fov):
    targets = []
    agents = []
    # Spawn targets and required agents
    for _ in range(num_targets):
        target = spawn_candidate(world_size, agents, targets)
        targets.append(target)
        visible_agents = sum(is_in_fov(agent, target, radius_fov) for agent in agents)
        for _ in range(3 - visible_agents):
            agents.append(spawn_agent_near_target(target, world_size, radius_fov, agents, targets))
    # Add remaining agents randomly
    total_agents_needed = int(num_targets * ratio_at)
    while len(agents) < total_agents_needed:
        candidate = spawn_candidate(world_size, agents, targets)
        agents.append(candidate)
    if len(agents) > total_agents_needed:
        warnings.warn(f"\033[38;5;214mNumber of agents ({len(agents)}) exceeds the required number ({total_agents_needed}).\033[0m")
    return np.array(targets), np.array(agents)

def get_distances(agents, targets, noise_level=0.0, bias=0.0):
    distances = []
    for agent in agents:
        agent_distance = []
        for target in targets:
            agent_distance.append(np.linalg.norm(agent - target))
        distances.append(agent_distance)
    noisy_distances = np.array(distances) + np.random.normal(bias, noise_level, np.array(distances).shape)
    return np.array(distances), noisy_distances

def ensure_Adj_doubly_stocasticity(num_agents, Adj):
    A = Adj + np.eye(num_agents)
    ONES = np.ones((num_agents, num_agents))
    while any(abs(np.sum(A, axis=0) - 1) > 1e-10):
        A = A / (A @ ONES)      # Guarantees row stochasticity
        A = A / (ONES.T @ A)    # Guarantees column stochasticity
        A = np.abs(A)
    return A

def ensure_connected_graph(G):

    if nx.is_connected(G):
        return G

    components = list(nx.connected_components(G))
    
    for i in range(len(components) - 1):
        u = list(components[i])[0]
        v = list(components[i + 1])[0]
        G.add_edge(u, v)
        
    return G

def generate_graph(num_agents, type, p_er=0.5):
    if type == 'path':
        G = nx.path_graph(num_agents)
    elif type == 'cycle':
        G = nx.path_graph(num_agents)
        G.add_edge(0, num_agents-1) # Add an edge between the first and last node
    elif type == 'star':
        G = nx.star_graph(num_agents - 1)
        visualize_graph(G)
    elif type == 'erdos_renyi':
        # Create a random graph with N nodes and probability of edge creation 0.5
        G = nx.erdos_renyi_graph(num_agents, p=p_er, seed=0) 
        G = ensure_connected_graph(G)
    else:
        raise ValueError("Unknown graph type. Use 'cycle', 'star', or 'erdos_renyi'.")
    
    Adj = nx.adjacency_matrix(G).toarray()
    A = ensure_Adj_doubly_stocasticity(num_agents, Adj)
    return G, Adj, A

def get_targets_real_positions(targets):
    return np.array([target for target in targets])

def local_cost_function(z, p_i, distances_i):
    # p_i:  position of the agent
    # z:    position of the target
    # distances_i:  distances to the targets
    num_targets = len(distances_i)
    local_cost = 0
    local_cost_gradient = np.zeros((num_targets, len(p_i)))
    for target in range(num_targets):
        # Cost function evaluation
        estimated_distance_squared = np.linalg.norm(z[target] - p_i)**2
        measured_distance_squared = distances_i[target]**2
        local_cost += (measured_distance_squared - estimated_distance_squared)**2 
        # Gradient evaluation
        local_cost_gradient[target, :] = 4 * (estimated_distance_squared - measured_distance_squared) * (z[target] - p_i)
    return local_cost, local_cost_gradient
    
def visualize_graph(G):
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True)
    plt.show()
    
def visualize_world(agents, targets, world_size):
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


def animate_world_evolution(agents, targets, world_size, z_hystory, speed=0):
    """
    Animate the evolution of z_history showing the paths from agents to targets.

    Parameters:
        agents (numpy.ndarray): Initial positions of agents, shape (n_agents, 2).
        targets (numpy.ndarray): Positions of targets, shape (n_targets, 2).
        world_size (tuple): Size of the world as (width, height).
        z_hystory (numpy.ndarray): Array with shape (T, n_agents, n_targets, 2).
        speed (int): Speed factor for animation frame skipping.
    """
    T, n_agents, n_targets, _ = z_hystory.shape

    frame_skip = int(speed)
    frame_skip += 1
    num_steps = T // frame_skip
    positions = z_hystory[::frame_skip]  # Reduce data for animation

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Agents to Targets Animation")
    padding = 0.2
    x_min, x_max = 0, world_size[0]
    y_min, y_max = 0, world_size[1]
    ax.set_xlim(x_min - padding * x_max, x_max + padding * x_max)
    ax.set_ylim(y_min - padding * y_max, y_max + padding * y_max)
    ax.set_aspect('equal')
    ax.grid(True)

    # Plot static agents and targets
    ax.scatter(agents[:, 0], agents[:, 1], c='blue', marker='o', label='Agent')
    ax.scatter(targets[:, 0], targets[:, 1], c='red', marker='x', label='Target')
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

    animation = FuncAnimation(
        fig, update,
        frames=num_steps,
        init_func=init,
        blit=True,
        interval=50
    )

    plt.show()