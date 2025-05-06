import numpy as np
import matplotlib.pyplot as plt
import warnings
import networkx as nx


def get_default_params():
    return {
        'num_targets': 2,
        'ratio_at': 5,
        'world_size': [5, 5],
        'radius_fov': np.inf,
        'noise_level': 0.0,
        'bias': 0.0
    }

def is_in_fov(agent_pos, target_pos, radius_fov):
    return np.linalg.norm(agent_pos - target_pos) <= radius_fov
        
def doll_cost_function(z, Q, z_ref):
    diff = z - z_ref
    cost = np.linalg.norm(diff.T @ Q @ diff, 'fro')**2
    grad = 2 * Q @ diff
    return cost, grad

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
        G = nx.star_graph(num_agents)
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