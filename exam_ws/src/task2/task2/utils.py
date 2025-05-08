import warnings
import numpy as np
import networkx as nx
from config import PARAMETERS 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_default_params():
    return PARAMETERS

def debug_spawn_agent_near_target(target, existing_agents, existing_targets, world_size=PARAMETERS['world_size'], radius_spawn_target=PARAMETERS['radius_spawn_target']):
    while True:
        candidate = np.random.uniform(0, world_size[0], size=2)
        center = np.array([world_size[0]/2, world_size[1]/2])
        if (np.linalg.norm(candidate - target) <= radius_spawn_target and
            np.linalg.norm(candidate - center) < np.linalg.norm(target - center) - 3 and
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_targets)):
            return candidate

def debug_spawn_candidate(existing_agents, existing_targets, world_size=PARAMETERS['world_size'], target_radius=PARAMETERS['target_radius']):
    while True:
        angle = np.random.uniform(0, 2 * np.pi)
        noise = 0
        candidate = np.array([
            target_radius * np.sin(angle) + world_size[0]/2 + noise * np.sin(angle),
            target_radius * np.cos(angle) + world_size[0]/2 + noise * np.cos(angle)
        ])
        if (not any(np.allclose(candidate, a, atol=2) for a in existing_agents) and 
            not any(np.allclose(candidate, t, atol=2) for t in existing_targets)):
            return candidate

def spawn_agent_near_target(target, existing_agents, existing_targets, world_size=PARAMETERS['world_size'], radius_spawn_target=PARAMETERS['radius_spawn_target']):
    while True:
        candidate = np.random.uniform(0, world_size[0], size=2)
        if (np.linalg.norm(candidate - target) <= radius_spawn_target and
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_targets)):
            return candidate
        
def spawn_candidate(existing_agents, existing_targets, world_size=PARAMETERS['world_size']):
    while True:
        candidate = np.random.uniform(0, world_size[0], size=2)
        if (not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and 
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_targets)):
            return candidate

def generate_agents_and_targets(num_targets=PARAMETERS['num_targets'], world_size=PARAMETERS['world_size']):
    targets = []
    agents = []
    for _ in range(num_targets):
        # Genera un nuovo target
        target = debug_spawn_candidate(agents, targets, world_size=world_size)
        targets.append(target)
        # Genera un agente vicino al proprio target
        agent = debug_spawn_agent_near_target(target, agents, targets, world_size=world_size)
        agents.append(agent)
    return np.array(targets), np.array(agents)

def get_distances(agents, targets):
    distances = []
    for agent in range(len(agents)):
        distances.append(np.linalg.norm(agents[agent] - targets[agent]))
    return np.array(distances)

def get_default_params():
    return PARAMETERS

def ensure_connected_graph(G):
    if nx.is_connected(G):
        return G
    components = list(nx.connected_components(G))
    for i in range(len(components) - 1):
        u = list(components[i])[0]
        v = list(components[i + 1])[0]
        G.add_edge(u, v)
    return G

def generate_graph(num_agents, type, p_er=PARAMETERS['p_er']):
    if type == 'path':
        G = nx.path_graph(num_agents)
    elif type == 'cycle':
        G = nx.path_graph(num_agents)
        G.add_edge(0, num_agents-1) # Add an edge between the first and last node
    elif type == 'star':
        G = nx.star_graph(num_agents - 1)
    elif type == 'erdos_renyi':
        # Create a random graph with N nodes and probability of edge creation 0.5
        G = nx.erdos_renyi_graph(num_agents, p=p_er, seed=0) 
        G = ensure_connected_graph(G)
    else:
        raise ValueError("Unknown graph type. Use 'cycle', 'star', or 'erdos_renyi'.")
    
    Adj = nx.adjacency_matrix(G).toarray()
    A = metropolis_hastings_weights(G)
    return G, Adj, A

def metropolis_hastings_weights(G):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    degrees = {node: G.degree(node) for node in G.nodes()}
    A = np.zeros((n, n))
    
    # Fill non-diagonal elements (i != j)
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j and G.has_edge(node_i, node_j):
                d_i = degrees[node_i]
                d_j = degrees[node_j]
                A[i, j] = 1 / (1 + max(d_i, d_j))
    
    # Fill diagonal elements (i == j)
    for i, node_i in enumerate(nodes):
        neighbors = list(G.neighbors(node_i))
        neighbor_indices = [node_to_idx[neigh] for neigh in neighbors]
        A[i, i] = 1 - sum(A[i, j] for j in neighbor_indices)
    return A

def compute_aggregative_variale(agents):
    print(f"agents: {agents}")
    sigma = np.mean(agents, axis=0)
    return sigma

def local_cost_function(z, p_i, sigma, weight_ratio=PARAMETERS['weight_ratio']):
    agent_to_sigma_distance_squared = np.linalg.norm(z - sigma)**2
    agent_to_target_distance_squared = weight_ratio * np.linalg.norm(z - p_i)**2
    local_cost = (agent_to_target_distance_squared + agent_to_sigma_distance_squared)**2
    local_cost_gradient = 4 * (agent_to_target_distance_squared + agent_to_sigma_distance_squared) * (z + p_i)
    
    return local_cost, local_cost_gradient
        

def visualize_graph(G):
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True)
    plt.show()
    
def visualize_world(agents, targets, sigma, world_size=PARAMETERS['world_size']):
    plt.figure(figsize=(8, 8))    
    plt.scatter(agents[:, 0], agents[:, 1], c='blue', marker='o', label='Agent')
    plt.scatter(targets[:, 0], targets[:, 1], c='red', marker='x', label='Target')
    plt.scatter(sigma[0], sigma[1], c='mediumseagreen', marker='^', s=200, label='Sigma')
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

def plot_graph_with_connections(G):
    pos = nx.spring_layout(G)  
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
    # Mostriamo il titolo
    plt.title("Grafo con Connessioni")
    plt.axis('off')  
    plt.show()