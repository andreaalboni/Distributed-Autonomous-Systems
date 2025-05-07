import warnings
import numpy as np
import networkx as nx
from config import PARAMETERS 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_default_params():
    return PARAMETERS

def is_in_fov(agent_pos, target_pos, radius_fov=PARAMETERS['radius_fov']):
    return np.linalg.norm(agent_pos - target_pos) <= radius_fov
        

def spawn_agent_near_target(target, existing_agents, existing_targets, world_size=PARAMETERS['world_size'], radius_fov=PARAMETERS['radius_fov']):
    while True:
        candidate = np.random.uniform(0, world_size[0], size=2)
        if (is_in_fov(candidate, target) and 
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_targets)):
            return candidate
        
def spawn_candidate(existing_agents, existing_targets, world_size=PARAMETERS['world_size']):
    while True:
        candidate = np.random.uniform(0, world_size[0], size=2)
        if (not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and 
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_targets)):
            return candidate

def generate_agents_and_targets(num_targets=PARAMETERS['num_targets'], ratio_at=PARAMETERS['ratio_at'], world_size=PARAMETERS['world_size'], radius_fov=PARAMETERS['radius_fov']):
    targets = []
    agents = []

    for _ in range(num_targets):
        # Genera un nuovo target
        target = spawn_candidate(agents, targets, world_size=world_size)
        targets.append(target)
        
        # Genera un agente vicino al proprio target
        agent = spawn_agent_near_target(target, agents, targets, world_size=world_size)
        agents.append(agent)

    return np.array(targets), np.array(agents)

def get_distances(agents, targets, noise_level=PARAMETERS['noise_level'], bias=PARAMETERS['bias']):
    distances = []
    for agent in agents:
        agent_distance = []
        for target in targets:
            agent_distance.append(np.linalg.norm(agent - target))
        distances.append(agent_distance)
    noisy_distances = np.array(distances) + np.random.normal(bias, noise_level, np.array(distances).shape)
    return np.array(distances), noisy_distances
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
    """
    A_ij = 1/(1 + max(d_i, d_j)) if (i,j) ∈ E and i ≠ j
           1 - ∑(A_ih) for h ∈ N_i\{i} if i = j
           0 otherwise
    """
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

def visualize_graph(G):
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True)
    plt.show()
    
def visualize_world(agents, targets, world_size=PARAMETERS['world_size']):
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

