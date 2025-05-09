import warnings
import numpy as np
import networkx as nx
from config import PARAMETERS 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_default_params():
    return PARAMETERS

def debug_spawn_agent_near_intruder(intruder, existing_agents, existing_intruders, world_size=PARAMETERS['world_size'], radius_spawn_agent=PARAMETERS['radius_spawn_agent']):
    while True:
        candidate = np.random.uniform(0, world_size[0], size=2)
        center = np.array([world_size[0]/2, world_size[1]/2])
        if (np.linalg.norm(candidate - intruder) <= radius_spawn_agent and
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_intruders)):
            return candidate

def debug_spawn_candidate(existing_agents, existing_intruders, world_size=PARAMETERS['world_size'], intruder_radius=PARAMETERS['intruder_radius']):
    while True:
        angle = np.random.uniform(0, 2 * np.pi)
        noise = 0
        candidate = np.array([
            intruder_radius * np.sin(angle) + world_size[0]/2 + noise * np.sin(angle),
            intruder_radius * np.cos(angle) + world_size[0]/2 + noise * np.cos(angle)
        ])
        if (not any(np.allclose(candidate, a, atol=3.0) for a in existing_agents) and 
            not any(np.allclose(candidate, t, atol=3.0) for t in existing_intruders)):
            return candidate

def spawn_agent_near_intruder(intruder, existing_agents, existing_intruders, world_size=PARAMETERS['world_size'], radius_spawn_agent=PARAMETERS['radius_spawn_agent']):
    while True:
        candidate = np.random.uniform(0, world_size[0], size=2)
        if (np.linalg.norm(candidate - intruder) <= radius_spawn_agent and
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_intruders)):
            return candidate
        
def spawn_candidate(existing_agents, existing_intruders, world_size=PARAMETERS['world_size']):
    while True:
        candidate = np.random.uniform(0, world_size[0], size=2)
        if (not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and 
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_intruders)):
            return candidate

def generate_agents_and_intruders(num_intruders=PARAMETERS['num_intruders'], world_size=PARAMETERS['world_size']):
    intruders = []
    agents = []
    for _ in range(num_intruders):
        # Genera un nuovo intruder
        intruder = debug_spawn_candidate(agents, intruders, world_size=world_size)
        intruders.append(intruder)
        # Genera un agente vicino al proprio intruder
        agent = debug_spawn_agent_near_intruder(intruder, agents, intruders, world_size=world_size)
        agents.append(agent)
    return np.array(intruders), np.array(agents)

def get_distances(agents, intruders):
    distances = []
    for agent in range(len(agents)):
        distances.append(np.linalg.norm(agents[agent] - intruders[agent]))
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

def compute_aggregative_variable(agents):
    sigma = np.mean(agents, axis=0)
    return sigma

def local_cost_function(z, p_i, sigma, weight_ratio=PARAMETERS['weight_ratio']):
    
    '''
    Cost function a lezione:
    l_i = gamma_i * (norma della distanza agente - intruder)**2 +                       --> garantisce agente vada sull'intruder
            gamma_bar_i * (norma della distanza agente - baricentro)**2 +               --> garantisce formazione sia il più possibile compatta
            norma della distanza baricentro - intruder (che io farei che sia il baricentro degli intruder)
    sigma(z) = sum (phi_i(z_i)) / N     --> calcolo baricentro
    phi_i(z_i): per calcolo baricentro normale: = z_i, per weighted barycentre: = weight_i * z_i
    '''
    
    agent_to_sigma_distance_squared = np.linalg.norm(z - sigma)**2
    agent_to_intruder_distance_squared = weight_ratio * np.linalg.norm(z - p_i)**2
    local_cost = (agent_to_intruder_distance_squared + agent_to_sigma_distance_squared)**2
    local_cost_gradient = 4 * (agent_to_intruder_distance_squared + agent_to_sigma_distance_squared) * (z + p_i)
    
    return local_cost, local_cost_gradient
        

def visualize_graph(G):
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True)
    plt.show()
    
def visualize_world(agents, intruders, world_size=PARAMETERS['world_size']):
    plt.figure(figsize=(8, 8))    
    plt.scatter(agents[:, 0], agents[:, 1], c='black', marker='o', s=50, label='Agent')
    plt.scatter(intruders[:, 0], intruders[:, 1], c='none', marker='s', s=50, label='Intruder', edgecolors='cyan')
    sigma = compute_aggregative_variable(agents)
    plt.scatter(sigma[0], sigma[1], c='none', marker='^', s=50, label='Sigma', edgecolors='mediumseagreen')
    intruder = compute_aggregative_variable(intruders)
    plt.scatter(intruder[0], intruder[1], c='red', marker='x', s=50, label=r'$r_0$')
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