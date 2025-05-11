import numpy as np
import networkx as nx
from launch_ros.actions import Node
from launch import LaunchDescription


PARAMETERS = {
    'num_intruders': 5,
    'world_size': [20, 20],
    'intruder_radius': 10.0,
    'radius_spawn_agent': 5.0,
    'noise_r_0': 0.0,
    'p_er': 0.5,
}


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

def compute_agents_barycenter(agents):
    sigma = np.mean(agents, axis=0)
    return sigma

def compute_r_0(intruders, noise_radius=PARAMETERS['noise_r_0'], world_size=PARAMETERS['world_size']):
    barycenter_intruders = compute_agents_barycenter(intruders)
    if (noise_radius == 0.0):
        return barycenter_intruders
    while True:
        r_0_candidate = np.random.uniform(0, world_size[0], size=2)
        if (np.linalg.norm(r_0_candidate - barycenter_intruders) <= noise_radius and
            np.linalg.norm(r_0_candidate - barycenter_intruders) >= noise_radius/10):
                return r_0_candidate


def generate_launch_description():
    ALPHA = 0.0001
    MAX_ITERS = 2000
    COMM_TIME = 5e-2
    N = PARAMETERS['num_intruders']

    graph_type = 'cycle'
    _, adj, A = generate_graph(N, type=graph_type)

    intruders, agents = generate_agents_and_intruders()
    r_0 = compute_r_0(intruders).tolist()
    z_0 = agents

    gamma = 15 * np.ones(len(agents))
    gamma_bar = 3 * np.ones(len(agents))

    node_list = []
    package_name = "task22"

    for i in range(N):
        gamma_i = gamma[i]
        A_i = A[i].tolist()
        z_0_i = z_0[i].tolist()
        gamma_bar_i = gamma_bar[i]
        intruder_i = intruders[i].tolist()
        N_i = np.nonzero(adj[:, i])[0].tolist()

        node_list.append(
            Node(
                package=package_name,
                namespace=f"agent_{i}",
                executable="agent",
                parameters=[
                    {
                        "id": i,
                        "A_i": A_i,
                        "r_0": r_0,
                        "z0": z_0_i,
                        "alpha": ALPHA,
                        "gamma": gamma_i,
                        "neighbors": N_i,
                        "max_iters": MAX_ITERS,
                        "intruder": intruder_i,
                        "gamma_bar": gamma_bar_i,
                        "communication_time": COMM_TIME,
                    }
                ],
                output="screen",
                prefix=f'xterm -title "agent_{i}"  -fg white -bg black -fs 12 -fa "Monospace" -hold -e',
            )
        )
    return LaunchDescription(node_list)

generate_launch_description()