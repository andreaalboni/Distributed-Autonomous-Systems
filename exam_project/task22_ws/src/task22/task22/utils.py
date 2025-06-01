import numpy as np
import networkx as nx

def debug_safety_control_spawn_agent_near_intruder(intruder, existing_agents, existing_intruders, world_size, radius_spawn_agent, d):
    if len(existing_agents) < 2 and d == 2:
        if len(existing_agents) == 0: return [2, 2]
        if len(existing_agents) == 1: return [1, 1]
    if len(existing_agents) < 2 and d == 3:
            if len(existing_agents) == 0: return [2, 2, 2]
            if len(existing_agents) == 1: return [1, 1, 1]
    while True:
        candidate = np.random.uniform(0, world_size, size=d)
        if (np.linalg.norm(candidate - intruder) <= radius_spawn_agent and
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_intruders)):
            return candidate

def debug_safety_control_spawn_candidate(existing_agents, existing_intruders, world_size, intruder_radius, d):
    if len(existing_intruders) < 2 and d == 2:
        if len(existing_intruders) == 0: return [0, 0]
        if len(existing_intruders) == 1: return [3, 3]
    if len(existing_intruders) < 2 and d == 3:
        if len(existing_intruders) == 0: return [0, 0, 0]
        if len(existing_intruders) == 1: return [3, 3, 3]
    while True:
        if d == 2:
            theta = np.random.uniform(0, 2 * np.pi)
            candidate = np.array([
                intruder_radius * np.cos(theta),
                intruder_radius * np.sin(theta)
            ])
        elif d == 3:
            theta = np.random.uniform(0, 2 * np.pi)  # azimuthal angle
            phi = np.random.uniform(0, np.pi)        # polar angle
            candidate = np.array([
                intruder_radius * np.sin(phi) * np.cos(theta),
                intruder_radius * np.sin(phi) * np.sin(theta),
                intruder_radius * np.cos(phi)
            ])
        else:
            raise ValueError("Only 2D and 3D spaces are supported")
        candidate += world_size/2
        if (not any(np.allclose(candidate, a, atol=3.0) for a in existing_agents) and 
            not any(np.allclose(candidate, t, atol=3.0) for t in existing_intruders)):
            return candidate

def debug_spawn_agent_near_intruder(intruder, existing_agents, existing_intruders, world_size, 
                                    radius_spawn_agent, d):
    while True:
        candidate = np.random.uniform(0, world_size, size=d)
        center = world_size/2 * np.ones_like(candidate)
        if (np.linalg.norm(candidate - intruder) <= radius_spawn_agent and
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_intruders)):
            return candidate

def debug_spawn_candidate(existing_agents, existing_intruders, world_size, intruder_radius, d):
    while True:
        if d == 2:
            theta = np.random.uniform(0, 2 * np.pi)
            candidate = np.array([
                intruder_radius * np.cos(theta),
                intruder_radius * np.sin(theta)
            ])
        elif d == 3:
            theta = np.random.uniform(0, 2 * np.pi)  # azimuthal angle
            phi = np.random.uniform(0, np.pi)        # polar angle
            candidate = np.array([
                intruder_radius * np.sin(phi) * np.cos(theta),
                intruder_radius * np.sin(phi) * np.sin(theta),
                intruder_radius * np.cos(phi)
            ])
        else:
            raise ValueError("Only 2D and 3D spaces are supported")
        candidate += world_size/2
        if (not any(np.allclose(candidate, a, atol=3.0) for a in existing_agents) and 
            not any(np.allclose(candidate, t, atol=3.0) for t in existing_intruders)):
            return candidate

def spawn_agent_near_intruder(intruder, existing_agents, existing_intruders, world_size,
                               radius_spawn_agent, d):
    while True:
        candidate = np.random.uniform(0, world_size, size=d)
        if (np.linalg.norm(candidate - intruder) <= radius_spawn_agent and
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_intruders)):
            return candidate
        
def spawn_candidate(existing_agents, existing_intruders, world_size, d):
    while True:
        candidate = np.random.uniform(0, world_size, size=d)
        if (not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and 
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_intruders)):
            return candidate

def generate_agents_and_intruders(num_intruders, world_size, radius_spawn_agent, intruder_radius, d):
    intruders = []
    agents = []
    for _ in range(num_intruders):
        # def debug_spawn_candidate(existing_agents, existing_intruders, world_size, intruder_radius, d):

        intruder = debug_safety_control_spawn_candidate(agents, intruders, world_size, intruder_radius, d)
        intruders.append(intruder)
        agent = debug_safety_control_spawn_agent_near_intruder(intruder, agents, intruders, world_size, radius_spawn_agent, d)
        agents.append(agent)
    print("\033[92m" + f"intruders: {intruders}" + "\033[0m")
    print("\033[94m" + f"agents: {agents}" + "\033[0m")
    return np.array(intruders), np.array(agents)

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
    elif type == 'erdos_renyi':
        # Create a random graph with N nodes and probability of edge creation 0.5
        G = nx.erdos_renyi_graph(num_agents, p=p_er, seed=0) 
        G = ensure_connected_graph(G)
    else:
        raise ValueError("Unknown graph type. Use 'cycle', 'path', 'star', or 'erdos_renyi'.")
    
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

def compute_r_0(intruders, noise_radius, world_size, d):
    barycenter_intruders = compute_agents_barycenter(intruders)
    if (noise_radius == 0.0):
        return barycenter_intruders
    while True:
        r_0_candidate = np.random.uniform(0, world_size, size=d)
        if (np.linalg.norm(r_0_candidate - barycenter_intruders) <= noise_radius and
            np.linalg.norm(r_0_candidate - barycenter_intruders) >= noise_radius/10):
                return r_0_candidate
            
def calculate_heading_from_movement(current_pos, previous_pos, d, threshold=1e-8):
    if previous_pos is None:
        return 0.0 if d == 2 else [0.0, 0.0]
    diff = np.array(current_pos) - np.array(previous_pos)
    if d == 2:
        dx, dy = diff[0], diff[1]
        if abs(dx) > threshold or abs(dy) > threshold:
            return np.arctan2(dy, dx)
        return None  # No significant movement
    else:  # d == 3
        dx, dy, dz = diff[0], diff[1], diff[2]
        if abs(dx) > threshold or abs(dy) > threshold or abs(dz) > threshold:
            azimuth = np.arctan2(dy, dx)
            elevation = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
            return [azimuth, elevation]
        return None  # No significant movement

def simulate_lidar_scan(agents, heading, fov_horizontal, fov_vertical, fov_range, d):
    # Simulate a LIDAR scan -> return matrix of distances between agents
    num_agents = len(agents)
    distances = np.zeros((num_agents, num_agents))
    horizontal_angles = np.zeros((num_agents, num_agents))
    vertical_angles = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                dist = np.linalg.norm(agents[i] - agents[j])
                if d == 2:
                    angle = np.arctan2(agents[j][1] - agents[i][1], agents[j][0] - agents[i][0])
                    if (abs(angle-heading[i]) <= np.deg2rad(fov_horizontal / 2)) and (dist <= fov_range):
                        distances[i, j] = dist
                        horizontal_angles[i, j] = angle
                    else:
                        distances[i, j] = np.nan
                        horizontal_angles[i, j] = np.nan
                elif d == 3:
                    dx, dy, dz = agents[j] - agents[i]
                    azimuth = np.arctan2(dy, dx)
                    elevation = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
                    if (abs(azimuth-heading[i,0]) <= np.deg2rad(fov_horizontal / 2)) and \
                       (abs(elevation-heading[i,1]) <= np.deg2rad(fov_vertical / 2)) and \
                       (dist <= fov_range):
                        distances[i, j] = dist
                        horizontal_angles[i, j] = azimuth
                        vertical_angles[i, j] = elevation
                    else:
                        distances[i, j] = np.nan
                        horizontal_angles[i, j] = np.nan
                        vertical_angles[i, j] = np.nan
    return distances, horizontal_angles, vertical_angles

