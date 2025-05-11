import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from launch_ros.actions import Node
from launch import LaunchDescription
from matplotlib.animation import FuncAnimation


PARAMETERS = {
    'num_intruders': 5,
    'world_size': [20, 20],
    'intruder_radius': 10.0,
    'radius_spawn_agent': 5.0,
    'noise_r_0': 0.0,
    'p_er': 0.5,
}


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

def local_cost_function(agent_i, intruder_i, sigma, r_0, gamma_i, gamma_bar_i):
    
    '''
    Cost function a lezione:
    l_i = gamma_i * (norma della distanza agente - intruder)**2 +                       --> garantisce agente vada sull'intruder
            gamma_bar_i * (norma della distanza agente - sigma)**2 +               --> garantisce formazione sia il più possibile compatta
            norma della distanza r_0 - intruder (che io farei che sia il baricentro degli intruder)
    sigma(z) = sum (phi_i(z_i)) / N     --> calcolo sigma
    phi_i(z_i): per calcolo sigma normale: = z_i, per weighted barycenter: = weight_i * z_i
    
    # agents has shape (2,)
    # intruder has shape (2,)
    # sigma has shape (2,)
    
    # grad_1 is the gradient of the cost function with respect to the intruder
    # grad_2 is the gradient of the cost function with respect to the barycenter
    '''
    agent_to_intruder = np.linalg.norm(agent_i - intruder_i)**2  
    agent_to_sigma = np.linalg.norm(agent_i - sigma)**2
    intruder_to_r_0 = np.linalg.norm(intruder_i - r_0)**2
    
    local_cost = gamma_i * agent_to_intruder + gamma_bar_i * agent_to_sigma + intruder_to_r_0
    
    # grad_1 is the gradient of the cost function with respect to the agent
    grad_1 = 2 * (gamma_i * (agent_i - intruder_i) + gamma_bar_i * (agent_i - sigma))
    # grad_2 is the gradient of the cost function with respect to sigma
    grad_2 = - 2 * gamma_bar_i * (agent_i - sigma)
    
    return local_cost, grad_1, grad_2

def local_phi_function(agent_i):
    phi_i = agent_i
    grad_phi_i = 1
    return phi_i, grad_phi_i

def visualize_graph(G):
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True)
    plt.show()
    
def visualize_world(agents, intruders, world_size=PARAMETERS['world_size']):
    plt.figure(figsize=(8, 8))    
    plt.scatter(agents[:, 0], agents[:, 1], c='black', marker='o', s=50, label='Agent')
    plt.scatter(intruders[:, 0], intruders[:, 1], c='none', marker='s', s=50, label='Intruder', edgecolors='cyan')
    sigma = compute_agents_barycenter(agents)
    plt.scatter(sigma[0], sigma[1], c='none', marker='^', s=50, label='Sigma', edgecolors='mediumseagreen')
    target = compute_r_0(intruders)
    intruders_barycenter = compute_agents_barycenter(intruders)
    if not np.array_equal(intruders_barycenter, target):
        plt.scatter(intruders_barycenter[0], intruders_barycenter[1], c='purple', alpha=0.35, marker='x', s=50, label='Intruders\' CoG')
    plt.scatter(target[0], target[1], c='red', marker='x', s=50, label=r'$r_0$')
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
    
def animate_world_evolution(intruders, z_history, s, sigma, r_0, world_size=PARAMETERS['world_size'], speed=10):
    T, n_agents, *_ = z_history.shape
    frame_skip = int(speed) + 1
    positions = z_history[::frame_skip]
    s_positions = s[::frame_skip]
    pause_frames = int(3 * 20)
    positions = np.concatenate([positions, np.repeat(positions[-1:], pause_frames, axis=0)])
    s_positions = np.concatenate([s_positions, np.repeat(s_positions[-1:], pause_frames, axis=0)])
    num_steps = len(positions)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Agents to Targets Animation')
    padding = 0.2
    x_min, x_max = 0, world_size[0]
    y_min, y_max = 0, world_size[1]
    ax.set_xlim(x_min - padding * x_max, x_max + padding * x_max)
    ax.set_ylim(y_min - padding * y_max, y_max + padding * y_max)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.scatter(intruders[:, 0], intruders[:, 1], c='red', marker='x', s=50, label='Intruder')
    # ax.scatter(sigma[0], sigma[1], c='black', marker='s', s=50, label=r'$\sigma$')
    ax.scatter(r_0[0], r_0[1], c='none', marker='o', s=50, edgecolors='purple', label=r'$r_0$')
    path_lines = [[] for _ in range(n_agents)]
    ax.legend()
    z_scatters = [ax.scatter([], [], c='blue', s=50) for _ in range(n_agents)]
    s_scatters = [ax.scatter([], [], c='orange', marker='x', s=35) for _ in range(n_agents)]
    def init():
        for scatter in z_scatters + s_scatters:
            scatter.set_offsets(np.empty((0, 2)))
        return z_scatters + s_scatters + sum(path_lines, [])
    def update(frame):
        pos = positions[frame]
        s_pos = s_positions[frame]
        for i in range(n_agents):
            z_scatters[i].set_offsets(pos[i])
            s_scatters[i].set_offsets(s_pos[i])
            if len(path_lines[i]) > 0:
                for line in path_lines[i]:
                    line.remove()
                path_lines[i] = []
            if frame > 0:
                line, = ax.plot(positions[:frame+1, i, 0], positions[:frame+1, i, 1], 
                               'gray', linestyle='--', alpha=0.5)
                path_lines[i].append(line)
        return z_scatters + s_scatters + sum(path_lines, [])
    anim = FuncAnimation(
        fig, update,
        frames=num_steps,
        init_func=init,
        blit=True,
        interval=50,
        repeat=True
    )
    plt.show()
    return anim



def generate_launch_description():
    ALPHA = 0.0001
    MAXITERS = 2000
    COMM_TIME = 5e-2
    N = PARAMETERS['num_intruders']

    graph_type = 'cycle'
    _, adj, A = generate_graph(N, type=graph_type)

    intruders, agents = generate_agents_and_intruders()
    z_0 = agents

    node_list = []
    package_name = "task22"

    for i in range(N):
        A_i = A[i].tolist()
        z_0_i = z_0[i].tolist()
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
                        "z0": z_0_i,
                        "alpha": ALPHA,
                        "neighbors": N_i,
                        "max_iters": MAXITERS,
                        "intruder": intruder_i,
                        "communication_time": COMM_TIME,
                    }
                ],
                output="screen",
                prefix=f'xterm -title "agent_{i}"  -fg white -bg black -fs 12 -fa "Monospace" -hold -e',
            )
        )
    return LaunchDescription(node_list)

generate_launch_description()