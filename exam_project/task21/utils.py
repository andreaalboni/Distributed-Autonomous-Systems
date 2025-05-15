import matplotlib
import numpy as np
import networkx as nx
from config import PARAMETERS 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')

def get_default_params():
    return PARAMETERS

def debug_spawn_agent_near_intruder(intruder, existing_agents, existing_intruders, world_size=PARAMETERS['world_size'], radius_spawn_agent=PARAMETERS['radius_spawn_agent'], d=PARAMETERS['d']):
    while True:
        candidate = np.random.uniform(0, world_size, size=d)
        center = world_size/2 * np.ones_like(candidate)
        if (np.linalg.norm(candidate - intruder) <= radius_spawn_agent and
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_intruders)):
            return candidate

def debug_spawn_candidate(existing_agents, existing_intruders, world_size=PARAMETERS['world_size'], intruder_radius=PARAMETERS['intruder_radius'], d=PARAMETERS['d']):
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

def spawn_agent_near_intruder(intruder, existing_agents, existing_intruders, world_size=PARAMETERS['world_size'], radius_spawn_agent=PARAMETERS['radius_spawn_agent'], d=PARAMETERS['d']):
    while True:
        candidate = np.random.uniform(0, world_size, size=d)
        if (np.linalg.norm(candidate - intruder) <= radius_spawn_agent and
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_intruders)):
            return candidate
        
def spawn_candidate(existing_agents, existing_intruders, world_size=PARAMETERS['world_size'], d=PARAMETERS['d']):
    while True:
        candidate = np.random.uniform(0, world_size, size=d)
        if (not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and 
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_intruders)):
            return candidate

def generate_agents_and_intruders(num_intruders=PARAMETERS['num_intruders'], world_size=PARAMETERS['world_size']):
    intruders = []
    agents = []
    for _ in range(num_intruders):
        intruder = debug_spawn_candidate(agents, intruders, world_size=world_size)
        intruders.append(intruder)
        agent = debug_spawn_agent_near_intruder(intruder, agents, intruders, world_size=world_size)
        agents.append(agent)
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

def compute_r_0(intruders, noise_radius=PARAMETERS['noise_r_0'], world_size=PARAMETERS['world_size'], d=PARAMETERS['d']):
    barycenter_intruders = compute_agents_barycenter(intruders)
    if (noise_radius == 0.0):
        return barycenter_intruders
    while True:
        r_0_candidate = np.random.uniform(0, world_size, size=d)
        if (np.linalg.norm(r_0_candidate - barycenter_intruders) <= noise_radius and
            np.linalg.norm(r_0_candidate - barycenter_intruders) >= noise_radius/10):
                return r_0_candidate

def local_cost_function(agent_i, intruder_i, sigma, r_0, gamma_i, gamma_bar_i, gamma_hat_i):
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
    
    local_cost = gamma_i * agent_to_intruder + gamma_bar_i * agent_to_sigma + gamma_hat_i * intruder_to_r_0
    
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
    
def visualize_world(agents, intruders, world_size=PARAMETERS['world_size'], d=PARAMETERS['d']):
    if d <= 3 and d > 1:
        fig = plt.figure(figsize=(10, 8))
        if d == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        ax.set_title('World visualization')
        padding = 0.2
        
        if d == 2:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_ylim(-padding * world_size, world_size * (1 + padding))
            ax.scatter(agents[:, 0], agents[:, 1], c='black', marker='o', s=50, label='Agent')
            ax.scatter(intruders[:, 0], intruders[:, 1], c='cyan', marker='s', s=50, label='Intruder', edgecolors='cyan', facecolors='none')
            
            sigma = compute_agents_barycenter(agents)
            ax.scatter(sigma[0], sigma[1], c='green', marker='^', s=50, label='Sigma', edgecolors='mediumseagreen', facecolors='none')
            
            target = compute_r_0(intruders)
            intruders_barycenter = compute_agents_barycenter(intruders)
            
            if not np.array_equal(intruders_barycenter, target):
                ax.scatter(intruders_barycenter[0], intruders_barycenter[1], c='purple', alpha=0.35, marker='x', s=50, label='Intruders\' CoG')
            
            ax.scatter(target[0], target[1], c='red', marker='x', s=50, label=r'$r_0$')
            ax.grid(True)
            ax.set_aspect('equal')
            
        elif d == 3:
            # Set limits for all three axes with equal range
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_ylim(-padding * world_size, world_size * (1 + padding))
            ax.set_zlim(-padding * world_size, world_size * (1 + padding))
            
            # Plot agents as black circles
            ax.scatter(agents[:, 0], agents[:, 1], agents[:, 2], c='black', marker='o', s=50, label='Agent')
            
            # Plot intruders as cyan squares with transparent fill
            ax.scatter(intruders[:, 0], intruders[:, 1], intruders[:, 2], facecolors='none', edgecolors='cyan', marker='s', s=50, label='Intruder')
            
            # Calculate and plot the center of gravity of agents (sigma)
            sigma = compute_agents_barycenter(agents)
            ax.scatter(sigma[0], sigma[1], sigma[2], facecolors='none', edgecolors='mediumseagreen', marker='^', s=50, label='Sigma')
            
            # Calculate and plot the target point (r_0)
            target = compute_r_0(intruders)
            intruders_barycenter = compute_agents_barycenter(intruders)
            
            # Plot the barycenter of intruders if different from target
            if not np.allclose(intruders_barycenter, target):  # Using allclose instead of array_equal for floating point comparison
                ax.scatter(intruders_barycenter[0], intruders_barycenter[1], intruders_barycenter[2], 
                           c='purple', alpha=0.35, marker='x', s=50, label='Intruders\' CoG')
            
            # Plot the target point
            ax.scatter(target[0], target[1], target[2], c='red', marker='x', s=50, label=r'$r_0$')
            
            # Set equal aspect ratio for 3D plots
            # Get the limits for normalization
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()
            
            # Get the range of each dimension
            x_range = abs(x_limits[1] - x_limits[0])
            y_range = abs(y_limits[1] - y_limits[0])
            z_range = abs(z_limits[1] - z_limits[0])
            
            # Find the greatest range for normalization
            max_range = max(x_range, y_range, z_range)
            
            # Set the axes centrally in the figure using the max range
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
        plt.tight_layout()
        plt.show()
    else:
        print(f"Visualization only supports dimensions 1-3. Current dimension: {d}")
        return None

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
    
def animate_world_evolution(intruders, z_history, s, sigma, r_0, world_size, d, speed=10):
    """
    Animate the evolution of agents in the world.
    
    Parameters:
    - intruders: Array of intruder positions
    - z_history: History of agent positions (T, n_agents, d)
    - s: History of target positions
    - sigma: Initial sigma (not used, will be computed from agents)
    - r_0: Reference point
    - world_size: Size of the world
    - d: Dimensionality of the space (2 or 3)
    - speed: Animation speed
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    # Compute barycenter for each timestep
    def compute_agents_barycenter(positions):
        return np.mean(positions, axis=0)
    
    T, n_agents, *_ = z_history.shape
    frame_skip = int(speed) + 1
    positions = z_history[::frame_skip]
    s_positions = s[::frame_skip]
    
    # Calculate sigma (barycenter) for each frame
    sigma_positions = np.array([compute_agents_barycenter(pos) for pos in positions])
    
    # Add pause at the end
    pause_frames = int(3 * 20)
    positions = np.concatenate([positions, np.repeat(positions[-1:], pause_frames, axis=0)])
    s_positions = np.concatenate([s_positions, np.repeat(s_positions[-1:], pause_frames, axis=0)])
    sigma_positions = np.concatenate([sigma_positions, np.repeat(sigma_positions[-1:], pause_frames, axis=0)])
    
    num_steps = len(positions)
    fig = plt.figure(figsize=(10, 8))
    
    if d == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel('Z')
    else:
        ax = fig.add_subplot(111)
        
    ax.set_title('Agents to Targets Animation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    padding = 0.2
    x_min, x_max = 0, world_size
    y_min, y_max = 0, world_size
    z_min, z_max = 0, world_size if d == 3 else 0
    
    ax.set_xlim(x_min - padding * x_max, x_max + padding * x_max)
    ax.set_ylim(y_min - padding * y_max, y_max + padding * y_max)
    if d == 3:
        ax.set_zlim(z_min - padding * z_max, z_max + padding * z_max)

    # Initial plots
    if d == 3:
        # Plot intruders
        intruder_plot = ax.scatter(
            intruders[:, 0], intruders[:, 1], intruders[:, 2], 
            c='cyan', marker='s', s=50, label='Intruder', 
            edgecolors='cyan', facecolors='none'
        )
        
        # Plot reference point
        ref_point = ax.scatter(
            r_0[0], r_0[1], r_0[2], 
            c='red', marker='x', s=50, label=r'$r_0$'
        )
        
        # Plot initial sigma (barycenter)
        sigma_scatter = ax.scatter(
            sigma_positions[0, 0], sigma_positions[0, 1], sigma_positions[0, 2],
            facecolors='none', edgecolors='mediumseagreen', marker='^', s=50, label='Sigma'
        )
        
        # Plot agents
        agent_scatter = ax.scatter(
            positions[0, :, 0], positions[0, :, 1], positions[0, :, 2], 
            c='black', marker='o', s=50, label='Agent'
        )
        
        # Create path lines for agents
        path_lines = []
        for i in range(n_agents):
            line = ax.plot(
                [positions[0, i, 0]], [positions[0, i, 1]], [positions[0, i, 2]],
                'gray', linestyle='--', alpha=0.5
            )[0]
            path_lines.append(line)
            
        # Create path line for sigma
        sigma_path = ax.plot(
            [sigma_positions[0, 0]], [sigma_positions[0, 1]], [sigma_positions[0, 2]],
            'mediumseagreen', linestyle='--', alpha=0.5
        )[0]
        
        # Update function
        def update(frame):
            # Update agent positions
            agent_scatter._offsets3d = (
                positions[frame, :, 0], 
                positions[frame, :, 1], 
                positions[frame, :, 2]
            )
            
            # Update sigma position
            sigma_scatter._offsets3d = (
                [sigma_positions[frame, 0]], 
                [sigma_positions[frame, 1]], 
                [sigma_positions[frame, 2]]
            )
            
            # Update agent path lines
            for i, line in enumerate(path_lines):
                x_data = positions[:frame+1, i, 0]
                y_data = positions[:frame+1, i, 1]
                z_data = positions[:frame+1, i, 2]
                line.set_data(x_data, y_data)
                line.set_3d_properties(z_data)
                
            # Update sigma path line
            x_data = sigma_positions[:frame+1, 0]
            y_data = sigma_positions[:frame+1, 1]
            z_data = sigma_positions[:frame+1, 2]
            sigma_path.set_data(x_data, y_data)
            sigma_path.set_3d_properties(z_data)
            
            return [agent_scatter, sigma_scatter, sigma_path] + path_lines
            
    else:  # 2D case
        # Plot intruders
        intruder_plot = ax.scatter(
            intruders[:, 0], intruders[:, 1], 
            c='cyan', marker='s', s=50, label='Intruder', 
            edgecolors='cyan', facecolors='none'
        )
        
        # Plot reference point
        ref_point = ax.scatter(
            r_0[0], r_0[1], 
            c='red', marker='x', s=50, label=r'$r_0$'
        )
        
        # Plot initial sigma (barycenter)
        sigma_scatter = ax.scatter(
            sigma_positions[0, 0], sigma_positions[0, 1],
            facecolors='none', edgecolors='mediumseagreen', marker='^', s=50, label='Sigma'
        )
        
        # Plot agents
        agent_scatter = ax.scatter(
            positions[0, :, 0], positions[0, :, 1], 
            c='black', marker='o', s=50, label='Agent'
        )
        
        # Create path lines for agents
        path_lines = []
        for i in range(n_agents):
            line = ax.plot(
                [positions[0, i, 0]], [positions[0, i, 1]],
                'gray', linestyle='--', alpha=0.5
            )[0]
            path_lines.append(line)
            
        # Create path line for sigma
        sigma_path = ax.plot(
            [sigma_positions[0, 0]], [sigma_positions[0, 1]],
            'mediumseagreen', linestyle='--', alpha=0.5
        )[0]
        
        # Update function
        def update(frame):
            # Update agent positions
            agent_scatter.set_offsets(positions[frame])
            
            # Update sigma position
            sigma_scatter.set_offsets([sigma_positions[frame]])
            
            # Update agent path lines
            for i, line in enumerate(path_lines):
                x_data = positions[:frame+1, i, 0]
                y_data = positions[:frame+1, i, 1]
                line.set_data(x_data, y_data)
                
            # Update sigma path line
            x_data = sigma_positions[:frame+1, 0]
            y_data = sigma_positions[:frame+1, 1]
            sigma_path.set_data(x_data, y_data)
            
            return [agent_scatter, sigma_scatter, sigma_path] + path_lines
    
    ax.legend()
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=num_steps, 
        blit=True, interval=50, repeat=True
    )
    
    if d == 3:
        ax.view_init(elev=30, azim=45)
        
    plt.tight_layout()
    plt.show()
    
    return anim
