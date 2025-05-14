import warnings
import numpy as np
import networkx as nx
from config import PARAMETERS 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')


def get_default_params():
    return PARAMETERS

def is_in_fov(agent_pos, target_pos, radius_fov=PARAMETERS['radius_fov']):
    return np.linalg.norm(agent_pos - target_pos) <= radius_fov
        
def doll_cost_function(z, Q, z_ref):
    diff = z - z_ref
    cost = np.linalg.norm(diff.T @ Q @ diff, 'fro')**2
    grad = 2 * Q @ diff
    return cost, grad

def spawn_agent_near_target(target, existing_agents, existing_targets, world_size=PARAMETERS['world_size'], d=PARAMETERS['d']):
    while True:
        candidate = np.random.uniform(0, world_size, size=d)
        if (is_in_fov(candidate, target) and 
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_targets)):
            return candidate
        
def spawn_candidate(existing_agents, existing_targets, world_size=PARAMETERS['world_size'], d=PARAMETERS['d']):
    while True:
        candidate = np.random.uniform(0, world_size, size=d)
        if (not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and 
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_targets)):
            return candidate

def generate_agents_and_targets(num_targets=PARAMETERS['num_targets'], ratio_at=PARAMETERS['ratio_at'], world_size=PARAMETERS['world_size']):
    targets = []
    agents = []
    # Spawn targets and required agents
    for _ in range(num_targets):
        target = spawn_candidate(agents, targets)
        targets.append(target)
        visible_agents = sum(is_in_fov(agent, target) for agent in agents)
        for _ in range(3 - visible_agents):
            agents.append(spawn_agent_near_target(target, agents, targets))
    # Add remaining agents randomly
    total_agents_needed = int(num_targets * ratio_at)
    while len(agents) < total_agents_needed:
        candidate = spawn_candidate(agents, targets)
        agents.append(candidate)
    if len(agents) > total_agents_needed:
        warnings.warn(f"\033[38;5;214mNumber of agents ({len(agents)}) exceeds the required number ({total_agents_needed}).\033[0m")
    
    # Normalization:
    targets = np.array(targets) / world_size
    agents = np.array(agents) / world_size
    return targets, agents

def get_distances(agents, targets, noise_level=PARAMETERS['noise_level'], bias_param=PARAMETERS['bias'], radius_fov=PARAMETERS['radius_fov'], world_size=PARAMETERS['world_size']):
    distances = []
    noisy_distances = []
    for agent in agents:
        agent_distance = []
        noisy_distance_to_target = []
        for target in targets:
            dist = np.linalg.norm(agent - target)
            if dist > radius_fov:
                agent_distance.append(np.nan)
            else:
                agent_distance.append(dist)
                bias = np.random.uniform(-bias_param, bias_param)
                var = np.random.normal(0, noise_level)
                noise = (bias + var) / world_size
                noisy_distance_to_target.append(dist + noise)
        distances.append(agent_distance)
        noisy_distances.append(noisy_distance_to_target)
    return np.array(distances), np.array(noisy_distances)

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
        if np.isnan(distances_i[target]):
            local_cost += 0
            local_cost_gradient[target, :] = np.nan * np.zeros_like(p_i)
        else:
            estimated_distance_squared = np.linalg.norm(z[target] - p_i)**2
            measured_distance_squared = distances_i[target]**2
            local_cost += (measured_distance_squared - estimated_distance_squared)**2 
            
            # Gradient evaluation
            #print(4 * (estimated_distance_squared - measured_distance_squared) * (z[target] - p_i))
            local_cost_gradient[target, :] = 4 * (estimated_distance_squared - measured_distance_squared) * (z[target] - p_i)
    return local_cost, local_cost_gradient
    
def visualize_graph(G):
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True)
    plt.show()
    
def visualize_world(agents, targets, world_size=PARAMETERS['world_size'], d=PARAMETERS['d']):
    if d <= 3 and d > 0:
        # De-Normalization:
        agents = agents * world_size
        targets = targets * world_size
        
        fig = plt.figure(figsize=(8, 8))
        if d == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
            
        ax.set_title('World visualization')
        padding = 0.2
        
        if d == 1:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_yticks([])  # Hide y-axis ticks
            ax.scatter(agents, np.zeros_like(agents), c='blue', marker='o', s=50, label='Agent')
            ax.scatter(targets, np.zeros_like(targets), c='red', marker='x', s=50, label='Target')
            ax.grid(False)
            ax.set_aspect('equal')
            
        elif d == 2:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_ylim(-padding * world_size, world_size * (1 + padding))
            ax.scatter(agents[:, 0], agents[:, 1], c='blue', marker='o', s=50, label='Agent')
            ax.scatter(targets[:, 0], targets[:, 1], c='red', marker='x', s=50, label='Target')
            ax.grid(True)
            ax.set_aspect('equal')
            
        elif d == 3:
            # Set limits for all three axes with equal range
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_ylim(-padding * world_size, world_size * (1 + padding))
            ax.set_zlim(-padding * world_size, world_size * (1 + padding))
            
            # Plot agents and targets
            ax.scatter(agents[:, 0], agents[:, 1], agents[:, 2], c='blue', marker='o', s=50, label='Agent')
            ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c='red', marker='x', s=50, label='Target')
            
            # Set equal aspect ratio for 3D plots
            # This is the key fix - we need to set equal scaling for all three axes
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
            
            # Update the limits
            ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
            ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
            ax.set_zlim3d([mid_z - max_range/2, mid_z + max_range/2])
            
            # Enable grid
            ax.grid(True)
            
            # Add axis labels for better orientation
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
        ax.legend()
        plt.show()
    else:
        print(f"Visualization only supports dimensions 1-3. Current dimension: {d}")
        return None

def metropolis_hastings_weights(G):
    r"""
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

def animate_world_evolution(agents, targets, z_history, type, world_size=PARAMETERS['world_size'], d=PARAMETERS['d'], speed=4):
    if d > 0 and d <= 3:
        agents = agents * world_size
        targets = targets * world_size

        z_history = z_history * world_size
        T, n_agents, n_targets, _ = z_history.shape
        frame_skip = int(speed)
        frame_skip += 1
        num_steps = T // frame_skip
        positions = z_history[::frame_skip]
        pause_frames = int(3 * 20)
        positions = np.concatenate([positions, np.repeat(positions[-1:], pause_frames, axis=0)])
        num_steps = len(positions)
        fig = plt.figure(figsize=(8, 8))
        if d == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        ax.set_title(f'Agents to Targets Animation - {type} Graph')
        padding = 0.2
        if d == 1:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_yticks([])
            ax.scatter(agents, np.zeros_like(agents), c='blue', marker='o', s=50, label='Agent')
            ax.scatter(targets, np.zeros_like(targets), c='red', marker='x', s=50, label='Target')
            ax.grid(False)
        elif d == 2:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_ylim(-padding * world_size, world_size * (1 + padding))
            ax.scatter(agents[:, 0], agents[:, 1], c='blue', marker='o', s=50, label='Agent')
            ax.scatter(targets[:, 0], targets[:, 1], c='red', marker='x', s=50, label='Target')
            ax.grid(True)
        elif d == 3:
            ax.set_xlim(-padding * world_size, world_size * (1 + padding))
            ax.set_ylim(-padding * world_size, world_size * (1 + padding))
            ax.set_zlim(-padding * world_size, world_size * (1 + padding))
            ax.scatter(agents[:, 0], agents[:, 1], agents[:, 2], c='blue', marker='o', s=50, label='Agent')
            ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c='red', marker='x', s=50, label='Target')
            ax.grid(True)
            ax.scatter([], [], [], c='green', s=20, label='Estimation')
        ax.legend()
        if d != 3:
            ax.set_aspect('equal')
        if d == 3:
            estimation_points = None
            def update(frame):
                nonlocal estimation_points
                if estimation_points is not None:
                    for point in estimation_points:
                        point.remove()
                
                pos = positions[frame]
                x_data = []
                y_data = []
                z_data = []
                for i in range(n_agents):
                    for j in range(n_targets):
                        x_data.append(pos[i, j, 0])
                        y_data.append(pos[i, j, 1])
                        z_data.append(pos[i, j, 2])
                
                estimation_points = [ax.scatter(x_data, y_data, z_data, c='green', s=20)]
                return estimation_points
            anim = FuncAnimation(
                fig, update,
                frames=num_steps,
                interval=50,
                repeat=True
            )
        else:
            scatters = []
            for _ in range(n_agents * n_targets):
                if d == 1:
                    scatter = ax.scatter([], [], c='green', s=20)
                elif d == 2:
                    scatter = ax.scatter([], [], c='green', s=20)
                scatters.append(scatter)
            def init():
                for scatter in scatters:
                    scatter.set_offsets(np.empty((0, 2)))
                return scatters
            def update(frame):
                pos = positions[frame]
                index = 0
                for i in range(n_agents):
                    for j in range(n_targets):
                        if d == 1:
                            scatters[index].set_offsets(np.column_stack((pos[i, j], 0)))
                        elif d == 2:
                            scatters[index].set_offsets(pos[i, j])
                        index += 1
                return scatters
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
    else:
        print(f"Animation only supports dimensions 1-3. Current dimension: {d}")
        return None