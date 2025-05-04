import numpy as np
import matplotlib.pyplot as plt
import warnings
import networkx as nx


def get_default_params():
    return {
        'num_targets': 5,
        'ratio_at': 5,
        'world_size': [10, 10],
        'radius_fov': 3,
        'noise_level': 0.1,
        'bias': 0.0
    }


def is_in_fov(agent_pos, target_pos, radius_fov):
    return np.linalg.norm(agent_pos - target_pos) <= radius_fov


def spawn_agent_near_target(target, world_size, radius_fov, existing_agents, existing_targets):
    while True:
        candidate = np.random.randint(0, world_size[0], size=2)
        if (is_in_fov(candidate, target, radius_fov) and 
            not any(np.array_equal(candidate, a) for a in existing_agents) and
            not any(np.array_equal(candidate, t) for t in existing_targets)):
            return candidate
        
        
def spawn_candidate(world_size, existing_agents, existing_targets):
    while True:
        candidate = np.random.randint(0, world_size[0], size=2)
        if (not any(np.array_equal(candidate, a) for a in existing_agents) and 
            not any(np.array_equal(candidate, t) for t in existing_targets)):
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

def generate_adjacency_matrix(distances, radius_fov):
    number_of_agents = len(distances)
    number_of_targets = len(distances[0])
    number_of_nodes = number_of_agents + number_of_targets
    # the adjacency matrix will be composed by
    # the first part being the agents and 
    # the second part being the targets
    adj = np.zeros((number_of_nodes, number_of_nodes))
    for agent in range(number_of_agents):
        for target in range(number_of_targets):
            if distances[agent][target] <= radius_fov:
                adj[agent,number_of_agents + target] = 1
                adj[number_of_agents + target, agent] = 1
        
    return adj

def generate_graph(adj):
    G = nx.from_numpy_array(adj)
    return G

def visualize_graph(G, num_targets):
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42, k=0.7, iterations=100)

    num_nodes = G.number_of_nodes()
    num_agents = num_nodes - num_targets

    agents = list(range(num_agents))
    targets = list(range(num_agents, num_nodes))

    nx.draw_networkx_nodes(G, pos, nodelist=agents, node_color='skyblue', node_shape='o', label='Agents')
    nx.draw_networkx_nodes(G, pos, nodelist=targets, node_color='salmon', node_shape='s', label='Targets')
    nx.draw_networkx_edges(G, pos)

    labels = {i: f"A{i}" if i < num_agents else f"T{i - num_agents}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title("Agent-Target Graph")
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
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



def main():
    params = get_default_params()
    targets, agents = generate_agents_and_targets(
        num_targets=params['num_targets'],
        ratio_at=params['ratio_at'],
        world_size=params['world_size'],
        radius_fov=params['radius_fov']
    )

    visualize_world(agents, targets, world_size=params['world_size'])

    real_distances, noisy_distances = get_distances(agents, targets)

    print("Distances:\n", real_distances)
    
    adj = generate_adjacency_matrix(real_distances, params['radius_fov'])
    
    G = generate_graph(adj)
    visualize_graph(G, params['num_targets'])
    
if __name__ == "__main__":
    main()
