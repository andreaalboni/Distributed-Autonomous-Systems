import numpy as np
import matplotlib.pyplot as plt
import warnings
import networkx as nx
import matplotlib.animation as animation

def get_params():
    return {
        'num_targets': 1, # Number of targets
        'ratio_at':4 , # Ratio of agents to targets
        'world_size': [10, 10],
        'radius_fov': np.inf,
        'graph_type': 'cycle',
        'alpha': 1.0,
        'beta': 1.0,
        'num_iters': 1000,   
        'step_size': 0.05
    }


def spawn_candidate(world_size, existing_points):
    while True:
        candidate = np.random.uniform(0, world_size[0], size=2)
        if not any(np.allclose(candidate, p, atol=1e-1) for p in existing_points):
            return candidate

def generate_agents_and_targets(num_targets, ratio_at, world_size):
    targets = []
    agents = []
    for _ in range(num_targets):
        target = spawn_candidate(world_size, targets)
        targets.append(target)
        for _ in range(ratio_at):
            agent = spawn_candidate(world_size, agents + targets)
            agents.append(agent)
    return np.array(targets), np.array(agents)

def generate_agents_and_private_targets(num_agents, world_size):
    agents = []
    targets = []
    for _ in range(num_agents):
        target = spawn_candidate(world_size, agents + targets)
        agent = spawn_candidate(world_size, agents + targets + [target])
        targets.append(target)
        agents.append(agent)
    return np.array(targets), np.array(agents)


def ensure_doubly_stochastic(A):
    A += np.eye(A.shape[0])
    ONE = np.ones_like(A)
    for _ in range(100):
        A = A / (A @ ONE)
        A = A / (ONE.T @ A)
    return np.abs(A)

def generate_graph(num_agents, graph_type='cycle', p_er=0.4):
    if graph_type == 'cycle':
        G = nx.cycle_graph(num_agents)
    elif graph_type == 'path':
        G = nx.path_graph(num_agents)
    elif graph_type == 'erdos_renyi':
        while True:
            G = nx.erdos_renyi_graph(num_agents, p=p_er)
            if nx.is_connected(G):
                break
    else:
        raise ValueError("Unsupported graph type.")
    Adj = nx.to_numpy_array(G)
    A = ensure_doubly_stochastic(Adj)
    return G, Adj, A


def plot_world(agents, targets, world_size):
    plt.figure(figsize=(6,6))
    plt.scatter(agents[:,0], agents[:,1], c='blue', label='Agents')
    plt.scatter(targets[:,0], targets[:,1], c='red', marker='x', label='Targets')
    plt.xlim(0, world_size[0])
    plt.ylim(0, world_size[1])
    plt.legend()
    plt.title('Initial world state')
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()

def plot_graph(G):
    plt.figure(figsize=(4, 4))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title('Communication graph')
    plt.show()

def plot_world_with_connections(agents, targets, world_size):
    plt.figure(figsize=(6, 6))
    
    # Disegna agenti
    plt.scatter(agents[:, 0], agents[:, 1], c='blue', label='Agents', s=60)
    
    # Disegna target
    plt.scatter(targets[:, 0], targets[:, 1], c='red', marker='x', label='Targets', s=60)
    
    # Connessioni agent-target
    for i in range(len(agents)):
        plt.plot(
            [agents[i, 0], targets[i, 0]],
            [agents[i, 1], targets[i, 1]],
            'gray', linestyle='--', linewidth=1
        )
    
    # Aspetto grafico
    plt.xlim(0, world_size[0])
    plt.ylim(0, world_size[1])
    plt.gca().set_aspect('equal')
    plt.title("Agents, Targets, and Private Connections")
    plt.legend()
    plt.grid(True)
    plt.show()




def main():
    params = get_params()
    num_targets = params['num_targets']
    # num_agents = params['num_agents']
    ratio_at = params['ratio_at']
    world_size = params['world_size']

    
    targets, agents = generate_agents_and_targets(num_targets, ratio_at, world_size)   
    G, Adj, A = generate_graph(len(agents), graph_type=params['graph_type'])
    
    print('\n Adj_Matrix:\n ', Adj)
    
    #plot_world(agents, targets, world_size)
    #plot_graph(G)

    targets_one, agents_one = generate_agents_and_private_targets(ratio_at, world_size)
    G_one, Adj_one, A_one = generate_graph(len(agents_one), graph_type=params['graph_type'])

    print('\n Adj_Matrix_one:\n ', Adj_one)

    plot_world(agents_one, targets_one, world_size)
    plot_world_with_connections(agents_one, targets_one, world_size)
    plot_graph(G_one)

  



if __name__ == "__main__":
    main()