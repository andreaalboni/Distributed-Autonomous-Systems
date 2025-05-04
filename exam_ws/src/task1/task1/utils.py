import numpy as np
import matplotlib.pyplot as plt

def get_default_params():
    return {
        'num_targets': 3,
        'ratio_at': 5,
        'world_size': [10, 10],
        'noise_level': 0.1
    }


def generate_agents_and_targets(num_targets, ratio_at=5, world_size=[10,10]):
    targets = np.random.randint(0, world_size[0], size=(num_targets, 2))
    num_agents = int(num_targets * ratio_at)
    while True:
        agents = np.random.randint(0, world_size[0], size=(num_agents, 2))
        if not np.any(np.all(agents[:, None] == targets, axis=2)):
            break
    return targets, agents


def visualize_world(agents, targets, world_size=[10,10]):
    plt.figure(figsize=(8, 8))
    for agent in agents:
        plt.scatter(agent[0], agent[1], c='blue', marker='o', label='Agent')
    for target in targets:
        plt.scatter(target[0], target[1], c='red', marker='+', label='Target')
    plt.xlim(0, world_size[0])
    plt.ylim(0, world_size[1])
    plt.title('World Visualization')
    plt.show()
    
    
def get_distances(agents, targets, normal=True, noise_level=0.1, bias=0.0):
    distances = []
    for agent in agents:
        agent_distance = []
        for target in targets:
            agent_distance.append(np.linalg.norm(agent - target))
        distances.append(agent_distance)
    noisy_distances = np.array(distances) + np.random.normal(bias, noise_level, np.array(distances).shape)
    return distances, noisy_distances


def main():
    t, a = generate_agents_and_targets(5)
    visualize_world(a, t)
    distances, noisy_distances = get_distances(a, t, noise_level=0.1)
    
    
if __name__ == "__main__":
    main()