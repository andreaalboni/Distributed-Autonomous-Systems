import numpy as np
import matplotlib.pyplot as plt

def get_default_params():
    return {
        'num_targets': 5,
        'ratio_at': 5,
        'world_size': [10, 10],
        'noise_level': 0.1,
        'bias': 0.0,
        'radius_fov': np.inf,
    }

_params = get_default_params()

def set_params(params_to_update):
    global _params
    valid_params = set(_params.keys())
    invalid_params = set(params_to_update.keys()) - valid_params
    
    if invalid_params:
        raise ValueError(f"Invalid parameters: {invalid_params}. Valid parameters are: {valid_params}")
    
    _params.update(params_to_update)

def generate_targets():
    num_targets = _params['num_targets']
    world_size = _params['world_size']
    
    targets = np.random.randint(0, world_size[0], size=(num_targets, 2))
    return targets


def generate_agents_for_targets(targets, min_agents_per_target=5):
    agents = []
    world_size = _params['world_size']
    
    for target in targets:
        for _ in range(min_agents_per_target):
            
            offset = np.random.uniform(-1.5, 1.5, size=2)
            new_agent = target + offset
            new_agent = np.clip(new_agent, 0, world_size[0])  
            agents.append(new_agent)
    
    return np.array(agents)


def visualize_world(agents, targets):
    world_size = _params['world_size']
    
    plt.figure(figsize=(8, 8))
    for agent in agents:
        plt.scatter(agent[0], agent[1], c='blue', marker='o', label='Agent')
    for target in targets:
        plt.scatter(target[0], target[1], c='red', marker='+', label='Target')
    plt.xlim(0, world_size[0])
    plt.ylim(0, world_size[1])
    plt.title('World Visualization')
    plt.show()


def get_distances(agents, targets):
    radius_fov = _params['radius_fov']
    noise_level = _params['noise_level']
    bias = _params['bias']
    
    distances = []
    for agent in agents:
        agent_distance = []
        for target in targets:
            dist = np.linalg.norm(agent - target)
            if dist <= radius_fov:
                agent_distance.append(dist)
            else:
                agent_distance.append(np.nan)
        distances.append(agent_distance)
    noisy_distances = np.array(distances) + np.random.normal(bias, noise_level, np.array(distances).shape)
    return np.array(distances), np.array(noisy_distances)



def main():
    set_params({
        'num_targets': 5,
        'world_size': [50,50],
        'radius_fov': 2.0
    })
    
    targets = generate_targets()
    agents = generate_agents_for_targets(targets, min_agents_per_target=3)

    distances, noisy_distances = get_distances(agents, targets)
    visualize_world(agents, targets)

if __name__ == "__main__":
    main()