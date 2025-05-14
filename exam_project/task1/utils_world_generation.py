import warnings
import numpy as np

def is_in_fov(agent_pos, target_pos, radius_fov):
    return np.linalg.norm(agent_pos - target_pos) <= radius_fov

def spawn_agent_near_target(target, existing_agents, existing_targets, world_size, d, radius_fov):
    while True:
        candidate = np.random.uniform(0, world_size, size=d)
        if (is_in_fov(candidate, target, radius_fov) and 
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_targets)):
            return candidate
        
def spawn_candidate(existing_agents, existing_targets, world_size, d):
    while True:
        candidate = np.random.uniform(0, world_size, size=d)
        if (not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and 
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_targets)):
            return candidate

def generate_agents_and_targets(num_targets, ratio_at, world_size, d, radius_fov):
    targets = []
    agents = []
    # Spawn targets and required agents
    for _ in range(num_targets):
        target = spawn_candidate(agents, targets, world_size, d)
        targets.append(target)
        visible_agents = sum(is_in_fov(agent, target, radius_fov) for agent in agents)
        for _ in range(3 - visible_agents):
            agents.append(spawn_agent_near_target(target, agents, targets, world_size, d, radius_fov))
    # Add remaining agents randomly
    total_agents_needed = int(num_targets * ratio_at)
    while len(agents) < total_agents_needed:
        candidate = spawn_candidate(agents, targets, world_size, d)
        agents.append(candidate)
    if len(agents) > total_agents_needed:
        warnings.warn(f"\033[38;5;214mNumber of agents ({len(agents)}) exceeds the required number ({total_agents_needed}).\033[0m")
    
    # Normalization:
    targets = np.array(targets) / world_size
    agents = np.array(agents) / world_size
    return targets, agents

def get_distances(agents, targets, noise_level, bias_param, radius_fov, world_size):
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
