import warnings
import numpy as np

def is_in_fov(agent_pos, target_pos, radius_fov):
    """Check if the target position is within the agent's field of view radius."""
    return np.linalg.norm(agent_pos - target_pos) <= radius_fov

def spawn_agent_near_target(target, existing_agents, existing_targets, world_size, d, radius_fov):
    """Spawns an agent near a target within a specified field of view, avoiding overlap with existing agents and targets."""
    while True:
        candidate = np.random.uniform(0, world_size, size=d)
        if (is_in_fov(candidate, target, radius_fov) and 
            not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_targets)):
            return candidate
        
def spawn_candidate(existing_agents, existing_targets, world_size, d):
    """Generates a random candidate position within the world that does not overlap with existing agents or targets."""
    while True:
        candidate = np.random.uniform(0, world_size, size=d)
        if (not any(np.allclose(candidate, a, atol=1e-1) for a in existing_agents) and 
            not any(np.allclose(candidate, t, atol=1e-1) for t in existing_targets)):
            return candidate

def generate_agents_and_targets(num_targets, ratio_at, world_size, d, radius_fov):
    """
    Generate positions for agents and targets in a 2D world.
    Args:
        num_targets (int): Number of targets to generate.
        ratio_at (float): Ratio of agents to targets.
        world_size (float): Size of the world (assumed square).
        d (float): Minimum distance between entities.
        radius_fov (float): Field of view radius for agents.
    Returns:
        tuple: Normalized numpy arrays of target positions and agent positions.
    """
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

def get_distances(agents, targets, noise_level, bias_param, radius_fov, world_size, task):
    """
    Compute true and noisy distances between agents and targets.

    Args:
        agents (np.ndarray): Array of agent positions.
        targets (np.ndarray): Array of target positions.
        noise_level (float): Standard deviation of Gaussian noise.
        bias_param (float): Maximum absolute value for uniform bias.
        radius_fov (float): Field of view radius; distances beyond this are set to NaN.
        world_size (float): Size of the world, used to scale noise.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - distances: True distances between agents and targets (NaN if out of FOV).
            - noisy_distances: Noisy distances with bias and noise added (NaN if out of FOV).
    """
    distances = []
    noisy_distances = []
    for agent in agents:
        agent_distance = []
        noisy_distance_to_target = []
        for target in targets:
            dist = np.linalg.norm(agent - target)
            if dist > radius_fov:
                agent_distance.append(np.nan)
                noisy_distance_to_target.append(np.nan)
            else:
                agent_distance.append(dist)
                bias = np.random.uniform(-bias_param, bias_param)
                var = np.random.normal(0, noise_level)
                noise = (bias + var) / world_size
                noisy_distance_to_target.append(dist + noise)
            if task == '1.1':
                break
        distances.append(agent_distance)
        noisy_distances.append(noisy_distance_to_target)
    return np.array(distances), np.array(noisy_distances)
