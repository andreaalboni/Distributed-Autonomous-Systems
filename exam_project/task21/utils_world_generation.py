import numpy as np

def debug_spawn_agent_near_intruder(intruder, existing_agents, existing_intruders, world_size, radius_spawn_agent, d):
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

def spawn_agent_near_intruder(intruder, existing_agents, existing_intruders, world_size, radius_spawn_agent, d):
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

def generate_agents_and_intruders(num_intruders, world_size, radius_spawn_agent, d, intruder_radius=None):
    intruders = []
    agents = []
    for _ in range(num_intruders):
        intruder = debug_spawn_candidate(agents, intruders, world_size, intruder_radius, d)
        intruders.append(intruder)
        agent = debug_spawn_agent_near_intruder(intruder, agents, intruders, world_size, radius_spawn_agent, d)
        agents.append(agent)
    return np.array(intruders), np.array(agents)