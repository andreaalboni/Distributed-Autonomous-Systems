import os
import pickle
from utils import animate_world_evolution

file_to_load = 'evolution_3.pkl'

def get_next_id(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    existing = [f for f in os.listdir(save_dir) if f.endswith('.pkl') and f.startswith('evolution_')]
    ids = [int(f.split('_')[1].split('.')[0]) for f in existing if f.split('_')[1].split('.')[0].isdigit()]
    return max(ids, default=0) + 1

def save_evolution_data(agents, targets, z_history, type, world_size, save_dir='./exam_project/task1/evolution_data'):
    # De-Normalization:
    agents = agents * world_size[0]
    targets = targets * world_size[0]
    z_hystory = z_hystory * world_size[0]

    run_id = get_next_id(save_dir)

    data = {
        'agents': agents,
        'targets': targets,
        'z_history': z_history,
        'type': type
    }
    
    filename = f"id-{run_id}-{len(data['targets'])}-{data['type']}.pkl"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved evolution as {filename}")
    return filename

def load_and_animate_evolution(filename, save_dir='evolution_data'):
    print(f"Loading evolution data from {filename}...")
    filepath = os.path.join(save_dir, filename)

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found in {save_dir}.")
        return

    animate_world_evolution(
        data['agents'],
        data['targets'],
        data['z_history'],
        data['type']
    )


def main():
    load_and_animate_evolution(file_to_load)
    
if __name__ == "__main__":
    main()