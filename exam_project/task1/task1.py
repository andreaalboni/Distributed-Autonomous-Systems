import json
import numpy as np
import tkinter as tk
from utils_graph import *
from cost_functions import *
from utils_visualization import *
from tkinter import ttk, messagebox
from utils_world_generation import *
from gradient_tracking import gradient_tracking_method

class Task1:
    def __init__(self, root):
        self.root = root
        self.root.title("Task 1")
        self.root.geometry("600x600")
        
        # Default parameters
        self.default_params = {
            'num_targets': 2,
            'ratio_at': 5,
            'd': 4,
            'world_size': 5,
            'radius_fov': np.inf,
            'noise_level': 0.0,
            'bias': 0.0,
            'p_er': 0.5,
            'graph_type': 'cycle',
        }
        self.params = self.default_params.copy()
        self.task_has_run = False
        self.setup_ui()
        
        
    def setup_ui(self):
        # Main frame
        self.frame = ttk.Frame(self.root, padding="20")
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(self.frame, text="Task 1", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Parameters section
        params_frame = ttk.LabelFrame(self.frame, text="Parameters")
        params_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
        # Create parameter input fields
        self.param_entries = {}
        row = 0
        
        for i, (param, value) in enumerate(self.params.items()):
            ttk.Label(params_frame, text=f"{param}:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
            
            # Special handling for infinity
            if param == 'radius_fov' and value == np.inf:
                display_value = "Infinity"
            else:
                display_value = value
                
            entry = ttk.Entry(params_frame)
            entry.insert(0, str(display_value))
            entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
            self.param_entries[param] = entry
            
            row += 1
        
        # Add padding to all children in the parameters frame
        for child in params_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)
            
        # Buttons Frame
        button_frame = ttk.Frame(self.frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Parameter management buttons
        ttk.Button(button_frame, text="Update Parameters", command=self.update_parameters).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Reset to Default", command=self.reset_to_default).grid(row=0, column=1, padx=5)
        
        # Visualization and Task buttons
        visualization_frame = ttk.LabelFrame(self.frame, text="Visualization & Tasks")
        visualization_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
        # Create buttons but store references to them
        self.visualize_world_btn = ttk.Button(visualization_frame, text="Visualize World", command=self.visualize_world, state=tk.DISABLED)
        self.visualize_world_btn.grid(row=0, column=0, padx=10, pady=10)
        
        self.visualize_graph_btn = ttk.Button(visualization_frame, text="Visualize Graph", command=self.visualize_graph, state=tk.DISABLED)
        self.visualize_graph_btn.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Button(visualization_frame, text="Task 1.1", command=self.run_task_1_1).grid(row=1, column=0, padx=10, pady=10)
        ttk.Button(visualization_frame, text="Task 1.2", command=self.run_task_1_2).grid(row=1, column=1, padx=10, pady=10)
        
        # Configure grid weights for resizing
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(1, weight=1)
        visualization_frame.columnconfigure(0, weight=1)
        visualization_frame.columnconfigure(1, weight=1)
        
    def update_parameters(self):
        for param, entry in self.param_entries.items():
            value = entry.get()
            if value.lower() == "infinity" or value.lower() == "inf":
                self.params[param] = np.inf
                continue
            try:
                if param in ['num_targets', 'd', 'world_size']:
                    self.params[param] = int(value)
                else:
                    self.params[param] = float(value)
            except ValueError:
                messagebox.showerror("Error", f"Invalid value for {param}. Please enter a number.")
                return
        messagebox.showinfo("Success", "Parameters updated successfully!")
        
    def reset_to_default(self):
        self.params = self.default_params.copy()
        for param, value in self.params.items():
            if param == 'radius_fov' and value == np.inf:
                self.param_entries[param].delete(0, tk.END)
                self.param_entries[param].insert(0, "Infinity")
            else:
                self.param_entries[param].delete(0, tk.END)
                self.param_entries[param].insert(0, str(value))
        self.save_parameters()
        messagebox.showinfo("Success", "Parameters reset to default!")
    
    def enable_visualization_buttons(self):
        if not self.task_has_run:
            self.task_has_run = True
            self.visualize_world_btn.config(state=tk.NORMAL)
            self.visualize_graph_btn.config(state=tk.NORMAL)
    
    def visualize_world(self):
        """Visualize the world using current parameters"""
        messagebox.showinfo("Visualize World", f"Visualizing world with parameters:\n{json.dumps(self.get_display_params(), indent=2)}")
    
    def visualize_graph(self):
        """Visualize the graph using current parameters"""
        # Placeholder for graph visualization functionality
        messagebox.showinfo("Visualize Graph", f"Visualizing graph with parameters:\n{json.dumps(self.get_display_params(), indent=2)}")
        # Here you would add the actual graph visualization code
        
    def run_task_1_1(self):
        messagebox.showinfo("Task 1.1", f"Running Task 1.1 with parameters:\n{json.dumps(self.get_display_params(), indent=2)}")
        targets, agents = generate_agents_and_targets(self.params['num_targets'],
                                                  self.params['ratio_at'],
                                                  self.params['world_size'],
                                                  self.params['d'],
                                                  self.params['radius_fov'])

        G, adj, A = generate_graph(len(agents), self.params['graph_type'])
    
        real_distances, noisy_distances = get_distances(agents, 
                                                    targets, 
                                                    self.params['noise_level'],
                                                    self.params['bias'],
                                                    self.params['radius_fov'],
                                                    self.params['world_size'])
    
        cost_function = local_cost_function_task2
        z_hystory, cost, norm_grad_cost, prova, norm_error = gradient_tracking_method(agents,
                                                                                      targets,
                                                                                      noisy_distances,
                                                                                      adj,
                                                                                      A,
                                                                                      cost_function)
        
        # Visualization
        plot_gradient_tracking_results(z_hystory, cost, norm_grad_cost, prova, agents, targets, norm_error)
        animate_world_evolution(agents, targets, z_hystory, self.params['graph_type'], self.params['world_size'], self.params['d'])
        self.enable_visualization_buttons()
    
    def run_task_1_2(self):
        messagebox.showinfo("Task 1.2", f"Running Task 1.2 with parameters:\n{json.dumps(self.get_display_params(), indent=2)}")
        targets, agents = generate_agents_and_targets(self.params['num_targets'],
                                                  self.params['ratio_at'],
                                                  self.params['world_size'],
                                                  self.params['d'],
                                                  self.params['radius_fov'])

        G, adj, A = generate_graph(len(agents), self.params['graph_type'])
    
        real_distances, noisy_distances = get_distances(agents, 
                                                    targets, 
                                                    self.params['noise_level'],
                                                    self.params['bias'],
                                                    self.params['radius_fov'],
                                                    self.params['world_size'])
    
        cost_function = local_cost_function_task2
        z_hystory, cost, norm_grad_cost, prova, norm_error = gradient_tracking_method(agents,
                                                                                      targets,
                                                                                      noisy_distances,
                                                                                      adj,
                                                                                      A,
                                                                                      cost_function)
        
        # Visualization
        plot_gradient_traking_results(z_hystory, cost, norm_grad_cost, prova, agents, targets, norm_error)
        animate_world_evolution(agents, targets, z_hystory, self.params['graph_type'], self.params['world_size'], self.params['d'])
        self.enable_visualization_buttons()
    
    def get_display_params(self):
        """Return parameters in a displayable format"""
        display_params = self.params.copy()
        if display_params.get('radius_fov') == np.inf:
            display_params['radius_fov'] = "Infinity"
        return display_params
    
if __name__ == "__main__":
    root = tk.Tk()
    app = Task1(root)
    root.mainloop()