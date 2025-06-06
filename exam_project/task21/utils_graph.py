import numpy as np
import networkx as nx

def ensure_connected_graph(G):
    """Ensures the input graph is connected by adding edges between disconnected components."""
    if nx.is_connected(G):
        return G
    components = list(nx.connected_components(G))
    for i in range(len(components) - 1):
        u = list(components[i])[0]
        v = list(components[i + 1])[0]
        G.add_edge(u, v)
    return G

def metropolis_hastings_weights(G):
    r"""
    Compute the Metropolis-Hastings weight matrix for a given graph.
    The Metropolis-Hastings weights are used to construct a symmetric, doubly-stochastic
    matrix suitable for consensus algorithms on undirected graphs.
    
    A_ij = 1/(1 + max(d_i, d_j)) if (i,j) ∈ E and i ≠ j
           1 - ∑(A_ih) for h ∈ N_i\{i} if i = j
           0 otherwise
           
    Args:
        G (networkx.Graph): An undirected graph.
    Returns:
        numpy.ndarray: The Metropolis-Hastings weight matrix of shape (n, n), where n is the number of nodes in G.
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

def generate_graph(num_agents, type, p_er=0.5):
    """
    Generate a graph and its adjacency matrix and Metropolis-Hastings weights.
    Args:
        num_agents (int): Number of nodes (agents) in the graph.
        type (str): Type of graph to generate. Options are 'path', 'cycle', 'star', or 'erdos_renyi'.
        p_er (float, optional): Probability of edge creation for Erdos-Renyi graphs. Defaults to 0.5.
    Returns:
        tuple: A tuple containing:
            - G (networkx.Graph): The generated graph.
            - Adj (numpy.ndarray): The adjacency matrix of the graph.
            - A (numpy.ndarray): The Metropolis-Hastings weight matrix.
    Raises:
        ValueError: If an unknown graph type is provided.
    """
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
        raise ValueError("Unknown graph type. Use 'cycle', 'path', 'star', or 'erdos_renyi'.")
    
    Adj = nx.adjacency_matrix(G).toarray()
    A = metropolis_hastings_weights(G)
    return G, Adj, A

