import json
import numpy as np
import networkx as nx

def load_graph_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Load the graph into NetworkX
    G = nx.Graph()
    for node in data['nodes']:
        G.add_node(node['id'])
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    return G

def compute_balancedness_metrics(G):
    # Edge Weight Uniformity
    weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    weight_mean = weights.mean()
    weight_std_dev = weights.std()
    
    # Node Degree Uniformity
    degrees = np.array([G.degree(n) for n in G.nodes()])
    degree_mean = degrees.mean()
    degree_std_dev = degrees.std()

    # Connectivity Strength Distribution
    strength_variance = np.var([sum(G[u][v]['weight'] for v in G.neighbors(u)) for u in G.nodes()])
    
    metrics = {
        "Edge Weight Mean": weight_mean,
        "Edge Weight Std Dev (Uniformity)": weight_std_dev,
        "Node Degree Mean": degree_mean,
        "Node Degree Std Dev (Uniformity)": degree_std_dev,
        "Connectivity Strength Variance": strength_variance
    }
    return metrics

# Load the graph from JSON and compute metrics
test_topics = ['Manchester United']
for topic in test_topics:
    file_path = f"./llm/graph_data_Recent News about {topic}.json"  # Update with the actual path to your JSON file
    G = load_graph_from_json(file_path)
    metrics = compute_balancedness_metrics(G)

    # Output the balancedness metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
