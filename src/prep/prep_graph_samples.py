"""
Builds a graph from the edges (training set) and performs random walk sampling from the graph
- Currently returns 10 samples of sequence length 10 for each node (this is a parameter in create_random_walk_samples)
"""
import argparse
import random

import networkx
import numpy as np
import scipy as sp

from src.config import DATA_PATH
from src.utils.io_utils import save_model
from src.utils.logger import logger


def load_network(edgelist_path):
    graph = networkx.read_weighted_edgelist(edgelist_path)
    logger.info('No of nodes ({:,}) and edges ({:,})'.format(graph.number_of_nodes(), graph.number_of_edges()))

    # Get dictionary mapping of integer to nodes
    node_dict = {i: key for i, key in enumerate(graph.nodes.keys())}

    return graph, node_dict


def create_transition_matrix(graph):
    """
    https://stackoverflow.com/questions/37311651/get-node-list-from-random-walk-in-networkx
    https://stackoverflow.com/questions/15330380/probability-to-visit-nodes-in-a-random-walk-on-graph

    Args:
        graph:

    Returns:

    """
    adjacency_mat = networkx.adj_matrix(graph)
    logger.info('Adjacency matrix shape: {}'.format(adjacency_mat.shape))
    graph = None

    degree_vector = sp.sparse.csr_matrix(1 / np.sum(adjacency_mat, axis=0))

    transition_matrix = adjacency_mat.multiply(degree_vector).T  # Need to transpose so each row probability sum to 1
    logger.info('Transition matrix shape: {}'.format(transition_matrix.shape))

    return transition_matrix


def create_transition_dict(transition_matrix):
    transition_dict = {}
    rows, cols = transition_matrix.nonzero()

    # Create dictionary of transition product and probabilities for each product
    prev_row = -1
    for row, col in zip(rows, cols):
        if row != prev_row:
            transition_dict.setdefault(row, {})
            transition_dict[row].setdefault('product', [])
            transition_dict[row].setdefault('probability', [])

        transition_dict[row]['product'].append(col)
        transition_dict[row]['probability'].append(transition_matrix[row, col])
        prev_row = row

    return transition_dict


def create_random_walk_samples(node_dict, transition_dict, samples_per_node=10, sequence_len=10):
    random.seed(42)
    n_nodes = len(node_dict)

    sample_array = np.zeros((n_nodes * samples_per_node, sequence_len), dtype=int)
    logger.info('Sample array shape: {}'.format(sample_array.shape))

    # For each node
    for node_idx in range(n_nodes):

        if node_idx % 100000 == 0:
            logger.info('Getting samples for node: {:,}/{:,}'.format(node_idx, n_nodes))

        # For each sample
        for sample_idx in range(samples_per_node):
            node = node_idx

            # For each event in sequence
            for seq_idx in range(sequence_len):
                sample_array[node_idx * samples_per_node + sample_idx, seq_idx] = node
                node = random.choices(population=transition_dict[node]['product'],
                                      weights=transition_dict[node]['probability'], k=1)[0]

    return sample_array


def get_samples(edgelist_path):
    graph, node_dict = load_network(edgelist_path)
    logger.info('Network loaded')

    transition_matrix = create_transition_matrix(graph)
    logger.info('Transition matrix created')
    graph = None

    transition_dict = create_transition_dict(transition_matrix)
    logger.info('Transition dict created')
    transition_matrix = None

    sample_array = create_random_walk_samples(node_dict, transition_dict)
    logger.info('Random walk samples created')

    # Convert array of nodeIDs back to product IDs
    sample_array = np.vectorize(node_dict.get)(sample_array)
    logger.info('Converted back to product IDs')

    return sample_array, node_dict, transition_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparing graph samples via random walk')
    parser.add_argument('read_path', type=str, help='Path to input graph edgelist')
    parser.add_argument('write_path', type=str, help='Path to output samples (.npy format)')
    parser.add_argument('graph_name', type=str, help='Name for node dict and transition dict')
    args = parser.parse_args()

    sample_array, node_dict, transition_dict = get_samples(args.read_path)

    np.save(args.write_path, sample_array)
    logger.info('Sample array saved to {}'.format(args.write_path))
    sample_array = None

    save_model(node_dict, '{}/{}_node_dict.tar.gz'.format(DATA_PATH, args.graph_name))
    node_dict = None

    save_model(transition_dict, '{}/{}_transition_dict.tar.gz'.format(DATA_PATH, args.graph_name))
    transition_dict = None
