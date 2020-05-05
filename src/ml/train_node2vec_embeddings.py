import argparse

import networkx as nx
from node2vec import Node2Vec

from src.config import DATA_PATH
from src.utils.logger import logger


def train_embeddings(edgelist_path, embedding_path):
    # Create path
    graph = nx.read_weighted_edgelist(edgelist_path)
    logger.info('Graph created!')
    assert graph.get_edge_data('0000013714', '0005064295')['weight'] == 3.2, 'Expected edge weight of 3.2'

    # Precomput probabilities and generate walks
    node2vec = Node2Vec(graph, dimensions=128, walk_length=30, num_walks=10, workers=10, temp_folder=DATA_PATH)
    logger.info('Computed probabilities and generated walks')
    graph = None  # We don't need graph anymore since probabilities have been precomputed

    # Embed nodes
    model = node2vec.fit(window=5, min_count=1, batch_words=128)
    logger.info('Nodes embedded')

    # Save embeddings for later use
    model.wv.save_word2vec_format(embedding_path)
    logger.info('Embedding saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create embeddings using node2vec package')
    parser.add_argument('read_path', type=str, help='Path to input (train) graph edgelist')
    parser.add_argument('write_path', type=str, help='Path to output embeddings')
    args = parser.parse_args()

    train_embeddings(args.read_path, args.write_path)
