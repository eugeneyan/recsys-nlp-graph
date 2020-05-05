import argparse
import datetime

import numpy as np
from gensim.models import Word2Vec

from src.config import MODEL_PATH
from src.utils.logger import logger


def load_sequences(sequence_path):
    """
    Expects a numpy array at sequence_path

    Args:
        sequence_path:

    Returns:

    """
    sequences = np.load(sequence_path)
    logger.info('Sequences shape: {}'.format(sequences.shape))

    # Convert sequences to string and list of list
    sequences = sequences.astype(str).tolist()

    return sequences


def train_embeddings(sequences, workers, dimension=128, window=5, min_count=1, negative=5, epochs=3, seed=42):
    # Logging specific to gensim training
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Initialize model
    model = Word2Vec(sequences, workers=workers,
                     size=dimension, window=window, min_count=min_count, negative=negative, seed=seed)
    logger.info('Model initialized')

    # Train model (No need to retrain model as initialization includes training)
    # model.train(sequences, total_examples=len(sequences), epochs=epochs)
    # logger.info('Model trained!')

    return model


def save_model(model):
    # Save model and keyedvectors
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
    model.save('{}/gensim-w2v-{}.model'.format(MODEL_PATH, current_datetime))
    model.wv.save('{}/gensim-w2v-{}.kv'.format(MODEL_PATH, current_datetime))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create embeddings using gensim package')
    parser.add_argument('read_path', type=str, help='Path to input sequences')
    parser.add_argument('n_workers', type=int, help='Number of workers')
    args = parser.parse_args()

    sequences = load_sequences(args.read_path)

    start_time = datetime.datetime.now()
    model = train_embeddings(sequences, workers=args.n_workers)
    end_time = datetime.datetime.now()
    time_diff = round((end_time - start_time).total_seconds() / 60, 2)
    logger.info('Total time taken: {:,} minutes'.format(time_diff))
    save_model(model)
