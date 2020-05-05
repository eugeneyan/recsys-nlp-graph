"""
Converts edge relationships (e.g., bought together, also bought) to numeric weights between two nodes.
"""
import argparse

import numpy as np
import pandas as pd

from src.utils.logger import logger

relationship_weights = {'bought_together': 1.2,
                        'also_bought': 1.0,
                        'also_viewed': 0.5}


def create_product_pair(df, col_list):
    pairs = df[col_list].values
    pairs.sort(axis=1)
    df['product_pair'] = ['|'.join(arr) for arr in pairs]

    return df


def split_product_pair(product_pair):
    result = product_pair.split('|')
    return result[0], result[1]


def get_relationship_weights(df, relationship_weights):
    df['weight'] = 0
    for relationship, weight in relationship_weights.items():
        df.loc[df['relationship'] == relationship, 'weight'] += weight

    return df


def get_edges(df):
    """
    Returns a dataframe of products and the weights of the edges between them.

    Args:
        df:

    Returns:

    """
    logger.info('Relationship distribution: \n{}'.format(df['relationship'].value_counts()))

    df = create_product_pair(df, col_list=['asin', 'related'])
    logger.info('Product pairs created')

    df = get_relationship_weights(df, relationship_weights)
    logger.info('Relationship weights updated')

    # Aggregate to remove duplicates
    logger.info('Original no. of edges: {:,}'.format(df.shape[0]))
    df = df.groupby('product_pair').agg({'weight': 'sum'}).reset_index()
    logger.info('Deduplicated no. of edges: {:,}'.format(df.shape[0]))

    # Save edge list
    df['product1'], df['product2'] = zip(*df['product_pair'].apply(split_product_pair))

    df = df[['product1', 'product2', 'weight', 'product_pair']]
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparing edges and associated weights')
    parser.add_argument('read_path', type=str, help='Path to input csv (of node relationships)')
    parser.add_argument('write_path', type=str, help='Path to output edges')
    parser.add_argument('--sample_size', type=int, help='Sample size (default: no sampling)',
                        default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.read_path, error_bad_lines=False, warn_bad_lines=True,
                     dtype={'asin': 'str', 'related': 'str'})
    logger.info('DF shape: {}'.format(df.shape))

    # Sample for development efficiency
    if args.sample_size:
        sample_idx = np.random.choice(df.shape[0], size=args.sample_size, replace=False)
        df = df.iloc[sample_idx]

    df = get_edges(df)

    df.to_csv(args.write_path, index=False)
    logger.info('Csv saved to {}'.format(args.write_path))
