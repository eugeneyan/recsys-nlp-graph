"""
Splits all ground truth edges into train and validation set, with some constraints
- The validation set should only contain edges where both products are in the train set

For the validation set, negative samples are created by randomly selecting a pair of nodes and creating a negative edge.
- From these samples, we exclude valid edges from either the train or validation set.
"""
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH
from src.prep.prep_edges import create_product_pair
from src.utils.logger import logger


def train_val_split(df, n_val_samples: int, filter_out_unseen: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if filter_out_unseen:
        # First split to get some test samples
        train, val = train_test_split(df, test_size=int(1.1 * n_val_samples), random_state=42)  # Need slightly more
        logger.info('Train shape: {}, val shape: {}'.format(train.shape, val.shape))

        # Get set of products in train
        train_product_set = set(train['product1']).union(set(train['product2']))
        logger.info('No. of unique products in train: {:,}'.format(len(train_product_set)))

        # Only keep val where both products are in train product set
        val = val[(val['product1'].isin(train_product_set)) & (val['product2'].isin(train_product_set))]
        logger.info('Updated val shape: {}'.format(val.shape))

        # Split again to only get n_val_samples
        val = val.iloc[:n_val_samples].copy()
        logger.info('Final val shape: {}'.format(val.shape))

        # Get train set
        train = df[~df.index.isin(set(val.index))].copy()
        logger.info('Final train shape: {}'.format(train.shape))

    else:
        # First split to get some test samples
        train, val = train_test_split(df, test_size=int(n_val_samples), random_state=42)
        logger.info('Train shape: {}, val shape: {}'.format(train.shape, val.shape))

    return train, val


def get_sample(item_array, n_iter=None, sample_size=2):
    np.random.seed(42)
    n = len(item_array)

    # find the index we last sampled from
    start_idx = (n_iter * sample_size) % n
    if (start_idx + sample_size >= n) or (start_idx <= sample_size):
        # shuffle array if we have reached the end and repeat again
        np.random.shuffle(item_array)

    return item_array[start_idx:start_idx + sample_size]


def collect_samples(item_array, sample_size, n_samples):
    samples = []

    for i in range(0, n_samples):
        if i % 1000000 == 0:
            logger.info('Neg sample: {:,}'.format(i))

        sample = get_sample(item_array, n_iter=i, sample_size=sample_size)
        samples.append(sample)

    return samples


def create_negative_edges(df, val, n_val_samples):
    # Get set of valid product edges (across both train and val)
    valid_product_pairs = set(df['product_pair'])
    logger.info('No. of valid product pairs: {:,}'.format(len(valid_product_pairs)))

    # Get set of products in val (to generate edges)
    val_product_arr = np.array(list(set(val['product1']).union(set(val['product2']))))
    logger.info('No. of unique products in val: {:,}'.format(len(val_product_arr)))

    # Create negative samples
    neg_samples = collect_samples(val_product_arr, sample_size=2, n_samples=int(1.1 * n_val_samples))
    neg_samples_df = pd.DataFrame(neg_samples, columns=['product1', 'product2'])
    neg_samples_df.dropna(inplace=True)
    neg_samples_df = create_product_pair(neg_samples_df, col_list=['product1', 'product2'])
    logger.info('No. of negative samples: {:,}'.format(neg_samples_df.shape[0]))

    # Exclude neg samples that are valid pairs
    neg_samples_df = neg_samples_df[~neg_samples_df['product_pair'].isin(valid_product_pairs)].copy()
    logger.info('Updated no. of negative samples: {:,}'.format(neg_samples_df.shape[0]))

    # Only keep no. of val samples required
    neg_samples_df = neg_samples_df.iloc[:n_val_samples].copy()
    logger.info('Final no. of negative samples: {:,}'.format(neg_samples_df.shape[0]))

    return neg_samples_df


def combine_val_and_neg_edges(val, neg_samples):
    neg_samples['edge'] = 0
    val['edge'] = 1

    VAL_COLS = ['product1', 'product2', 'edge']
    neg = neg_samples[VAL_COLS].copy()
    val = val[VAL_COLS].copy()
    logger.info('Val shape: {}, Neg edges shape: {}, Ratio: {}'.format(val.shape, neg.shape,
                                                                       val.shape[0] / (val.shape[0] + neg.shape[0])))

    val = pd.concat([val, neg])
    logger.info('Final val shape: {}'.format(val.shape))

    return val


def get_train_and_val(df, val_prop: float):
    """
    Splits into training and validation set, where validation set has 50% negative edges

    Args:
        df:
        val_prop:

    Returns:

    """
    n_val_samples = int(val_prop * df.shape[0])
    logger.info('Eventual required val samples (proportion: {}): {:,}'.format(val_prop, n_val_samples))

    train, val = train_val_split(df, n_val_samples)
    logger.info('Ratio of train to val: {:,}:{:,} ({:.2f})'.format(train.shape[0], val.shape[0],
                                                                   val.shape[0] / (train.shape[0] + val.shape[0])))

    neg_samples = create_negative_edges(df, val, n_val_samples)

    val = combine_val_and_neg_edges(val, neg_samples)
    train = train[['product1', 'product2', 'weight']].copy()

    return train, val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splitting into train and val set')
    parser.add_argument('read_path', type=str, help='Path to input csv of edges')
    parser.add_argument('val_prop', type=float, help='Proportion of validation set (e.g., 0.33)')
    args = parser.parse_args()

    df = pd.read_csv(args.read_path, error_bad_lines=False, warn_bad_lines=True,
                     dtype={'product1': 'str', 'product2': 'str'})
    logger.info('DF shape: {}'.format(df.shape))

    train, val = get_train_and_val(df, val_prop=args.val_prop)

    # Save to train, val, and train edgelist
    input_filename = Path(args.read_path).resolve().stem
    train.to_csv('{}/{}_train.csv'.format(DATA_PATH, input_filename), index=False)
    logger.info('Train saved as: {}/{}_train.csv'.format(DATA_PATH, input_filename))
    val.to_csv('{}/{}_val.csv'.format(DATA_PATH, input_filename), index=False)
    logger.info('Val saved as: {}/{}_val.csv'.format(DATA_PATH, input_filename))

    train.to_csv('{}/{}_train.edgelist'.format(DATA_PATH, input_filename), sep=' ', index=False, header=False)
    logger.info('Train edgelist saved as: {}/{}_train.edgelist'.format(DATA_PATH, input_filename))
