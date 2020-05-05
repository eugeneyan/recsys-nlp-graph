"""
Parses item to item relationships in 'related' field and explodes it such that each relationship is a single row.
"""
import argparse

import numpy as np
import pandas as pd

from src.utils.logger import logger


def get_also_bought_count(related):
    try:
        return len(related['also_bought'])
    except KeyError:
        return -1


def explode_on_related(df: pd.DataFrame, relationship: str) -> pd.DataFrame:
    # Filter on relationship
    df = df[df['related'].apply(lambda x: relationship in x.keys())].copy()

    # Get value (list) from relationship dict
    df['related'] = df['related'].apply(lambda x: x[relationship])

    # Explode efficiently using numpy
    vals = df['related'].values.tolist()
    lens = [len(val_list) for val_list in vals]
    vals_array = np.repeat(df['asin'], lens)
    exploded_df = pd.DataFrame(np.column_stack((vals_array, np.concatenate(vals))), columns=df.columns)

    # Add relationship
    exploded_df['relationship'] = relationship
    logger.info('Exploded for relationship: {}'.format(relationship))

    return exploded_df


def get_node_relationship(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe of products and their relationships (e.g., bought together, also bought, also viewed)

    Args:
        df:

    Returns:

    """
    # Keep only rows with related data
    df = df[~df['related'].isnull()].copy()
    logger.info('DF shape after dropping empty related: {}'.format(df.shape))

    df = df[~df['title'].isnull()].copy()
    logger.info('DF shape after dropping empty title: {}'.format(df.shape))
    df = df[['asin', 'related']].copy()

    # Evaluate related str into dict
    df['related'] = df['related'].apply(eval)
    logger.info('Completed eval on "related" string')

    # Exclude products where also bought relationships less than 2
    df['also_bought_count'] = df['related'].apply(get_also_bought_count)
    df = df[df['also_bought_count'] >= 2].copy()
    logger.info('DF shape after dropping products with <2 edges: {}'.format(df.shape))
    df.drop(columns='also_bought_count', inplace=True)

    # Explode columns
    bought_together_df = explode_on_related(df, relationship='bought_together')
    also_bought_df = explode_on_related(df, relationship='also_bought')
    also_viewed_df = explode_on_related(df, relationship='also_viewed')

    # Concatenate df
    combined_df = pd.concat([bought_together_df, also_bought_df, also_viewed_df], axis=0)
    logger.info('Distribution of relationships: \n{}'.format(combined_df['relationship'].value_counts()))

    return combined_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparing node relationships')
    parser.add_argument('read_path', type=str, help='Path to input csv')
    parser.add_argument('write_path', type=str, help='Path to output csv (of nodes relationships)')
    args = parser.parse_args()

    df = pd.read_csv(args.read_path, error_bad_lines=False, warn_bad_lines=True,
                     dtype={'asin': 'str', 'title': 'str', 'brand': 'str'})
    logger.info('DF shape: {}'.format(df.shape))

    exploded_df = get_node_relationship(df)

    exploded_df.to_csv(args.write_path, index=False)
    logger.info('Csv saved to {}'.format(args.write_path))
