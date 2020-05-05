"""
Parses the raw json data into csv file for faster loading into pd.DataFrame.
"""
import argparse
import csv
import gzip
from typing import List

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype

from src.utils.logger import logger


def parse(path: str):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def parse_json_to_df(path: str) -> pd.DataFrame:
    i = 0
    df_dict = {}
    for d in parse(path):
        df_dict[i] = d
        i += 1
        if i % 10000 == 0:
            logger.info('Rows processed: {:,}'.format(i))

    df = pd.DataFrame.from_dict(df_dict, orient='index')

    # Lowercase
    df['related'] = df['related'].astype(str)
    df['categories'] = df['categories'].astype(str)
    df['salesRank'] = df['salesRank'].astype(str)
    df = lowercase_df(df)

    return df


# Lowercase Functions
def lowercase_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase characters from all columns in a dataframe.

    Args:
        df: Pandas dataframe

    Returns:
        Lowercased dataframe
    """
    df = df.copy()
    for col in df.columns:
        if is_object_dtype(df[col]):
            df = lowercase_cols(df, [col])
    return df


def lowercase_cols(df: pd.DataFrame, colnames: List[str]) -> pd.DataFrame:
    """
    Lowercase characters from specified columns in a dataframe

    Args:
        df: Pandas dataframe
        colnames (List): Names of columns to be lowercased

    Returns: Lowercased dataframe

    """
    df = df.copy()
    for col in colnames:
        assert df[col].dtype != np.float64 and df[col].dtype != np.int64, \
            'Trying to lowercase a non-string column: {}'.format(col)
        df[col] = df[col].str.lower()
    return df


def parse_json_to_csv(read_path: str, write_path: str) -> None:
    """
    Note: This assumes that the first json in the path has all the keys, which could be WRONG

    Args:
        read_path:
        write_path:

    Returns:

    """
    csv_writer = csv.writer(open(write_path, 'w'))
    i = 0
    for d in parse(read_path):
        if i == 0:
            header = d.keys()
            csv_writer.writerow(header)

        csv_writer.writerow(d.values().lower())
        i += 1
        if i % 10000 == 0:
            logger.info('Rows processed: {:,}'.format(i))

    logger.info('Csv saved to {}'.format(write_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing json (gzipped) to csv')
    parser.add_argument('read_path', type=str, help='Path to input gzipped json')
    parser.add_argument('write_path', type=str, help='Path to output csv')
    args = parser.parse_args()

    df = parse_json_to_df(args.read_path)
    df.to_csv(args.write_path, index=False)
    logger.info('Csv saved to {}'.format(args.write_path))
