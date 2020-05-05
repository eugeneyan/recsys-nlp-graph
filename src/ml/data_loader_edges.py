import itertools
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import MODEL_PATH
from src.utils.io_utils import save_model
from src.utils.logger import logger


class Edges:
    NEGATIVE_SAMPLE_TABLE_SIZE = 1e7

    def __init__(self, edge_path: str, val_path: str, power: float = 0.75):
        """
        Initializes an Edges object for use in a Dataset.

        Args:
            edge_path: Path to numpy array of sequences, where each row is a sequence
            power: Negative sampling parameter; suggested 0.75
        """
        self.power = power
        self.negative_idx = 0
        self.n_unique_tokens = 0

        self.edges = pd.read_csv(edge_path)
        self.n_edges = len(self.edges)
        logger.info('Edges loaded (length = {:,})'.format(self.n_edges))

        self.val = pd.read_csv(val_path)
        logger.info('Validation set loaded: {}'.format(self.val.shape))

        self.product_set = self.get_product_set()
        self.word2id, self.id2word = self.get_mapping_dicts()
        self.get_product_id_func = np.vectorize(self.get_product_id)
        self.n_unique_tokens = len(self.word2id)
        logger.info('No. of unique tokens: {}'.format(self.n_unique_tokens))
        save_model(self.word2id, '{}/word2id_edge'.format(MODEL_PATH))
        save_model(self.id2word, '{}/id2word_edge'.format(MODEL_PATH))
        logger.info('Word2Id and Id2Word created and saved')

        # Convert product ID strings to integers
        self.edges = self.prep_edges()
        logger.info('Edges prepared')

        # Prepare negative sampling table
        self.word_freq = self.get_word_freq(self.edges[:, :2])
        self.neg_table = self.get_negative_sample_table(self.power)

    def get_product_set(self):
        product_set = set(self.edges['product1'].tolist() + self.edges['product2'].tolist() +
                          self.val['product1'].tolist() + self.val['product2'].tolist())

        return product_set

    def get_mapping_dicts(self):
        word2id = dict()
        id2word = dict()

        wid = 0
        for w in self.product_set:
            word2id[w] = wid
            id2word[wid] = w
            wid += 1

        return word2id, id2word

    def get_product_id(self, x):
        return self.word2id.get(x, -1)

    def prep_edges(self):
        self.edges['product1_id'] = self.get_product_id_func(self.edges['product1']).astype(int)
        self.edges['product2_id'] = self.get_product_id_func(self.edges['product2']).astype(int)
        edges = self.edges[['product1_id', 'product2_id', 'weight']].copy().values

        return edges

    def get_word_freq(self, edges):
        product_counts = list(itertools.chain.from_iterable(edges))
        word_freq = Counter(product_counts)
        return word_freq

    def get_negative_sample_table(self, power=0.75) -> np.array:
        """
        Returns a table (size = NEGATIVE_SAMPLE_TABLE_SIZE) of negative samples which can be selected via indexing.

        Args:
            power:

        Returns:

        """
        # Convert to array
        word_freq = np.array(list(self.word_freq.items()), dtype=np.float64)

        # Adjust by power
        word_freq[:, 1] = word_freq[:, 1] ** power

        # Get probabilities
        word_freq_sum = word_freq[:, 1].sum()
        word_freq[:, 1] = word_freq[:, 1] / word_freq_sum

        # Multiply probabilities by sample table size
        word_freq[:, 1] = np.round(word_freq[:, 1] * self.NEGATIVE_SAMPLE_TABLE_SIZE)

        # Convert to int
        word_freq = word_freq.astype(int).tolist()

        # Create sample table
        sample_table = [[tup[0]] * tup[1] for tup in word_freq]
        sample_table = np.array(list(itertools.chain.from_iterable(sample_table)))
        np.random.shuffle(sample_table)

        return sample_table

    def get_negative_samples(self, context, sample_size=5) -> np.array:
        """
        Returns a list of negative samples, where len = sample_size.

        Args:
            sample_size:

        Returns:

        """
        while True:
            # Get a batch from the shuffled table
            neg_sample = self.neg_table[self.negative_idx:self.negative_idx + sample_size]

            # Update negative index
            self.negative_idx = (self.negative_idx + sample_size) % len(self.neg_table)

            # Check if batch insufficient
            if len(neg_sample) != sample_size:
                neg_sample = np.concatenate((neg_sample, self.neg_table[:self.negative_idx]))

            # Check if context in negative sample
            if not context in neg_sample:
                return neg_sample


class EdgesDataset(Dataset):
    def __init__(self, edges, neg_sample_size=5):
        self.edges = edges
        self.neg_sample_size = neg_sample_size

    def __len__(self):
        return self.edges.n_edges

    def __getitem__(self, idx):
        pair = self.edges.edges[idx]
        neg_samples = self.edges.get_negative_samples(context=pair[1])

        return pair, neg_samples

    @staticmethod
    def collate(batches):
        logger.debug('Batches: {}'.format(batches))
        batch_list = []

        for batch in batches:
            pair = np.array(batch[0])
            negs = np.array(batch[1])
            negs = np.vstack((pair[0].repeat(negs.shape[0]), negs)).T

            # Create arrays
            pair_arr = np.ones((pair.shape[0]), dtype=int)  # This sets label to 1  # TODO: Leave label as continuous
            pair_arr[:-1] = pair[:-1]
            negs_arr = np.zeros((negs.shape[0], negs.shape[1] + 1), dtype=int)
            negs_arr[:, :-1] = negs
            all_arr = np.vstack((pair_arr, negs_arr))
            batch_list.append(all_arr)

        batch_array = np.vstack(batch_list)

        # Return item1, item2, label
        return (torch.LongTensor(batch_array[:, 0]), torch.LongTensor(batch_array[:, 1]),
                torch.FloatTensor(batch_array[:, 2]))

    @staticmethod
    def collate_continuous(batches):
        logger.debug('Batches: {}'.format(batches))
        batch_list = []

        for batch in batches:
            pair = np.array(batch[0])
            negs = np.array(batch[1])
            negs = np.vstack((pair[0].repeat(negs.shape[0]), negs)).T

            # Create arrays
            pair_arr = pair
            negs_arr = np.zeros((negs.shape[0], negs.shape[1] + 1), dtype=int)
            negs_arr[:, :-1] = negs
            all_arr = np.vstack((pair_arr, negs_arr))
            batch_list.append(all_arr)

        batch_array = np.vstack(batch_list)

        # Return item1, item2, label
        return (torch.LongTensor(batch_array[:, 0]), torch.LongTensor(batch_array[:, 1]),
                torch.FloatTensor(batch_array[:, 2]))
