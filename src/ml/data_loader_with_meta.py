import itertools
from collections import Counter, OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from category_encoders import OrdinalEncoder
from torch.utils.data import Dataset

from src.config import MODEL_PATH
from src.utils.io_utils import save_model
from src.utils.logger import logger

# META_COLS = ['asin', 'price', 'category_lvl_2', 'category_lvl_3', 'category_lvl_4', 'brand']


def round_up(num, divisor=5):
    return ((num + divisor - 1) // divisor) * divisor


def bin_price(price):
    if price < 25:
        return price
    elif 25 < price < 50:
        return round_up(price, divisor=5)
    elif 50 < price < 500:
        return round_up(price, divisor=10)
    else:
        return 500


def prep_price(price_col):
    price = np.round(price_col)
    price.fillna(-1, inplace=True)
    price = price.astype(int)
    price = price.apply(bin_price)
    price = price + 1

    return price


def prep_categorical(cat_col, min_threshold=100):
    counts = cat_col.value_counts()
    category_set = counts[counts > min_threshold].index

    return np.where(cat_col.isin(category_set), cat_col, 'MISC')


def get_dict_values(meta, META_COLS):
    return [meta[col] for col in META_COLS]


class Sequences:
    NEGATIVE_SAMPLE_TABLE_SIZE = 1e7
    WINDOW = 5

    def __init__(self, sequence_path: str, val_path: str, meta_path: str, subsample: float = 0.001,
                 power: float = 0.75):
        """
        Initializes a Sequence object for use in a Dataset.

        Args:
            sequence_path: Path to numpy array of sequences, where each row is a sequence
            subsample: Subsampling parameter; suggested range (0, 1e-5)
            power: Negative sampling parameter; suggested 0.75
        """
        self.negative_idx = 0
        self.n_unique_tokens = 0
        # META_COLS = ['asin', 'price', 'category_lvl_2', 'category_lvl_3', 'category_lvl_4', 'brand']
        self.META_COLS = ['category_lvl_3', 'brand']  # Add meta columns here

        self.sequences = np.load(sequence_path).tolist()
        self.n_sequences = len(self.sequences)
        logger.info('Sequences loaded (length = {:,})'.format(self.n_sequences))

        self.val = pd.read_csv(val_path)
        logger.info('Validation set loaded: {}'.format(self.val.shape))

        self.word_freq = self.get_word_freq()
        logger.info('Word frequency calculated')

        self.word2id, self.id2word = self.get_mapping_dicts()
        self.add_val_product_to_mapping_dicts()
        self.n_unique_tokens = len(self.word2id)
        logger.info('No. of unique tokens: {}'.format(self.n_unique_tokens))
        save_model(self.word2id, '{}/word2id'.format(MODEL_PATH))
        save_model(self.id2word, '{}/id2word'.format(MODEL_PATH))
        logger.info('Word2Id and Id2Word created and saved')

        self.meta = pd.read_csv(meta_path, dtype={'asin': 'object'})
        self.meta.drop_duplicates(subset='asin', inplace=True)
        self.meta['productid'] = self.meta['asin'].copy()
        self.meta = self.prep_meta()
        self.meta_dict, self.emb_sizes = self.convert_meta_to_dict()
        self.emb_sizes['product'] = len(self.word2id)
        logger.info('Embedding dimensions: {}'.format(self.emb_sizes))
        save_model(self.meta_dict, '{}/meta_dict'.format(MODEL_PATH))
        self.meta = None

        self.sequences = self.convert_sequence_to_id()
        self.word_freq = self.convert_word_freq_to_id()
        logger.info('Convert sequence and wordfreq to ID')

        self.discard_probs = self.get_discard_probs(sample=subsample)
        logger.info('Discard probability calculated')

        self.neg_table = self.get_negative_sample_table(power=power)
        logger.info('Negative sample table created')

        # Used to preload all center context pairs (very memory heavy)
        # self.pairs = self.get_all_center_context_pairs(window=window)
        # self.n_pairs = len(self.pairs)
        # logger.info('Center Context pairs created')

    def get_word_freq(self) -> Counter:
        """
        Returns a dictionary of word frequencies.

        Returns:

        """
        # Flatten list
        seq_flat = list(itertools.chain.from_iterable(self.sequences))

        # Get word frequency
        word_freq = Counter(seq_flat)

        return word_freq

    def get_mapping_dicts(self):
        word2id = dict()
        id2word = dict()

        wid = 0
        for w, c in self.word_freq.items():
            word2id[w] = wid
            id2word[wid] = w
            wid += 1

        return word2id, id2word

    def add_val_product_to_mapping_dicts(self):
        val_product_set = set(self.val['product1'].values).union(set(self.val['product2'].values))

        logger.info('Adding val products to word2id, original size: {}'.format(len(self.word2id)))
        wid = max(self.word2id.values()) + 1
        for w in val_product_set:
            if w not in self.word2id:
                self.word2id[w] = wid
                self.id2word[wid] = w
                wid += 1

        self.val = None  # Release memory
        logger.info('Added val products to word2id, updated size: {}'.format(len(self.word2id)))

    def convert_sequence_to_id(self):
        return np.vectorize(self.word2id.get)(self.sequences)

    def get_product_id(self, x):
        return self.word2id.get(x, -1)

    def convert_word_freq_to_id(self):
        return {self.word2id[k]: v for k, v in self.word_freq.items()}

    def prep_meta(self):
        logger.info('No. of rows in meta before filter by word2id: {}'.format(self.meta.shape[0]))
        meta = self.meta[self.meta['asin'].isin(self.word2id.keys())].copy()
        logger.info('No. of rows in meta after filter by word2id: {}'.format(meta.shape[0]))

        meta['price'] = prep_price(meta['price'])
        meta['category_lvl_2'] = prep_categorical(meta['category_lvl_2'])
        meta['category_lvl_3'] = prep_categorical(meta['category_lvl_3'])
        meta['category_lvl_4'] = prep_categorical(meta['category_lvl_4'])
        meta['brand'] = prep_categorical(meta['brand'])

        return meta

    def convert_meta_to_dict(self):
        meta = self.meta[['productid'] + self.META_COLS].copy()

        # Encode to int
        encoder = OrdinalEncoder(cols=self.META_COLS)
        meta = encoder.fit_transform(meta)
        save_model(encoder, '{}/encoder'.format(MODEL_PATH))

        meta['values'] = meta.apply(get_dict_values, args=(self.META_COLS,), axis=1)
        meta_dict = meta.set_index('productid')['values'].to_dict()
        meta_dict = {self.word2id[k]: v for k, v in meta_dict.items()}

        meta_counts_dict = (meta[self.META_COLS].max() + 1).to_dict()  # Need to +1 to account for index starting from zero
        # Without +1 the embedding size will be insufficient by 1
        ordered_meta_counts_dict = OrderedDict()
        for col in ['product'] + self.META_COLS:
            ordered_meta_counts_dict[col] = meta_counts_dict.get(col, 0)

        return meta_dict, ordered_meta_counts_dict

    def get_discard_probs(self, sample=0.001) -> Dict[int, float]:
        """
        Returns a dictionary of words and their associated discard probability, where the word should be discarded
        if np.random.rand() < probability.

        Args:
            sample:

        Returns:

        """
        # Convert to array
        word_freq = np.array(list(self.word_freq.items()), dtype=np.float64)

        # Convert to probabilities
        word_freq[:, 1] = word_freq[:, 1] / word_freq[:, 1].sum()

        # Perform subsampling
        # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        word_freq[:, 1] = (np.sqrt(word_freq[:, 1] / sample) + 1) * (sample / word_freq[:, 1])

        # Get dict
        discard_probs = {int(k): v for k, v in word_freq.tolist()}

        return discard_probs

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

    def get_meta(self, idx):
        return self.meta_dict.get(idx, [0] * len(self.META_COLS))

    # Works on per sequence
    def get_pairs(self, idx, window=5):
        pairs = []
        sequence = self.sequences[idx]

        for center_idx, center in enumerate(sequence):
            for i in range(-window, window + 1):
                context_idx = center_idx + i
                if context_idx > 0 and context_idx < len(sequence) and center != sequence[
                    context_idx] and np.random.rand() < self.discard_probs[sequence[context_idx]]:
                    context = sequence[context_idx]
                    center_meta = self.get_meta(center)
                    context_meta = self.get_meta(center)
                    pairs.append(([center] + center_meta, [context] + context_meta))

        return pairs

    def get_all_center_context_pairs(self, window=5) -> List[Tuple[int, int]]:
        """
        Returns a list of tuples (center, context).

        Args:
            window:

        Returns:

        """

        pairs = []

        for sequence in self.sequences:
            for center_idx, node in enumerate(sequence):
                for i in range(-window, window + 1):
                    context_idx = center_idx + i
                    if (0 <= context_idx < len(sequence)) \
                        and node != sequence[context_idx] \
                        and np.random.rand() < self.discard_probs[sequence[context_idx]]:
                        pairs.append((node, sequence[context_idx]))

        return pairs

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
                return [[samp] + self.get_meta(samp) for samp in neg_sample]


class EdgesDataset(Dataset):
    def __init__(self, sequences: Sequences, neg_sample_size=5):
        self.sequences = sequences
        self.neg_sample_size = neg_sample_size

    def __len__(self):
        return self.sequences.n_sequences

    def __getitem__(self, idx):
        pairs = self.sequences.get_pairs(idx)
        neg_samples = []
        for center, context in pairs:
            neg_samples.append(self.sequences.get_negative_samples(context))

        return pairs, neg_samples

    @staticmethod
    def collate(batches):
        # logger.info('Batches: {}'.format(batches))
        pairs_batch = [batch[0] for batch in batches]
        neg_contexts_batch = [batch[1] for batch in batches]

        pairs_batch = list(itertools.chain.from_iterable(pairs_batch))
        neg_contexts = list(itertools.chain.from_iterable(neg_contexts_batch))

        centers = [center for center, _ in pairs_batch]
        contexts = [context for _, context in pairs_batch]

        return torch.LongTensor(centers), torch.LongTensor(contexts), torch.LongTensor(neg_contexts)
