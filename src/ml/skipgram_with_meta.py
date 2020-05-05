import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import logger

torch.manual_seed(1368)


class SkipGram(nn.Module):

    def __init__(self, emb_sizes, emb_dim):
        super().__init__()
        self.emb_sizes = emb_sizes
        self.emb_dim = emb_dim

        # Create embedding layers
        self.center_embeddings = nn.ModuleList()
        for k, v in self.emb_sizes.items():
            self.center_embeddings.append(nn.Embedding(v, emb_dim, sparse=True))

        self.context_embeddings = nn.ModuleList()
        for k, v in self.emb_sizes.items():
            self.context_embeddings.append(nn.Embedding(v, emb_dim, sparse=True))

        self.init_emb()

    def init_emb(self):
        """
        Init embeddings like word2vec

        Center embeddings have uniform distribution in [-0.5/emb_dim , 0.5/emb_dim].
        Context embeddings are initialized with 0s.

        Returns:

        """
        emb_range = 0.5 / self.emb_dim

        # Initializing embeddings:
        # https://stackoverflow.com/questions/55276504/different-methods-for-initializing-embedding-layer-weights-in-pytorch
        for emb in self.center_embeddings:
            emb.weight.data.uniform_(-emb_range, emb_range)

        for emb in self.context_embeddings:
            emb.weight.data.uniform_(0, 0)

    def forward(self, centers, contexts, neg_contexts):
        """

        Args:
            center: List of center words
            context: List of context words
            neg_context: List of list of negative context words

        Returns:

        """
        # Calculate positive score
        emb_centers = []
        for i in range(centers.shape[1]):
            logger.debug('center i: {}'.format(i))
            emb_centers.append(self.center_embeddings[i](centers[:, i]))
        emb_center = torch.mean(torch.stack(emb_centers), axis=0)

        emb_contexts = []
        for i in range(contexts.shape[1]):
            logger.debug('context i: {}'.format(i))
            emb_contexts.append(self.context_embeddings[i](contexts[:, i]))
        emb_context = torch.mean(torch.stack(emb_contexts), axis=0)

        emb_neg_contexts = []
        neg_contexts = neg_contexts.view(-1, len(self.context_embeddings))
        for i in range(neg_contexts.shape[1]):
            logger.debug('neg context i: {}, {}'.format(i, neg_contexts[:, i]))
            emb_neg_contexts.append(self.context_embeddings[i](neg_contexts[:, i]))
        emb_neg_context = torch.mean(torch.stack(emb_neg_contexts), axis=0)

        # Next two lines equivalent to torch.dot(emb_center, emb_context) but for batch
        score = torch.mul(emb_center, emb_context)  # Get dot product (part 1)
        score = torch.sum(score, dim=1)  # Get dot product (part2)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)  # Get score for the positive pairs

        # Calculate negative score (for negative samples)
        neg_score = torch.bmm(emb_neg_context.view(emb_center.shape[0], -1, emb_center.shape[1]),
                              emb_center.unsqueeze(2)).squeeze()  # Get dot product
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        # Return combined score
        return torch.mean(score + neg_score)

    def get_center_emb(self, centers):
        emb_centers = []
        for row_idx, center in enumerate(centers):
            emb_center = []
            for col_idx, center_ in enumerate(center):
                emb_center.append(self.center_embeddings[col_idx](center_))

            emb_centers.append(torch.mean(torch.stack(emb_center), axis=0))

        return torch.stack(emb_centers)

    def save_embeddings(self, file_name):
        embedding = self.center_embeddings.weight.cpu().data.numpy()
        np.save(file_name, embedding)
