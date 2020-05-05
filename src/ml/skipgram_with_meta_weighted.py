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

        # Create embedding weighting layer
        self.emb_weights = nn.Embedding(emb_sizes['product'], len(emb_sizes),
                                        sparse=True)  # emb_sizes['product'] is total number of products
        self.emb_weights_softmax = nn.Softmax(dim=1)

        self.init_emb()

        logger.info('Model initialized: {}'.format(self))

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

        emb_weights_init = 1 / len(self.emb_sizes)
        self.emb_weights.weight.data.uniform_(emb_weights_init)

    def get_embedding(self, nodes):
        embs = []
        emb_weight = self.emb_weights(nodes[:, 0])
        emb_weight_norm = self.emb_weights_softmax(emb_weight)

        for i in range(nodes.shape[1]):
            logger.debug('center i: {}'.format(i))
            embs.append(self.center_embeddings[i](nodes[:, i]))
        emb_stack = torch.stack(embs)
        embs_weighted = emb_stack * emb_weight_norm.T.unsqueeze(2).expand_as(emb_stack)
        emb = torch.sum(embs_weighted, axis=0)

        return emb

    def forward(self, centers, contexts, neg_contexts):
        """

        Args:
            center: List of center words
            context: List of context words
            neg_context: List of list of negative context words

        Returns:

        """
        emb_center = self.get_embedding(centers)
        emb_context = self.get_embedding(contexts)

        neg_contexts = neg_contexts.view(-1, len(self.context_embeddings))  # Need to expand this first
        emb_neg_context = self.get_embedding(neg_contexts)

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
