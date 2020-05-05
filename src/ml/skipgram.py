import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1368)


class SkipGram(nn.Module):

    def __init__(self, emb_size, emb_dim):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.center_embeddings = nn.Embedding(emb_size, emb_dim, sparse=True)
        self.context_embeddings = nn.Embedding(emb_size, emb_dim, sparse=True)
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
        self.center_embeddings.weight.data.uniform_(-emb_range, emb_range)
        self.context_embeddings.weight.data.uniform_(0, 0)

    def forward(self, center, context, neg_context):
        """

        Args:
            center: List of center words
            context: List of context words
            neg_context: List of list of negative context words

        Returns:

        """
        # Calculate positive score
        emb_center = self.center_embeddings(center)  # Get embeddings for center word
        emb_context = self.context_embeddings(context)  # Get embeddings for context word
        emb_neg_context = self.context_embeddings(neg_context)  # Get embeddings for negative context words

        # Next two lines equivalent to torch.dot(emb_center, emb_context) but for batch
        score = torch.mul(emb_center, emb_context)  # Get dot product (part 1)
        score = torch.sum(score, dim=1)  # Get dot product (part2)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)  # Get score for the positive pairs

        # Calculate negative score (for negative samples)
        neg_score = torch.bmm(emb_neg_context, emb_center.unsqueeze(2)).squeeze()  # Get dot product
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        # Return combined score
        return torch.mean(score + neg_score)

    def get_center_emb(self, center):
        return self.center_embeddings(center)

    def save_embeddings(self, file_name):
        embedding = self.center_embeddings.weight.cpu().data.numpy()
        np.save(file_name, embedding)
