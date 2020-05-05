import torch
import torch.nn as nn

from src.utils.logger import logger

torch.manual_seed(1368)


def regularize_l2(array):
    loss = torch.sum(array ** 2.0)
    return loss


class MFContinuous(nn.Module):
    def __init__(self, emb_size, emb_dim, c_vector=1e-6):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.c_vector = c_vector

        # Layers
        self.embedding = nn.Embedding(emb_size, emb_dim)
        self.sig = nn.Sigmoid()

        # Loss
        self.mse = nn.MSELoss()

        logger.info('Model initialized: {}'.format(self))

    def forward(self, product1, product2):
        emb_product1 = self.embedding(product1)
        emb_product2 = self.embedding(product2)
        interaction = torch.sum(emb_product1 * emb_product2, dim=1, dtype=torch.float)

        return interaction

    def predict(self, product1, product2):
        emb_product1 = self.embedding(product1)
        emb_product2 = self.embedding(product2)
        interaction = self.sig(torch.sum(emb_product1 * emb_product2, dim=1, dtype=torch.float))  # Add sigmoid

        return interaction

    def loss(self, pred, label):
        mf_loss = self.mse(pred, label)

        # L2 regularization
        product_prior = regularize_l2(self.embedding.weight) * self.c_vector

        loss_total = mf_loss + product_prior

        return loss_total
