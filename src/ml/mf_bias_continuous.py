import torch
import torch.nn as nn

from src.utils.logger import logger

torch.manual_seed(1368)


def regularize_l2(array):
    loss = torch.sum(array ** 2.0)
    return loss


class MFBiasContinuous(nn.Module):
    def __init__(self, emb_size, emb_dim, c_vector=1e-6, c_bias=1e-6):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.c_vector = c_vector
        self.c_bias = c_bias

        # Layers
        self.product_embedding = nn.Embedding(emb_size, emb_dim)
        self.sig = nn.Sigmoid()

        # Bias
        self.product_bias = nn.Embedding(emb_size, 1)
        self.bias = nn.Parameter(torch.ones(1))

        # Loss
        self.mse = nn.MSELoss()

        logger.info('Model initialized: {}'.format(self))

    def forward(self, product1, product2):
        emb_product1 = self.product_embedding(product1)
        emb_product2 = self.product_embedding(product2)
        interaction = torch.sum(emb_product1 * emb_product2, dim=1, dtype=torch.float)

        bias_product1 = self.product_bias(product1).squeeze()
        bias_product2 = self.product_bias(product2).squeeze()
        biases = self.bias + bias_product1 + bias_product2

        prediction = (interaction + biases)

        return prediction

    def predict(self, product1, product2):
        emb_product1 = self.product_embedding(product1)
        emb_product2 = self.product_embedding(product2)
        interaction = torch.sum(emb_product1 * emb_product2, dim=1, dtype=torch.float)

        bias_product1 = self.product_bias(product1).squeeze()
        bias_product2 = self.product_bias(product2).squeeze()
        biases = self.bias + bias_product1 + bias_product2

        prediction = self.sig((interaction + biases))

        return prediction

    def loss(self, pred, label):
        mf_loss = self.mse(pred, label)

        # L2 regularization
        product_prior = regularize_l2(self.product_embedding.weight) * self.c_vector
        product_bias_prior = regularize_l2(self.product_bias.weight) * self.c_bias

        loss_total = mf_loss + product_prior + product_bias_prior

        return loss_total
