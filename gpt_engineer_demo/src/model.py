import torch
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=100):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)
