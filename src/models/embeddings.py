"""燃料 Embedding 层。"""
import torch
import torch.nn as nn


class FuelEmbedding(nn.Module):
    """将离散燃料 ID 映射为连续向量。

    fuel_id: 0=参考火焰(乙烯), 1=甲苯, 2=甲醇
    """

    def __init__(self, num_fuels: int = 3, embed_dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(num_fuels, embed_dim)

    def forward(self, fuel_id: torch.Tensor) -> torch.Tensor:
        """fuel_id: (B,) → (B, embed_dim)"""
        return self.embedding(fuel_id)
