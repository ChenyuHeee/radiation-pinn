"""输入特征变换：傅里叶编码 + 归一化。"""
import torch
import torch.nn as nn
import math


class FourierFeatures(nn.Module):
    """位置傅里叶特征编码，解决 PINN 频谱偏差。

    γ(p) = [sin(2^0 π p), cos(2^0 π p), ..., sin(2^{L-1} π p), cos(2^{L-1} π p)]
    """

    def __init__(self, L: int = 6):
        super().__init__()
        self.L = L
        # 预计算频率 2^0 ... 2^{L-1}
        freqs = torch.tensor([2.0 ** i for i in range(L)]) * math.pi
        self.register_buffer("freqs", freqs)  # (L,)

    @property
    def out_dim(self) -> int:
        return 2 * self.L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 1) → (..., 2L)"""
        # x * freqs → (..., L)
        xf = x * self.freqs  # broadcast
        return torch.cat([torch.sin(xf), torch.cos(xf)], dim=-1)


class InputNormalizer:
    """将物理量归一化到 [0, 1] 或 [-1, 1]。"""

    def __init__(self, phi_range=(0.8, 1.4), z_range=(0.0, 0.090),
                 r_range=(0.0, 0.015), x_prec_range=(0.0, 1.0)):
        self.ranges = {
            "phi": phi_range,
            "z": z_range,
            "r": r_range,
            "x_prec": x_prec_range,
        }

    def normalize(self, name: str, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.ranges[name]
        return (x - lo) / (hi - lo + 1e-12)

    def denormalize(self, name: str, x_norm: torch.Tensor) -> torch.Tensor:
        lo, hi = self.ranges[name]
        return x_norm * (hi - lo) + lo
