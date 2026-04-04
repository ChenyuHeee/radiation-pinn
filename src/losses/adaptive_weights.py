"""不确定度自适应损失权重 (Kendall et al.)。

L = Σ_k (1 / 2σ_k²) · L_k + log(σ_k)

σ_k 为各任务的可学习参数，自动平衡多个损失项。
"""
import torch
import torch.nn as nn


class AdaptiveWeights(nn.Module):
    """自适应损失权重。

    每个损失项关联一个可学习的 log_sigma 参数。
    """

    def __init__(self, loss_names: list):
        super().__init__()
        self.loss_names = loss_names
        # 初始化 log(σ) = 0 → σ = 1 → 权重 = 0.5
        self.log_sigmas = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.0))
            for name in loss_names
        })

    def forward(self, losses: dict) -> tuple:
        """加权合并损失。

        Args:
            losses: {name: scalar_loss}

        Returns:
            (total_loss, weights_info_dict)
        """
        total = None
        info = {}

        for name in self.loss_names:
            if name not in losses:
                continue
            log_sigma = self.log_sigmas[name]
            sigma_sq = torch.exp(2 * log_sigma)
            weighted = 0.5 / sigma_sq * losses[name] + log_sigma
            total = weighted if total is None else total + weighted
            info[name] = {
                "raw": losses[name].item(),
                "sigma": torch.exp(log_sigma).item(),
                "weighted": weighted.item(),
            }

        if total is None:
            total = torch.tensor(0.0, requires_grad=True)
        return total, info

    @staticmethod
    def _get_device(losses: dict) -> torch.device:
        for v in losses.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")


class FixedWeights(nn.Module):
    """固定权重（Phase 1/2 使用）。"""

    def __init__(self, weights: dict):
        super().__init__()
        self.weights = weights

    def forward(self, losses: dict) -> tuple:
        total = None
        info = {}
        for name, w in self.weights.items():
            if name not in losses:
                continue
            term = w * losses[name]
            total = term if total is None else total + term
            info[name] = {
                "raw": losses[name].item(),
                "weight": w,
                "weighted": term.item(),
            }
        if total is None:
            total = torch.tensor(0.0, requires_grad=True)
        return total, info
