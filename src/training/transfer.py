"""燃料迁移学习 (Phase 4)。

策略：冻结共享 Trunk，仅训练:
  - 新燃料 Embedding 向量
  - 碳烟 Arrhenius 参数
  - 子网络最后一层
"""
import torch

from ..models.pinn import RadiationPINN


def setup_transfer(model: RadiationPINN, new_fuel_id: int = 2):
    """为 Phase 4 迁移学习配置模型。

    Args:
        model: 已训练好的 PINN 模型
        new_fuel_id: 新燃料的 ID (默认 2=甲醇)

    Returns:
        可训练参数列表（用于创建新优化器）
    """
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻：新燃料 Embedding
    model.fuel_embed.embedding.weight.requires_grad = True

    # 解冻：Arrhenius 参数
    model.log_C_alpha.requires_grad = True
    model.T_alpha.requires_grad = True
    model.log_C_beta.requires_grad = True
    model.T_beta.requires_grad = True
    model.log_C_omega.requires_grad = True
    model.T_omega.requires_grad = True

    # 解冻：各子网络最后一层
    _unfreeze_last_linear(model.T_net.net)
    _unfreeze_last_linear(model.Y_net.net)
    _unfreeze_last_linear(model.soot_net.net)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"迁移学习: {sum(p.numel() for p in trainable)} 可训练参数 "
          f"(总参数: {sum(p.numel() for p in model.parameters())})")

    return trainable


def _unfreeze_last_linear(module):
    """解冻 Sequential 中最后一个 Linear 层。"""
    for child in reversed(list(module.children())):
        if isinstance(child, torch.nn.Linear):
            for param in child.parameters():
                param.requires_grad = True
            return
