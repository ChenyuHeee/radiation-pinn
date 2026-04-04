"""数据损失：实验测量值 MSE。"""
import torch
import torch.nn as nn


class DataLoss(nn.Module):
    """数据损失 L_data = w_T·MSE_T + w_fv·MSE_fv + w_rad·MSE_rad。"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, predictions: dict, targets: dict,
                masks: dict) -> dict:
        """计算有标注样本的 MSE 损失。

        Args:
            predictions: model output dict {T: (B,1), fv: (B,1), q_rad: (B,)}
            targets: {T_K: (B,), fv: (B,), q_rad: (B,)}
            masks: {T_K: (B,), fv: (B,), q_rad: (B,)} bool masks

        Returns:
            dict of individual losses
        """
        losses = {}

        # 温度
        if masks["T_K"].any():
            m = masks["T_K"]
            T_pred = predictions["T"][m].squeeze(-1)
            T_true = targets["T_K"][m]
            losses["T"] = self.mse(T_pred, T_true)

        # 碳烟
        if masks["fv"].any():
            m = masks["fv"]
            fv_pred = predictions["fv"][m].squeeze(-1)
            fv_true = targets["fv"][m]
            losses["fv"] = self.mse(fv_pred, fv_true)

        # 辐射
        if masks["q_rad"].any():
            m = masks["q_rad"]
            q_pred = predictions["q_rad"][m]
            q_true = targets["q_rad"][m]
            losses["rad"] = self.mse(q_pred, q_true)

        return losses
