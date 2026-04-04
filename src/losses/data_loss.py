"""数据损失：实验测量值 MSE。"""
import torch
import torch.nn as nn


class DataLoss(nn.Module):
    """数据损失 L_data = w_T·MSE_T + w_fv·MSE_fv + w_rad·MSE_rad。"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    @staticmethod
    def _safe_mse(pred, target):
        """NaN-safe MSE: 跳过含 NaN 的样本(初始化阶段可能出现)。"""
        valid = torch.isfinite(pred) & torch.isfinite(target)
        if not valid.any():
            # 返回带梯度的零损失，避免 autograd 断裂
            return (pred * 0.0).sum()
        return torch.nn.functional.mse_loss(pred[valid], target[valid])

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

        # 温度 (归一化：除以 T_scale² 使梯度量级合理)
        if masks["T_K"].any():
            m = masks["T_K"]
            T_pred = predictions["T"][m].squeeze(-1) / 1000.0
            T_true = targets["T_K"][m] / 1000.0
            losses["T"] = self._safe_mse(T_pred, T_true)

        # 碳烟
        if masks["fv"].any():
            m = masks["fv"]
            fv_pred = predictions["fv"][m].squeeze(-1)
            fv_true = targets["fv"][m]
            losses["fv"] = self._safe_mse(fv_pred, fv_true)

        # 辐射 (q_rad 仅在 compute_radiation() 后可用，归一化)
        if "q_rad" in predictions and masks["q_rad"].any():
            m = masks["q_rad"]
            q_pred = predictions["q_rad"][m] / 1000.0
            q_true = targets["q_rad"][m] / 1000.0
            losses["rad"] = self._safe_mse(q_pred, q_true)

        # 组分 (只比较有标注的组分通道)
        if masks.get("species") is not None and masks["species"].any():
            m = masks["species"]  # (B,) 有组分标注的样本
            Y_pred = predictions["Y"][m]  # (N, K)
            Y_true = targets["species"][m]  # (N, K)
            sp_mask = targets["species_mask"][m]  # (N, K) 每个组分是否有值
            if sp_mask.any():
                losses["species"] = self._safe_mse(Y_pred[sp_mask], Y_true[sp_mask])

        # 消光法：tau>=1 → fv 应为 0; tau<1 → -ln(tau) 提供相对碳烟约束
        if masks.get("extinction") is not None and masks["extinction"].any():
            m = masks["extinction"]
            fv_pred = predictions["fv"][m].squeeze(-1)  # (N,)
            ext_target = targets["ext_target"][m]         # (N,)
            ext_type = targets["ext_type"][m]             # (N,) 0=zero-soot, 1=has-soot

            # type 0: 未检测到碳烟 → fv 应为 0
            m0 = ext_type == 0
            if m0.any():
                losses["ext_zero"] = self._safe_mse(
                    fv_pred[m0], torch.zeros_like(fv_pred[m0]))

            # type 1: 检测到碳烟 → -ln(tau) 与 fv 成正比，
            # 用归一化排序损失：fv/max(fv) ≈ -ln(tau)/max(-ln(tau))
            m1 = ext_type == 1
            if m1.any() and m1.sum() > 1:
                nlnt = ext_target[m1]
                fv1 = fv_pred[m1]
                nlnt_n = nlnt / (nlnt.max() + 1e-8)
                fv1_n = fv1 / (fv1.max().detach() + 1e-8)
                losses["ext_rank"] = self._safe_mse(fv1_n, nlnt_n)

        return losses
