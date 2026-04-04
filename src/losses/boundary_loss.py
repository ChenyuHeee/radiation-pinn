"""边界条件损失。"""
import torch
import torch.nn as nn


class BoundaryLoss(nn.Module):
    """边界条件损失 L_BC。

    四类边界：
    1. z=0 (inlet): T = T_inlet, f_v = 0
    2. r=0 (axis):  ∂T/∂r = 0, ∂f_v/∂r = 0 (轴对称)
    3. r=r_max (farfield): T → T_amb, f_v → 0
    4. z=z_max (outlet): 自然边界（弱约束）
    """

    def __init__(self, T_amb: float = 300.0, T_inlet: float = 400.0):
        super().__init__()
        self.T_amb = T_amb
        self.T_inlet = T_inlet

    def forward(self, model, boundary_data: dict) -> dict:
        """计算边界损失。

        Args:
            model: RadiationPINN
            boundary_data: dict from BoundarySampler.sample()
                {inlet: (inputs, fuel_ids), axis: ..., farfield: ..., outlet: ...}
        """
        losses = {}

        # ─── 入口 z=0 ───
        inp, fid = boundary_data["inlet"]
        out = model(inp, fid)
        losses["inlet_T"] = torch.mean((out["T"] - self.T_inlet) ** 2)
        losses["inlet_fv"] = torch.mean(out["fv"] ** 2)

        # ─── 轴对称 r=0 ───
        inp, fid = boundary_data["axis"]
        inp = inp.requires_grad_(True)
        out = model(inp, fid)

        # ∂T/∂r = 0
        dT_dr = torch.autograd.grad(
            out["T"], inp,
            grad_outputs=torch.ones_like(out["T"]),
            create_graph=True,
        )[0][:, 2:3]
        losses["axis_dTdr"] = torch.mean(dT_dr ** 2)

        # ∂fv/∂r = 0
        dfv_dr = torch.autograd.grad(
            out["fv"], inp,
            grad_outputs=torch.ones_like(out["fv"]),
            create_graph=True,
        )[0][:, 2:3]
        losses["axis_dfvdr"] = torch.mean(dfv_dr ** 2)

        # ─── 远场 r=r_max ───
        inp, fid = boundary_data["farfield"]
        out = model(inp, fid)
        losses["far_T"] = torch.mean((out["T"] - self.T_amb) ** 2)
        losses["far_fv"] = torch.mean(out["fv"] ** 2)

        # 汇总
        total = sum(losses.values())
        losses["total"] = total
        return losses
