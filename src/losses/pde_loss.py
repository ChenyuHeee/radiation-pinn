"""PDE 残差损失：能量守恒 + 碳烟输运方程。

通过 PyTorch autograd 对网络输出求高阶空间导数，零残差代表物理一致。
"""
import torch
import torch.nn as nn


class PDELoss(nn.Module):
    """PDE 物理残差损失。

    energy_residual: ρ·cp·vz·∂T/∂z - (1/r)·∂(r·k·∂T/∂r)/∂r - k·∂²T/∂z² - q_chem + q_rad = 0
    soot_residual:   vz·∂fv/∂z - S_nuc - S_grow + S_ox = 0
    """

    def __init__(self, rho: float = 0.3, cp: float = 1200.0,
                 vz: float = 0.5, k_thermal: float = 0.06):
        """简化假设的物性参数。

        Args:
            rho: 气体密度 kg/m³ (高温近似)
            cp: 比热 J/(kg·K)
            vz: 轴向速度 m/s (层流扩散火焰典型值)
            k_thermal: 导热系数 W/(m·K) (1000K 级别)
        """
        super().__init__()
        self.rho = rho
        self.cp = cp
        self.vz = vz
        self.k_thermal = k_thermal

    def forward(self, model, inputs: torch.Tensor,
                fuel_ids: torch.Tensor) -> dict:
        """计算 PDE 残差。

        Args:
            model: RadiationPINN 模型
            inputs: (N, 4) 需要梯度的配点坐标 [phi, z, r, x_prec]
            fuel_ids: (N,)

        Returns:
            dict: {energy_residual, soot_residual, total}
        """
        inputs = inputs.requires_grad_(True)
        out = model(inputs, fuel_ids)

        T = out["T"]       # (N, 1)
        fv = out["fv"]     # (N, 1)
        Y = out["Y"]       # (N, K)
        Y_prec = Y[:, 3:4]
        Y_O2 = Y[:, 4:5]

        z = inputs[:, 1:2]
        r = inputs[:, 2:3]

        # ─── 能量方程残差 ───
        dT_dz = self._grad(T, inputs, 1)    # ∂T/∂z
        dT_dr = self._grad(T, inputs, 2)    # ∂T/∂r
        d2T_dz2 = self._grad(dT_dz, inputs, 1)  # ∂²T/∂z²
        d2T_dr2 = self._grad(dT_dr, inputs, 2)  # ∂²T/∂r²

        # 轴对称: (1/r)·∂(r·∂T/∂r)/∂r = ∂²T/∂r² + (1/r)·∂T/∂r
        # 注意 r=0 时用 L'Hôpital: lim(r→0) (1/r)·∂T/∂r = ∂²T/∂r²
        r_safe = r.clamp(min=1e-6)  # 避免除零
        laplacian_r = d2T_dr2 + dT_dr / r_safe

        # 简化：忽略化学源项和辐射散热项（在训练初期）
        energy_res = (self.rho * self.cp * self.vz * dT_dz
                      - self.k_thermal * laplacian_r
                      - self.k_thermal * d2T_dz2)

        # ─── 碳烟输运方程残差 ───
        dfv_dz = self._grad(fv, inputs, 1)  # ∂fv/∂z

        # Arrhenius 源项
        src = model.soot_source_terms(T, Y_prec, Y_O2, fv)
        soot_res = (self.vz * dfv_dz
                    - src["nucleation"]
                    - src["growth"]
                    + src["oxidation"])

        # MSE 残差
        loss_energy = torch.mean(energy_res ** 2)
        loss_soot = torch.mean(soot_res ** 2)
        total = loss_energy + loss_soot

        return {
            "energy": loss_energy,
            "soot": loss_soot,
            "total": total,
        }

    @staticmethod
    def _grad(y: torch.Tensor, x: torch.Tensor,
              dim: int) -> torch.Tensor:
        """对 x 的第 dim 个分量求 y 的梯度。"""
        grad = torch.autograd.grad(
            y, x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grad[:, dim:dim + 1]
