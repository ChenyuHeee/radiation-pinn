"""物理约束层：可微分 RTE 辐射传递积分。

将碳烟场 f_v 和温度场 T 转化为辐射热通量 q_rad。
全程使用 PyTorch 张量运算，支持自动微分反向传播。
"""
import torch
import torch.nn as nn
import math


class RTEIntegrationLayer(nn.Module):
    """辐射传递方程 (RTE) 可微分视线积分层。

    物理流程:
    1. κ(r,z) = 6πE(m)/λ · f_v(r,z)       碳烟吸收系数
    2. S(r,z) = κ · σ · T⁴ / π             局部辐射源项
    3. q''_rad(z) = 2 ∫₀^rmax S·exp(-∫_r^rmax κ dr') dr   视线积分

    这是模型的核心创新层——辐射物理嵌入前向路径，
    使辐射实测值的梯度可以直接回传到温度和碳烟子网络。
    """

    def __init__(self, E_m: float = 0.37, lam: float = 633e-9,
                 sigma: float = 5.67e-8, r_max: float = 0.015,
                 r_points: int = 32):
        super().__init__()
        self.register_buffer("E_m", torch.tensor(E_m))
        self.register_buffer("lam", torch.tensor(lam))
        self.register_buffer("sigma", torch.tensor(sigma))
        self.r_max = r_max
        self.r_points = r_points

        # 预计算径向网格（均匀）
        r_grid = torch.linspace(0, r_max, r_points)
        self.register_buffer("r_grid", r_grid)

        # 碳烟吸收系数前因子
        coeff = 6.0 * math.pi * E_m / lam
        self.register_buffer("kappa_coeff", torch.tensor(coeff))

    def compute_kappa(self, fv: torch.Tensor) -> torch.Tensor:
        """碳烟吸收系数 κ = 6πE(m)/λ · f_v。"""
        return self.kappa_coeff * fv

    def compute_source(self, kappa: torch.Tensor,
                       T: torch.Tensor) -> torch.Tensor:
        """局部辐射源项 S = κ σ T⁴ / π。"""
        return kappa * self.sigma * T.pow(4) / math.pi

    def forward(self, T_field: torch.Tensor,
                fv_field: torch.Tensor) -> torch.Tensor:
        """计算辐射热通量。

        Args:
            T_field:  (B, N_r) 沿径向的温度分布 (K)
            fv_field: (B, N_r) 沿径向的碳烟体积分数

        Returns:
            q_rad: (B,) 辐射热通量 (W/m²)
        """
        kappa = self.compute_kappa(fv_field)          # (B, N_r)
        source = self.compute_source(kappa, T_field)  # (B, N_r)

        # 累积光学深度（从外向内）
        # τ(r) = ∫_r^r_max κ(r') dr'
        kappa_flip = torch.flip(kappa, dims=[-1])     # 翻转为从 r_max 到 0
        r_flip = torch.flip(self.r_grid, dims=[0])

        # cumulative_trapezoid: 沿翻转方向积分
        tau_flip = self._cumtrapz(kappa_flip, r_flip)
        # 补零（r_max 处光学深度为 0）
        tau_flip = torch.cat([
            torch.zeros_like(tau_flip[..., :1]), tau_flip
        ], dim=-1)
        tau = torch.flip(tau_flip, dims=[-1])         # 翻回 0→r_max 方向

        transmittance = torch.exp(-tau)               # (B, N_r)

        # 视线积分: q = 2 ∫₀^rmax S(r) · T(r) dr
        integrand = source * transmittance            # (B, N_r)
        q_rad = 2.0 * torch.trapezoid(integrand, self.r_grid, dim=-1)

        return q_rad  # (B,)

    def _cumtrapz(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """可微分累积梯形积分。

        y: (..., N), x: (N,)
        返回 (..., N-1) 的累积积分值。
        """
        dx = x[1:] - x[:-1]  # (N-1,)
        avg = 0.5 * (y[..., :-1] + y[..., 1:])  # (..., N-1)
        increments = avg * dx  # (..., N-1)
        return torch.cumsum(increments, dim=-1)


class PlanckAbsorption(nn.Module):
    """Planck 平均吸收系数: κ_P = 3.83 f_v T / C2。

    用于全波段灰体近似。
    """

    def __init__(self, C2: float = 1.4388e-2):
        super().__init__()
        self.register_buffer("C2", torch.tensor(C2))

    def forward(self, fv: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return 3.83 * fv * T / self.C2
