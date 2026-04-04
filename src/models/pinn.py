"""RadiationPINN 主模型 —— 多燃料辐射编程 PINN。

架构:
  输入(phi, z, r, x_prec, fuel_id)
    → 傅里叶编码 + 燃料Embedding + Concat
    → 共享 Trunk (ResBlock × 3)
    → 温度子网络 N_T → T(r,z)
    → 组分子网络 N_Y → Y_k(r,z)
    → 碳烟子网络 N_soot(h, T, Y_prec, Y_O2) → f_v(r,z)
    → 物理约束层 RTE → q_rad(z)
"""
import torch
import torch.nn as nn

from .embeddings import FuelEmbedding
from .subnetworks import SharedTrunk, TemperatureNet, SpeciesNet, SootNet
from .physics_layer import RTEIntegrationLayer
from ..data.transforms import FourierFeatures


class RadiationPINN(nn.Module):
    """多燃料辐射编程 PINN 模型。"""

    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        pcfg = cfg["physics"]

        # ─── 输入编码 ───
        self.fourier_r = FourierFeatures(L=mcfg["fourier_L"])
        self.fourier_z = FourierFeatures(L=mcfg["fourier_L"])
        self.fuel_embed = FuelEmbedding(
            num_fuels=mcfg["num_fuels"],
            embed_dim=mcfg["fuel_embed_dim"],
        )

        # d_in = 1(phi) + 2L(r) + 2L(z) + 1(x_prec) + embed_dim
        fourier_dim = self.fourier_r.out_dim  # 2L
        d_in = 1 + fourier_dim + fourier_dim + 1 + mcfg["fuel_embed_dim"]

        # ─── 共享 Trunk ───
        self.trunk = SharedTrunk(
            d_in=d_in,
            hidden=mcfg["trunk_hidden"],
            num_resblocks=mcfg["trunk_num_resblocks"],
        )
        hidden = mcfg["trunk_hidden"]

        # ─── 子网络 ───
        self.T_net = TemperatureNet(
            hidden=hidden,
            sub_hidden=mcfg["sub_hidden"],
            T_scale=mcfg["T_scale"],
            T_offset=mcfg["T_offset"],
        )
        self.Y_net = SpeciesNet(
            hidden=hidden,
            sub_hidden=mcfg["sub_hidden"][0],
            num_species=mcfg["num_species"],
        )
        self.soot_net = SootNet(
            hidden=hidden,
            sub_hidden=mcfg["sub_hidden"],
        )

        # ─── 物理约束层 ───
        self.rte_layer = RTEIntegrationLayer(
            E_m=pcfg["E_m"],
            lam=pcfg["lambda_laser"],
            sigma=pcfg["sigma"],
            r_max=pcfg["r_max"],
            r_points=pcfg["r_points"],
        )

        # ─── 碳烟 Arrhenius 可学习参数 ───
        # 成核: C_alpha * exp(-T_alpha / T)
        # 生长: C_beta * exp(-T_beta / T)
        # 氧化: C_omega * T^0.5 * exp(-T_omega / T)
        self.log_C_alpha = nn.Parameter(torch.tensor(0.0))
        self.T_alpha = nn.Parameter(torch.tensor(21000.0))
        self.log_C_beta = nn.Parameter(torch.tensor(0.0))
        self.T_beta = nn.Parameter(torch.tensor(12100.0))
        self.log_C_omega = nn.Parameter(torch.tensor(0.0))
        self.T_omega = nn.Parameter(torch.tensor(19680.0))

        self._pcfg = pcfg

    def encode_input(self, inputs: torch.Tensor,
                     fuel_ids: torch.Tensor) -> torch.Tensor:
        """输入编码: (phi, z, r, x_prec) → 拼接傅里叶特征 + 燃料 Embedding。"""
        phi = inputs[:, 0:1]
        z = inputs[:, 1:2]
        r = inputs[:, 2:3]
        x_prec = inputs[:, 3:4]

        z_enc = self.fourier_z(z)      # (B, 2L)
        r_enc = self.fourier_r(r)      # (B, 2L)
        fuel_enc = self.fuel_embed(fuel_ids)  # (B, embed_dim)

        return torch.cat([phi, z_enc, r_enc, x_prec, fuel_enc], dim=-1)

    def forward(self, inputs: torch.Tensor,
                fuel_ids: torch.Tensor) -> dict:
        """前向传播，返回所有中间场。

        Args:
            inputs: (B, 4) — [phi_norm, z_norm, r_norm, x_prec]
            fuel_ids: (B,) — 燃料 ID

        Returns:
            dict with keys: T, Y, fv, q_rad, h
        """
        # 编码
        x = self.encode_input(inputs, fuel_ids)

        # 共享 Trunk
        h = self.trunk(x)

        # 温度
        T = self.T_net(h)       # (B, 1)

        # 组分
        Y = self.Y_net(h)       # (B, K)
        # Y[:, 3] = 碳烟前驱体 (C2H2/C7H8)
        # Y[:, 4] = O2
        Y_prec = Y[:, 3:4]
        Y_O2 = Y[:, 4:5]

        # 碳烟（因果连接）
        fv = self.soot_net(h, T, Y_prec, Y_O2)  # (B, 1)

        return {
            "T": T,         # (B, 1) 温度 K
            "Y": Y,         # (B, K) 组分质量分数
            "fv": fv,       # (B, 1) 碳烟体积分数
            "h": h,         # (B, hidden) 共享特征
        }

    def compute_radiation(self, inputs: torch.Tensor,
                          fuel_ids: torch.Tensor,
                          n_r: int = None) -> dict:
        """完整前向传播（含 RTE 积分计算辐射）。

        对每个样本，在径向 r ∈ [0, r_max] 上均匀采样 n_r 个点，
        分别计算 T(r,z) 和 f_v(r,z)，然后做 RTE 积分得到 q_rad(z)。
        """
        if n_r is None:
            n_r = self._pcfg["r_points"]

        B = inputs.shape[0]
        device = inputs.device

        # 径向网格 (归一化)
        r_vals = torch.linspace(0, 1, n_r, device=device)  # (n_r,)

        # 展开：每个样本在 n_r 个径向位置重复
        # inputs: (B, 4) → (B*n_r, 4)
        inputs_expanded = inputs.unsqueeze(1).expand(-1, n_r, -1)  # (B, n_r, 4)
        r_expanded = r_vals.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)  # (B, n_r, 1)

        # 替换 r 维度
        inputs_r = inputs_expanded.clone()
        inputs_r[:, :, 2:3] = r_expanded
        inputs_r = inputs_r.reshape(B * n_r, 4)

        fuel_ids_r = fuel_ids.unsqueeze(1).expand(-1, n_r).reshape(B * n_r)

        # 前向传播
        out = self.forward(inputs_r, fuel_ids_r)

        # reshape 回 (B, n_r, ...)
        T_field = out["T"].reshape(B, n_r)      # (B, n_r) — T(r) at fixed z
        fv_field = out["fv"].reshape(B, n_r)     # (B, n_r) — f_v(r) at fixed z

        # RTE 积分
        q_rad = self.rte_layer(T_field, fv_field)  # (B,)

        return {
            "T": out["T"].reshape(B, n_r, 1),
            "Y": out["Y"].reshape(B, n_r, -1),
            "fv": out["fv"].reshape(B, n_r, 1),
            "q_rad": q_rad,
            "T_field": T_field,
            "fv_field": fv_field,
        }

    def soot_source_terms(self, T: torch.Tensor, Y_prec: torch.Tensor,
                          Y_O2: torch.Tensor, fv: torch.Tensor,
                          P: float = 101325.0, R: float = 8.314) -> dict:
        """计算碳烟 Arrhenius 源项（用于 PDE 残差）。

        成核: S_nuc = C_α (P/RT) x_prec exp(-T_α/T)
        生长: S_grow = C_β f_v (P/RT) x_prec exp(-T_β/T)
        氧化: S_ox = C_ω f_v (P/RT) x_O2 T^0.5 exp(-T_ω/T)
        """
        C_alpha = torch.exp(self.log_C_alpha)
        C_beta = torch.exp(self.log_C_beta)
        C_omega = torch.exp(self.log_C_omega)

        PRT = P / (R * T.clamp(min=300.0))

        S_nuc = C_alpha * PRT * Y_prec * torch.exp(-self.T_alpha / T.clamp(min=300.0))
        S_grow = C_beta * fv * PRT * Y_prec * torch.exp(-self.T_beta / T.clamp(min=300.0))
        S_ox = C_omega * fv * PRT * Y_O2 * T.sqrt() * torch.exp(-self.T_omega / T.clamp(min=300.0))

        return {"nucleation": S_nuc, "growth": S_grow, "oxidation": S_ox}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
