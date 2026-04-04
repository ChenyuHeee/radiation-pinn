"""子网络：温度 N_T, 组分 N_Y, 碳烟 N_soot。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差块: FC → SiLU → LayerNorm → FC → 残差连接。"""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.silu(self.fc1(x))
        out = self.fc2(out)
        return self.norm(out + residual)


class SharedTrunk(nn.Module):
    """共享特征提取器。

    FC(d_in→hidden) → SiLU → LayerNorm → [ResBlock × N] → FC → SiLU → LayerNorm
    输出: h ∈ ℝ^hidden
    """

    def __init__(self, d_in: int, hidden: int = 128, num_resblocks: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(d_in, hidden)
        self.norm_in = nn.LayerNorm(hidden)
        self.resblocks = nn.ModuleList([ResBlock(hidden) for _ in range(num_resblocks)])
        self.output_proj = nn.Linear(hidden, hidden)
        self.norm_out = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_in(F.silu(self.input_proj(x)))
        for block in self.resblocks:
            h = block(h)
        h = self.norm_out(F.silu(self.output_proj(h)))
        return h


class TemperatureNet(nn.Module):
    """温度子网络 N_T。

    h(hidden) → FC→SiLU→FC→SiLU→FC → Softplus → ×T_scale + T_offset
    保证 T > 0。
    """

    def __init__(self, hidden: int = 128, sub_hidden: list = None,
                 T_scale: float = 2000.0, T_offset: float = 300.0):
        super().__init__()
        if sub_hidden is None:
            sub_hidden = [64, 32]
        self.T_scale = T_scale
        self.T_offset = T_offset

        layers = []
        in_dim = hidden
        for h in sub_hidden:
            layers.extend([nn.Linear(in_dim, h), nn.SiLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """返回温度 T (K)，形状 (B, 1)。"""
        raw = self.net(h)
        return F.softplus(raw) * self.T_scale + self.T_offset


class SpeciesNet(nn.Module):
    """组分子网络 N_Y。

    h(hidden) → FC→SiLU→FC → Softmax
    保证 Y_k ∈ [0,1] 且 ΣY_k = 1。
    输出 K 种组分: CO2, H2O, CO, C2H2/C7H8, O2, N2
    """

    def __init__(self, hidden: int = 128, sub_hidden: int = 64,
                 num_species: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, sub_hidden),
            nn.SiLU(),
            nn.Linear(sub_hidden, num_species),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """返回组分质量分数 (B, K)，各行和为1。"""
        return F.softmax(self.net(h), dim=-1)


class SootNet(nn.Module):
    """碳烟子网络 N_soot。

    输入: concat[h(hidden), T(1), Y_prec(1), Y_O2(1)]
    → FC→SiLU→FC→SiLU→FC → Softplus
    保证 f_v ≥ 0。

    跨子网络因果连接：碳烟生成依赖温度和前驱体浓度。
    """

    def __init__(self, hidden: int = 128, sub_hidden: list = None):
        super().__init__()
        if sub_hidden is None:
            sub_hidden = [64, 32]
        in_dim = hidden + 3  # h + T + Y_prec + Y_O2

        layers = []
        for h in sub_hidden:
            layers.extend([nn.Linear(in_dim, h), nn.SiLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, T: torch.Tensor,
                Y_prec: torch.Tensor, Y_O2: torch.Tensor) -> torch.Tensor:
        """返回碳烟体积分数 f_v (B, 1)。"""
        x = torch.cat([h, T, Y_prec, Y_O2], dim=-1)
        return F.softplus(self.net(x))
