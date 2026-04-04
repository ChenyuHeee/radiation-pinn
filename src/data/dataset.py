"""PyTorch Dataset 和配点采样器。"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class FlameDataset(Dataset):
    """火焰实验数据集，用于 PINN 数据损失。

    每个样本包含：
      - inputs: (phi_norm, z_norm, r, x_prec, fuel_id)
      - targets: dict of available measurements {T_K, fv, q_rad}
    """

    def __init__(self, temperature: pd.DataFrame,
                 radiation: pd.DataFrame,
                 soot: pd.DataFrame,
                 species: pd.DataFrame = None,
                 phi_range=(0.8, 1.4),
                 z_max=0.090):
        super().__init__()
        self.samples = []
        self._build_temperature_samples(temperature, phi_range, z_max)
        self._build_radiation_samples(radiation, phi_range, z_max)
        self._build_soot_samples(soot, phi_range, z_max)
        self._build_species_samples(species, phi_range, z_max)

    def _build_temperature_samples(self, df, phi_range, z_max):
        if df is None or len(df) == 0:
            return
        for _, row in df.iterrows():
            phi_norm = (row["phi"] - phi_range[0]) / (phi_range[1] - phi_range[0])
            z_norm = (row["hab_mm"] / 1000.0) / z_max  # mm → m → 归一化
            fuel_id = int(row["fuel_id"])
            # x_prec: 甲苯=1.0, 其他=0.0（简化编程因子）
            x_prec = 1.0 if fuel_id == 1 else 0.0
            self.samples.append({
                "phi": phi_norm,
                "z": z_norm,
                "r": 0.0,  # 轴线测量 r=0
                "x_prec": x_prec,
                "fuel_id": fuel_id,
                "T_K": row["T_K"],
                "has_T": True,
                "has_fv": False,
                "has_rad": False,
                "has_species": False,
                "species_vec": [0.0] * 6,
            })

    def _build_radiation_samples(self, df, phi_range, z_max):
        if df is None or len(df) == 0:
            return
        for _, row in df.iterrows():
            phi_norm = (row["phi"] - phi_range[0]) / (phi_range[1] - phi_range[0])
            # HAB index → 估算高度（辐射传感器位置）
            z_norm = min((row["hab_index"] + 1) * 5.0 / 1000.0 / z_max, 1.0)
            fuel_id = int(row["fuel_id"])
            x_prec = 1.0 if fuel_id == 1 else 0.0
            self.samples.append({
                "phi": phi_norm,
                "z": z_norm,
                "r": 0.0,
                "x_prec": x_prec,
                "fuel_id": fuel_id,
                "q_rad": row["q_rad_Wm2"],
                "has_T": False,
                "has_fv": False,
                "has_rad": True,
                "has_species": False,
                "species_vec": [0.0] * 6,
            })

    def _build_soot_samples(self, df, phi_range, z_max):
        if df is None or len(df) == 0:
            return
        from .preprocess import FUEL_NAME_TO_ID
        for _, row in df.iterrows():
            fuel_name = row.get("fuel_name", "")
            fuel_id = FUEL_NAME_TO_ID.get(fuel_name, 0)
            phi_norm = (row["phi"] - phi_range[0]) / (phi_range[1] - phi_range[0])
            z_norm = (row["hab_mm"] / 1000.0) / z_max
            x_prec = 1.0 if fuel_id == 1 else 0.0
            self.samples.append({
                "phi": phi_norm,
                "z": z_norm,
                "r": 0.0,
                "x_prec": x_prec,
                "fuel_id": fuel_id,
                "fv": row["fv_mean"],
                "has_T": False,
                "has_fv": True,
                "has_rad": False,
                "has_species": False,
                "species_vec": [0.0] * 6,
            })

    # 组分索引映射：SpeciesNet 输出 [CO2, H2O, CO, C2H2/C7H8, O2, N2]
    SPECIES_COL_TO_IDX = {"CO2": 0, "H2O": 1, "CO": 2, "O2": 4}

    def _build_species_samples(self, df, phi_range, z_max):
        if df is None or len(df) == 0:
            return
        for _, row in df.iterrows():
            eq = row["equiv_ratio"]
            # 当量比 = 1/过量空气系数，映射到 phi (=过量空气系数)
            # 数据中 equiv_ratio 即当量比，对应的 phi = 1/equiv_ratio
            phi = 1.0 / eq if eq > 0 else 1.0
            phi_norm = (phi - phi_range[0]) / (phi_range[1] - phi_range[0])
            z_norm = (row["hab_mm"] / 1000.0) / z_max
            # 这些是甲醇燃料数据，fuel_id=2 (甲醇)
            fuel_id = 2
            x_prec = 0.0

            # 构建 6 维目标向量和掩码 (只有有效组分计入损失)
            sp_vec = [0.0] * 6
            sp_mask = [False] * 6
            for col, idx in self.SPECIES_COL_TO_IDX.items():
                val = row.get(col, np.nan)
                if pd.notna(val):
                    # 数据单位是百分比体积，需要除以 100 转换为小数
                    sp_vec[idx] = float(val) / 100.0
                    sp_mask[idx] = True

            if not any(sp_mask):
                continue

            self.samples.append({
                "phi": phi_norm,
                "z": z_norm,
                "r": 0.0,
                "x_prec": x_prec,
                "fuel_id": fuel_id,
                "has_T": False,
                "has_fv": False,
                "has_rad": False,
                "has_species": True,
                "species_vec": sp_vec,
                "species_mask": sp_mask,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        inputs = torch.tensor(
            [s["phi"], s["z"], s["r"], s["x_prec"]], dtype=torch.float32)
        fuel_id = torch.tensor(s["fuel_id"], dtype=torch.long)

        targets = {}
        mask = {}
        for key in ("T_K", "fv", "q_rad"):
            if s.get(f"has_{key.split('_')[0] if '_' in key else key}", False) or \
               (key == "T_K" and s.get("has_T", False)) or \
               (key == "q_rad" and s.get("has_rad", False)) or \
               (key == "fv" and s.get("has_fv", False)):
                targets[key] = torch.tensor(s.get(key, 0.0), dtype=torch.float32)
                mask[key] = True
            else:
                targets[key] = torch.tensor(0.0, dtype=torch.float32)
                mask[key] = False

        # 组分
        targets["species"] = torch.tensor(s.get("species_vec", [0.0] * 6), dtype=torch.float32)
        mask["species"] = s.get("has_species", False)
        targets["species_mask"] = torch.tensor(
            s.get("species_mask", [False] * 6), dtype=torch.bool)

        return inputs, fuel_id, targets, mask


def collate_flame(batch):
    """自定义 collate，处理异构 targets。"""
    inputs = torch.stack([b[0] for b in batch])
    fuel_ids = torch.stack([b[1] for b in batch])

    targets = {}
    masks = {}
    for key in ("T_K", "fv", "q_rad"):
        targets[key] = torch.stack([b[2][key] for b in batch])
        masks[key] = torch.tensor([b[3][key] for b in batch], dtype=torch.bool)

    targets["species"] = torch.stack([b[2]["species"] for b in batch])
    targets["species_mask"] = torch.stack([b[2]["species_mask"] for b in batch])
    masks["species"] = torch.tensor([b[3]["species"] for b in batch], dtype=torch.bool)

    return inputs, fuel_ids, targets, masks


class CollocationSampler:
    """PDE 配点采样器：在火焰域内均匀随机采样。"""

    def __init__(self, n_points: int = 10000,
                 phi_range=(0.0, 1.0),
                 z_range=(0.0, 1.0),
                 r_range=(0.0, 1.0),
                 num_fuels: int = 3,
                 device: str = "cpu"):
        self.n_points = n_points
        self.phi_range = phi_range
        self.z_range = z_range
        self.r_range = r_range
        self.num_fuels = num_fuels
        self.device = device

    def sample(self) -> tuple:
        """返回 (inputs, fuel_ids)，inputs 需要梯度。"""
        phi = torch.rand(self.n_points, 1, device=self.device) * \
              (self.phi_range[1] - self.phi_range[0]) + self.phi_range[0]
        z = torch.rand(self.n_points, 1, device=self.device) * \
            (self.z_range[1] - self.z_range[0]) + self.z_range[0]
        r = torch.rand(self.n_points, 1, device=self.device) * \
            (self.r_range[1] - self.r_range[0]) + self.r_range[0]
        x_prec = torch.rand(self.n_points, 1, device=self.device)

        inputs = torch.cat([phi, z, r, x_prec], dim=-1)
        inputs.requires_grad_(True)

        fuel_ids = torch.randint(0, self.num_fuels, (self.n_points,),
                                 device=self.device)
        return inputs, fuel_ids


class BoundarySampler:
    """边界条件采样器。"""

    def __init__(self, n_points: int = 2000, num_fuels: int = 3,
                 device: str = "cpu"):
        self.n_points = n_points
        self.num_fuels = num_fuels
        self.device = device

    def sample(self) -> dict:
        """采样四类边界点。"""
        n = self.n_points // 4
        device = self.device

        # z=0（入口），r 随机, phi 随机
        inlet = self._make_inputs(
            phi=torch.rand(n, 1, device=device),
            z=torch.zeros(n, 1, device=device),
            r=torch.rand(n, 1, device=device),
            x_prec=torch.rand(n, 1, device=device),
        )

        # r=0（轴对称），z 随机
        axis = self._make_inputs(
            phi=torch.rand(n, 1, device=device),
            z=torch.rand(n, 1, device=device),
            r=torch.zeros(n, 1, device=device),
            x_prec=torch.rand(n, 1, device=device),
        )

        # r=r_max（远场），z 随机
        farfield = self._make_inputs(
            phi=torch.rand(n, 1, device=device),
            z=torch.rand(n, 1, device=device),
            r=torch.ones(n, 1, device=device),  # 归一化 r_max=1
            x_prec=torch.rand(n, 1, device=device),
        )

        # z=z_max（出口补充）
        outlet = self._make_inputs(
            phi=torch.rand(n, 1, device=device),
            z=torch.ones(n, 1, device=device),
            r=torch.rand(n, 1, device=device),
            x_prec=torch.rand(n, 1, device=device),
        )

        fuel_ids = torch.randint(0, self.num_fuels, (n,), device=device)
        return {
            "inlet": (inlet, fuel_ids),
            "axis": (axis, fuel_ids),
            "farfield": (farfield, fuel_ids),
            "outlet": (outlet, fuel_ids),
        }

    def _make_inputs(self, phi, z, r, x_prec):
        inputs = torch.cat([phi, z, r, x_prec], dim=-1)
        inputs.requires_grad_(True)
        return inputs
