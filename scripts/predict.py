"""推理脚本：加载训练好的 PINN 进行预测。

Usage:
    python scripts/predict.py --checkpoint checkpoints/radiation_pinn_final.pt
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np

from src.models.pinn import RadiationPINN
from src.training.trainer import get_device


def main():
    parser = argparse.ArgumentParser(description="RadiationPINN 推理")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--config", type=str,
                        default="src/configs/default.yaml")
    parser.add_argument("--phi", type=float, default=1.0,
                        help="当量比")
    parser.add_argument("--fuel", type=int, default=0,
                        help="燃料 ID (0=参考, 1=甲苯, 2=甲醇)")
    parser.add_argument("--x_prec", type=float, default=0.0,
                        help="碳烟前驱体掺混比")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg)

    # 加载模型
    model = RadiationPINN(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # 预测：沿轴线的温度和碳烟分布
    phi_range = (0.8, 1.4)
    phi_norm = (args.phi - phi_range[0]) / (phi_range[1] - phi_range[0])

    n_z = 50
    z_norm = torch.linspace(0, 1, n_z, device=device)

    inputs = torch.zeros(n_z, 4, device=device)
    inputs[:, 0] = phi_norm
    inputs[:, 1] = z_norm
    inputs[:, 2] = 0.0  # r=0
    inputs[:, 3] = args.x_prec

    fuel_ids = torch.full((n_z,), args.fuel, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inputs, fuel_ids)

    z_mm = z_norm.cpu().numpy() * cfg["physics"]["z_max"] * 1000
    T = out["T"].squeeze().cpu().numpy()
    fv = out["fv"].squeeze().cpu().numpy()

    print(f"\n{'HAB(mm)':>10} {'T(K)':>10} {'fv':>12}")
    print("-" * 35)
    for i in range(n_z):
        print(f"{z_mm[i]:10.1f} {T[i]:10.1f} {fv[i]:12.6e}")

    # 辐射预测
    with torch.no_grad():
        rad_out = model.compute_radiation(inputs[:1], fuel_ids[:1])
        print(f"\n辐射热通量 q_rad = {rad_out['q_rad'].item():.2f} W/m²")


if __name__ == "__main__":
    main()
