"""逆向辐射编程：给定目标辐射分布，反推最优燃料配方。

冻结模型权重，将 (phi, x_prec) 作为可学习参数进行梯度优化，
最小化预测辐射与目标辐射之间的差异。

Usage:
    python scripts/inverse.py --checkpoint checkpoints/best.pt --target_qrad 5000
    python scripts/inverse.py --checkpoint checkpoints/best.pt --target_profile target.csv
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


def inverse_optimize(model, cfg, device,
                     target_qrad, fuel_id=0,
                     n_z=20, n_steps=500, lr=0.05):
    """逆向优化：搜索最优 (phi, x_prec)，使 q_rad 逼近目标。

    Args:
        model: 训练好的 RadiationPINN
        target_qrad: float 或 array。如果是 float，则所有高度目标一致；
                      如果是 array，则 len == n_z，逐高度匹配。
        fuel_id: 燃料 ID (0=参考, 1=甲苯, 2=甲醇)
        n_z: 轴向采样点数
        n_steps: 优化步数
        lr: 学习率

    Returns:
        dict with optimal parameters and predicted profiles
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 可学习参数（用 sigmoid 约束到合理范围）
    # phi_raw → phi = 0.8 + 0.6 * sigmoid(phi_raw) → [0.8, 1.4]
    # xp_raw  → x_prec = sigmoid(xp_raw) → [0, 1]
    phi_raw = torch.tensor(0.0, device=device, requires_grad=True)
    xp_raw = torch.tensor(0.0, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([phi_raw, xp_raw], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    phi_range = (0.8, 1.4)
    z_norm = torch.linspace(0.05, 0.95, n_z, device=device)

    # 目标辐射
    if isinstance(target_qrad, (int, float)):
        target = torch.full((n_z,), float(target_qrad), device=device)
    else:
        target = torch.tensor(target_qrad, dtype=torch.float32, device=device)

    fuel_ids = torch.full((n_z,), fuel_id, dtype=torch.long, device=device)

    history = {"loss": [], "phi": [], "x_prec": []}

    for step in range(n_steps):
        optimizer.zero_grad()

        phi_norm = torch.sigmoid(phi_raw)  # [0, 1] 归一化
        x_prec = torch.sigmoid(xp_raw)     # [0, 1]

        inputs = torch.zeros(n_z, 4, device=device)
        inputs[:, 0] = phi_norm
        inputs[:, 1] = z_norm
        inputs[:, 2] = 0.0
        inputs[:, 3] = x_prec

        # 逐点计算辐射
        qrad_list = []
        for i in range(n_z):
            out = model.compute_radiation(
                inputs[i:i+1], fuel_ids[i:i+1])
            qrad_list.append(out["q_rad"])
        q_pred = torch.cat(qrad_list)

        loss = torch.nn.functional.mse_loss(q_pred, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 记录
        phi_val = phi_range[0] + (phi_range[1] - phi_range[0]) * torch.sigmoid(phi_raw).item()
        xp_val = torch.sigmoid(xp_raw).item()
        history["loss"].append(loss.item())
        history["phi"].append(phi_val)
        history["x_prec"].append(xp_val)

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{n_steps}  loss={loss.item():.2f}  "
                  f"phi={phi_val:.3f}  x_prec={xp_val:.4f}")

    # 最终结果
    phi_opt = phi_range[0] + (phi_range[1] - phi_range[0]) * torch.sigmoid(phi_raw).item()
    xp_opt = torch.sigmoid(xp_raw).item()

    # 最终预测
    with torch.no_grad():
        inputs_final = torch.zeros(n_z, 4, device=device)
        inputs_final[:, 0] = torch.sigmoid(phi_raw)
        inputs_final[:, 1] = z_norm
        inputs_final[:, 3] = torch.sigmoid(xp_raw)
        fuel_final = torch.full((n_z,), fuel_id, dtype=torch.long, device=device)

        out_final = model(inputs_final, fuel_final)
        T_profile = out_final["T"].squeeze().cpu().numpy()
        fv_profile = out_final["fv"].squeeze().cpu().numpy()

        qrad_final = []
        for i in range(n_z):
            r = model.compute_radiation(inputs_final[i:i+1], fuel_final[i:i+1])
            qrad_final.append(r["q_rad"].item())

    z_mm = z_norm.cpu().numpy() * cfg["physics"]["z_max"] * 1000

    return {
        "phi_opt": phi_opt,
        "x_prec_opt": xp_opt,
        "z_mm": z_mm,
        "T_profile": T_profile,
        "fv_profile": fv_profile,
        "qrad_profile": np.array(qrad_final),
        "target_qrad": target.cpu().numpy(),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="逆向辐射编程")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="src/configs/default.yaml")
    parser.add_argument("--target_qrad", type=float, default=5000.0,
                        help="目标辐射热通量 (W/m²)")
    parser.add_argument("--fuel", type=int, default=0,
                        help="燃料 ID (0=参考, 1=甲苯, 2=甲醇)")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg)

    model = RadiationPINN(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"目标辐射: {args.target_qrad} W/m²")
    print(f"燃料: {args.fuel}")
    print(f"优化步数: {args.steps}")
    print()

    result = inverse_optimize(
        model, cfg, device,
        target_qrad=args.target_qrad,
        fuel_id=args.fuel,
        n_steps=args.steps,
    )

    print(f"\n{'='*40}")
    print(f"最优当量比 φ = {result['phi_opt']:.4f}")
    print(f"最优掺混比 x_prec = {result['x_prec_opt']:.4f}")
    print(f"\n轴向分布:")
    print(f"{'HAB(mm)':>10} {'T(K)':>10} {'fv':>12} {'q_rad':>12} {'target':>12}")
    print("-" * 60)
    for i in range(len(result["z_mm"])):
        print(f"{result['z_mm'][i]:10.1f} {result['T_profile'][i]:10.1f} "
              f"{result['fv_profile'][i]:12.6e} {result['qrad_profile'][i]:12.2f} "
              f"{result['target_qrad'][i]:12.2f}")


if __name__ == "__main__":
    main()
