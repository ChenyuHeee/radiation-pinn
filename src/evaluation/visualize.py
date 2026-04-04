"""可视化：温度剖面、碳烟分布、辐射编程曲线、PDE 残差场。"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_temperature_profile(model, cfg, device, save_dir="results"):
    """温度剖面 T(z)：不同 φ 下的预测 vs 实验值对比。"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    phi_values = cfg["data"]["equiv_ratios"]
    z_norm = torch.linspace(0, 1, 100, device=device)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(phi_values)))

    for i, phi in enumerate(phi_values):
        phi_range = (0.8, 1.4)
        phi_norm = (phi - phi_range[0]) / (phi_range[1] - phi_range[0])

        inputs = torch.zeros(100, 4, device=device)
        inputs[:, 0] = phi_norm
        inputs[:, 1] = z_norm
        inputs[:, 2] = 0.0  # r=0 轴线
        inputs[:, 3] = 0.0  # 参考火焰无掺混

        fuel_ids = torch.zeros(100, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(inputs, fuel_ids)
            T_pred = out["T"].squeeze().cpu().numpy()

        z_mm = z_norm.cpu().numpy() * cfg["physics"]["z_max"] * 1000
        ax.plot(z_mm, T_pred, color=colors[i], label=f"φ={phi}")

    ax.set_xlabel("HAB (mm)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("PINN 温度预测剖面")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "temperature_profile.png"), dpi=150)
    plt.close()


def plot_soot_field(model, cfg, device, phi=1.0, fuel_id=1, save_dir="results"):
    """碳烟分布 f_v(r,z) 2D 伪彩色图。"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    n_r, n_z = 50, 80
    r_norm = torch.linspace(0, 1, n_r, device=device)
    z_norm = torch.linspace(0, 1, n_z, device=device)
    R, Z = torch.meshgrid(r_norm, z_norm, indexing="ij")

    phi_range = (0.8, 1.4)
    phi_norm = (phi - phi_range[0]) / (phi_range[1] - phi_range[0])

    inputs = torch.stack([
        torch.full_like(R.flatten(), phi_norm),
        Z.flatten(),
        R.flatten(),
        torch.full_like(R.flatten(), 1.0 if fuel_id == 1 else 0.0),
    ], dim=-1).to(device)

    fuel_ids = torch.full((n_r * n_z,), fuel_id, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inputs, fuel_ids)
        fv = out["fv"].reshape(n_r, n_z).cpu().numpy()

    r_mm = r_norm.cpu().numpy() * cfg["physics"]["r_max"] * 1000
    z_mm = z_norm.cpu().numpy() * cfg["physics"]["z_max"] * 1000

    fig, ax = plt.subplots(figsize=(6, 10))
    c = ax.pcolormesh(r_mm, z_mm, fv.T, cmap="hot", shading="auto")
    plt.colorbar(c, ax=ax, label="$f_v$ (碳烟体积分数)")
    ax.set_xlabel("r (mm)")
    ax.set_ylabel("HAB (mm)")
    ax.set_title(f"碳烟分布 (φ={phi}, fuel_id={fuel_id})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"soot_field_phi{phi}.png"), dpi=150)
    plt.close()


def plot_radiation_programming(model, cfg, device, save_dir="results"):
    """辐射编程曲线：q_rad vs x_prec（掺混比）。"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    x_prec_values = torch.linspace(0, 1, 20, device=device)
    z_positions = [0.3, 0.5, 0.7]  # 归一化高度

    fig, ax = plt.subplots(figsize=(8, 6))

    for z_norm in z_positions:
        q_rads = []
        for xp in x_prec_values:
            inputs = torch.tensor([[0.5, z_norm, 0.0, xp.item()]],
                                  device=device)
            fuel_ids = torch.tensor([1], dtype=torch.long, device=device)

            with torch.no_grad():
                out = model.compute_radiation(inputs, fuel_ids)
                q_rads.append(out["q_rad"].item())

        z_mm = z_norm * cfg["physics"]["z_max"] * 1000
        ax.plot(x_prec_values.cpu().numpy(), q_rads,
                marker="o", label=f"HAB={z_mm:.0f}mm")

    ax.set_xlabel("$x_{prec}$（碳烟前驱体掺混比）")
    ax.set_ylabel("$q''_{rad}$ (W/m²)")
    ax.set_title("辐射编程曲线：掺混比 vs 辐射通量")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "radiation_programming.png"), dpi=150)
    plt.close()


def plot_training_history(history: dict, save_dir="results"):
    """训练损失曲线。"""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(history["epoch"], history["loss"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.set_title("训练损失曲线")
    ax.grid(True, alpha=0.3)

    # 标注阶段分界
    phases_seen = set()
    for i, p in enumerate(history["phase"]):
        if p not in phases_seen:
            ax.axvline(x=history["epoch"][i], color="red", linestyle="--",
                       alpha=0.5)
            ax.text(history["epoch"][i], ax.get_ylim()[1], p,
                    rotation=90, va="top", fontsize=8)
            phases_seen.add(p)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=150)
    plt.close()
