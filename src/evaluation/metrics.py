"""精度指标 + 物理一致性指标。"""
import torch
import numpy as np


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def r2_score(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def mape(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.abs(true) > 1e-6
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100)


def evaluate_predictions(model, dataset, device, cfg) -> dict:
    """对整个数据集评估模型。"""
    from torch.utils.data import DataLoader
    from ..data.dataset import collate_flame

    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False,
                        collate_fn=collate_flame)

    all_T_pred, all_T_true = [], []
    all_fv_pred, all_fv_true = [], []
    all_rad_pred, all_rad_true = [], []

    with torch.no_grad():
        for inputs, fuel_ids, targets, masks in loader:
            inputs = inputs.to(device)
            fuel_ids = fuel_ids.to(device)
            out = model(inputs, fuel_ids)

            # 温度
            if masks["T_K"].any():
                m = masks["T_K"]
                all_T_pred.append(out["T"][m].squeeze(-1).cpu().numpy())
                all_T_true.append(targets["T_K"][m].numpy())

            # 碳烟
            if masks["fv"].any():
                m = masks["fv"]
                all_fv_pred.append(out["fv"][m].squeeze(-1).cpu().numpy())
                all_fv_true.append(targets["fv"][m].numpy())

    results = {}
    if all_T_pred:
        T_p = np.concatenate(all_T_pred)
        T_t = np.concatenate(all_T_true)
        results["T_rmse"] = rmse(T_p, T_t)
        results["T_r2"] = r2_score(T_p, T_t)
        results["T_mape"] = mape(T_p, T_t)

    if all_fv_pred:
        fv_p = np.concatenate(all_fv_pred)
        fv_t = np.concatenate(all_fv_true)
        results["fv_rmse"] = rmse(fv_p, fv_t)
        results["fv_r2"] = r2_score(fv_p, fv_t)

    return results


def check_physics_consistency(model, cfg, device, n_points=5000) -> dict:
    """检查物理一致性指标。"""
    from ..data.dataset import CollocationSampler
    from ..losses.pde_loss import PDELoss

    sampler = CollocationSampler(
        n_points=n_points,
        num_fuels=cfg["model"]["num_fuels"],
        device=device,
    )
    pde_fn = PDELoss()

    model.eval()
    inputs, fuel_ids = sampler.sample()

    with torch.enable_grad():
        pde_losses = pde_fn(model, inputs, fuel_ids)

    # 检查非物理输出
    with torch.no_grad():
        out = model(inputs, fuel_ids)
        T_negative = (out["T"] < 0).float().mean().item()
        fv_negative = (out["fv"] < 0).float().mean().item()

    return {
        "energy_residual_L2": pde_losses["energy"].item() ** 0.5,
        "soot_residual_L2": pde_losses["soot"].item() ** 0.5,
        "T_negative_rate": T_negative,
        "fv_negative_rate": fv_negative,
    }
