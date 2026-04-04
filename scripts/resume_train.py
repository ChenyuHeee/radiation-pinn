"""从 checkpoint 恢复训练（使用已修复的 PDE loss）。

Usage:
    python scripts/resume_train.py
    python scripts/resume_train.py --ckpt checkpoints/radiation_pinn_epoch2000.pt
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from src.data.preprocess import build_training_dataset
from src.data.dataset import FlameDataset
from src.models.pinn import RadiationPINN
from src.training.trainer import Trainer, get_device


def main():
    parser = argparse.ArgumentParser(description="RadiationPINN 恢复训练")
    parser.add_argument("--config", type=str,
                        default="src/configs/default.yaml")
    parser.add_argument("--ckpt", type=str,
                        default="checkpoints/radiation_pinn_epoch2000.pt",
                        help="要恢复的 checkpoint 路径")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg)
    print(f"设备: {device}")

    torch.manual_seed(cfg["training"]["seed"])

    # ─── 数据 ───
    print("\n[Step 1] 数据预处理...")
    data = build_training_dataset(cfg)

    val_phi = cfg["training"].get("val_phi", 1.1)
    val_data, train_data = {}, {}
    for key in ("temperature", "radiation", "soot", "extinction"):
        df = data[key]
        if df is None or len(df) == 0:
            val_data[key] = df
            train_data[key] = df
            continue
        phi_col = "phi" if "phi" in df.columns else None
        if phi_col and val_phi is not None:
            mask_val = (df[phi_col] - val_phi).abs() < 0.01
            val_data[key] = df[mask_val].reset_index(drop=True)
            train_data[key] = df[~mask_val].reset_index(drop=True)
        else:
            val_data[key] = None
            train_data[key] = df
    train_data["species"] = data["species"]
    val_data["species"] = None

    dataset = FlameDataset(
        temperature=train_data["temperature"],
        radiation=train_data["radiation"],
        soot=train_data["soot"],
        species=train_data["species"],
        extinction=train_data["extinction"],
    )
    val_dataset = FlameDataset(
        temperature=val_data["temperature"],
        radiation=val_data["radiation"],
        soot=val_data["soot"],
        extinction=val_data["extinction"],
    )
    print(f"  训练样本: {len(dataset)}, 验证样本: {len(val_dataset)}")

    # ─── 模型 ───
    print("\n[Step 2] 加载模型...")
    model = RadiationPINN(cfg)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    resume_epoch = ckpt["epoch"] + 1
    print(f"  从 {args.ckpt} 恢复 (epoch {resume_epoch})")
    print(f"  恢复时 loss = {ckpt['history']['loss'][-1]:.6f}")
    print(f"  可训练参数: {model.count_parameters():,}")

    # ─── 构建 Trainer 并恢复状态 ───
    print("\n[Step 3] 恢复训练...")
    trainer = Trainer(model, dataset, cfg, device, val_dataset=val_dataset)

    # 恢复优化器状态（Adam 动量等）
    try:
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        print("  优化器状态已恢复")
    except Exception as e:
        print(f"  优化器状态恢复失败 ({e})，使用新优化器")

    # 恢复自适应权重
    try:
        trainer.adaptive_weights.load_state_dict(ckpt["adaptive_weights_state"])
        print("  自适应权重已恢复")
    except Exception:
        print("  自适应权重使用默认值")

    # 恢复历史
    trainer.history = ckpt["history"]

    # ─── 修改训练循环：从 resume_epoch 开始 ───
    import time
    total_epochs = trainer.scheduler.total_epochs
    print(f"\n  将从 epoch {resume_epoch} 训练到 {total_epochs}")
    print(f"  PDE loss 已使用修复版本（tanh 压缩 + clamp）")

    prev_phase_name = None
    start_time = time.time()

    for epoch in range(resume_epoch, total_epochs):
        phase = trainer.scheduler.get_phase(epoch)

        if phase["name"] != prev_phase_name:
            print(f"\n{'='*60}")
            print(f"进入阶段: {phase['name']} (epoch {phase['start']}~{phase['end']-1})")
            print(f"  lr={phase['lr']}, use_pde={phase['use_pde']}, "
                  f"use_bc={phase['use_bc']}")
            print(f"{'='*60}")
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = phase["lr"]
            trainer._set_trunk_frozen(phase.get("freeze_trunk", False))
            prev_phase_name = phase["name"]

        use_lbfgs = trainer.scheduler.should_use_lbfgs(epoch)
        if use_lbfgs and trainer.lbfgs is None:
            print("  → 切换到 L-BFGS 优化器")
            trainer.lbfgs = trainer._build_lbfgs()

        loss_val = trainer._train_epoch(epoch, phase, use_lbfgs)

        trainer.history["epoch"].append(epoch)
        trainer.history["loss"].append(loss_val)
        trainer.history["phase"].append(phase["name"])

        val_loss = None
        if trainer.val_dataloader is not None and (epoch + 1) % trainer.tcfg["log_every"] == 0:
            val_loss = trainer._eval_val()
        trainer.history["val_loss"].append(val_loss)

        if (epoch + 1) % trainer.tcfg["log_every"] == 0:
            elapsed = time.time() - start_time
            val_str = f" val={val_loss:.6f}" if val_loss is not None else ""
            print(f"  [epoch {epoch+1:5d}/{total_epochs}] "
                  f"loss={loss_val:.6f}{val_str} | "
                  f"phase={phase['name']} | "
                  f"time={elapsed:.1f}s")

        if (epoch + 1) % trainer.tcfg["save_every"] == 0:
            trainer._save_checkpoint(epoch)

    print(f"\n训练完成! 总耗时: {time.time()-start_time:.1f}s")
    trainer._save_checkpoint(total_epochs - 1, final=True)

    # ─── 评估 ───
    print("\n[Step 4] 评估...")
    from src.evaluation.metrics import evaluate_predictions, check_physics_consistency
    results = evaluate_predictions(model, dataset, device, cfg)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    physics = check_physics_consistency(model, cfg, device)
    print("  物理一致性:")
    for k, v in physics.items():
        print(f"    {k}: {v:.6f}")

    print("\n完成!")


if __name__ == "__main__":
    main()
