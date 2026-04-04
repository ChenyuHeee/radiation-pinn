"""训练入口脚本。

Usage:
    python scripts/train.py
    python scripts/train.py --config src/configs/default.yaml
"""
import sys
import os
import argparse

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from src.data.preprocess import build_training_dataset
from src.data.dataset import FlameDataset
from src.models.pinn import RadiationPINN
from src.training.trainer import Trainer, get_device
from src.evaluation.metrics import evaluate_predictions, check_physics_consistency
from src.evaluation.visualize import (
    plot_temperature_profile,
    plot_soot_field,
    plot_radiation_programming,
    plot_training_history,
)


def main():
    parser = argparse.ArgumentParser(description="RadiationPINN 训练")
    parser.add_argument("--config", type=str,
                        default="src/configs/default.yaml",
                        help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 设备
    device = get_device(cfg)
    print(f"设备: {device}")

    # 设置随机种子
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ─── 1. 数据预处理 ───
    print("\n[Step 1] 数据预处理...")
    data = build_training_dataset(cfg)
    print(f"  温度样本: {len(data['temperature'])} 条")
    print(f"  辐射样本: {len(data['radiation'])} 条")
    print(f"  碳烟样本: {len(data['soot'])} 条")
    print(f"  组分样本: {len(data['species'])} 条")

    # ─── 2. 构建数据集 ───
    print("\n[Step 2] 构建 FlameDataset...")
    dataset = FlameDataset(
        temperature=data["temperature"],
        radiation=data["radiation"],
        soot=data["soot"],
        species=data["species"],
    )
    print(f"  总样本数: {len(dataset)}")

    if len(dataset) == 0:
        print("错误: 数据集为空! 请检查数据路径配置。")
        return

    # ─── 3. 构建模型 ───
    print("\n[Step 3] 构建 RadiationPINN...")
    model = RadiationPINN(cfg)
    print(f"  可训练参数: {model.count_parameters():,}")

    # ─── 4. 训练 ───
    print("\n[Step 4] 开始训练...")
    trainer = Trainer(model, dataset, cfg, device)
    trainer.train()

    # ─── 5. 评估 ───
    print("\n[Step 5] 评估...")
    results = evaluate_predictions(model, dataset, device, cfg)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    physics = check_physics_consistency(model, cfg, device)
    print("  物理一致性:")
    for k, v in physics.items():
        print(f"    {k}: {v:.6f}")

    # ─── 6. 可视化 ───
    print("\n[Step 6] 生成可视化...")
    save_dir = "results"
    plot_temperature_profile(model, cfg, device, save_dir)
    plot_soot_field(model, cfg, device, save_dir=save_dir)
    plot_radiation_programming(model, cfg, device, save_dir)
    plot_training_history(trainer.history, save_dir)
    print(f"  可视化已保存到 {save_dir}/")

    print("\n完成!")


if __name__ == "__main__":
    main()
