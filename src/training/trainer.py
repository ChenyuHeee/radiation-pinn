"""四阶段课程式训练循环。"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.pinn import RadiationPINN
from ..losses.data_loss import DataLoss
from ..losses.pde_loss import PDELoss
from ..losses.boundary_loss import BoundaryLoss
from ..losses.adaptive_weights import AdaptiveWeights, FixedWeights
from ..data.dataset import FlameDataset, collate_flame, CollocationSampler, BoundarySampler
from .curriculum import CurriculumScheduler


class Trainer:
    """RadiationPINN 训练器。"""

    def __init__(self, model: RadiationPINN, dataset: FlameDataset,
                 cfg: dict, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.tcfg = cfg["training"]

        # 数据加载
        self.dataloader = DataLoader(
            dataset,
            batch_size=len(dataset) if self.tcfg["batch_size"] == -1 else self.tcfg["batch_size"],
            shuffle=True,
            collate_fn=collate_flame,
            num_workers=self.tcfg["num_workers"],
        )

        # 损失函数
        self.data_loss_fn = DataLoss()
        self.pde_loss_fn = PDELoss()
        self.bc_loss_fn = BoundaryLoss(T_amb=cfg["physics"]["T_amb"])

        # 采样器
        self.collocation_sampler = CollocationSampler(
            n_points=self.tcfg["n_collocation"],
            num_fuels=cfg["model"]["num_fuels"],
            device=device,
        )
        self.boundary_sampler = BoundarySampler(
            n_points=self.tcfg["n_boundary"],
            num_fuels=cfg["model"]["num_fuels"],
            device=device,
        )

        # 课程调度器
        self.scheduler = CurriculumScheduler(self.tcfg["phases"])

        # 自适应权重（Phase 3 使用）
        self.adaptive_weights = AdaptiveWeights(
            ["T", "fv", "rad", "energy", "soot", "bc"]
        ).to(device)

        # 固定权重（Phase 1/2 使用）
        self.fixed_weights = FixedWeights({
            "T": 1.0, "fv": 1.0, "rad": 1.0,
            "energy": 0.01, "soot": 0.01, "bc": 0.1,
        })

        # 优化器
        self.optimizer = self._build_optimizer()
        self.lbfgs = None  # 延迟创建

        # 训练历史
        self.history = {"epoch": [], "loss": [], "phase": []}

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = list(self.model.parameters()) + list(self.adaptive_weights.parameters())
        return torch.optim.Adam(
            params,
            lr=self.tcfg["phases"]["warmup"]["lr"],
            weight_decay=self.tcfg["weight_decay"],
        )

    def _build_lbfgs(self) -> torch.optim.LBFGS:
        params = list(self.model.parameters()) + list(self.adaptive_weights.parameters())
        return torch.optim.LBFGS(
            params,
            lr=1e-4,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

    def train(self):
        """主训练循环。"""
        total_epochs = self.scheduler.total_epochs
        print(f"开始训练: 总共 {total_epochs} epochs, "
              f"模型参数量: {self.model.count_parameters():,}")

        prev_phase_name = None
        start_time = time.time()

        for epoch in range(total_epochs):
            phase = self.scheduler.get_phase(epoch)

            # 阶段切换日志
            if phase["name"] != prev_phase_name:
                print(f"\n{'='*60}")
                print(f"进入阶段: {phase['name']} (epoch {phase['start']}~{phase['end']-1})")
                print(f"  lr={phase['lr']}, use_pde={phase['use_pde']}, "
                      f"use_bc={phase['use_bc']}")
                print(f"{'='*60}")
                # 更新学习率
                for pg in self.optimizer.param_groups:
                    pg["lr"] = phase["lr"]
                # 冻结/解冻 Trunk
                self._set_trunk_frozen(phase.get("freeze_trunk", False))
                prev_phase_name = phase["name"]

            # L-BFGS 切换
            use_lbfgs = self.scheduler.should_use_lbfgs(epoch)
            if use_lbfgs and self.lbfgs is None:
                print("  → 切换到 L-BFGS 优化器")
                self.lbfgs = self._build_lbfgs()

            # 训练一个 epoch
            loss_val = self._train_epoch(epoch, phase, use_lbfgs)

            self.history["epoch"].append(epoch)
            self.history["loss"].append(loss_val)
            self.history["phase"].append(phase["name"])

            # 日志
            if (epoch + 1) % self.tcfg["log_every"] == 0:
                elapsed = time.time() - start_time
                print(f"  [epoch {epoch+1:5d}/{total_epochs}] "
                      f"loss={loss_val:.6f} | "
                      f"phase={phase['name']} | "
                      f"time={elapsed:.1f}s")

            # 保存 checkpoint
            if (epoch + 1) % self.tcfg["save_every"] == 0:
                self._save_checkpoint(epoch)

        print(f"\n训练完成! 总耗时: {time.time()-start_time:.1f}s")
        self._save_checkpoint(total_epochs - 1, final=True)

    def _train_epoch(self, epoch: int, phase: dict,
                     use_lbfgs: bool) -> float:
        """单个 epoch 训练。"""
        self.model.train()
        all_losses = {}

        # ─── 数据损失 ───
        for inputs, fuel_ids, targets, masks in self.dataloader:
            inputs = inputs.to(self.device)
            fuel_ids = fuel_ids.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            masks = {k: v.to(self.device) for k, v in masks.items()}

            def closure():
                nonlocal all_losses
                all_losses = {}
                self.optimizer.zero_grad()
                out = self.model(inputs, fuel_ids)
                data_losses = self.data_loss_fn(out, targets, masks)
                all_losses.update(data_losses)

                # PDE 残差
                pde_weight = self.scheduler.get_pde_weight(epoch)
                if phase["use_pde"] and pde_weight > 0:
                    col_inputs, col_fids = self.collocation_sampler.sample()
                    pde_losses = self.pde_loss_fn(self.model, col_inputs, col_fids)
                    all_losses["energy"] = pde_losses["energy"] * pde_weight
                    all_losses["soot"] = pde_losses["soot"] * pde_weight

                # 边界损失
                if phase["use_bc"]:
                    bc_data = self.boundary_sampler.sample()
                    bc_losses = self.bc_loss_fn(self.model, bc_data)
                    all_losses["bc"] = bc_losses["total"]

                # 加权合并
                if phase["name"] == "joint":
                    total, _ = self.fixed_weights(all_losses)
                else:
                    total, _ = self.fixed_weights(all_losses)

                total.backward()
                # 梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                return total

            if use_lbfgs and self.lbfgs is not None:
                loss = self.lbfgs.step(closure)
            else:
                loss = closure()
                self.optimizer.step()

        return loss.item() if isinstance(loss, torch.Tensor) else loss

    def _set_trunk_frozen(self, freeze: bool):
        """冻结/解冻共享 Trunk。"""
        for param in self.model.trunk.parameters():
            param.requires_grad = not freeze
        state = "冻结" if freeze else "解冻"
        print(f"  → 共享 Trunk 已{state}")

    def _save_checkpoint(self, epoch: int, final: bool = False):
        """保存模型 checkpoint。"""
        ckpt_dir = self.tcfg["checkpoint_dir"]
        os.makedirs(ckpt_dir, exist_ok=True)
        suffix = "final" if final else f"epoch{epoch+1}"
        path = os.path.join(ckpt_dir, f"radiation_pinn_{suffix}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "adaptive_weights_state": self.adaptive_weights.state_dict(),
            "history": self.history,
        }, path)
        print(f"  → Checkpoint 已保存: {path}")


def get_device(cfg: dict) -> torch.device:
    """根据配置选择设备。"""
    dev = cfg["training"]["device"]
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(dev)
