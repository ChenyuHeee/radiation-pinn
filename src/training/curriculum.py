"""课程式训练调度器：控制物理约束注入节奏。"""


class CurriculumScheduler:
    """四阶段课程式训练调度。

    Phase 1 (warmup):     仅数据损失
    Phase 2 (physics):    数据 + PDE（权重线性增长）
    Phase 3 (joint):      数据 + PDE + BC（自适应权重）
    Phase 4 (transfer):   冻结 trunk，仅微调新燃料
    """

    def __init__(self, phases_cfg: dict):
        self.phases = []
        total = 0
        for name in ("warmup", "physics_injection", "joint", "transfer"):
            cfg = phases_cfg[name]
            start = total
            end = total + cfg["epochs"]
            self.phases.append({
                "name": name,
                "start": start,
                "end": end,
                "epochs": cfg["epochs"],
                "lr": cfg["lr"],
                "use_pde": cfg.get("use_pde", False),
                "use_bc": cfg.get("use_bc", False),
                "pde_weight_start": cfg.get("pde_weight_start", 0.0),
                "pde_weight_end": cfg.get("pde_weight_end", 1.0),
                "use_lbfgs": cfg.get("use_lbfgs", False),
                "lbfgs_start_epoch": cfg.get("lbfgs_start_epoch", 0),
                "freeze_trunk": cfg.get("freeze_trunk", False),
            })
            total = end
        self.total_epochs = total

    def get_phase(self, epoch: int) -> dict:
        """返回当前 epoch 所属的阶段配置。"""
        for phase in self.phases:
            if phase["start"] <= epoch < phase["end"]:
                return phase
        return self.phases[-1]

    def get_pde_weight(self, epoch: int) -> float:
        """计算当前 PDE 权重（物理注入阶段线性增长）。"""
        phase = self.get_phase(epoch)
        if not phase["use_pde"]:
            return 0.0

        local_epoch = epoch - phase["start"]
        alpha = min(local_epoch / max(phase["epochs"], 1), 1.0)
        w_start = phase["pde_weight_start"]
        w_end = phase["pde_weight_end"]
        return w_start + alpha * (w_end - w_start)

    def should_use_lbfgs(self, epoch: int) -> bool:
        """判断是否切换 L-BFGS 优化器。"""
        phase = self.get_phase(epoch)
        if not phase["use_lbfgs"]:
            return False
        local_epoch = epoch - phase["start"]
        return local_epoch >= phase["lbfgs_start_epoch"]

    def should_freeze_trunk(self, epoch: int) -> bool:
        """判断是否冻结共享 Trunk。"""
        return self.get_phase(epoch).get("freeze_trunk", False)
