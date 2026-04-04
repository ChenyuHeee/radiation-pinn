# 🔥 RadiationPINN — 多燃料辐射编程 PINN 模型

基于物理知情神经网络 (PINN) 的多燃料火焰辐射建模系统。

> 项目：辐射编程——从"被动燃烧"到"主动编程"的燃烧器绿色智能革命  
> 定位：核心技术 2-2（多燃料辐射特性建模）

## 核心思路

在绿色燃料（甲醇）中掺混微量碳烟前驱体（甲苯），通过 PINN 建立 **掺混比 → 碳烟场 → 辐射场** 的定量映射，实现辐射的精准"编程"控制。

```
燃料输入(φ, x_prec) → [共享Trunk] → 温度场 T(r,z)
                                   → 组分场 Y_k(r,z)
                                   → 碳烟场 f_v(r,z) → [RTE积分] → 辐射通量 q_rad(z)
```

## 项目结构

```
intelligenergy/
├── src/
│   ├── configs/default.yaml      # 全局配置
│   ├── data/
│   │   ├── preprocess.py         # 数据预处理（温度修正、消光法、CSV/Excel读取）
│   │   ├── dataset.py            # FlameDataset + PDE配点采样器
│   │   └── transforms.py         # 傅里叶特征编码
│   ├── models/
│   │   ├── pinn.py               # RadiationPINN 主模型
│   │   ├── subnetworks.py        # 共享Trunk + T/Y/Soot 子网络
│   │   ├── physics_layer.py      # 可微分 RTE 辐射积分层
│   │   └── embeddings.py         # 燃料 Embedding
│   ├── losses/
│   │   ├── data_loss.py          # 实验数据 MSE 损失
│   │   ├── pde_loss.py           # 能量守恒 + 碳烟输运 PDE 残差
│   │   ├── boundary_loss.py      # 边界条件损失
│   │   └── adaptive_weights.py   # 不确定度自适应权重
│   ├── training/
│   │   ├── trainer.py            # 四阶段课程式训练
│   │   ├── curriculum.py         # 物理注入调度器
│   │   └── transfer.py           # 燃料迁移学习
│   └── evaluation/
│       ├── metrics.py            # RMSE / R² / MAPE + 物理一致性
│       └── visualize.py          # 温度剖面、碳烟场、辐射编程曲线
├── scripts/
│   ├── train.py                  # 训练入口
│   └── predict.py                # 推理脚本
├── hh/                           # 实验原始数据（不要修改）
├── docs/                         # 设计文档
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python scripts/train.py
```

训练分四个阶段自动执行：

| 阶段 | Epoch | 做什么 |
|------|-------|--------|
| **Phase 1** 数据预热 | 0 ~ 2000 | 仅用实验数据训练，让网络学会基本映射 |
| **Phase 2** 物理注入 | 2000 ~ 7000 | 逐步加入 PDE 残差约束（能量守恒 + 碳烟方程） |
| **Phase 3** 联合微调 | 7000 ~ 10000 | 数据 + PDE + 边界条件全开，自适应权重，后段切 L-BFGS |
| **Phase 4** 燃料迁移 | 10000 ~ 11000 | 冻结主干，仅微调新燃料（甲醇）的参数 |

训练过程会自动：
- 每 100 epoch 打印损失
- 每 1000 epoch 保存 checkpoint 到 `checkpoints/`
- 训练结束后生成可视化图表到 `results/`

### 3. 🌐 启动交互式 Web 应用（推荐）

训练完成后，一键启动可视化交互平台：

```bash
streamlit run app.py
```

浏览器自动打开，包含三大功能页面：

| 页面 | 功能 | 说明 |
|------|------|------|
| **🔬 正向预测** | 参数 → 火焰场 → 辐射 | 拖动滑条调节当量比、掺混比，实时查看温度/碳烟/辐射/组分分布 |
| **🎯 逆向设计** | 目标辐射 → 最优配方 | 输入想要的辐射热通量，模型自动搜索最优的当量比和掺混比 |
| **🗺️ 编程地图** | 参数空间全局视图 | 展示 q_rad 如何随 (φ, x_prec) 变化的等值线图 |

> 不会用命令行？也可以用 `bash start.sh` 一键启动。

### 4. 命令行预测（可选）

```bash
# 正向预测
python scripts/predict.py --checkpoint checkpoints/radiation_pinn_final.pt --phi 1.0 --fuel 0

# 逆向设计：搜索使辐射 = 5000 W/m² 的最优参数
python scripts/inverse.py --checkpoint checkpoints/radiation_pinn_final.pt --target_qrad 5000
```

### 5. 自定义配置

编辑 `src/configs/default.yaml` 修改：

```yaml
# 改训练轮数（想快速试一下的话把每阶段调小）
training:
  phases:
    warmup:
      epochs: 200        # 原 2000，改小快速验证
    physics_injection:
      epochs: 500        # 原 5000
    joint:
      epochs: 300        # 原 3000
    transfer:
      epochs: 100        # 原 1000

# 改设备
training:
  device: "cpu"          # 可选 auto / cuda / mps / cpu
```

## 实验数据说明

数据都在 `hh/` 目录下：

| 数据 | 位置 | 格式 | 本次加载量 |
|------|------|------|-----------|
| 参考火焰温度（原始） | `hh/原始数据/参考火焰/{φ}/温度/{HAB}/*.csv` | 时间序列 CSV | 自动修正后 ~370 条 |
| 甲苯温度（修正后） | `hh/温度/甲苯/*.xlsx` | Excel | ~112 条 |
| 甲醇温度（修正后） | `hh/温度/甲醇/*.xlsx` | Excel | ~126 条 |
| 辐射通量 | `hh/原始数据/{燃料}/{φ}/辐射/*.csv` | 时间序列 CSV | ~196 条 |
| 碳烟体积分数 | `hh/碳烟/碳烟数据（绝密!)2.xlsx` | Excel | ~111 条 |

数据预处理自动完成：
- 温度 CSV → 稳态段提取（后80%）→ 时间平均 → 热电偶辐射修正
- 辐射 CSV → 稳态段提取 → 时间平均
- 碳烟 Excel → 解析复杂表格结构

## 模型架构简述

```
输入 (φ, z, r, x_prec, fuel_id)
  ↓ 傅里叶编码 + 燃料Embedding
  ↓
共享 Trunk: FC → [ResBlock × 3] → FC   (128维)
  ├─→ 温度子网络 → Softplus → T(r,z)     保证 T > 0
  ├─→ 组分子网络 → Softmax → Y_k(r,z)    保证 ΣY=1
  └─→ 碳烟子网络(h,T,Y_prec,Y_O2) → Softplus → f_v(r,z)  保证 f_v ≥ 0
                    ↓
        物理约束层（RTE 可微分积分）
                    ↓
              辐射热通量 q_rad(z)
```

参数量：**150,950**

## 输出文件

训练完成后产生：

```
checkpoints/
  └── radiation_pinn_final.pt     # 模型权重

results/
  ├── temperature_profile.png     # 不同 φ 的温度预测曲线
  ├── soot_field_phi1.0.png       # 碳烟 2D 分布图
  ├── radiation_programming.png   # 辐射编程曲线（核心！掺混比 vs 辐射通量）
  └── training_history.png        # 训练损失曲线
```

## 展示方式

### 给不会技术的人使用

1. 训练完成后，把整个 `radiation-pinn/` 文件夹发给对方
2. 对方只需安装 Python，然后执行：
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. 浏览器自动打开交互界面，通过拖动滑条和点击按钮即可操作

### Web 应用截图功能

- **正向预测**：调参数 → 看四张图（温度/碳烟/辐射/组分） → 下载数据 CSV
- **逆向设计**：设目标辐射 → 自动优化 → 看结果对比图 → 下载结果
- **编程地图**：一键生成辐射编程等值线图

## 常见问题

**Q: 训练太慢了？**  
在 `default.yaml` 里把四个阶段的 epochs 都调小，比如各除以 10 先跑通。

**Q: 没有 GPU？**  
在 Mac 上会自动使用 MPS 加速；没有的话退回 CPU，也能跑，只是慢一些。

**Q: 想加新燃料？**  
1. 在 `default.yaml` 里 `num_fuels` 加 1
2. 准备新燃料的温度/辐射数据放到 `hh/` 对应目录
3. 用 Phase 4 迁移学习，只需少量数据即可扩展
