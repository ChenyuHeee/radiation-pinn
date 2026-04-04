"""RadiationPINN 交互式 Web 应用。

启动：
    streamlit run app.py

无需命令行知识，所有操作通过网页完成。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.models.pinn import RadiationPINN
from src.training.trainer import get_device

# ─── 中文字体 ───
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS",
                                    "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ======================== 页面配置 ========================
st.set_page_config(
    page_title="🔥 辐射编程 PINN",
    page_icon="🔥",
    layout="wide",
)

FUEL_NAMES = {0: "参考火焰 (乙烯)", 1: "甲苯混合", 2: "甲醇"}
PHI_RANGE = (0.8, 1.4)


# ======================== 缓存模型加载 ========================
@st.cache_resource
def load_model(ckpt_path, config_path="src/configs/default.yaml"):
    """加载模型和配置（只执行一次）。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = "cpu"  # Streamlit 中用 CPU 更稳定
    model = RadiationPINN(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, cfg, device


def predict_profiles(model, cfg, device, phi, x_prec, fuel_id, n_z=60):
    """正向预测：给定参数，返回轴向分布。"""
    phi_norm = (phi - PHI_RANGE[0]) / (PHI_RANGE[1] - PHI_RANGE[0])
    z_norm = torch.linspace(0.02, 0.98, n_z, device=device)

    inputs = torch.zeros(n_z, 4, device=device)
    inputs[:, 0] = phi_norm
    inputs[:, 1] = z_norm
    inputs[:, 2] = 0.0
    inputs[:, 3] = x_prec

    fuel_ids = torch.full((n_z,), fuel_id, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inputs, fuel_ids)
        T = out["T"].squeeze().cpu().numpy()
        fv = out["fv"].squeeze().cpu().numpy()
        Y = out["Y"].cpu().numpy()

        # 辐射（逐点计算）
        qrad = []
        for i in range(n_z):
            r = model.compute_radiation(inputs[i:i+1], fuel_ids[i:i+1])
            qrad.append(r["q_rad"].item())

    z_mm = z_norm.cpu().numpy() * cfg["physics"]["z_max"] * 1000
    return z_mm, T, fv, np.array(qrad), Y


def inverse_optimize(model, cfg, device, target_qrad, fuel_id,
                     n_z=20, n_steps=300, lr=0.05):
    """逆向优化：搜索最优 (phi, x_prec)。"""
    for p in model.parameters():
        p.requires_grad_(False)

    phi_raw = torch.tensor(0.0, device=device, requires_grad=True)
    xp_raw = torch.tensor(0.0, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([phi_raw, xp_raw], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    z_norm = torch.linspace(0.05, 0.95, n_z, device=device)

    if isinstance(target_qrad, (int, float)):
        target = torch.full((n_z,), float(target_qrad), device=device)
    else:
        target = torch.tensor(target_qrad, dtype=torch.float32, device=device)

    fuel_ids = torch.full((n_z,), fuel_id, dtype=torch.long, device=device)
    history = {"loss": [], "phi": [], "x_prec": []}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for step in range(n_steps):
        optimizer.zero_grad()

        phi_norm = torch.sigmoid(phi_raw)
        x_prec = torch.sigmoid(xp_raw)

        inputs = torch.zeros(n_z, 4, device=device)
        inputs[:, 0] = phi_norm
        inputs[:, 1] = z_norm
        inputs[:, 3] = x_prec

        qrad_list = []
        for i in range(n_z):
            out = model.compute_radiation(inputs[i:i+1], fuel_ids[i:i+1])
            qrad_list.append(out["q_rad"])
        q_pred = torch.cat(qrad_list)

        loss = torch.nn.functional.mse_loss(q_pred, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        phi_val = PHI_RANGE[0] + (PHI_RANGE[1] - PHI_RANGE[0]) * torch.sigmoid(phi_raw).item()
        xp_val = torch.sigmoid(xp_raw).item()
        history["loss"].append(loss.item())
        history["phi"].append(phi_val)
        history["x_prec"].append(xp_val)

        progress_bar.progress((step + 1) / n_steps)
        if (step + 1) % 20 == 0:
            status_text.text(f"步骤 {step+1}/{n_steps}  损失={loss.item():.2f}  "
                             f"φ={phi_val:.3f}  x_prec={xp_val:.4f}")

    progress_bar.empty()
    status_text.empty()

    phi_opt = PHI_RANGE[0] + (PHI_RANGE[1] - PHI_RANGE[0]) * torch.sigmoid(phi_raw).item()
    xp_opt = torch.sigmoid(xp_raw).item()

    z_mm, T, fv, qrad_pred, Y = predict_profiles(
        model, cfg, device, phi_opt, xp_opt, fuel_id, n_z)

    return {
        "phi_opt": phi_opt,
        "x_prec_opt": xp_opt,
        "z_mm": z_mm,
        "T": T, "fv": fv, "qrad": qrad_pred,
        "target": target.cpu().numpy(),
        "history": history,
    }


# ======================== 绘图函数 ========================
def plot_forward_results(z_mm, T, fv, qrad, Y, species_names):
    """绘制正向预测的 4 张子图。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 温度
    ax = axes[0, 0]
    ax.plot(z_mm, T, "r-", linewidth=2)
    ax.set_xlabel("火焰高度 HAB (mm)")
    ax.set_ylabel("温度 T (K)")
    ax.set_title("轴线温度分布")
    ax.grid(True, alpha=0.3)

    # 碳烟
    ax = axes[0, 1]
    ax.plot(z_mm, fv, "k-", linewidth=2)
    ax.set_xlabel("火焰高度 HAB (mm)")
    ax.set_ylabel("碳烟体积分数 fv")
    ax.set_title("轴线碳烟分布")
    ax.grid(True, alpha=0.3)

    # 辐射
    ax = axes[1, 0]
    ax.plot(z_mm, qrad, "b-", linewidth=2)
    ax.set_xlabel("火焰高度 HAB (mm)")
    ax.set_ylabel("辐射热通量 q_rad (W/m²)")
    ax.set_title("轴向辐射通量")
    ax.grid(True, alpha=0.3)

    # 组分
    ax = axes[1, 1]
    for i, name in enumerate(species_names):
        ax.plot(z_mm, Y[:, i], linewidth=1.5, label=name)
    ax.set_xlabel("火焰高度 HAB (mm)")
    ax.set_ylabel("质量分数")
    ax.set_title("轴线组分分布")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_inverse_results(result):
    """绘制逆向优化结果。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 辐射对比
    ax = axes[0, 0]
    ax.plot(result["z_mm"], result["target"], "r--", linewidth=2, label="目标")
    ax.plot(result["z_mm"], result["qrad"], "b-", linewidth=2, label="优化结果")
    ax.set_xlabel("火焰高度 HAB (mm)")
    ax.set_ylabel("辐射热通量 (W/m²)")
    ax.set_title("辐射分布：目标 vs 优化")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 温度
    ax = axes[0, 1]
    ax.plot(result["z_mm"], result["T"], "r-", linewidth=2)
    ax.set_xlabel("火焰高度 HAB (mm)")
    ax.set_ylabel("温度 (K)")
    ax.set_title("优化后温度分布")
    ax.grid(True, alpha=0.3)

    # 碳烟
    ax = axes[1, 0]
    ax.plot(result["z_mm"], result["fv"], "k-", linewidth=2)
    ax.set_xlabel("火焰高度 HAB (mm)")
    ax.set_ylabel("碳烟体积分数")
    ax.set_title("优化后碳烟分布")
    ax.grid(True, alpha=0.3)

    # 优化历史
    ax = axes[1, 1]
    ax.semilogy(result["history"]["loss"], "g-", linewidth=1)
    ax.set_xlabel("优化步数")
    ax.set_ylabel("损失 (log)")
    ax.set_title("优化收敛过程")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_programming_map(model, cfg, device, fuel_id):
    """辐射编程地图：q_rad 随 (phi, x_prec) 的变化。"""
    phi_vals = np.linspace(0.8, 1.4, 8)
    xp_vals = np.linspace(0, 1, 8)
    z_norm_mid = 0.5  # 中间高度

    qrad_map = np.zeros((len(phi_vals), len(xp_vals)))

    for i, phi in enumerate(phi_vals):
        for j, xp in enumerate(xp_vals):
            phi_norm = (phi - PHI_RANGE[0]) / (PHI_RANGE[1] - PHI_RANGE[0])
            inp = torch.tensor([[phi_norm, z_norm_mid, 0.0, xp]],
                               dtype=torch.float32, device=device)
            fid = torch.tensor([fuel_id], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.compute_radiation(inp, fid)
                qrad_map[i, j] = out["q_rad"].item()

    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(xp_vals, phi_vals)
    c = ax.contourf(X, Y, qrad_map, levels=20, cmap="hot")
    plt.colorbar(c, ax=ax, label="q_rad (W/m²)")
    ax.set_xlabel("碳烟前驱体掺混比 x_prec")
    ax.set_ylabel("当量比 φ")
    ax.set_title("辐射编程地图 (HAB = 45mm)")
    plt.tight_layout()
    return fig


# ======================== 主界面 ========================
def main():
    st.title("🔥 辐射编程 PINN 交互平台")
    st.markdown('**多燃料火焰辐射建模与逆向设计** — 从"被动燃烧"到"主动编程"')

    # ─── 侧边栏：模型加载 ───
    st.sidebar.header("📂 模型配置")

    # 查找可用的 checkpoint
    ckpt_dir = "checkpoints"
    ckpt_files = []
    if os.path.exists(ckpt_dir):
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        ckpt_files.sort()

    if not ckpt_files:
        st.warning("⚠️ 未找到训练好的模型！请先运行训练：\n\n"
                    "```bash\npython scripts/train.py\n```\n\n"
                    "训练完成后刷新此页面。")
        st.stop()

    selected_ckpt = st.sidebar.selectbox(
        "选择模型权重", ckpt_files,
        format_func=lambda x: x.replace(".pt", ""))
    ckpt_path = os.path.join(ckpt_dir, selected_ckpt)

    model, cfg, device = load_model(ckpt_path)
    st.sidebar.success(f"✅ 模型已加载 ({model.count_parameters():,} 参数)")

    # ─── Tab 切换 ───
    tab1, tab2, tab3 = st.tabs([
        "🔬 正向预测", "🎯 逆向设计", "🗺️ 编程地图"])

    # ======================== Tab 1: 正向预测 ========================
    with tab1:
        st.header("正向预测：参数 → 火焰场 → 辐射")
        st.markdown("调节燃料参数，实时查看温度、碳烟、辐射和组分分布。")

        col1, col2, col3 = st.columns(3)
        with col1:
            phi = st.slider("当量比 φ", 0.8, 1.4, 1.0, 0.05,
                             help="燃料/空气比。φ>1 为富燃，φ<1 为贫燃")
        with col2:
            x_prec = st.slider("前驱体掺混比 x_prec", 0.0, 1.0, 0.0, 0.01,
                                help="甲苯等碳烟前驱体的掺混比例")
        with col3:
            fuel_id = st.selectbox("燃料类型", [0, 1, 2],
                                    format_func=lambda x: FUEL_NAMES[x])

        # 自动预测（参数变化即刷新）
        with st.spinner("计算中..."):
            z_mm, T, fv, qrad, Y = predict_profiles(
                model, cfg, device, phi, x_prec, fuel_id)

        species_names = ["CO₂", "H₂O", "CO", "C₂H₂/C₇H₈", "O₂", "N₂"]
        fig = plot_forward_results(z_mm, T, fv, qrad, Y, species_names)
        st.pyplot(fig)
        plt.close(fig)

        # 关键指标卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌡️ 峰值温度", f"{T.max():.0f} K",
                    f"位置 {z_mm[T.argmax()]:.0f} mm")
        col2.metric("💨 峰值碳烟", f"{fv.max():.2e}",
                    f"位置 {z_mm[fv.argmax()]:.0f} mm")
        col3.metric("☀️ 峰值辐射", f"{qrad.max():.0f} W/m²",
                    f"位置 {z_mm[qrad.argmax()]:.0f} mm")
        col4.metric("📊 平均辐射", f"{qrad.mean():.0f} W/m²")

        # 数据表格
        with st.expander("📋 查看详细数据"):
            df = pd.DataFrame({
                "HAB (mm)": z_mm.round(1),
                "T (K)": T.round(1),
                "fv": fv,
                "q_rad (W/m²)": qrad.round(2),
            })
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("💾 下载数据 CSV", csv,
                               "prediction.csv", "text/csv")

    # ======================== Tab 2: 逆向设计 ========================
    with tab2:
        st.header("逆向设计：目标辐射 → 最优燃料配方")
        st.markdown("设定想要的辐射热通量，模型自动搜索最优的当量比和掺混比。")

        col1, col2 = st.columns(2)
        with col1:
            target_mode = st.radio("目标模式",
                                    ["均匀辐射", "自定义分布"],
                                    horizontal=True)
        with col2:
            inv_fuel = st.selectbox("燃料类型 ", [0, 1, 2],
                                     format_func=lambda x: FUEL_NAMES[x],
                                     key="inv_fuel")

        if target_mode == "均匀辐射":
            target_val = st.slider("目标辐射 q_rad (W/m²)",
                                    100, 50000, 5000, 100)
            target_qrad = float(target_val)
        else:
            st.markdown("输入各高度处的目标辐射（逗号分隔，20 个值）：")
            target_str = st.text_area(
                "目标辐射分布",
                "1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, "
                "8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, "
                "800, 600, 400, 200",
                height=80)
            try:
                target_qrad = [float(x.strip()) for x in target_str.split(",")]
            except ValueError:
                st.error("格式错误，请输入逗号分隔的数值")
                st.stop()

        col1, col2 = st.columns(2)
        with col1:
            inv_steps = st.slider("优化步数", 100, 1000, 300, 50)
        with col2:
            inv_lr = st.slider("学习率", 0.01, 0.2, 0.05, 0.01)

        if st.button("🎯 开始逆向优化", key="inverse_btn"):
            result = inverse_optimize(
                model, cfg, device,
                target_qrad=target_qrad,
                fuel_id=inv_fuel,
                n_steps=inv_steps,
                lr=inv_lr,
            )

            # 结果卡片
            st.success("✅ 优化完成！")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔧 最优当量比 φ", f"{result['phi_opt']:.4f}")
            col2.metric("🧪 最优掺混比 x_prec", f"{result['x_prec_opt']:.4f}")
            col3.metric("📉 最终损失", f"{result['history']['loss'][-1]:.2f}")

            fig = plot_inverse_results(result)
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("📋 查看详细数据"):
                df = pd.DataFrame({
                    "HAB (mm)": result["z_mm"].round(1),
                    "T (K)": result["T"].round(1),
                    "fv": result["fv"],
                    "q_rad 预测": result["qrad"].round(2),
                    "q_rad 目标": result["target"].round(2),
                })
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("💾 下载数据 CSV", csv,
                                   "inverse_result.csv", "text/csv")

    # ======================== Tab 3: 编程地图 ========================
    with tab3:
        st.header("辐射编程地图")
        st.markdown('展示辐射热通量如何随当量比和掺混比变化 —— 这就是"辐射编程"的核心图。')

        map_fuel = st.selectbox("燃料类型  ", [0, 1, 2],
                                 format_func=lambda x: FUEL_NAMES[x],
                                 key="map_fuel")

        # 自动生成编程地图
        with st.spinner("扫描参数空间..."):
            fig = plot_programming_map(model, cfg, device, map_fuel)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("""
        **如何阅读此图：**
        - **横轴**：碳烟前驱体掺混比 x_prec（0 = 纯燃料，1 = 最大掺混）
        - **纵轴**：当量比 φ（0.8~1.4）
        - **颜色**：辐射热通量大小
        - 颜色越亮 → 辐射越强 → 碳烟越多
        """)


if __name__ == "__main__":
    main()
