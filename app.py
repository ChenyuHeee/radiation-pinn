"""RadiationPINN 交互式 Web 应用。
Author: Chenyu He, Zhejiang University, 2026
E-mail: hechenyu@zju.edu.cn
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
import matplotlib.font_manager as fm
# 优先使用 Noto Sans CJK（Streamlit Cloud apt 安装），其次 macOS 字体
_cjk_fonts = ["Noto Sans CJK SC", "PingFang SC", "SimHei",
              "Arial Unicode MS", "DejaVu Sans"]
# 刷新字体缓存以识别新安装的字体
fm._load_fontmanager(try_read_cache=False)
plt.rcParams["font.sans-serif"] = _cjk_fonts
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
def load_model(ckpt_path, config_path=None):
    """加载模型和配置（只执行一次）。"""
    if config_path is None:
        config_path = os.path.join(_BASE_DIR, "src/configs/default.yaml")
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

    # 恢复模型参数梯度状态
    for p in model.parameters():
        p.requires_grad_(True)

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
    ckpt_dir = os.path.join(_BASE_DIR, "checkpoints")
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 正向预测", "🎯 逆向设计", "🗺️ 编程地图", "📊 模型验证"])

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


    # ======================== Tab 4: 模型验证 ========================
    with tab4:
        st.header("模型验证：预测 vs 实验")
        st.markdown("用训练数据评估模型拟合质量。展示逐点对比、分条件剖面、误差统计。")

        # 加载实验数据
        data_dir = os.path.join(_BASE_DIR, "data")
        temp_path = os.path.join(data_dir, "temperature.csv")
        rad_path = os.path.join(data_dir, "radiation.csv")
        soot_path = os.path.join(data_dir, "soot.csv")

        if not os.path.exists(temp_path):
            st.warning("未找到 data/ 目录下的实验数据 CSV，无法进行验证。")
        else:
            df_temp = pd.read_csv(temp_path)
            df_rad = pd.read_csv(rad_path) if os.path.exists(rad_path) else None
            df_soot = pd.read_csv(soot_path) if os.path.exists(soot_path) else None

            z_max = cfg["physics"]["z_max"]
            phi_lo, phi_hi = PHI_RANGE

            # ---------- 温度预测 ----------
            phi_norms_t = ((df_temp["phi"] - phi_lo) / (phi_hi - phi_lo)).values
            z_norms_t = (df_temp["hab_mm"].values / 1000.0) / z_max
            fuel_ids_t = df_temp["fuel_id"].values.astype(int)
            x_prec_t = np.where(fuel_ids_t == 1, 1.0, 0.0)
            T_exp = df_temp["T_K"].values

            inp_t = torch.zeros(len(df_temp), 4)
            inp_t[:, 0] = torch.tensor(phi_norms_t, dtype=torch.float32)
            inp_t[:, 1] = torch.tensor(z_norms_t, dtype=torch.float32)
            inp_t[:, 2] = 0.0
            inp_t[:, 3] = torch.tensor(x_prec_t, dtype=torch.float32)
            fid_t = torch.tensor(fuel_ids_t, dtype=torch.long)

            with torch.no_grad():
                out_t = model(inp_t.to(device), fid_t.to(device))
            T_pred = out_t["T"].squeeze().cpu().numpy()

            # ---------- 辐射预测 ----------
            q_exp, q_pred = None, None
            if df_rad is not None and len(df_rad) > 0:
                phi_norms_r = ((df_rad["phi"] - phi_lo) / (phi_hi - phi_lo)).values
                z_norms_r = np.minimum((df_rad["hab_index"].values + 1) * 5.0 / 1000.0 / z_max, 1.0)
                fuel_ids_r = df_rad["fuel_id"].values.astype(int)
                x_prec_r = np.where(fuel_ids_r == 1, 1.0, 0.0)
                q_exp = df_rad["q_rad_Wm2"].values

                qp_list = []
                for k in range(len(df_rad)):
                    inp_r = torch.tensor([[phi_norms_r[k], z_norms_r[k], 0.0, x_prec_r[k]]], dtype=torch.float32, device=device)
                    fid_r = torch.tensor([fuel_ids_r[k]], dtype=torch.long, device=device)
                    with torch.no_grad():
                        rr = model.compute_radiation(inp_r, fid_r)
                    qp_list.append(rr["q_rad"].item())
                q_pred = np.array(qp_list)

            # ---------- 碳烟预测 ----------
            fv_exp, fv_pred = None, None
            if df_soot is not None and len(df_soot) > 0:
                from src.data.preprocess import FUEL_NAME_TO_ID
                df_sv = df_soot.dropna(subset=["hab_mm"]).copy()
                if len(df_sv) > 0:
                    sv_fuel = df_sv["fuel_name"].map(FUEL_NAME_TO_ID).fillna(0).astype(int).values
                    sv_phi = ((df_sv["phi"].values - phi_lo) / (phi_hi - phi_lo))
                    sv_z = (df_sv["hab_mm"].values / 1000.0) / z_max
                    sv_xp = np.where(sv_fuel == 1, 1.0, 0.0)
                    fv_exp = df_sv["fv_mean"].values

                    inp_s = torch.zeros(len(df_sv), 4)
                    inp_s[:, 0] = torch.tensor(sv_phi, dtype=torch.float32)
                    inp_s[:, 1] = torch.tensor(sv_z, dtype=torch.float32)
                    inp_s[:, 3] = torch.tensor(sv_xp, dtype=torch.float32)
                    fid_s = torch.tensor(sv_fuel, dtype=torch.long)
                    with torch.no_grad():
                        out_s = model(inp_s.to(device), fid_s.to(device))
                    fv_pred = out_s["fv"].squeeze().cpu().numpy()

            # ==================== 可视化 ====================
            # --- 1. Parity Plots ---
            st.subheader("1 · 预测值 vs 实测值 (Parity Plot)")
            n_parity = 1 + (1 if q_exp is not None else 0) + (1 if fv_exp is not None else 0)
            fig_p, axes_p = plt.subplots(1, n_parity, figsize=(5 * n_parity, 5))
            if n_parity == 1:
                axes_p = [axes_p]

            def _parity(ax, y_true, y_pred, label, unit, color):
                ax.scatter(y_true, y_pred, s=10, alpha=0.6, c=color, edgecolors="none")
                lo = min(y_true.min(), y_pred.min())
                hi = max(y_true.max(), y_pred.max())
                margin = (hi - lo) * 0.05
                ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                        "k--", linewidth=1, label="理想线 y=x")
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                ax.set_xlabel(f"实测 {label} ({unit})")
                ax.set_ylabel(f"预测 {label} ({unit})")
                ax.set_title(f"{label}\nR²={r2:.4f}  RMSE={rmse:.1f}")
                ax.legend(fontsize=8)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.3)

            idx = 0
            _parity(axes_p[idx], T_exp, T_pred, "温度 T", "K", "#e74c3c")
            idx += 1
            if q_exp is not None:
                _parity(axes_p[idx], q_exp, q_pred, "辐射 q_rad", "W/m²", "#3498db")
                idx += 1
            if fv_exp is not None:
                _parity(axes_p[idx], fv_exp, fv_pred, "碳烟 fv", "ppm", "#2c3e50")

            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close(fig_p)

            # --- 2. 逐 φ 温度剖面 ---
            st.subheader("2 · 逐当量比温度剖面对比")
            val_fuel = st.selectbox("选择燃料", [0, 1, 2],
                                    format_func=lambda x: FUEL_NAMES[x],
                                    key="val_fuel")
            df_tf = df_temp[df_temp["fuel_id"] == val_fuel]
            phis_avail = sorted(df_tf["phi"].unique())
            n_phi = len(phis_avail)
            if n_phi > 0:
                ncols = min(n_phi, 4)
                nrows = (n_phi + ncols - 1) // ncols
                fig_t, axes_t = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                                              squeeze=False)
                for pi, phi_val in enumerate(phis_avail):
                    ax = axes_t[pi // ncols][pi % ncols]
                    sub = df_tf[df_tf["phi"] == phi_val].sort_values("hab_mm")
                    hab_exp = sub["hab_mm"].values
                    t_exp_sub = sub["T_K"].values
                    # 模型预测曲线
                    z_dense = np.linspace(hab_exp.min(), hab_exp.max(), 60)
                    phi_n = (phi_val - phi_lo) / (phi_hi - phi_lo)
                    xp = 1.0 if val_fuel == 1 else 0.0
                    inp_d = torch.zeros(60, 4)
                    inp_d[:, 0] = phi_n
                    inp_d[:, 1] = torch.tensor(z_dense / 1000.0 / z_max, dtype=torch.float32)
                    inp_d[:, 3] = xp
                    fid_d = torch.full((60,), val_fuel, dtype=torch.long)
                    with torch.no_grad():
                        t_dense = model(inp_d.to(device), fid_d.to(device))["T"].squeeze().cpu().numpy()
                    ax.plot(z_dense, t_dense, "r-", linewidth=1.5, label="PINN 预测")
                    ax.scatter(hab_exp, t_exp_sub, c="k", s=20, zorder=5, label="实验")
                    ax.set_title(f"φ = {phi_val}")
                    ax.set_xlabel("HAB (mm)")
                    ax.set_ylabel("T (K)")
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
                for pi in range(n_phi, nrows * ncols):
                    axes_t[pi // ncols][pi % ncols].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_t)
                plt.close(fig_t)

            # --- 3. 误差统计表 ---
            st.subheader("3 · 误差统计汇总")
            stats = []

            def _stats(name, y_true, y_pred):
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                mask = np.abs(y_true) > 1e-6
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float("nan")
                return {"指标": name, "样本数": len(y_true), "RMSE": f"{rmse:.2f}",
                        "R²": f"{r2:.4f}", "MAPE (%)": f"{mape:.1f}"}

            stats.append(_stats("温度 T (K)", T_exp, T_pred))
            if q_exp is not None:
                stats.append(_stats("辐射 q_rad (W/m²)", q_exp, q_pred))
            if fv_exp is not None:
                stats.append(_stats("碳烟 fv (ppm)", fv_exp, fv_pred))

            st.table(pd.DataFrame(stats))

            # --- 4. 残差分布 ---
            st.subheader("4 · 残差分布直方图")
            fig_h, axes_h = plt.subplots(1, n_parity, figsize=(5 * n_parity, 4))
            if n_parity == 1:
                axes_h = [axes_h]

            idx = 0
            res_T = T_pred - T_exp
            axes_h[idx].hist(res_T, bins=30, color="#e74c3c", alpha=0.7, edgecolor="white")
            axes_h[idx].axvline(0, color="k", linestyle="--")
            axes_h[idx].set_xlabel("温度残差 (K)")
            axes_h[idx].set_ylabel("计数")
            axes_h[idx].set_title(f"温度残差  均值={res_T.mean():.1f}  标准差={res_T.std():.1f}")
            idx += 1

            if q_exp is not None:
                res_q = q_pred - q_exp
                axes_h[idx].hist(res_q, bins=20, color="#3498db", alpha=0.7, edgecolor="white")
                axes_h[idx].axvline(0, color="k", linestyle="--")
                axes_h[idx].set_xlabel("辐射残差 (W/m²)")
                axes_h[idx].set_ylabel("计数")
                axes_h[idx].set_title(f"辐射残差  均值={res_q.mean():.1f}  标准差={res_q.std():.1f}")
                idx += 1

            if fv_exp is not None:
                res_fv = fv_pred - fv_exp
                axes_h[idx].hist(res_fv, bins=20, color="#2c3e50", alpha=0.7, edgecolor="white")
                axes_h[idx].axvline(0, color="k", linestyle="--")
                axes_h[idx].set_xlabel("碳烟残差 (ppm)")
                axes_h[idx].set_ylabel("计数")
                axes_h[idx].set_title(f"碳烟残差  均值={res_fv.mean():.2f}  标准差={res_fv.std():.2f}")

            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)

            st.markdown("""
            **说明：**
            - **Parity Plot**：点越贴近对角线，拟合越好
            - **R²**：决定系数，1.0 为完美拟合
            - **RMSE**：均方根误差，越小越好
            - **MAPE**：平均绝对百分比误差，越小越好
            - **残差直方图**：以 0 为中心、分布越窄越好，偏移说明系统偏差
            """)


if __name__ == "__main__":
    main()
