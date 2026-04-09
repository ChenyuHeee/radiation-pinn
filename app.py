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
import glob as _glob

# Streamlit Cloud (Linux): fonts-noto-cjk 安装 OTF 到 /usr/share/fonts
# matplotlib 默认不扫描 OTF 目录，需要手动注册
for _d in ["/usr/share/fonts/opentype/noto", "/usr/share/fonts/noto",
           "/usr/share/fonts/truetype/noto"]:
    for _f in _glob.glob(_d + "/*CJK*"):
        fm.fontManager.addfont(_f)

_cjk_fonts = ["Noto Sans CJK SC", "Noto Sans CJK JP",
              "PingFang SC", "Heiti SC", "STHeiti", "Hiragino Sans GB",
              "SimHei", "Arial Unicode MS", "DejaVu Sans"]
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔬 正向预测", "🎯 逆向设计", "🗺️ 编程地图", "📊 模型验证",
        "🔍 模型可解释性", "📖 模型说明"])

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
        st.header("模型拟合与验证")

        # 加载实验数据
        data_dir = os.path.join(_BASE_DIR, "data")
        temp_path = os.path.join(data_dir, "temperature.csv")
        rad_path = os.path.join(data_dir, "radiation.csv")
        soot_path = os.path.join(data_dir, "soot.csv")

        if not os.path.exists(temp_path):
            st.warning("未找到 data/ 目录下的实验数据 CSV，无法进行验证。")
        else:
            # 训练时的验证当量比
            VAL_PHI = cfg["training"].get("val_phi", 1.1)
            PHI_TOL = 0.01

            df_temp = pd.read_csv(temp_path)
            df_rad = pd.read_csv(rad_path) if os.path.exists(rad_path) else None
            df_soot = pd.read_csv(soot_path) if os.path.exists(soot_path) else None

            z_max = cfg["physics"]["z_max"]
            phi_lo, phi_hi = PHI_RANGE

            # ---- 辅助函数 ----
            def _compute_metrics(y_true, y_pred):
                """返回 RMSE, R², MAPE"""
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                mask = np.abs(y_true) > 1e-6
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float("nan")
                return rmse, r2, mape

            def _is_val(phi_arr):
                """判断哪些样本属于验证集 (φ ≈ VAL_PHI)"""
                return np.abs(phi_arr - VAL_PHI) < PHI_TOL

            # ---------- 温度预测 ----------
            phi_norms_t = ((df_temp["phi"] - phi_lo) / (phi_hi - phi_lo)).values
            z_norms_t = (df_temp["hab_mm"].values / 1000.0) / z_max
            fuel_ids_t = df_temp["fuel_id"].values.astype(int)
            x_prec_t = np.where(fuel_ids_t == 1, 1.0, 0.0)
            T_exp = df_temp["T_K"].values
            T_is_val = _is_val(df_temp["phi"].values)

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
            q_exp, q_pred, q_is_val = None, None, None
            if df_rad is not None and len(df_rad) > 0:
                phi_norms_r = ((df_rad["phi"] - phi_lo) / (phi_hi - phi_lo)).values
                z_norms_r = np.minimum((df_rad["hab_index"].values + 1) * 5.0 / 1000.0 / z_max, 1.0)
                fuel_ids_r = df_rad["fuel_id"].values.astype(int)
                x_prec_r = np.where(fuel_ids_r == 1, 1.0, 0.0)
                q_exp = df_rad["q_rad_Wm2"].values
                q_is_val = _is_val(df_rad["phi"].values)

                qp_list = []
                for k in range(len(df_rad)):
                    inp_r = torch.tensor([[phi_norms_r[k], z_norms_r[k], 0.0, x_prec_r[k]]], dtype=torch.float32, device=device)
                    fid_r = torch.tensor([fuel_ids_r[k]], dtype=torch.long, device=device)
                    with torch.no_grad():
                        rr = model.compute_radiation(inp_r, fid_r)
                    qp_list.append(rr["q_rad"].item())
                q_pred = np.array(qp_list)

            # ---------- 碳烟预测 ----------
            fv_exp, fv_pred, fv_is_val = None, None, None
            if df_soot is not None and len(df_soot) > 0:
                from src.data.preprocess import FUEL_NAME_TO_ID
                df_sv = df_soot.dropna(subset=["hab_mm"]).copy()
                if len(df_sv) > 0:
                    sv_fuel = df_sv["fuel_name"].map(FUEL_NAME_TO_ID).fillna(0).astype(int).values
                    sv_phi = ((df_sv["phi"].values - phi_lo) / (phi_hi - phi_lo))
                    sv_z = (df_sv["hab_mm"].values / 1000.0) / z_max
                    sv_xp = np.where(sv_fuel == 1, 1.0, 0.0)
                    fv_exp = df_sv["fv_mean"].values
                    fv_is_val = _is_val(df_sv["phi"].values)

                    inp_s = torch.zeros(len(df_sv), 4)
                    inp_s[:, 0] = torch.tensor(sv_phi, dtype=torch.float32)
                    inp_s[:, 1] = torch.tensor(sv_z, dtype=torch.float32)
                    inp_s[:, 3] = torch.tensor(sv_xp, dtype=torch.float32)
                    fid_s = torch.tensor(sv_fuel, dtype=torch.long)
                    with torch.no_grad():
                        out_s = model(inp_s.to(device), fid_s.to(device))
                    fv_pred = out_s["fv"].squeeze().cpu().numpy()

            # ---- 汇总各数据的 train/val 分割情况 ----
            n_val_T = int(T_is_val.sum())
            n_val_q = int(q_is_val.sum()) if q_is_val is not None else 0
            n_val_fv = int(fv_is_val.sum()) if fv_is_val is not None else 0
            has_val = (n_val_T + n_val_q + n_val_fv) > 0

            # ---- 说明信息 ----
            st.info(f"""
**数据划分说明** — 训练时以 φ = {VAL_PHI} 的数据作为验证集（未参与训练），其余为训练集。

| 数据类型 | 训练集样本 | 验证集样本 (φ={VAL_PHI}) |
|---------|-----------|------------------------|
| 温度 T | {len(T_exp) - n_val_T} | {n_val_T} {'⚠️ 无此 φ' if n_val_T == 0 else ''} |
| 辐射 q_rad | {len(q_exp) - n_val_q if q_exp is not None else '—'} | {n_val_q if q_exp is not None else '—'} |
| 碳烟 fv | {len(fv_exp) - n_val_fv if fv_exp is not None else '—'} | {n_val_fv if fv_exp is not None else '—'} |

训练集指标反映**拟合能力**，验证集指标反映**泛化能力**（模型未见过这些数据）。
""")

            # ==================== 可视化 ====================
            # --- 1. Parity Plots (训练集/验证集分色) ---
            st.subheader("1 · 预测值 vs 实测值 (Parity Plot)")
            n_parity = 1 + (1 if q_exp is not None else 0) + (1 if fv_exp is not None else 0)
            fig_p, axes_p = plt.subplots(1, n_parity, figsize=(5 * n_parity, 5))
            if n_parity == 1:
                axes_p = [axes_p]

            def _parity(ax, y_true, y_pred, is_val_mask, label, unit, color_train, color_val):
                mask_tr = ~is_val_mask
                mask_vl = is_val_mask
                # 训练集
                if mask_tr.any():
                    ax.scatter(y_true[mask_tr], y_pred[mask_tr], s=10, alpha=0.5,
                               c=color_train, edgecolors="none", label="训练集")
                # 验证集
                if mask_vl.any():
                    ax.scatter(y_true[mask_vl], y_pred[mask_vl], s=30, alpha=0.9,
                               c=color_val, edgecolors="k", linewidths=0.5,
                               marker="D", label=f"验证集 (φ={VAL_PHI})")
                lo = min(y_true.min(), y_pred.min())
                hi = max(y_true.max(), y_pred.max())
                margin = (hi - lo) * 0.05
                ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                        "k--", linewidth=1, label="理想线 y=x")
                # 训练集指标
                rmse_tr, r2_tr, _ = _compute_metrics(y_true[mask_tr], y_pred[mask_tr]) if mask_tr.any() else (0, 0, 0)
                title = f"{label}\n训练 R²={r2_tr:.4f}  RMSE={rmse_tr:.1f}"
                if mask_vl.any():
                    rmse_vl, r2_vl, _ = _compute_metrics(y_true[mask_vl], y_pred[mask_vl])
                    title += f"\n验证 R²={r2_vl:.4f}  RMSE={rmse_vl:.1f}"
                ax.set_xlabel(f"实测 {label} ({unit})")
                ax.set_ylabel(f"预测 {label} ({unit})")
                ax.set_title(title, fontsize=9)
                ax.legend(fontsize=7)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.3)

            idx = 0
            _parity(axes_p[idx], T_exp, T_pred, T_is_val, "温度 T", "K", "#e74c3c", "#ff7979")
            idx += 1
            if q_exp is not None:
                _parity(axes_p[idx], q_exp, q_pred, q_is_val, "辐射 q_rad", "W/m²", "#3498db", "#74b9ff")
                idx += 1
            if fv_exp is not None:
                _parity(axes_p[idx], fv_exp, fv_pred, fv_is_val, "碳烟 fv", "ppm", "#2c3e50", "#636e72")

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
                    is_val_phi = abs(phi_val - VAL_PHI) < PHI_TOL
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
                    sc_color = "#ff7979" if is_val_phi else "k"
                    sc_label = "实验 (验证集)" if is_val_phi else "实验 (训练集)"
                    ax.scatter(hab_exp, t_exp_sub, c=sc_color, s=20, zorder=5, label=sc_label)
                    title_suffix = " [验证]" if is_val_phi else " [训练]"
                    ax.set_title(f"φ = {phi_val}{title_suffix}")
                    ax.set_xlabel("HAB (mm)")
                    ax.set_ylabel("T (K)")
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
                for pi in range(n_phi, nrows * ncols):
                    axes_t[pi // ncols][pi % ncols].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_t)
                plt.close(fig_t)

            # --- 3. 误差统计表 (训练 / 验证分开) ---
            st.subheader("3 · 误差统计汇总")

            def _stats_row(name, y_true, y_pred, split_label):
                rmse, r2, mape = _compute_metrics(y_true, y_pred)
                return {"指标": name, "数据集": split_label, "样本数": len(y_true),
                        "RMSE": f"{rmse:.2f}", "R²": f"{r2:.4f}", "MAPE (%)": f"{mape:.1f}"}

            stats = []
            # 温度
            mask_tr_T = ~T_is_val
            if mask_tr_T.any():
                stats.append(_stats_row("温度 T (K)", T_exp[mask_tr_T], T_pred[mask_tr_T], "训练"))
            if T_is_val.any():
                stats.append(_stats_row("温度 T (K)", T_exp[T_is_val], T_pred[T_is_val], "验证"))
            # 辐射
            if q_exp is not None:
                mask_tr_q = ~q_is_val
                if mask_tr_q.any():
                    stats.append(_stats_row("辐射 q_rad (W/m²)", q_exp[mask_tr_q], q_pred[mask_tr_q], "训练"))
                if q_is_val.any():
                    stats.append(_stats_row("辐射 q_rad (W/m²)", q_exp[q_is_val], q_pred[q_is_val], "验证"))
            # 碳烟
            if fv_exp is not None:
                mask_tr_fv = ~fv_is_val
                if mask_tr_fv.any():
                    stats.append(_stats_row("碳烟 fv (ppm)", fv_exp[mask_tr_fv], fv_pred[mask_tr_fv], "训练"))
                if fv_is_val.any():
                    stats.append(_stats_row("碳烟 fv (ppm)", fv_exp[fv_is_val], fv_pred[fv_is_val], "验证"))

            st.table(pd.DataFrame(stats))

            # --- 4. 残差分布 ---
            st.subheader("4 · 残差分布直方图")
            fig_h, axes_h = plt.subplots(1, n_parity, figsize=(5 * n_parity, 4))
            if n_parity == 1:
                axes_h = [axes_h]

            def _residual_hist(ax, y_true, y_pred, is_val_mask, label, unit, color_tr, color_vl):
                mask_tr = ~is_val_mask
                res_all = y_pred - y_true
                if mask_tr.any():
                    res_tr = res_all[mask_tr]
                    ax.hist(res_tr, bins=30, color=color_tr, alpha=0.7, edgecolor="white", label="训练集")
                if is_val_mask.any():
                    res_vl = res_all[is_val_mask]
                    ax.hist(res_vl, bins=15, color=color_vl, alpha=0.8, edgecolor="white", label="验证集")
                ax.axvline(0, color="k", linestyle="--")
                ax.set_xlabel(f"{label}残差 ({unit})")
                ax.set_ylabel("计数")
                ax.set_title(f"{label}残差  均值={res_all.mean():.1f}  标准差={res_all.std():.1f}")
                ax.legend(fontsize=7)

            idx = 0
            _residual_hist(axes_h[idx], T_exp, T_pred, T_is_val, "温度", "K", "#e74c3c", "#ff7979")
            idx += 1
            if q_exp is not None:
                _residual_hist(axes_h[idx], q_exp, q_pred, q_is_val, "辐射", "W/m²", "#3498db", "#74b9ff")
                idx += 1
            if fv_exp is not None:
                _residual_hist(axes_h[idx], fv_exp, fv_pred, fv_is_val, "碳烟", "ppm", "#2c3e50", "#636e72")

            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)

            st.markdown(f"""
**说明：**
- **训练集**（圆点）= 模型训练时使用的数据；**验证集**（菱形）= φ={VAL_PHI}，训练时被留出未参与训练
- **训练集 R²** 反映拟合能力，**验证集 R²** 反映泛化能力
- 目前仅辐射数据含 φ={VAL_PHI} 验证样本；温度和碳烟无此 φ 的实验数据，全部为训练集
- 若需更可靠的泛化评估，建议补充独立实验数据或采用 k-fold 交叉验证（需重新训练）
""")


    # ======================== Tab 5: 模型可解释性 ========================
    with tab5:
        st.header("模型可解释性：从输入到输出的信号流")
        st.markdown("选择工况参数，实时追踪数据在网络每一层中的变换过程。")

        # ---- 用户选择工况 ----
        ei_col1, ei_col2, ei_col3, ei_col4 = st.columns(4)
        with ei_col1:
            ei_phi = st.slider("当量比 φ", 0.8, 1.4, 1.0, 0.05, key="ei_phi")
        with ei_col2:
            ei_xp = st.slider("掺混比 x_prec", 0.0, 1.0, 0.0, 0.05, key="ei_xp")
        with ei_col3:
            ei_fuel = st.selectbox("燃料", [0, 1, 2],
                                   format_func=lambda x: FUEL_NAMES[x], key="ei_fuel")
        with ei_col4:
            ei_hab = st.slider("HAB (mm)", 5, 90, 45, 5, key="ei_hab")

        z_max = cfg["physics"]["z_max"]
        phi_lo, phi_hi = PHI_RANGE
        phi_n = (ei_phi - phi_lo) / (phi_hi - phi_lo)
        z_n = (ei_hab / 1000.0) / z_max
        xp_val = ei_xp

        inp = torch.tensor([[phi_n, z_n, 0.0, xp_val]], dtype=torch.float32, device=device)
        fid = torch.tensor([ei_fuel], dtype=torch.long, device=device)

        # ========== Stage 1: 输入编码 ==========
        st.subheader("Stage 1 · 输入编码")

        with torch.no_grad():
            phi_raw = inp[:, 0:1]
            z_raw = inp[:, 1:2]
            r_raw = inp[:, 2:3]
            xp_raw = inp[:, 3:4]
            z_enc = model.fourier_z(z_raw)
            r_enc = model.fourier_r(r_raw)
            fuel_enc = model.fuel_embed(fid)
            encoded = torch.cat([phi_raw, z_enc, r_enc, xp_raw, fuel_enc], dim=-1)

        enc_np = encoded.cpu().numpy().flatten()

        col_enc1, col_enc2 = st.columns([2, 3])
        with col_enc1:
            st.markdown(f"""
**原始输入** → **编码后 {len(enc_np)} 维向量**

| 分量 | 维度 | 值 |
|------|------|-----|
| φ (标量) | 1 | {phi_n:.3f} |
| z 傅里叶 | 12 | sin/cos 编码 |
| r 傅里叶 | 12 | sin/cos 编码 |
| x_prec | 1 | {xp_val:.2f} |
| 燃料 Embed | 8 | 学习向量 |
""")
        with col_enc2:
            fig_enc, ax_enc = plt.subplots(figsize=(8, 2))
            colors = (["#e74c3c"] * 1 +
                      ["#3498db"] * 12 +
                      ["#2ecc71"] * 12 +
                      ["#f39c12"] * 1 +
                      ["#9b59b6"] * 8)
            ax_enc.bar(range(len(enc_np)), enc_np, color=colors, width=1.0, edgecolor="none")
            ax_enc.set_xlabel("编码维度索引")
            ax_enc.set_ylabel("值")
            ax_enc.set_title("输入编码向量 (34维)")
            # 标注分段
            for label, start, end, c in [("φ", 0, 1, "#e74c3c"),
                                          ("z-Fourier", 1, 13, "#3498db"),
                                          ("r-Fourier", 13, 25, "#2ecc71"),
                                          ("x_prec", 25, 26, "#f39c12"),
                                          ("Fuel Embed", 26, 34, "#9b59b6")]:
                mid = (start + end) / 2
                ax_enc.annotate(label, xy=(mid, ax_enc.get_ylim()[1]),
                                ha="center", fontsize=7, color=c, weight="bold")
            ax_enc.grid(True, alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig_enc)
            plt.close(fig_enc)

        # ========== 燃料 Embedding 空间 ==========
        st.subheader("Stage 1b · 燃料 Embedding 空间")
        with torch.no_grad():
            all_fuel_ids = torch.arange(3, device=device)
            all_embeds = model.fuel_embed(all_fuel_ids).cpu().numpy()

        fig_emb, axes_emb = plt.subplots(1, 2, figsize=(10, 3.5))
        # 热力图
        im = axes_emb[0].imshow(all_embeds, aspect="auto", cmap="RdBu_r")
        axes_emb[0].set_yticks([0, 1, 2])
        axes_emb[0].set_yticklabels(["乙烯", "甲苯", "甲醇"])
        axes_emb[0].set_xlabel("Embedding 维度")
        axes_emb[0].set_title("燃料 Embedding 权重")
        plt.colorbar(im, ax=axes_emb[0], shrink=0.8)
        # 相似度
        from numpy.linalg import norm as np_norm
        sim_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                ni, nj = np_norm(all_embeds[i]), np_norm(all_embeds[j])
                sim_matrix[i, j] = np.dot(all_embeds[i], all_embeds[j]) / (ni * nj + 1e-8)
        im2 = axes_emb[1].imshow(sim_matrix, cmap="YlOrRd", vmin=-1, vmax=1)
        axes_emb[1].set_xticks([0, 1, 2])
        axes_emb[1].set_xticklabels(["乙烯", "甲苯", "甲醇"])
        axes_emb[1].set_yticks([0, 1, 2])
        axes_emb[1].set_yticklabels(["乙烯", "甲苯", "甲醇"])
        for i in range(3):
            for j in range(3):
                axes_emb[1].text(j, i, f"{sim_matrix[i,j]:.2f}",
                                 ha="center", va="center", fontsize=10)
        axes_emb[1].set_title("燃料余弦相似度")
        plt.colorbar(im2, ax=axes_emb[1], shrink=0.8)
        plt.tight_layout()
        st.pyplot(fig_emb)
        plt.close(fig_emb)

        # ========== Stage 2: SharedTrunk ==========
        st.subheader("Stage 2 · SharedTrunk 特征提取")
        with torch.no_grad():
            # 逐层追踪
            trunk = model.trunk
            h_in = trunk.norm_in(torch.nn.functional.silu(trunk.input_proj(encoded.to(device))))
            activations = [("Input Proj + SiLU + LN", h_in.cpu().numpy().flatten())]
            h_curr = h_in
            for bi, block in enumerate(trunk.resblocks):
                h_curr = block(h_curr)
                activations.append((f"ResBlock {bi+1}", h_curr.cpu().numpy().flatten()))
            h_out = trunk.norm_out(torch.nn.functional.silu(trunk.output_proj(h_curr)))
            activations.append(("Output Proj + SiLU + LN", h_out.cpu().numpy().flatten()))

        n_acts = len(activations)
        fig_trunk, axes_trunk = plt.subplots(n_acts, 1, figsize=(10, 2 * n_acts), sharex=True)
        for i, (name, act) in enumerate(activations):
            ax = axes_trunk[i]
            ax.bar(range(len(act)), act, color="#3498db", alpha=0.7, width=1.0)
            ax.set_ylabel("激活值")
            ax.set_title(f"{name}  (128维)  |  均值={act.mean():.3f}  标准差={act.std():.3f}",
                         fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.axhline(0, color="k", linewidth=0.5)
        axes_trunk[-1].set_xlabel("神经元索引")
        plt.tight_layout()
        st.pyplot(fig_trunk)
        plt.close(fig_trunk)

        # ========== Stage 3: 子网络输出 ==========
        st.subheader("Stage 3 · 子网络输出")
        with torch.no_grad():
            out = model(inp, fid)
            T_val = out["T"].item()
            Y_val = out["Y"].cpu().numpy().flatten()
            fv_val = out["fv"].item()

        col_s3a, col_s3b, col_s3c = st.columns(3)
        with col_s3a:
            st.metric("🌡️ 温度 T", f"{T_val:.1f} K")
            st.caption("T_net: Softplus × 2000 + 300")
        with col_s3b:
            species_names = ["CO₂", "H₂O", "CO", "C₂H₂/C₇H₈", "O₂", "N₂"]
            fig_sp, ax_sp = plt.subplots(figsize=(4, 3))
            bars = ax_sp.barh(species_names, Y_val, color=["#e74c3c", "#3498db", "#95a5a6",
                                                            "#f39c12", "#2ecc71", "#9b59b6"])
            ax_sp.set_xlabel("质量分数")
            ax_sp.set_title("组分 Yₖ (Softmax)")
            for bar, v in zip(bars, Y_val):
                ax_sp.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           f"{v:.4f}", va="center", fontsize=8)
            ax_sp.set_xlim(0, max(Y_val) * 1.3)
            plt.tight_layout()
            st.pyplot(fig_sp)
            plt.close(fig_sp)
        with col_s3c:
            st.metric("🔥 碳烟 fv", f"{fv_val:.4f} ppm")
            st.caption("SootNet: 因果级联 (h, T, Y_prec, Y_O₂)")

        # ========== Stage 4: RTE 物理积分过程 ==========
        st.subheader("Stage 4 · RTE 辐射传递积分过程")
        st.markdown("展示辐射物理层如何将温度场和碳烟场转化为辐射热通量。")

        n_r = cfg["physics"]["r_points"]
        with torch.no_grad():
            rad_out = model.compute_radiation(inp, fid, n_r=n_r)
            T_field = rad_out["T_field"].cpu().numpy().flatten()
            fv_field = rad_out["fv_field"].cpu().numpy().flatten()
            q_rad_val = rad_out["q_rad"].item()

            # 手动重算中间物理量用于可视化
            fv_t = rad_out["fv_field"]
            T_t = rad_out["T_field"]
            kappa = model.rte_layer.compute_kappa(fv_t)
            source = model.rte_layer.compute_source(kappa, T_t)
            kappa_np = kappa.cpu().numpy().flatten()
            source_np = source.cpu().numpy().flatten()

        r_mm = np.linspace(0, cfg["physics"]["r_max"] * 1000, n_r)

        fig_rte, axes_rte = plt.subplots(2, 3, figsize=(14, 7))

        # 1: T(r) 温度径向分布
        axes_rte[0, 0].plot(r_mm, T_field, "r-", linewidth=2)
        axes_rte[0, 0].fill_between(r_mm, 300, T_field, alpha=0.15, color="red")
        axes_rte[0, 0].set_ylabel("T (K)")
        axes_rte[0, 0].set_title("① 温度径向分布 T(r)")
        axes_rte[0, 0].grid(True, alpha=0.3)

        # 2: fv(r) 碳烟径向分布
        axes_rte[0, 1].plot(r_mm, fv_field, "k-", linewidth=2)
        axes_rte[0, 1].fill_between(r_mm, 0, fv_field, alpha=0.2, color="gray")
        axes_rte[0, 1].set_ylabel("fv (ppm)")
        axes_rte[0, 1].set_title("② 碳烟径向分布 fv(r)")
        axes_rte[0, 1].grid(True, alpha=0.3)

        # 3: κ(r) 吸收系数
        axes_rte[0, 2].plot(r_mm, kappa_np, "b-", linewidth=2)
        axes_rte[0, 2].fill_between(r_mm, 0, kappa_np, alpha=0.15, color="blue")
        axes_rte[0, 2].set_ylabel("κ (1/m)")
        axes_rte[0, 2].set_title("③ 吸收系数 κ = 6πE(m)/λ · fv")
        axes_rte[0, 2].grid(True, alpha=0.3)

        # 4: S(r) 源项
        axes_rte[1, 0].plot(r_mm, source_np, "m-", linewidth=2)
        axes_rte[1, 0].fill_between(r_mm, 0, source_np, alpha=0.15, color="purple")
        axes_rte[1, 0].set_xlabel("r (mm)")
        axes_rte[1, 0].set_ylabel("S (W/m³/sr)")
        axes_rte[1, 0].set_title("④ 辐射源项 S = κσT⁴/π")
        axes_rte[1, 0].grid(True, alpha=0.3)

        # 5: 累积光学深度
        with torch.no_grad():
            kappa_flip = torch.flip(kappa, dims=[-1])
            r_flip = torch.flip(model.rte_layer.r_grid, dims=[0])
            tau_flip = model.rte_layer._cumtrapz(kappa_flip, r_flip)
            tau_flip = torch.cat([torch.zeros_like(tau_flip[..., :1]), tau_flip], dim=-1)
            tau = torch.flip(tau_flip, dims=[-1]).cpu().numpy().flatten()
            trans = np.exp(-tau)

        axes_rte[1, 1].plot(r_mm, tau, "g-", linewidth=2, label="光学深度 τ")
        axes_rte[1, 1].plot(r_mm, trans, "g--", linewidth=1.5, label="透射率 e⁻ᵗ")
        axes_rte[1, 1].set_xlabel("r (mm)")
        axes_rte[1, 1].set_ylabel("τ / 透射率")
        axes_rte[1, 1].set_title("⑤ 光学深度与透射率")
        axes_rte[1, 1].legend(fontsize=8)
        axes_rte[1, 1].grid(True, alpha=0.3)

        # 6: 被积函数 S·exp(-τ) → 积分得 q_rad
        integrand = source_np * trans
        axes_rte[1, 2].fill_between(r_mm, 0, integrand, alpha=0.4, color="#e67e22",
                                     label="S · e⁻ᵗ (被积函数)")
        axes_rte[1, 2].plot(r_mm, integrand, color="#e67e22", linewidth=2)
        axes_rte[1, 2].set_xlabel("r (mm)")
        axes_rte[1, 2].set_ylabel("S · e⁻ᵗ")
        axes_rte[1, 2].set_title(f"⑥ 积分 → q_rad = {q_rad_val:.2f} W/m²")
        axes_rte[1, 2].legend(fontsize=8)
        axes_rte[1, 2].grid(True, alpha=0.3)

        for ax in axes_rte.flat:
            ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig_rte)
        plt.close(fig_rte)

        st.info(f"**辐射物理链**: T(r) × fv(r) → κ(r) → S(r) → 视线积分 → "
                f"**q_rad = {q_rad_val:.2f} W/m²** (HAB={ei_hab}mm, φ={ei_phi}, "
                f"{FUEL_NAMES[ei_fuel]})")

        # ========== Stage 5: 权重分布 ==========
        st.subheader("Stage 5 · 网络权重分布统计")

        weight_stats = []
        weight_hists = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() >= 1:
                w = param.detach().cpu().numpy().flatten()
                weight_stats.append({
                    "层": name.replace(".", " → "),
                    "形状": str(list(param.shape)),
                    "参数量": param.numel(),
                    "均值": f"{w.mean():.4f}",
                    "标准差": f"{w.std():.4f}",
                    "最小": f"{w.min():.4f}",
                    "最大": f"{w.max():.4f}",
                })
                weight_hists.append((name, w))

        with st.expander("📋 各层权重统计表", expanded=False):
            st.dataframe(pd.DataFrame(weight_stats), use_container_width=True)

        # 权重分布直方图（选取关键层）
        key_layers = [wh for wh in weight_hists if any(k in wh[0] for k in
                      ["input_proj.weight", "resblocks.0.fc1.weight",
                       "resblocks.2.fc2.weight", "output_proj.weight",
                       "T_net.net.0.weight", "soot_net.net.0.weight",
                       "Y_net.net.0.weight"])]
        if len(key_layers) > 0:
            n_kl = min(len(key_layers), 8)
            ncols_w = min(n_kl, 4)
            nrows_w = (n_kl + ncols_w - 1) // ncols_w
            fig_w, axes_w = plt.subplots(nrows_w, ncols_w, figsize=(4 * ncols_w, 3 * nrows_w))
            if nrows_w == 1 and ncols_w == 1:
                axes_w = np.array([[axes_w]])
            elif nrows_w == 1:
                axes_w = axes_w[np.newaxis, :]
            elif ncols_w == 1:
                axes_w = axes_w[:, np.newaxis]
            for i, (name, w) in enumerate(key_layers[:n_kl]):
                ax = axes_w[i // ncols_w][i % ncols_w]
                ax.hist(w, bins=50, color="#3498db", alpha=0.7, edgecolor="white", density=True)
                ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
                short_name = name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name
                ax.set_title(f"{short_name}\nμ={w.mean():.3f} σ={w.std():.3f}", fontsize=8)
                ax.tick_params(labelsize=7)
            for i in range(n_kl, nrows_w * ncols_w):
                axes_w[i // ncols_w][i % ncols_w].set_visible(False)
            plt.suptitle("关键层权重分布", fontsize=11, y=1.01)
            plt.tight_layout()
            st.pyplot(fig_w)
            plt.close(fig_w)

        # ========== 信号流总结 ==========
        st.subheader("信号流总结")
        st.markdown(f"""
```
                        输入: φ={ei_phi}, HAB={ei_hab}mm, x_prec={ei_xp}, {FUEL_NAMES[ei_fuel]}
                              │
                        ┌─────▼─────┐
                        │ 傅里叶编码 │ → {len(enc_np)}维向量
                        │ +Embedding │
                        └─────┬─────┘
                              │
                 ┌────────────▼────────────┐
                 │    SharedTrunk (128维)    │
                 │ InputProj → 3×ResBlock    │
                 │ → OutputProj + LayerNorm  │
                 └────────────┬────────────┘
                              │ h (128维)
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌──────▼──────┐
        │   T_net    │  │   Y_net    │  │  SootNet    │
        │ T={T_val:.0f} K  │  │ 6组分      │  │ fv={fv_val:.4f}  │
        └─────┬─────┘  └─────┬─────┘  │(h+T+Y→fv) │
              │               │        └──────┬──────┘
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  RTE 物理积分层      │
                    │ κ→S→∫ S·e⁻ᵗ dr     │
                    │ q_rad={q_rad_val:.2f} W/m²  │
                    └────────────────────┘
```
""")


    # ======================== Tab 6: 模型说明 ========================
    with tab6:
        st.header("模型架构与训练方法")

        st.markdown(r"""
## 1 · 项目定位

**RadiationPINN** 是一个基于物理信息神经网络（Physics-Informed Neural Network）的
火焰辐射预测与编程系统。
它以碳烟前驱体掺混比 $x_{\mathrm{prec}}$ 和当量比 $\phi$ 作为"编程因子"，
通过可微分物理约束实现 **"辐射编程"** ——
即在给定燃料与工况下，正向预测或逆向优化辐射热通量 $q_{\mathrm{rad}}$。

---

## 2 · 网络架构

```
输入: (φ, z, r, x_prec, fuel_id)
  │
  ├─ φ (1维) ─────────────────┐
  ├─ z → 傅里叶编码 (12维) ───┤
  ├─ r → 傅里叶编码 (12维) ───┤ 拼接 → 34维
  ├─ x_prec (1维) ────────────┤
  └─ fuel_id → Embedding (8维)┘
           │
     ┌─────▼──────┐
     │ SharedTrunk │  3 × ResBlock (128维)
     │  + GELU     │
     └─────┬──────┘
           │ h (128维)
     ┌─────┼─────────────────┐
     │     │                 │
  ┌──▼──┐ ┌──▼──┐    ┌───────▼──────┐
  │ T_net│ │Y_net│    │  SootNet     │
  │→ T  │ │→ Yₖ │    │(h,T,Y_prec,  │
  └─────┘ └─────┘    │ Y_O₂) → fᵥ  │
                     └───────┬──────┘
                             │
                    ┌────────▼────────┐
                    │ RTE 物理约束层   │
                    │ κ → S → ∫ → q_rad│
                    └─────────────────┘
```

### 关键设计要点

| 模块 | 设计 | 说明 |
|------|------|------|
| **傅里叶编码** | L=6, 各生成 2L=12 维 | 捕捉空间高频特征 |
| **燃料 Embedding** | 3 种燃料 → 8 维向量 | 共享架构、差异化表示 |
| **SharedTrunk** | 128 维 × 3 ResBlock + GELU | 提取通用火焰特征 |
| **温度子网络 T_net** | Softplus 激活 × 2000 + 300 | 保证 T > 300K (物理约束) |
| **组分子网络 Y_net** | Softmax 归一化 (6组分) | 保证 ΣYₖ = 1 |
| **碳烟子网络 SootNet** | 因果级联: 接收 T, Y_prec, Y_O₂ | 碳烟生成依赖温度和前驱体浓度 |
| **RTE 物理层** | 可微分径向积分 | $q_{\mathrm{rad}} = 2\int S \cdot e^{-\tau} \, dr$ |

""")

        # 动态显示参数量
        n_params = model.count_parameters()
        st.metric("模型总参数量", f"{n_params:,}")

        st.markdown(r"""
---

## 3 · 物理约束体系

模型通过三大物理方程对网络施加约束，使预测满足物理规律：

### 3.1 能量守恒方程

$$\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (\lambda \nabla T) + \dot{Q}_{\mathrm{chem}} - \nabla \cdot q_{\mathrm{rad}}$$

- 热传导 + 化学反应放热 − 辐射散热 = 0
- 辐射项通过 RTE 层反向耦合

### 3.2 碳烟输运方程 (Moss-Brookes 模型)

$$\frac{\partial f_v}{\partial t} = \underbrace{C_\alpha e^{-T_\alpha/T}}_\text{成核} + \underbrace{C_\beta e^{-T_\beta/T} f_v}_\text{表面生长} - \underbrace{C_\omega T^{0.5} e^{-T_\omega/T} f_v}_\text{氧化}$$

- 包含 **6 个可学习 Arrhenius 参数** ($C_\alpha, T_\alpha, C_\beta, T_\beta, C_\omega, T_\omega$)
- 不同燃料通过梯度自适应调整动力学参数

### 3.3 辐射传递方程 (RTE)

$$q_{\mathrm{rad}}(z) = 2 \int_0^{r_{\max}} \kappa \sigma T^4 / \pi \cdot e^{-\int_0^r \kappa \, dr'} \, dr$$

- 吸收系数: $\kappa = \frac{6\pi E(m)}{\lambda} \cdot f_v$
- 其中 $E(m) = 0.37$, $\lambda = 633\,\mathrm{nm}$ (He-Ne 激光波长)
- 通过 PyTorch autograd **可微分积分**, 梯度可回传

""")

        st.markdown(r"""
---

## 4 · 损失函数

$$\mathcal{L} = \mathcal{L}_{\mathrm{data}} + w_{\mathrm{PDE}} \cdot \mathcal{L}_{\mathrm{PDE}} + \mathcal{L}_{\mathrm{BC}}$$

| 损失项 | 内容 | 说明 |
|--------|------|------|
| $\mathcal{L}_{\mathrm{data}}$ | 温度 + 辐射 + 碳烟 MSE | 监督实验数据拟合 |
| $\mathcal{L}_{\mathrm{PDE}}$ | 能量方程残差 + 碳烟输运残差 | 物理规律约束 |
| $\mathcal{L}_{\mathrm{BC}}$ | 边界条件 (T → T_amb, fv → 0) | 火焰边界物理一致性 |

- PDE 权重 $w_{\mathrm{PDE}}$ 在训练过程中从 0 线性增长到 1（课程式注入）
- Arrhenius 源项使用 tanh 压缩 + 梯度截断防止训练爆炸

""")

        st.markdown(r"""
---

## 5 · 四阶段课程式训练

训练采用分阶段策略，从纯数据驱动逐步过渡到物理约束联合优化：

""")

        # 训练阶段表格
        phases_data = [
            {"阶段": "① Warm-up", "Epochs": "0 – 2,000", "学习率": "1×10⁻³",
             "PDE": "❌", "BC": "❌", "说明": "纯数据预热，建立基础拟合"},
            {"阶段": "② Physics Injection", "Epochs": "2,000 – 7,000", "学习率": "3×10⁻⁴",
             "PDE": "0→1 线性", "BC": "❌", "说明": "物理约束渐进注入，避免冲突"},
            {"阶段": "③ Joint Fine-tuning", "Epochs": "7,000 – 10,000", "学习率": "1×10⁻⁴",
             "PDE": "✅", "BC": "✅", "说明": "全约束联合优化，后段切换 L-BFGS"},
            {"阶段": "④ Transfer", "Epochs": "10,000 – 11,000", "学习率": "1×10⁻⁵",
             "PDE": "✅", "BC": "✅", "说明": "冻结 Trunk，仅调子网络，新燃料迁移"},
        ]
        st.table(pd.DataFrame(phases_data))

        st.markdown(r"""
---

## 6 · 实验数据

| 数据类型 | 燃料覆盖 | 当量比 φ | 测点数 |
|---------|---------|---------|--------|
| 温度 T (K) | 乙烯、甲苯、甲醇 | 0.8 – 1.0 | ~732 |
| 辐射 q_rad (W/m²) | 乙烯、甲苯 | 0.8 – 1.2 | ~196 |
| 碳烟 fv (ppm) | 甲苯 | 0.8 – 0.9 | ~111 |

- 温度数据来自多 HAB (5–90 mm) 位置的热电偶测量
- 辐射数据来自多角度辐射计的侧向积分通量
- 碳烟体积分数来自激光消光法 / LII 测量
- 训练时 φ=1.1 的辐射数据被留出作为验证集

""")

        st.markdown(r"""
---

## 7 · 物理常数与超参数
""")

        col_phys, col_hyper = st.columns(2)
        with col_phys:
            st.markdown("**物理常数**")
            st.table(pd.DataFrame([
                {"参数": "σ (Stefan-Boltzmann)", "值": "5.67×10⁻⁸ W/(m²·K⁴)"},
                {"参数": "E(m) 碳烟吸收函数", "值": "0.37"},
                {"参数": "λ 激光波长", "值": "633 nm (He-Ne)"},
                {"参数": "r_max 径向范围", "值": "15 mm"},
                {"参数": "z_max 轴向范围", "值": "90 mm"},
                {"参数": "T_amb 环境温度", "值": "300 K"},
            ]))
        with col_hyper:
            st.markdown("**训练超参数**")
            st.table(pd.DataFrame([
                {"参数": "优化器", "值": "Adam → L-BFGS"},
                {"参数": "权重衰减", "值": "1×10⁻⁵"},
                {"参数": "调度器", "值": "Cosine Annealing"},
                {"参数": "PDE 配点/epoch", "值": "10,000"},
                {"参数": "边界点/epoch", "值": "2,000"},
                {"参数": "总 Epochs", "值": "11,000"},
            ]))

        st.markdown(r"""
---

## 8 · Arrhenius 可学习参数

碳烟动力学的 6 个 Arrhenius 参数由网络自动学习（初始值 → 训练后）：
""")

        arr_data = [
            {"参数": "C_α (成核系数)", "初始值": "exp(0)=1.0",
             "训练值": f"{torch.exp(model.log_C_alpha).item():.4f}",
             "物理意义": "碳烟粒子成核速率前因子"},
            {"参数": "T_α (成核活化温度)", "初始值": "21000 K",
             "训练值": f"{model.T_alpha.item():.0f} K",
             "物理意义": "成核反应活化能 / k_B"},
            {"参数": "C_β (生长系数)", "初始值": "exp(0)=1.0",
             "训练值": f"{torch.exp(model.log_C_beta).item():.4f}",
             "物理意义": "表面生长速率前因子"},
            {"参数": "T_β (生长活化温度)", "初始值": "12100 K",
             "训练值": f"{model.T_beta.item():.0f} K",
             "物理意义": "表面生长活化能 / k_B"},
            {"参数": "C_ω (氧化系数)", "初始值": "exp(0)=1.0",
             "训练值": f"{torch.exp(model.log_C_omega).item():.4f}",
             "物理意义": "氧化消耗速率前因子"},
            {"参数": "T_ω (氧化活化温度)", "初始值": "19680 K",
             "训练值": f"{model.T_omega.item():.0f} K",
             "物理意义": "氧化反应活化能 / k_B"},
        ]
        st.table(pd.DataFrame(arr_data))

        st.markdown(r"""
---

## 9 · 技术栈

- **深度学习框架**: PyTorch ≥ 2.0
- **Web 应用**: Streamlit
- **可视化**: Matplotlib
- **物理编码**: 自定义可微分 RTE 积分层 (PyTorch autograd)
- **部署**: Streamlit Cloud (CPU 推理, 单次预测 < 1 ms)
""")


if __name__ == "__main__":
    main()
