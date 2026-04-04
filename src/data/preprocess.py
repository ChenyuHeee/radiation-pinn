"""数据预处理管线：

1. 温度 CSV → 稳态段时间平均
2. 修正后温度 Excel → 直接读取
3. 辐射 CSV → 稳态段时间平均
4. 碳烟 Excel → 结构化解析
"""
import os
import glob
import numpy as np
import pandas as pd


# ─────────────────── 原始温度 CSV ───────────────────

def _read_csv_auto(path: str) -> pd.DataFrame:
    """自动尝试多种编码读取 CSV。"""
    for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding="latin-1")


def load_raw_temperature_csv(path: str) -> float:
    """读取单个温度 CSV，返回稳态段平均温度 (°C)。

    格式: 列0=时间(HH:MM:SS), 列1="通道 1 平均 (C)"
    稳态段：取后 80% 数据的平均值（丢弃初始升温段）。
    """
    df = _read_csv_auto(path)
    col = df.columns[1]  # "通道 1 平均 (C)" 可能有编码问题
    vals = pd.to_numeric(df[col], errors="coerce").dropna().values
    if len(vals) == 0:
        return float("nan")
    # 稳态段：后 80%
    start = int(len(vals) * 0.2)
    return float(np.mean(vals[start:]))


def load_raw_temperatures(base_dir: str, equiv_ratios: list, hab_list: list,
                          fuel_id: int) -> pd.DataFrame:
    """批量读取原始温度数据。

    返回 DataFrame: columns=[fuel_id, phi, hab_mm, T_measured_C, T_measured_K]
    """
    records = []
    for phi in equiv_ratios:
        phi_dir = os.path.join(base_dir, str(phi), "温度")
        if not os.path.isdir(phi_dir):
            continue
        for hab in hab_list:
            hab_dir = os.path.join(phi_dir, str(hab))
            if not os.path.isdir(hab_dir):
                continue
            csvs = sorted(glob.glob(os.path.join(hab_dir, "*.csv")))
            temps = []
            for csv_path in csvs:
                t = load_raw_temperature_csv(csv_path)
                if not np.isnan(t):
                    temps.append(t)
            if temps:
                for t in temps:
                    records.append({
                        "fuel_id": fuel_id,
                        "phi": phi,
                        "hab_mm": hab,
                        "T_measured_C": t,
                        "T_measured_K": t + 273.15,
                    })
    return pd.DataFrame(records)


# ─────────────────── 修正后温度 Excel ───────────────────

def load_corrected_temperatures(directory: str, fuel_id: int,
                                equiv_ratios: list) -> pd.DataFrame:
    """读取修正后温度 Excel 文件。

    文件格式: 无表头，第1列=HAB(mm)，后续列=重复测量的修正温度(K)。
    文件名如 0.8(2)_corrected.xlsx, 1_corrected.xlsx。
    """
    records = []
    for phi in equiv_ratios:
        # 匹配文件名
        pattern = os.path.join(directory, f"{phi}*_corrected.xlsx")
        files = glob.glob(pattern)
        if not files:
            # 尝试整数形式 (1.0 → "1")
            if phi == int(phi):
                pattern = os.path.join(directory, f"{int(phi)}*_corrected.xlsx")
                files = glob.glob(pattern)
        if not files:
            continue
        df = pd.read_excel(files[0], header=None)
        # 第0行是 HAB=5 的数据（被pandas误读为列名），需要特殊处理
        # 重新读取：所有行都是数据
        # 列0 = HAB(mm), 列1~N = 重复测量温度
        for idx in range(len(df)):
            row = df.iloc[idx]
            hab = float(row.iloc[0])
            temps = pd.to_numeric(row.iloc[1:], errors="coerce").dropna().values
            if len(temps) == 0:
                continue
            t_mean = float(np.mean(temps))
            records.append({
                "fuel_id": fuel_id,
                "phi": phi,
                "hab_mm": hab,
                "T_corrected_K": t_mean,
            })
    return pd.DataFrame(records)


# ─────────────────── 辐射 CSV ───────────────────

def load_radiation_csv(path: str) -> float:
    """读取单个辐射 CSV，返回稳态段平均辐射通量 (W/m²)。

    格式: 时间, 热通量(W/m2), 电压(mV)
    """
    try:
        df = _read_csv_auto(path)
    except Exception:
        df = pd.read_csv(path, encoding="latin-1")
    # 列名可能有编码问题，用位置索引
    col = df.columns[1]
    vals = pd.to_numeric(df[col], errors="coerce").dropna().values
    if len(vals) == 0:
        return float("nan")
    start = int(len(vals) * 0.2)
    return float(np.mean(vals[start:]))


def load_radiation_data(base_dir: str, equiv_ratios: list,
                        fuel_id: int) -> pd.DataFrame:
    """批量读取辐射数据。

    辐射 CSV 与 HAB 的对应关系：按文件名排序后依次对应 HAB 列表。
    """
    records = []
    for phi in equiv_ratios:
        rad_dir = os.path.join(base_dir, str(phi), "辐射")
        if not os.path.isdir(rad_dir):
            continue
        csvs = sorted(glob.glob(os.path.join(rad_dir, "*.csv")))
        for i, csv_path in enumerate(csvs):
            q = load_radiation_csv(csv_path)
            if not np.isnan(q):
                records.append({
                    "fuel_id": fuel_id,
                    "phi": phi,
                    "hab_index": i,  # 辐射传感器位置索引
                    "q_rad_Wm2": q,
                })
    return pd.DataFrame(records)


# ─────────────────── 碳烟数据 Excel ───────────────────

def load_soot_data(path: str) -> pd.DataFrame:
    """解析碳烟 Excel 数据。

    该文件结构复杂（合并单元格），按行扫描提取有效数值块。
    返回 DataFrame: [fuel_name, phi, hab_mm, fv_values...]
    """
    df_raw = pd.read_excel(path, header=None)
    records = []
    current_fuel = None
    current_phi = None

    i = 0
    while i < len(df_raw):
        row = df_raw.iloc[i]
        cell0 = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""

        # 检测燃料名称行
        if cell0 in ("甲苯", "甲醇", "参考火焰", "乙烯"):
            current_fuel = cell0
            i += 1
            continue

        # 检测当量比行
        try:
            phi_val = float(cell0)
            if 0.5 <= phi_val <= 2.0:
                current_phi = phi_val
                i += 1
                continue
        except (ValueError, TypeError):
            pass

        # 检测数据行：第0列是 HAB (数值)
        try:
            hab = float(row.iloc[0])
            if hab < 0.5 or hab > 200:
                i += 1
                continue
            vals = pd.to_numeric(row.iloc[1:], errors="coerce").dropna().values
            if len(vals) > 0 and current_fuel and current_phi:
                records.append({
                    "fuel_name": current_fuel,
                    "phi": current_phi,
                    "hab_mm": hab,
                    "fv_mean": float(np.mean(vals)),
                    "fv_values": vals.tolist(),
                })
        except (ValueError, TypeError):
            pass
        i += 1

    return pd.DataFrame(records)


# ─────────────────── 热电偶辐射修正 ───────────────────

def thermocouple_correction(T_m_K: np.ndarray,
                            T_w_K: float = 300.0,
                            T_surr_K: float = 300.0,
                            d: float = 0.5e-3,
                            epsilon: float = 0.3,
                            Re: float = 5.0) -> np.ndarray:
    """热电偶辐射散热修正。

    T_g = T_m + ε σ (T_m⁴ - T_w⁴) · d / (k · Nu)

    k = 0.026 (T_m / T_surr)^0.78
    Nu = (0.24 + 0.56 Re^0.45) (T_m / T_surr)^0.17
    """
    sigma = 5.67e-8
    T_m = np.asarray(T_m_K, dtype=np.float64)
    ratio = T_m / T_surr_K

    k = 0.026 * ratio ** 0.78
    Nu = (0.24 + 0.56 * Re ** 0.45) * ratio ** 0.17

    correction = epsilon * sigma * (T_m ** 4 - T_w_K ** 4) * d / (k * Nu)
    T_g = T_m + correction
    return T_g


# ─────────────────── 统一预处理接口 ───────────────────

FUEL_NAME_TO_ID = {"参考火焰": 0, "乙烯": 0, "甲苯": 1, "甲醇": 2}


def _load_from_csv(data_cfg: dict) -> dict | None:
    """尝试从预处理后的 CSV 文件直接加载数据。

    如果三个 CSV 文件都存在，返回数据 dict；否则返回 None。
    """
    t_csv = data_cfg.get("temperature_csv", "")
    r_csv = data_cfg.get("radiation_csv", "")
    s_csv = data_cfg.get("soot_csv", "")

    if not (t_csv and os.path.isfile(t_csv)
            and r_csv and os.path.isfile(r_csv)
            and s_csv and os.path.isfile(s_csv)):
        return None

    temperature = pd.read_csv(t_csv)
    radiation = pd.read_csv(r_csv)
    soot = pd.read_csv(s_csv)

    sp_csv = data_cfg.get("species_csv", "")
    species = pd.read_csv(sp_csv) if sp_csv and os.path.isfile(sp_csv) else pd.DataFrame()

    return {"temperature": temperature, "radiation": radiation, "soot": soot, "species": species}


def build_training_dataset(cfg: dict) -> dict:
    """构建统一训练数据集。

    优先从预处理后的 CSV 加载；若 CSV 不存在则回退到原始数据处理。

    返回 dict:
      - temperature: DataFrame [fuel_id, phi, hab_mm, T_K]
      - radiation:   DataFrame [fuel_id, phi, hab_index, q_rad_Wm2]
      - soot:        DataFrame [fuel_name, phi, hab_mm, fv_mean]
    """
    data_cfg = cfg["data"]

    # 快速路径：直接从 CSV 加载
    cached = _load_from_csv(data_cfg)
    if cached is not None:
        return cached

    # 慢速路径：从原始实验数据处理
    eq_ratios = data_cfg["equiv_ratios"]

    # 1) 修正后温度（优先使用）
    temp_dfs = []
    if os.path.isdir(data_cfg.get("corrected_temp_toluene", "")):
        df = load_corrected_temperatures(
            data_cfg["corrected_temp_toluene"], fuel_id=1, equiv_ratios=eq_ratios)
        if len(df) > 0:
            df = df.rename(columns={"T_corrected_K": "T_K"})
            temp_dfs.append(df)

    if os.path.isdir(data_cfg.get("corrected_temp_methanol", "")):
        df = load_corrected_temperatures(
            data_cfg["corrected_temp_methanol"], fuel_id=2, equiv_ratios=eq_ratios)
        if len(df) > 0:
            df = df.rename(columns={"T_corrected_K": "T_K"})
            temp_dfs.append(df)

    # 参考火焰原始温度 → 修正
    if os.path.isdir(data_cfg.get("raw_temp_ref", "")):
        hab_ref = data_cfg.get("hab_ref", list(range(5, 95, 5)))
        df_raw = load_raw_temperatures(
            data_cfg["raw_temp_ref"], eq_ratios, hab_ref, fuel_id=0)
        if len(df_raw) > 0:
            df_raw["T_K"] = thermocouple_correction(df_raw["T_measured_K"].values)
            df_raw = df_raw[["fuel_id", "phi", "hab_mm", "T_K"]]
            temp_dfs.append(df_raw)

    temperature = pd.concat(temp_dfs, ignore_index=True) if temp_dfs else pd.DataFrame()

    # 2) 辐射
    rad_dfs = []
    if os.path.isdir(data_cfg.get("raw_temp_ref", "")):
        rad_dfs.append(load_radiation_data(
            data_cfg["raw_temp_ref"], eq_ratios, fuel_id=0))
    if os.path.isdir(data_cfg.get("raw_temp_toluene", "")):
        rad_dfs.append(load_radiation_data(
            data_cfg["raw_temp_toluene"], eq_ratios, fuel_id=1))
    radiation = pd.concat(rad_dfs, ignore_index=True) if rad_dfs else pd.DataFrame()

    # 3) 碳烟
    soot = pd.DataFrame()
    soot_path = data_cfg.get("soot_file", "")
    if os.path.isfile(soot_path):
        soot = load_soot_data(soot_path)

    return {
        "temperature": temperature,
        "radiation": radiation,
        "soot": soot,
        "species": pd.DataFrame(),
    }
