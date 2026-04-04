"""导出预处理后的训练数据为 CSV，用于仓库归档。

只保留训练所需的数值数据，不包含原始敏感文件。
"""
import os
import sys
import yaml

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_dir = os.path.dirname(repo_dir)
sys.path.insert(0, project_dir)
os.chdir(project_dir)

from src.data.preprocess import build_training_dataset

with open("src/configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

data = build_training_dataset(cfg)

out_dir = os.path.join(repo_dir, "data")
os.makedirs(out_dir, exist_ok=True)

# 温度
if len(data["temperature"]) > 0:
    path = os.path.join(out_dir, "temperature.csv")
    data["temperature"].to_csv(path, index=False)
    print(f"温度: {len(data['temperature'])} 条 → {path}")

# 辐射
if len(data["radiation"]) > 0:
    path = os.path.join(out_dir, "radiation.csv")
    data["radiation"].to_csv(path, index=False)
    print(f"辐射: {len(data['radiation'])} 条 → {path}")

# 碳烟
if len(data["soot"]) > 0:
    # fv_values 是 list，序列化为字符串
    df = data["soot"].copy()
    df["fv_values"] = df["fv_values"].apply(str)
    path = os.path.join(out_dir, "soot.csv")
    df.to_csv(path, index=False)
    print(f"碳烟: {len(data['soot'])} 条 → {path}")

print("数据导出完成!")
