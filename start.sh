#!/bin/bash
# ═══════════════════════════════════════
#  🔥 辐射编程 PINN — 一键启动
# ═══════════════════════════════════════
#  双击此文件或在终端运行: bash start.sh

cd "$(dirname "$0")"

echo "🔥 辐射编程 PINN 交互平台"
echo "========================="
echo ""

# 检查 Python
if ! command -v python3 &>/dev/null; then
    echo "❌ 未找到 Python3，请先安装 Python"
    exit 1
fi

# 检查依赖
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "📦 首次运行，正在安装依赖..."
    pip3 install -r requirements.txt
fi

# 检查 checkpoint
if [ ! -d "checkpoints" ] || [ -z "$(ls checkpoints/*.pt 2>/dev/null)" ]; then
    echo "⚠️  未找到训练好的模型权重"
    echo "   请先训练: python3 scripts/train.py"
    echo "   训练完成后重新运行此脚本"
    exit 1
fi

echo "🚀 正在启动 Web 应用..."
echo "   浏览器会自动打开，如果没有请访问: http://localhost:8501"
echo ""

streamlit run app.py --server.headless=true --browser.gatherUsageStats=false
