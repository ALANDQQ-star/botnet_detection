# BOTNET HUNTER - 僵尸网络威胁分析平台

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📖 项目简介

BOTNET HUNTER 是一个基于图神经网络（GNN）的僵尸网络威胁分析平台，专为学术实验环境设计。该平台集成了多种机器学习方法，能够实时检测、分析和推演僵尸网络威胁。

### 核心功能

- **🧠 双模式检测方法**
  - **现有方法 (GAT+GCN)**: 基于图注意力网络和图卷积网络的混合架构
  - **基线方法 (Bot-AHGCN)**: 基于异构信息网络的僵尸网络检测方法

- **📊 实时可视化**
  - 全球僵尸网络态势地图
  - 中国区域威胁热力图
  - 网络拓扑结构可视化
  - 图神经网络推理过程动态展示

- **🔍 威胁分析**
  - C2 服务器识别与定位
  - 攻击链建模（HMM）
  - 时空关联分析
  - 15 步动态推演

- **📈 评估指标**
  - AUC、F1、精确率、召回率实时统计
  - 阈值自动优化
  - 运行历史对比

## 🚀 快速部署

### 环境要求

- Python 3.10+
- CUDA 11.x+ (可选，用于 GPU 加速)
- 内存: 建议 16GB+

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/your-username/botnet-hunter.git
cd botnet-hunter
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **下载 GeoIP 数据库**

下载 [GeoLite2-City.mmdb](https://dev.maxmind.com/geoip/geolite2-free-geolocation-data) 并放置在项目根目录。

5. **准备数据集**

下载 CTU-13 数据集并配置数据路径：
```bash
# 在参数配置页面设置数据目录
# 默认路径: /root/autodl-fs/CTU-13/CTU-13-Dataset
```

### 启动应用

```bash
streamlit run app.py
```

应用将在 `http://localhost:8501` 启动。

## 📁 项目结构

```
BOT/
├── app.py                      # Streamlit 主应用
├── main.py                     # 后端主程序入口
├── ui_state.py                 # UI 状态管理
├── viz_utils.py                # 可视化工具函数
│
├── data_loader.py              # CTU-13 数据加载器
├── heterogeneous_graph.py      # 异构图构建
├── graph_contrastive_learning.py  # GNN 模型定义与训练
├── bot_ahgcn_baseline.py       # Bot-AHGCN 基线方法
│
├── attack_chain_fsm.py         # HMM 攻击链推理
├── spatiotemporal_analysis.py  # 时空关联分析
├── llm_advisor.py              # LLM 安全建议模块
│
├── pages/                      # Streamlit 多页面
│   ├── 01_流量态势.py
│   ├── 02_威胁情报.py
│   ├── ...
│   └── 09_报告导出.py
│
├── requirements.txt            # Python 依赖
├── .gitignore                  # Git 忽略文件
└── README.md                   # 项目说明
```

## 🎮 使用指南

### 1. 配置参数

点击「配置」按钮进入参数配置页面：
- **数据目录**: 设置 CTU-13 数据集路径
- **训练场景**: 选择用于训练的场景（默认 1-10）
- **测试场景**: 选择测试场景（默认 13）
- **训练轮次**: 设置 GNN 训练轮次
- **学习率**: 设置优化器学习率
- **检测阈值**: 设置威胁判定阈值

### 2. 启动分析

1. 选择检测方法（现有方法/基线方法）
2. 点击「启动分析」按钮
3. 观察实时进度和可视化更新

### 3. 查看结果

- **统计卡片**: 显示威胁节点数、C2 服务器数、风险评分
- **地图视图**: 切换全球/中国视图查看威胁分布
- **拓扑概要**: 展开查看网络拓扑结构
- **详细报告**: 导航到各子页面查看详细分析

## 🔧 技术架构

### 检测方法

#### 现有方法 (GAT+GCN)
```
输入特征 → GAT 层 1 → GAT 层 2 → 全局池化 → 分类器
```
- 多头注意力机制捕获节点间关系
- 图卷积聚合邻居特征
- 端到端学习节点表示

#### 基线方法 (Bot-AHGCN)
```
AHIN 构建 → 元路径相似性 → 注意力聚合 → 分类
```
- 异构信息网络建模
- 元路径引导的相似性计算
- 类型感知的注意力机制

### HMM 攻击链建模

状态空间：启动 → DNS 查询 → C2 建立 → C2 维持 → 扫描 → 攻击 → 数据外泄 → 僵死

### C2 识别

基于时空特征的弱监督学习：
- 流量纯度 (Purity)
- 端口集中度 (Port Concentration)
- Bot 比例 (Bot Ratio)
- 通信间隔统计 (IAT)

## 📊 性能指标

| 方法 | AUC | F1 | 精确率 | 召回率 |
|------|-----|-----|--------|--------|
| GAT+GCN | ~0.95 | ~0.80 | ~0.85 | ~0.75 |
| Bot-AHGCN | ~0.92 | ~0.75 | ~0.80 | ~0.70 |

*注：实际性能取决于数据集和参数配置*

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目仅供学术研究使用。请参考 LICENSE 文件了解详情。

## 🙏 致谢

- [CTU-13 Dataset](https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Streamlit](https://streamlit.io/)

## 📧 联系方式

如有问题或建议，请通过 Issue 联系。

---

**⚠️ 免责声明**: 本工具仅用于学术研究和安全分析目的。使用者需遵守相关法律法规，不得用于非法活动。