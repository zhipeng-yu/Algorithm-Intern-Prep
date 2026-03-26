
## 📅 学习日志

### 🗓️ 2026/3/9 — 环境搭建
- ✅ 下载并安装 **Anaconda**
- ✅ 配置 **PyTorch** 环境（支持 GPU）
- ✅ 安装常用库：`pandas`, `numpy`, `matplotlib`, `seaborn`
-✅ 验证：
  - 所有库可正常导入
  - PyTorch 能检测到 CUDA GPU（`torch.cuda.is_available()` 返回 `True`）

### 🗓️ 2026/3/15 — 版本控制入门
- ✅ 创建 GitHub 仓库
- ✅ 初始化 `README.md`
- ✅ 学习 Git 基础命令：
  - `git init`, `git add`, `git commit`, `git push`
  - 分支管理与远程同步

### 🗓️ 2026/3/16 — 数据处理实践 (`data_handling_Iris.py`)
使用 [Kaggle Iris 数据集](https://www.kaggle.com/datasets/uciml/iris) 进行以下操作：
- ✅ 用 `pd.read_csv()` 加载数据
- ✅ 快速探索数据：
  - `.head()`：查看前几行
  - `.info()`：获取数据类型与缺失值概览
  - `.describe()`：统计摘要（均值、标准差等）
- ✅ 处理缺失值（本数据集无缺失，但练习了删除 `dropna()` 与填充 `fillna()` 方法）
- ✅ 数据筛选与分组：
  - 布尔索引（如 `df[df['species'] == 'setosa']`）
  - `groupby()` 按类别统计（如各类别的平均花瓣长度）

### 🗓️ 2026/3/19 — 数据可视化
- ✅ 使用 **Matplotlib** 绘制：
  - 直方图（特征分布）
  - 散点图（特征间关系，如 `sepal_length` vs `petal_width`）
- ✅ 使用 **Seaborn** 提升图表美观度：
  - `sns.pairplot()`：多变量关系矩阵
  - `sns.boxplot()`：按类别分布箱线图
  - 配色与样式优化（`sns.set_style("whitegrid")`）

  ### 🗓️ 2026/3/21 — titanic数据初步
使用 [Kaggle Iris 数据集](https://www.kaggle.com/datasets/uciml/iris) 进行以下操作：
- ✅ 用 `pd.read_csv()` 加载数据
- ✅ 快速探索数据：
  - `.head()`：查看前几行
  - `.info()`：获取数据类型与缺失值概览
  - `.describe()`：统计摘要（均值、标准差等）
- ✅ 处理缺失值（本数据集无缺失，但练习了删除 `dropna()` 与填充 `fillna()` 方法）
- ✅ 数据筛选与分组：
  - 布尔索引（如 `df[df['species'] == 'setosa']`）
  - `groupby()` 按类别统计（如各类别的平均花瓣长度）

---

## 🛠️ 技术栈
| 工具/库       | 用途                     |
|--------------|--------------------------|
| Anaconda     | 环境与包管理              |
| PyTorch      | 深度学习框架（GPU 支持）   |
| Pandas       | 数据加载与清洗            |
| Matplotlib   | 基础绘图                 |
| Seaborn      | 高级统计可视化            |
| Git / GitHub | 版本控制与代码托管        |


---

✨ *持续学习，点滴进步！*