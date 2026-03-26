import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/iris/Iris.csv')

# 1. 查看前5行
print("--- Head ---")
print(df.head())

# 2. 查看数据基本信息 (非常重要！)
print("\n--- Info ---")
print(df.info())

# 3. 查看数值型数据的统计摘要
print("\n--- Describe ---")
print(df.describe())

# 💡 思考题：观察describe的输出，哪个特征的方差最大？哪个最小？

#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 150 entries, 0 to 149
#Data columns (total 6 columns):
#含义：这是一个表格（DataFrame），共有 150 行数据（编号 0 到 149），6 列特征。

# #   Column         Non-Null Count  Dtype
#---  ------         --------------  -----
# 0   Id             150 non-null    int64
# ...
# 5   Species        150 non-null    object

#Non-Null Count (非空计数)：这是重点！
#每一列都是 150 non-null，意味着没有缺失值（没有空单元格）。
#实战意义：在真实工作中，这里经常会出现 148 non-null，那就说明有2个数据丢了，你需要决定是删除还是填充。今天你很幸运，数据很干净！

#Dtype (数据类型)：
#int64：整数（如 Id）。
#float64：小数（如 SepalLengthCm 花萼长度）。
#object：通常是字符串或文本（如 Species 鸢尾花的品种名）。
#实战意义：机器学习模型通常只能处理数字。看到 Species 是 object，你就知道后续建模前需要把它转换成数字（比如 0, 1, 2），这个过程叫编码（Encoding）

#dtypes: float64(4), int64(1), object(1)
#memory usage: 7.2+ KB
#含义：内存占用极小（7.2 KB），说明处理速度会非常快。

#📊 第二部分：.describe() —— 数据的“统计画像”
#这部分主要回答：数据分布如何？有没有异常值？大概范围是多少？
#注意：它默认只统计数值型列（所以 Species 和 Id 没出现在详细统计中，除非特意设置）。

#| 指标 | 数值 | 含义解读 (工程师思维) |
#| :--- | :--- | :--- |
#| count | 150.0 | 确认参与统计的数量，再次验证无缺失。 |
#| mean | 5.84 | 平均值。大部分鸢尾花的花萼长度在 5.84cm 左右。 |
#| std | 0.83 | 标准差。衡量数据的波动程度。0.83 说明数据比较集中，大家长得都差不多，差异不大。如果这个数很大（比如 5.0），说明数据很分散。 |
#| min | 4.30 | 最小值。检查是否有不合逻辑的负数或极小值（异常检测）。 |
#| 25% | 5.10 | 下四分位数。表示有 25% 的数据小于 5.10。 |
#| 50% | 5.80 | 中位数。表示一半的数据比 5.80 小，一半比它大。如果中位数和平均值(mean)差距很大，说明数据分布不均匀（偏态）。 |
#| 75% | 6.40 | 上四分位数。表示有 75% 的数据小于 6.40。 |
#| max | 7.90 | 最大值。检查是否有离谱的大数（比如 100cm，那肯定是录入错误）。 |


# 1. 模拟缺失值 (在真实工作中，数据里会有 NaN)
# 我们随机把几行数据改成 NaN
#第一行：找到第 10 行（索引为 10），将该行的 SepalWidthCm（花萼宽度）列的值修改为 None（即空值/NaN）。
#第二行：找到第 20 行（索引为 20），将该行的 PetalWidthCm（花瓣宽度）列的值修改为 None（即空值/NaN）。
df.loc[10, 'SepalWidthCm'] = None
df.loc[20, 'PetalWidthCm'] = None

#print("\n--- Info ---")
#print(df.info())

# 2. 检查缺失值
if df.isnull().sum().sum() > 0:
    print("发现缺失值，开始填充...")
    df = df.fillna(df.mean(numeric_only=True))
else:
    print("数据完美，无需填充！")

# 4. 数据类型转换 (比如把类别转换成类别类型)
# 将 'class' 列转换为类别类型 (Categorical)，节省内存且便于后续建模
df['Species'] = df['Species'].astype('category')

print("\n清洗并转换类型后的数据 info:")
print(df.info())

print("\n--- Describe ---")
print(df.describe())

# 1. 布尔索引筛选
# 任务：找出所有花萼长度 (sepal_length) 大于 5.0 的鸢尾花
filtered_df = df[df['SepalLengthCm'] > 5.0]
print("筛选结果 (前5行):")
print(filtered_df.head())

# 2. 复杂筛选 (逻辑与 & / 或 |)
# 任务：找出花萼长度 > 5.0 且 花瓣宽度 (petal_width) < 1.0 的数据
complex_filtered = df[(df['SepalLengthCm'] > 5.0) & (df['SepalWidthCm'] < 3.0)]
print("\n复杂筛选结果:")
print(complex_filtered)

# 3. GroupBy 分组聚合
# 任务：按 'class' (种类) 分组，计算每组的平均值
grouped = df.groupby('Species').mean()
print("\n按种类分组的平均值:")
print(grouped)

# 4. 进阶：按 'class' 分组，统计每组的数量
count_grouped = df.groupby('Species').size()
print("\n每种类别的数量:")
print(count_grouped)



# 1. 绘制直方图 (Histogram)

# 假设你已经读取了数据到 df 变量中
# df = pd.read_csv('your_data.csv')

# bins 参数表示把数据分成多少个区间（柱子）
plt.figure(figsize=(10, 6))  # 设置画布大小，可选

# 绘制直方图：展示 'sepal length (cm)' 这一列数据的分布情况
# 参数说明：
# - df['sepal length (cm)']: 从 DataFrame 中提取要分析的数据列（花萼长度）
# - bins=20: 将数据范围划分为 20 个区间（即画出 20 根柱子），数值越大柱子越细
# - color='skyblue': 设置柱子的填充颜色为天蓝色
# - edgecolor='black': 设置柱子边缘的颜色为黑色，让每根柱子界限更清晰
plt.hist(df['SepalLengthCm'], bins=20, color='skyblue', edgecolor='black')

# 2. 添加标题和标签（这是代码规范的重要部分！）
# fontsize设置标题字体大小为 X
plt.title('Distribution of Sepal Length', fontsize=16)
# 设置 x 轴的标签（横坐标说明）
plt.xlabel('Sepal Length (cm)', fontsize=12)
# 设置 y 轴的标签（纵坐标说明）
plt.ylabel('Frequency', fontsize=12)

# 3. 显示图形
plt.show()


# 1. 绘制散点图 (Scatter Plot)
# 创建一个新的图形窗口（Figure 对象），并设置其尺寸
# figsize=(宽度, 高度)，单位是英寸（inches）
plt.figure(figsize=(10, 6))


# x轴放一个特征，y轴放另一个特征
# 参数说明：
# - df['sepal length (cm)']：作为 x 轴的数据，取自 DataFrame 中的 'sepal length (cm)' 列
# - df['petal length (cm)']：作为 y 轴的数据，取自 DataFrame 中的 'petal length (cm)' 列
# - alpha=0.6：设置散点的透明度（opacity），取值范围 [0, 1]
# 0 = 完全透明，1 = 完全不透明；0.6 可以避免点重叠时完全遮挡，便于观察密度
plt.scatter(df['SepalLengthCm'], df['PetalLengthCm'], alpha=0.6)

# 2. 添加标题和标签
# fontsize=16：指定标题字体大小为 16 磅（points）
plt.title('Relationship between Sepal Length and Petal Length', fontsize=16)
# 设置 x 轴的标签（横坐标说明）
plt.xlabel('Sepal Length (cm)', fontsize=12)
# 设置 y 轴的标签（纵坐标说明）
plt.ylabel('Petal Length (cm)', fontsize=12)

# 3. 显示图形
plt.show()


#绘制更美观的“配对图”
#如果你想一次性看数据集中所有特征之间的关系，用 Matplotlib 要写很多循环，但 Seaborn 一行代码搞定。
#import seaborn as sns

# 1. 设置主题（Seaborn 的主题通常比默认的好看）
sns.set(style="whitegrid")

# 2. 绘制配对图 (Pairplot)
# 这个图会自动对数据集中的每一对数值特征画散点图，对角线上画直方图
# hue 参数可以让你根据类别（比如 Iris 的 Species）给点上色
sns.pairplot(df, hue="Species")

# 3. 显示图形
plt.show()
