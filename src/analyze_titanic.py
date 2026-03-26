import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/titanic/train_and_test2.csv')

# 1. 查看前5行
print("--- Head ---")
print(df.head())

'''
--- Head ---
   Passengerid   Age  ...  zero.18  2urvived
0            1  22.0  ...        0         0   
1            2  38.0  ...        0         1   
2            3  26.0  ...        0         1   
3            4  35.0  ...        0         1   
4            5  35.0  ...        0         0 
'''

# 2. 查看数据基本信息 (非常重要！)
print("\n--- Info ---")
print(df.info())

'''
--- Info ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1309 entries, 0 to 1308
Data columns (total 28 columns):
 #   Column       Non-Null Count  Dtype        
---  ------       --------------  -----        
 0   Passengerid  1309 non-null   int64        
 1   Age          1309 non-null   float64      
 2   Fare         1309 non-null   float64      
 3   Sex          1309 non-null   int64        
 4   sibsp        1309 non-null   int64        
 5   zero         1309 non-null   int64        
 6   zero.1       1309 non-null   int64        
 7   zero.2       1309 non-null   int64        
 8   zero.3       1309 non-null   int64        
 9   zero.4       1309 non-null   int64        
 10  zero.5       1309 non-null   int64        
 11  zero.6       1309 non-null   int64        
 12  Parch        1309 non-null   int64        
 13  zero.7       1309 non-null   int64        
 14  zero.8       1309 non-null   int64        
 15  zero.9       1309 non-null   int64        
 16  zero.10      1309 non-null   int64        
 17  zero.11      1309 non-null   int64        
 18  zero.12      1309 non-null   int64        
 19  zero.13      1309 non-null   int64        
 20  zero.14      1309 non-null   int64        
 21  Pclass       1309 non-null   int64        
 22  zero.15      1309 non-null   int64        
 23  zero.16      1309 non-null   int64        
 24  Embarked     1307 non-null   float64      
 25  zero.17      1309 non-null   int64        
 26  zero.18      1309 non-null   int64        
 27  2urvived     1309 non-null   int64        
dtypes: float64(3), int64(25)
memory usage: 286.5 KB
None
'''

# 3. 查看数值型数据的统计摘要
print("\n--- Describe ---")
print(df.describe())

'''
--- Describe ---
       Passengerid  ...     2urvived
count  1309.000000  ...  1309.000000
mean    655.000000  ...     0.261268
std     378.020061  ...     0.439494
min       1.000000  ...     0.000000
25%     328.000000  ...     0.000000
50%     655.000000  ...     0.000000
75%     982.000000  ...     1.000000
max    1309.000000  ...     1.000000

[8 rows x 28 columns]
'''

# 1. 模拟缺失值 (在真实工作中，数据里会有 NaN)
# 我们随机把几行数据改成 NaN
#第一行：找到第 10 行（索引为 10），将该行的 SepalWidthCm（花萼宽度）列的值修改为 None（即空值/NaN）。
#第二行：找到第 20 行（索引为 20），将该行的 PetalWidthCm（花瓣宽度）列的值修改为 None（即空值/NaN）。
#df.loc[10, 'SepalWidthCm'] = None
#df.loc[20, 'PetalWidthCm'] = None

# 2. 检查缺失值
if df.isnull().sum().sum() > 0:
    print("发现缺失值，开始填充...")
    df = df.fillna(df.mean(numeric_only=True))
else:
    print("数据完美，无需填充！")

# 4. 数据类型转换 (比如把类别转换成类别类型)
# 将 'class' 列转换为类别类型 (Categorical)，节省内存且便于后续建模
#df['Species'] = df['Species'].astype('category')

print("\n清洗并转换类型后的数据 info:")
print(df.info())

print("\n--- Describe ---")
print(df.describe())