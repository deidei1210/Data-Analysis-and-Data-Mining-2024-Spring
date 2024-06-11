import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 处理文件的函数
def process_file(input_path, output_path):
    # 读取Excel文件，指定引擎为openpyxl
    df = pd.read_excel(input_path, engine='openpyxl')

    # 连续型特征：均值填充或中位数填充
    df['CGM'].fillna(df['CGM'].mean(), inplace=True)

    # 分类特征：众数填充或填充特定值
    df['Non-insulin hypoglycemic agents'].fillna('Unknown', inplace=True)
    df['Non-insulin hypoglycemic drug'].fillna('Unknown', inplace=True)
    df['Insulin type-s.c.'].fillna('Unknown', inplace=True)

    # 特殊处理
    df['Insulin dose - s.c. (IU)'].fillna(0, inplace=True)
    df['Non-insulin hypoglycemic dose (mg)'].fillna(0, inplace=True)
    df['CSII - bolus insulin (Novolin R, IU)'].fillna(0, inplace=True)
    df['CSII - basal insulin (Novolin R, IU / H)'].fillna(0, inplace=True)
    df['Carbohydrate/g'].fillna(0, inplace=True)

    # 滚动统计量：可以用前面的值填充
    df['rolling_mean_CGM'].fillna(method='ffill', inplace=True)
    df['rolling_std_CGM'].fillna(method='ffill', inplace=True)

    # 日期特征通常不填充，如果有空值，可以考虑删除这些行
    df.dropna(subset=['Date'], inplace=True)
    columns_to_remove = [
        'Dietary intake', '饮食', 'food segmentation', 'hour', 'day_of_week','Non-insulin hypoglycemic agents'
    ]
    df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    # 处理rolling_mean_CGM和rolling_std_CGM前三行的数据
    if not df['rolling_mean_CGM'].isnull().any():
        df.loc[:2, 'rolling_mean_CGM'] = df['rolling_mean_CGM'].mean()
    if not df['rolling_std_CGM'].isnull().any():
        df.loc[:2, 'rolling_std_CGM'] = df['rolling_std_CGM'].mean()

    # 填充所有包含NaN值的列为0
    nan_columns = df.columns[df.isnull().any()]
    df[nan_columns] = df[nan_columns].fillna(0)

    # 列表中指定的列进行归一化
    columns_to_normalize = [
        'CGM',
        'CBG (mg / dl)',
        'Blood Ketone (mmol / L)',
        'Insulin dose - s.c. (IU)',
        'CSII - bolus insulin (Novolin R, IU)',
        'CSII - basal insulin (Novolin R, IU / H)',
        'rolling_mean_CGM',
        'rolling_std_CGM'
    ]

    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    # 保存处理后的文件
    df.to_excel(output_path, index=False, engine='openpyxl')

def process_folder(folder_path, output_folder):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            print(file_name)
            file_path = os.path.join(folder_path, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            process_file(file_path, output_file_path)
    print("Processing " + folder_path + " completed")

# 定义文件路径
input_folder_t1dm = 'datasets_new1/Shanghai_T1DM'
input_folder_t2dm = 'datasets_new1/Shanghai_T2DM'
output_folder_t1dm = 'DataProcessing_new/Shanghai_T1DM'
output_folder_t2dm = 'DataProcessing_new/Shanghai_T2DM'

# 创建新的文件夹
os.makedirs(output_folder_t1dm, exist_ok=True)
os.makedirs(output_folder_t2dm, exist_ok=True)

# 处理数据
process_folder(input_folder_t1dm, output_folder_t1dm)
process_folder(input_folder_t2dm, output_folder_t2dm)
print("数据处理完成，并已保存到新的文件夹中。")
