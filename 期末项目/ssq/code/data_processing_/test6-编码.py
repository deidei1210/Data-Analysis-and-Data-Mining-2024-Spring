import os
import pandas as pd
import numpy as np

# 输入和输出文件夹路径
input_folder = 'datasets/summary'
output_folder = 'datasets_new/summary'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 处理Alcohol Drinking History列
def process_alcohol_column(value):
    return 1 if value == 'drinker' else 0

# 处理Type of Diabetes列
def process_diabetes_column(value):
    if value == 'T1DM':
        return 1
    elif value == 'T2DM':
        return 2
    else:
        return value

# 处理Hypoglycemia列
def process_hypoglycemia_column(value):
    return 1 if value == 'yes' else 0

# 初始化存储编号映射的字典
multi_value_columns = [
    'Acute Diabetic Complications', 'Diabetic Macrovascular Complications',
    'Diabetic Microvascular Complications', 'Comorbidities',
    'Hypoglycemic Agents', 'Other Agents'
]
column_value_to_num_dict = {col: {'none': 0} for col in multi_value_columns}

# 提取和清理唯一值
for filename in os.listdir(input_folder):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_excel(file_path)
        # 重命名列
        df.rename(columns={
            'Alcohol Drinking History (drinker/non-drinker)': 'Alcohol Drinking History',
            'Diabetic Macrovascular  Complications': 'Diabetic Macrovascular Complications',
            'Patient Number': 'patient_id'
        }, inplace=True)
        # 处理多值列，收集所有唯一值
        for col in multi_value_columns:
            unique_values = set()
            df[col].dropna().apply(lambda x: unique_values.update([item.strip() for item in x.split(',')]))
            unique_values -= {'none'}
            unique_values = sorted(unique_values)
            for val in unique_values:
                if val not in column_value_to_num_dict[col]:
                    column_value_to_num_dict[col][val] = len(column_value_to_num_dict[col])

# 准备多标签编码
for col in multi_value_columns:
    column_value_to_num_dict[col] = {k: v for v, k in enumerate(sorted(column_value_to_num_dict[col]))}

# 辅助函数，将多标签数组转换为二进制字符串
def multi_label_to_binary(label_list):
    return ''.join(map(str, label_list))

# 遍历输入文件夹中的所有Excel文件，进行处理和编码替换
for filename in os.listdir(input_folder):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_excel(file_path)

        # 重命名列
        df.rename(columns={
            'Alcohol Drinking History (drinker/non-drinker)': 'Alcohol Drinking History',
            'Diabetic Macrovascular  Complications': 'Diabetic Macrovascular Complications',
            'Patient Number': 'patient_id'
        }, inplace=True)

        # 处理Alcohol Drinking History列
        df['Alcohol Drinking History'] = df['Alcohol Drinking History'].apply(process_alcohol_column)

        # 处理Type of Diabetes列
        df['Type of Diabetes'] = df['Type of Diabetes'].apply(process_diabetes_column)

        # 处理Hypoglycemia (yes/no)列
        df['Hypoglycemia (yes/no)'] = df['Hypoglycemia (yes/no)'].apply(process_hypoglycemia_column)

        # 填充包含“/”的缺失值为NaN
        df.replace('/', np.nan, inplace=True)

        # 处理多值列，进行多标签编码替换
        for col in multi_value_columns:
            max_num = max(column_value_to_num_dict[col].values())
            df[col] = df[col].apply(
                lambda x: multi_label_to_binary(
                    [0] * (max_num + 1) if pd.isna(x) else
                    [1 if i in [column_value_to_num_dict[col][item.strip()] for item in x.split(',')] else 0 for i in range(max_num + 1)]
                )
            )

        # 保存处理后的数据
        output_file_path = os.path.join(output_folder, filename)
        df.to_excel(output_file_path, index=False)

# 输出编号-内容对应表格
for col, num_dict in column_value_to_num_dict.items():
    num_df = pd.DataFrame(list(num_dict.items()), columns=['content', 'num'])
    num_df.to_excel(os.path.join(output_folder, f'{col}.xlsx'), index=False)

print("Data processing complete.")
