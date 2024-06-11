#最后用的是这个
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 输入和输出文件夹路径
static_folder = 'datasets_new/summary'
dynamic_folder_T1DM = 'datasets_new/Shanghai_T1DM'
dynamic_folder_T2DM = 'datasets_new/Shanghai_T2DM'
output_folder = 'datasets_new/combined6.4'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 处理静态特征数据
def process_static_data(file_path, diabetes_type):
    df = pd.read_excel(file_path)

    # 重命名列
    df.rename(columns={
        'Alcohol Drinking History (drinker/non-drinker)': 'Alcohol Drinking History',
        'Diabetic Macrovascular Complications': 'Diabetic Macrovascular Complications'
    }, inplace=True)

    # 处理特定列
    # df['Alcohol Drinking History'] = df['Alcohol Drinking History'].apply(lambda x: 1 if x == 'drinker' else 0)
    # df['Type of Diabetes'] = diabetes_type
    # df['Hypoglycemia (yes/no)'] = df['Hypoglycemia (yes/no)'].apply(lambda x: 1 if x == 'yes' else 0)
    df.replace('/', np.nan, inplace=True)

    # 重命名特定列
    if 'Duration of diabetes (years)' in df.columns:
        df.rename(columns={'Duration of diabetes (years)': 'Duration of Diabetes  (years)'}, inplace=True)

    if '2-hour Postprandial insulin (pmol/L)' in df.columns:
        df.rename(columns={'2-hour Postprandial insulin (pmol/L)': '2-hour Postprandial Insulin (pmol/L)'},
                  inplace=True)

    # 标准化数值型列
    #scaler = StandardScaler()
    #num_columns = df.select_dtypes(include=[np.number]).columns
    #df[num_columns] = scaler.fit_transform(df[num_columns])

    return df

# 读取静态特征数据
static_T1DM = process_static_data(os.path.join(static_folder, 'Shanghai_T1DM_Summary0.xlsx'), 1)
static_T2DM = process_static_data(os.path.join(static_folder, 'Shanghai_T2DM_Summary0.xlsx'), 2)
static_data = pd.concat([static_T1DM, static_T2DM], ignore_index=True)

# 处理动态血糖数据
def process_dynamic_data(dynamic_folder, static_df):
    dynamic_data_list = []

    for patient_id in static_df['patient_id']:
        dynamic_file_path = os.path.join(dynamic_folder, f'{patient_id}.xlsx')
        if os.path.exists(dynamic_file_path):
            dynamic_df = pd.read_excel(dynamic_file_path)

            # 计算动态血糖数据的统计量
            dynamic_df['Date'] = pd.to_datetime(dynamic_df['Date'])
            dynamic_df = dynamic_df.sort_values(by='Date')
            dynamic_df['CGM_diff'] = dynamic_df['CGM (mg / dl)'].diff()
            dynamic_df['Time_diff'] = dynamic_df['Date'].diff().dt.total_seconds() / 60.0  # 转换为分钟
            dynamic_df['CGM_rate'] = dynamic_df['CGM_diff'] / dynamic_df['Time_diff']

            # 按小时分组计算CGM变化速率的平均值
            dynamic_df['hour'] = dynamic_df['Date'].dt.floor('H')
            hourly_cgm_rate_mean = dynamic_df.groupby('hour')['CGM_rate'].mean().mean()

            cgm_stats = dynamic_df['CGM (mg / dl)'].agg(['mean', 'std', 'max', 'min']).to_dict()
            cbg_stats = dynamic_df['CBG (mg / dl)'].agg(['mean', 'std', 'max', 'min']).to_dict()
            blood_ketone_stats = dynamic_df['Blood Ketone (mmol / L)'].agg(['mean', 'std', 'max', 'min']).to_dict()
            carb_intake = dynamic_df['Carbohydrate/g'].sum()
            cgm_rate_stats = dynamic_df['CGM_rate'].agg(['mean', 'std', 'max', 'min']).to_dict()

            dynamic_stats = {
                'patient_id': patient_id,
                'CGM_mean': cgm_stats['mean'],
                'CGM_std': cgm_stats['std'],
                'CGM_max': cgm_stats['max'],
                'CGM_min': cgm_stats['min'],
                'CBG_mean': cbg_stats['mean'],
                'CBG_std': cbg_stats['std'],
                'CBG_max': cbg_stats['max'],
                'CBG_min': cbg_stats['min'],
                'Blood_Ketone_mean': blood_ketone_stats['mean'],
                'Blood_Ketone_std': blood_ketone_stats['std'],
                'Blood_Ketone_max': blood_ketone_stats['max'],
                'Blood_Ketone_min': blood_ketone_stats['min'],
                'Carb_intake': carb_intake,
                'CGM_rate_mean': cgm_rate_stats['mean'],
                'CGM_rate_std': cgm_rate_stats['std'],
                'CGM_rate_max': cgm_rate_stats['max'],
                'CGM_rate_min': cgm_rate_stats['min'],
                'Hourly_CGM_rate_mean': hourly_cgm_rate_mean
            }

            dynamic_data_list.append(dynamic_stats)

    dynamic_data = pd.DataFrame(dynamic_data_list)
    return dynamic_data

# 读取动态特征数据
dynamic_T1DM = process_dynamic_data(dynamic_folder_T1DM, static_T1DM)
dynamic_T2DM = process_dynamic_data(dynamic_folder_T2DM, static_T2DM)
dynamic_data = pd.concat([dynamic_T1DM, dynamic_T2DM], ignore_index=True)

# 合并静态和动态数据
combined_data = pd.merge(static_data, dynamic_data, on='patient_id')

# 存储合并后的数据
combined_file_path = os.path.join(output_folder, 'combined_data.xlsx')
combined_data.to_excel(combined_file_path, index=False)

# 特征工程和建模
features = combined_data.drop(['patient_id', 'CGM_mean'], axis=1)
target_cgm_mean = combined_data['CGM_mean']
target_cgm_std = combined_data['CGM_std']
target_cgm_rate_mean = combined_data['CGM_rate_mean']
target_hourly_cgm_rate_mean = combined_data['Hourly_CGM_rate_mean']
target_cbg_mean = combined_data['CBG_mean']

# 计算相关性矩阵
correlation_matrix_cgm_mean = features.corrwith(target_cgm_mean)
correlation_matrix_cgm_std = features.corrwith(target_cgm_std)
correlation_matrix_cgm_rate_mean = features.corrwith(target_cgm_rate_mean)
correlation_matrix_hourly_cgm_rate_mean = features.corrwith(target_hourly_cgm_rate_mean)
correlation_matrix_cbg_mean = features.corrwith(target_cbg_mean)

# 打印相关性
print("Correlation with CGM_mean:")
print(correlation_matrix_cgm_mean.sort_values(ascending=False))

print("\nCorrelation with CGM_std:")
print(correlation_matrix_cgm_std.sort_values(ascending=False))

print("\nCorrelation with CGM_rate_mean:")
print(correlation_matrix_cgm_rate_mean.sort_values(ascending=False))

print("\nCorrelation with Hourly_CGM_rate_mean:")
print(correlation_matrix_hourly_cgm_rate_mean.sort_values(ascending=False))

print("\nCorrelation with CBG_mean:")
print(correlation_matrix_cbg_mean.sort_values(ascending=False))

# 将相关性写入文件
correlation_df = pd.DataFrame({
    'Feature': features.columns,
    'Correlation_with_CGM_mean': correlation_matrix_cgm_mean.values,
    'Correlation_with_CGM_std': correlation_matrix_cgm_std.values,
    'Correlation_with_CGM_rate_mean': correlation_matrix_cgm_rate_mean.values,
    'Correlation_with_Hourly_CGM_rate_mean': correlation_matrix_hourly_cgm_rate_mean.values,
    'Correlation_with_CBG_mean': correlation_matrix_cbg_mean.values
})
correlation_df.to_excel(os.path.join(output_folder, 'correlation_analysis.xlsx'), index=False)

# 特征重要性分析
X = features
y = target_cgm_mean

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 特征重要性分析
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importances:")
for i in indices:
    print(f"{X.columns[i]}: {importances[i]}")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()
