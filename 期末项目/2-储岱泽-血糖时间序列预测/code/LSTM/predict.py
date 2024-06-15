import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 载入保存的模型
loaded_model = tf.keras.models.load_model('blood_glucose_prediction_model.h5')
print("Model loaded from 'blood_glucose_prediction_model.h5'")

time_series_features = [
    'CGM (mg / dl)',
    'Insulin dose - s.c.',
    'CSII - bolus insulin (Novolin R, IU)',
    'Carbohydrate/g'
]
static_features = [
    'type','patient_id','Age (years)', 'Weight (kg)', 'BMI (kg/m2)', 'Duration of Diabetes (years)',
    'HbA1c (mmol/mol)', 'Fasting Plasma Glucose (mg/dl)', '2-hour Postprandial C-peptide (nmol/L)',
    'Fasting C-peptide (nmol/L)', 'Glycated Albumin (%)', 'Acute Diabetic Complications',
    'Diabetic Macrovascular Complications', 'Diabetic Microvascular Complications',
    'Comorbidities', 'Hypoglycemic Agents', 'Other Agents'
]

# 定义时间步长
time_steps = 15


data_dir = './datasets_new/'
patient_id = '2054_0_20210524'
patient_file = os.path.join(data_dir, 'Shanghai_T2DM/', f'{patient_id}.csv')




# 读取Summary数据，并新增一个属性type表示糖尿病类型
t2dm_summary_file = os.path.join(data_dir, 'summary', 'Shanghai_T2DM_Summary0.xlsx')
t2dm_summary = pd.read_excel(t2dm_summary_file)
t2dm_summary['type'] = 2  # T1DM类型标记为1

# 从summary数据中获取病人的静态特征
static_data = t2dm_summary[t2dm_summary['patient_id'] == patient_id]

# 读取单个病人的详细数据
patient_data = pd.read_csv(patient_file)

# 提取前15个时间步的数据
input_time_series = patient_data[time_series_features].iloc[:15].values

# 提取前15个时间步的数据用于预测，第16-19个时间步的数据用于验证和可视化
target_cgm_values = patient_data['CGM (mg / dl)'].iloc[15:19].values

# 提取静态特征数据
static_data = static_data[static_features].values[0]


# 标准化静态特征
scaler_static = StandardScaler()
static_data_scaled = scaler_static.fit_transform(static_data.reshape(1, -1))

# 创建输入数据
input_time_series = input_time_series.reshape(1, 15, len(time_series_features))

# 进行预测
loaded_model = tf.keras.models.load_model('blood_glucose_prediction_model.h5')
print("Model loaded from 'blood_glucose_prediction_model.h5'")

predictions = []
for i in range(4):
    next_prediction = loaded_model.predict([input_time_series, static_data_scaled])[0][0]
    predictions.append(next_prediction)

    # 更新输入序列
    new_step = np.array([[next_prediction, 0, 0, 0]])  # 其他特征值设置为0
    input_time_series = np.append(input_time_series[:, 1:, :], new_step.reshape(1, 1, -1), axis=1)

# 可视化预测结果和实际值
time_points = [15,30,45,60]
plt.plot(time_points, predictions, label='Predicted', marker='o')
plt.plot(time_points, target_cgm_values, label='Actual', marker='x')
plt.xlabel('Time Point(min)')
plt.ylabel('Blood Glucose Level (mg/dl)')
plt.title('Predicted vs Actual Blood Glucose Levels for Patient')
plt.legend()
plt.grid(True)
plt.show()

# 输出预测结果
for i, (pred, actual) in enumerate(zip(predictions, target_cgm_values)):
    print(f'Time Point {i + 16}: Predicted = {pred}, Actual = {actual}, Error = {abs(pred - actual)}')
