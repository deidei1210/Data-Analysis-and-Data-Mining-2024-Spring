import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input, Dense, Dropout, concatenate, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

# 定义数据文件夹路径
data_dir = './datasets_new/'

# 定义需要读取的文件
t1dm_summary_file = os.path.join(data_dir, 'summary', 'Shanghai_T1DM_Summary0.xlsx')
t2dm_summary_file = os.path.join(data_dir, 'summary', 'Shanghai_T2DM_Summary0.xlsx')

# 读取Summary数据，并新增一个属性type表示糖尿病类型
t1dm_summary = pd.read_excel(t1dm_summary_file)
t1dm_summary['type'] = 1  # T1DM类型标记为1

t2dm_summary = pd.read_excel(t2dm_summary_file)
t2dm_summary['type'] = 2  # T2DM类型标记为2

# 读取详细数据
t1dm_files = [os.path.join(data_dir, 'Shanghai_T1DM', f) for f in os.listdir(os.path.join(data_dir, 'Shanghai_T1DM')) if
              f.endswith('.csv')]
t2dm_files = [os.path.join(data_dir, 'Shanghai_T2DM', f) for f in os.listdir(os.path.join(data_dir, 'Shanghai_T2DM')) if
              f.endswith('.csv')]

# 读取单个病人的详细数据
def read_patient_data(file):
    return pd.read_csv(file)

# 选择我们关心的特征
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
median_features = [
    'Age (years)', 'Weight (kg)', 'BMI (kg/m2)', 'Duration of Diabetes (years)',
    'HbA1c (mmol/mol)', 'Fasting Plasma Glucose (mg/dl)', '2-hour Postprandial C-peptide (nmol/L)',
    'Fasting C-peptide (nmol/L)', 'Glycated Albumin (%)',
]
# Calculate medians for T1DM and T2DM
t1dm_medians = t1dm_summary[median_features].dropna().median()
t2dm_medians = t2dm_summary[median_features].dropna().median()

# 标准化数据
scaler_time_series = StandardScaler()
scaler_static = StandardScaler()
scaler_target = StandardScaler()

# 定义时间步长
time_steps = 15

import numpy as np
import pandas as pd

def create_sequences(patient_id, patient_data, static_data, time_series_features, static_features, target, time_steps):
    sequences = []
    targets = []
    static = static_data[static_features].values[0]

    # 填充静态数据中的 NaN 值
    if pd.isna(static).sum() > 0:
        nan_indices = np.where(pd.isna(static))[0]
        for idx in nan_indices:
            feature_name = static_features[idx]
            static[idx] = t1dm_medians[feature_name]
        #print(f"NaN values in static features {nan_indices} for patient {patient_id} filled with mean")

    for i in range(len(patient_data) - time_steps):
        seq = patient_data.iloc[i:i + time_steps][time_series_features].values
        label = patient_data.iloc[i + time_steps][target]

        # 填充时间序列数据中的 NaN 值
        if np.isnan(seq).sum() > 0:
            nan_indices = np.where(np.isnan(seq))
            for row, col in zip(*nan_indices):
                feature_name = time_series_features[col]
                seq[row, col] = patient_data[feature_name].dropna().median()
            #print(f"NaN values in time series features at index {i} for patient {patient_id} filled with mean")

        # 填充目标数据中的 NaN 值
        if np.isnan(label):
            label = patient_data[target].mean()
            #print(f"NaN value in target at index {i + time_steps} for patient {patient_id} filled with mean")
        sequences.append((seq, static))
        targets.append(label)
    return sequences, targets

def build_model(time_steps, time_series_features, static_features):
    input_time_series = Input(shape=(time_steps, len(time_series_features)))
    input_static = Input(shape=(len(static_features),))

    # LSTM层
    lstm_out = LSTM(units=256, return_sequences=True, kernel_initializer=HeNormal())(input_time_series)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.4)(lstm_out)
    lstm_out = LSTM(units=256, return_sequences=True, kernel_initializer=HeNormal())(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.4)(lstm_out)
    lstm_out = LSTM(units=128, kernel_initializer=HeNormal())(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.4)(lstm_out)

    concat = concatenate([lstm_out, input_static])
    dense_out = Dense(512, activation='relu', kernel_initializer=GlorotUniform())(concat)
    dense_out = BatchNormalization()(dense_out)
    dense_out = Dropout(0.4)(dense_out)
    dense_out = Dense(256, activation='relu', kernel_initializer=GlorotUniform())(dense_out)
    dense_out = BatchNormalization()(dense_out)
    dense_out = Dropout(0.4)(dense_out)
    dense_out = Dense(128, activation='relu', kernel_initializer=GlorotUniform())(dense_out)
    dense_out = BatchNormalization()(dense_out)
    dense_out = Dropout(0.4)(dense_out)
    output = Dense(1, kernel_initializer=GlorotUniform())(dense_out)

    model = Model(inputs=[input_time_series, input_static], outputs=output)

    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

    return model

# 构建模型
model = build_model(time_steps, time_series_features, static_features)

# 定义学习率调度器
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)


# 收集所有数据
all_sequences = []
all_targets = []

# 处理一型糖尿病数据
for file in t1dm_files:
    patient_id = os.path.basename(file).replace('.csv', '')
    patient_data = read_patient_data(file)
    static_data = t1dm_summary[t1dm_summary['patient_id'] == patient_id]

    if not static_data.empty:
        sequences, targets = create_sequences(patient_id,patient_data, static_data, time_series_features, static_features, 'CGM (mg / dl)', time_steps)
        all_sequences.extend(sequences)
        all_targets.extend(targets)

# 处理二型糖尿病数据并训练模型
for file in t2dm_files:
    patient_id = os.path.basename(file).replace('.csv', '')
    patient_data = read_patient_data(file)
    static_data = t2dm_summary[t2dm_summary['patient_id'] == patient_id]
    if not static_data.empty:
        sequences, targets = create_sequences(patient_id,patient_data, static_data, time_series_features, static_features, 'CGM (mg / dl)', time_steps)
        all_sequences.extend(sequences)
        all_targets.extend(targets)

# 转换为numpy数组并标准化数据
if all_sequences and all_targets:
    X_time_series = np.array([seq[0] for seq in all_sequences])
    X_static = np.array([seq[1] for seq in all_sequences])
    y = np.array(all_targets)
    #
    # # 标准化时间序列数据
    # X_time_series_reshaped = X_time_series.reshape(-1, X_time_series.shape[-1])
    # X_time_series_scaled = scaler_time_series.fit_transform(X_time_series_reshaped)
    # X_time_series = X_time_series_scaled.reshape(X_time_series.shape)

    # 标准化静态数据
    X_static = scaler_static.fit_transform(X_static)

    df_X_static = pd.DataFrame(X_static)

    # Save the DataFrame to a CSV file
    df_X_static.to_csv('X_static.csv', index=False)



    # 创建一个新的索引数组用于确保静态数据和时间序列数据不会被打乱
    indices = np.arange(X_time_series.shape[0])
    X_train_indices, X_test_indices, y_train_indices, y_test_indices = train_test_split(
        indices, indices, test_size=0.2, shuffle=True, random_state=42)

    # 使用索引进行训练和测试数据的划分
    X_train_time_series = X_time_series[X_train_indices]
    X_test_time_series = X_time_series[X_test_indices]
    X_train_static = X_static[X_train_indices]
    X_test_static = X_static[X_test_indices]
    y_train = y[X_train_indices]
    y_test = y[X_test_indices]

    # 打印训练和测试数据形状以调试
    print(f'X_train_time_series shape: {X_train_time_series.shape}')
    print(f'X_train_static shape: {X_train_static.shape}')
    print(f'X_test_time_series shape: {X_test_time_series.shape}')
    print(f'X_test_static shape: {X_test_static.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')


    # 自定义回调函数
    class PrintDetailedMAE(Callback):
        def __init__(self, val_data):
            super(PrintDetailedMAE, self).__init__()
            self.val_data = val_data

        def on_epoch_end(self, epoch, logs=None):
            val_time_series, val_static, val_labels = self.val_data
            predictions = self.model.predict([val_time_series, val_static])
            absolute_error = np.abs(val_labels[0] - predictions[0])
            print(
                f'Example 0: True Value = {val_labels[0]}, Predicted Value = {predictions[0]}, Absolute Error = {absolute_error}')

            # 使用tf.keras.metrics.MeanAbsoluteError计算MAE
            mae_metric = tf.keras.metrics.MeanAbsoluteError()
            mae_metric.update_state(val_labels, predictions)
            mae = mae_metric.result().numpy()
            print(f'Epoch {epoch + 1}: Validation MAE = {mae:.4f}')


    # 准备验证数据
    val_split = 0.2
    split_index = int((1 - val_split) * len(X_train_time_series))

    # 验证数据
    X_val_time_series = X_train_time_series[split_index:]
    X_val_static = X_train_static[split_index:]
    y_val = y_train[split_index:]

    # 训练数据
    X_train_time_series = X_train_time_series[:split_index]
    X_train_static = X_train_static[:split_index]
    y_train = y_train[:split_index]

    # 创建并传递自定义回调函数
    print_detailed_mae_callback = PrintDetailedMAE((X_val_time_series, X_val_static, y_val))

    # 训练模型
    history = model.fit(
        [X_train_time_series, X_train_static], y_train,
        epochs=50, batch_size=64,
        validation_data=([X_val_time_series, X_val_static], y_val),
        callbacks=[reduce_lr, print_detailed_mae_callback]
    )

    # 评估模型
    y_pred = model.predict([X_test_time_series, X_test_static])
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # 打印评估结果
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')

    # 保存模型
    model.save('blood_glucose_prediction_model.h5')
    print("Model saved as 'blood_glucose_prediction_model.h5'")

    # 可视化训练过程中的损失和MAE
    plt.figure(figsize=(12, 6))

    # 绘制训练和验证损失
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 绘制训练和验证 MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()

    # 调整布局并保存图形
    plt.tight_layout()
    plt.savefig('training_validation_metrics.png')

    # 打印保存成功的消息
    print("Training and validation metrics plot saved as 'training_validation_metrics.png'")
