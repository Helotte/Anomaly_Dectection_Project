import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import json
from utils import *
from usad import *
from sklearn import preprocessing


# 读取Excel文件
data = pd.read_csv("usad_feature_analysis\smoothed_data\smoothed_data\smoothed_Singapore_data.csv")
data['summary_time'] = pd.to_datetime(data['summary_time'])
#data = data.drop(['average_wait_duration', 'average_berth_duration', 'average_stay_duration'], axis=1)


# 按照summary_time字段划分训练集和测试集
normal = data[(data['summary_time'] >= '2022-01-01') & (data['summary_time'] < '2024-01-01')]
attack = data[data['summary_time'] >= '2024-01-01']

# 删除不需要的列
normal = normal.drop(["summary_time"], axis=1)
attack = attack.drop(["summary_time"], axis=1)

# 打印数据集的形状
print(f"Normal data shape: {normal.shape}")
print(f"Attack data shape: {attack.shape}")

# 将所有列中的逗号替换为小数点，并转换为浮点数
for i in normal.columns:
    normal[i] = normal[i].apply(lambda x: str(x).replace(",", ".")).astype(float)
for i in attack.columns:
    attack[i] = attack[i].apply(lambda x: str(x).replace(",", ".")).astype(float)

# 数据标准化
min_max_scaler = preprocessing.MinMaxScaler()

# 对正常数据集进行拟合和转换
x = normal.values
x_scaled = min_max_scaler.fit_transform(x)
normal = pd.DataFrame(x_scaled, columns=normal.columns)

# 对攻击数据集进行转换
x = attack.values
x_scaled = min_max_scaler.transform(x)
attack = pd.DataFrame(x_scaled, columns=attack.columns)


window_size=6
windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
print(windows_normal.shape)

windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]
print(windows_attack.shape)

import torch.utils.data as data_utils

BATCH_SIZE =  7919
N_EPOCHS = 100
hidden_size = 100

w_size=windows_normal.shape[1]*windows_normal.shape[2]
z_size=windows_normal.shape[1]*hidden_size

windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = UsadModel(w_size, z_size)
model = to_device(model,device)

history = training(N_EPOCHS,model,train_loader,val_loader)

torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
            }, "usad_feature_analysis\model.pth")

# 测试模型
# 测试模型
checkpoint = torch.load("usad_feature_analysis\model.pth")
model.encoder.load_state_dict(checkpoint['encoder'])
model.decoder1.load_state_dict(checkpoint['decoder1'])
model.decoder2.load_state_dict(checkpoint['decoder2'])

results = testing(model, test_loader)

time_stamps = data[data['summary_time'] >= '2024-01-01']['summary_time'].values

# 计算每个时间点的异常分数
def calculate_anomaly_scores(results, window_size, data_length):
    anomaly_scores = np.zeros(data_length)
    for i, score in enumerate(results):
        for j in range(window_size):
            if i + j < data_length:
                anomaly_scores[i + j] += score
    # 每个点被累加了window_size次，需要除以window_size得到平均值
    anomaly_scores /= window_size
    return anomaly_scores


anomaly_scores = calculate_anomaly_scores(results, window_size, len(attack))

# 确定轻度异常的IQR Rule阈值
Q1 = np.percentile(anomaly_scores, 25)
Q3 = np.percentile(anomaly_scores, 75)
IQR = Q3 - Q1

# 轻度异常上界
light_upper_threshold = Q3 + 1.5 * IQR

# 分别筛选出轻度异常的时间戳和分数
light_anomalies_df = pd.DataFrame({'Anomaly Time': time_stamps, 'Anomaly Score': anomaly_scores})
light_anomalies = light_anomalies_df[(light_anomalies_df['Anomaly Score'] > light_upper_threshold)]

if not light_anomalies.empty:
    # 对轻度异常点应用3-sigma来确定重度异常
    mean_light = np.mean(light_anomalies['Anomaly Score'])
    std_light = np.std(light_anomalies['Anomaly Score'])

    # 重度异常上界（基于轻度异常点）
    heavy_upper_threshold = mean_light + 1.5 * std_light

    # 筛选出重度异常点
    heavy_anomalies = light_anomalies[light_anomalies['Anomaly Score'] > heavy_upper_threshold]

    # 获取所有异常点（包括轻度和重度）
    all_anomalies = pd.concat([light_anomalies, heavy_anomalies]).drop_duplicates().sort_values(by='Anomaly Score', ascending=False)
    # 提取异常点的索引
    anomaly_indices = all_anomalies.index.tolist()

    # 创建一个新的测试加载器，只包含异常点，batch_size设置为1以确保逐个处理
    anomaly_loader = torch.utils.data.DataLoader(
        data_utils.TensorDataset(torch.from_numpy(windows_attack[anomaly_indices]).float().view((-1, w_size))),
        batch_size=1, shuffle=False, num_workers=0
    )

    # 计算每个异常点的特征贡献度
    device = get_default_device()
    feature_contributions = feature_contribution(model, anomaly_loader, device)

    # 构建最终输出的DataFrame
    final_output = []
    for i, idx in enumerate(anomaly_indices):
        row = {
            'Anomaly Time': time_stamps[idx],
            'Anomaly Score': anomaly_scores[idx]
        }
        for j, feature_name in enumerate(attack.columns):
            row[f'Feature Contribution ({feature_name})'] = feature_contributions[i][j]
        final_output.append(row)

    final_df = pd.DataFrame(final_output)

    # 保存特征重要性和异常分数到Excel文件
    final_df.to_excel("usad_feature_analysis/anomaly_feature_importance.xlsx", index=False)

    # 可视化结果
    plt.figure(figsize=(15, 5))
    plt.plot(time_stamps, anomaly_scores, label='Anomaly Scores', color='blue')

    # 绘制轻度异常和重度异常的阈值线
    plt.axhline(y=light_upper_threshold, color='orange', linestyle='--', label=f'Light Anomaly Threshold (1.5*IQR: {light_upper_threshold:.2f})')
    if not heavy_anomalies.empty:
        plt.axhline(y=heavy_upper_threshold, color='r', linestyle='--', label=f'Heavy Anomaly Threshold (1.5*sigma on Light Anomalies: {heavy_upper_threshold:.2f})')

    # 填充轻度异常区域
    plt.fill_between(time_stamps, anomaly_scores, where=((anomaly_scores > light_upper_threshold) & 
                                                         (anomaly_scores <= heavy_upper_threshold)), 
                     color='orange', alpha=0.3, label='Light Anomalies')

    # 填充重度异常区域
    if not heavy_anomalies.empty:
        plt.fill_between(time_stamps, anomaly_scores, where=(anomaly_scores > heavy_upper_threshold), 
                         color='red', alpha=0.3, label='Heavy Anomalies')

    plt.xlabel('Time Step')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Detection Results with Anomaly Scores using Nested IQR Rule')
    plt.legend()
    plt.show()

    # 保存所有异常分数到Excel文件
    anomaly_df = pd.DataFrame({'Anomaly Time': time_stamps, 'Anomaly Score': anomaly_scores})
    anomaly_df.to_excel("usad_feature_analysis/anomaly_scores_IQR_upper.xlsx", index=False)

    # 打印被认为是异常的时间戳和分数
    print("Light Anomalies:")
    print(light_anomalies)
    print("\nHeavy Anomalies:")
    print(heavy_anomalies)

    # 将轻度异常的时间戳和分数保存到单独的Excel文件中
    light_anomalies.to_excel("usad_feature_analysis/detected_light_anomalies_IQR_upper.xlsx", index=False)

    # 将重度异常的时间戳和分数保存到单独的Excel文件中
    if not heavy_anomalies.empty:
        heavy_anomalies.to_excel("usad_feature_analysis/detected_heavy_anomalies_IQR_upper.xlsx", index=False)
    else:
        print("No heavy anomalies detected.")
else:
    print("No light anomalies detected, hence no heavy anomalies.")

    # 假设time_stamps是一个包含所有异常时间的时间戳列表
# 提取并格式化时间戳
anomaly_timestamps = [
    ts.strftime("%Y-%m-%d %H:%M:%S%z") if isinstance(ts, pd.Timestamp) else 
    np.datetime_as_string(ts, unit='s') + '+08:00'
    for ts in all_anomalies['Anomaly Time']
]

# 创建符合要求的JSON结构
json_data = {"anomaly_timestamps": anomaly_timestamps}

# 将时间戳保存到JSON文件中，不使用缩进以确保所有内容在一行
with open("usad_feature_analysis\output\Singapore_anomaly_timestamps.json", 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, separators=(', ', ': '))

