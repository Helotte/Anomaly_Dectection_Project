
# USAD 异常检测项目运行说明

## 一、运行环境要求

本项目代码基于 Python 3.7.9 版本进行开发和测试，建议使用相同版本的 Python 环境来运行代码，以避免可能出现的版本兼容性问题。

## 二、依赖库安装

项目所需的依赖库在 `requirements.txt` 文件中进行了罗列，你可以通过以下命令一键安装所有依赖：

```bash
pip install -r requirements.txt
```
其实只要torch==1.9.0,其余默认安装即可
执行上述命令后，将会自动安装项目运行所必需的各类 Python 库，包括但不限于 `numpy`、`pandas`、`matplotlib`、`seaborn`、`torch`、`sklearn` 等。

## 三、输入数据准备

代码默认从特定的文件目录下读取输入数据文件 `smoothed_Singapore_data.csv`，其期望的文件路径为 `usad_feature_analysis\smoothed_data\smoothed_data\`。如果你的数据文件存储在其他位置，请按照以下步骤修改输入文件目录：

打开代码文件，找到以下代码行：

```python
data = pd.read_csv("usad_feature_analysis\\smoothed_data\\smoothed_data\\smoothed_Singapore_data.csv")
```

将双引号内的文件路径修改为你实际存放数据文件的完整路径，例如：

```python
data = pd.read_csv("your_actual_path\\smoothed_Singapore_data.csv")
```

确保路径准确无误，否则代码在读取数据阶段将会报错。

## 四、运行代码




## 五、输出结果说明

代码运行成功后，将会在指定的 output 目录及其相关目录下生成一系列输出文件，用于保存不同阶段的分析结果，具体如下：

### （一）模型相关文件

在 `usad_feature_analysis` 目录下会保存训练好的模型参数文件 `model.pth`，该文件包含了模型的编码器（encoder）、解码器 1（decoder1）和解码器 2（decoder2）的相关参数，用于后续加载模型进行测试或其他应用场景。

### （二）异常分析结果文件（Excel 格式）

- `usad_feature_analysis/anomaly_feature_importance.xlsx`：此文件整合了异常点的相关信息，包括异常时间、异常分数以及各个特征对于每个异常点的贡献度，方便对异常情况进行详细分析，了解各特征在异常事件中的重要程度。
- `usad_feature_analysis/anomaly_scores_IQR_upper.xlsx`：保存了所有时间点对应的异常分数数据，可用于整体查看异常分数随时间的变化情况以及后续进一步的数据挖掘或统计分析等操作。
- `usad_feature_analysis/detected_light_anomalies_IQR_upper.xlsx`：专门存储被判定为轻度异常的时间戳和对应的异常分数，有助于聚焦分析轻度异常情况以及其出现的时间规律等。
- `usad_feature_analysis/detected_heavy_anomalies_IQR_upper.xlsx`（如果存在重度异常的情况下）：记录了重度异常的时间戳和异常分数信息，方便针对较为严重的异常事件进行单独的深入研究和处理。

### （三）异常时间戳文件（JSON 格式）

在 `usad_feature_analysis/output/` 目录下的 `Singapore_anomaly_timestamps.json` 文件，存储了所有异常点的时间戳信息，格式化为符合 JSON 规范的结构，便于与其他系统进行数据交互或者后续的数据处理流程中进行读取和解析使用。

同时，在代码运行过程中，还会弹出可视化图表展示异常检测结果，包括异常分数随时间的变化曲线，以及用不同颜色区域标注出的轻度异常和重度异常范围，方便直观地了解异常情况的分布和程度。

