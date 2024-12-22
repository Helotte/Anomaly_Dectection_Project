# To fit different datasets, you may need to adjust the city's name in line 96.

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from deepant.trainer import AnomalyDetector
from deepant.model import DeepAntPredictor
from utils.data_utils import DataModule

def load_full_dataset(file_path, window_size, device):
    data = pd.read_csv(file_path, parse_dates=["summary_time"], index_col="summary_time")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values)

    data_x = [data_scaled[i - window_size:i] for i in range(window_size, len(data_scaled))]
    data_x = np.array(data_x)

    dataset = DataModule(data_x, data_x[:, -1, :], device)
    loader = DataLoader(dataset, batch_size=data_x.shape[0], shuffle=False)
    return loader, data.index[window_size:]  # Return timestamps starting from the first complete window

def calculate_anomaly_scores(predictions, ground_truth):
    return [np.linalg.norm(pred - gt) for pred, gt in zip(predictions, ground_truth)]

def visualize_anomalies(timestamps, anomaly_scores, threshold, anomalies, output_path):
    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, anomaly_scores, label="Anomaly Scores", color='blue', linestyle='-', linewidth=1)
    plt.scatter([timestamps[i] for i in anomalies], [anomaly_scores[i] for i in anomalies], color='red', label='Anomaly Points', zorder=5)
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label="Threshold")
    plt.title("Anomaly Scores and Threshold on the Full Dataset", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(loc="upper left", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    
# separate the anomaly levels
def classify_anomaly_levels(anomaly_scores, mean_error, std_error):
    levels = []
    for score in anomaly_scores:
        if score <= mean_error + std_error:
            levels.append("Normal")
        elif score <= mean_error + 2 * std_error:
            levels.append("Mild Anomaly")
        elif score <= mean_error + 3 * std_error:  # by the "3-sigma rule", 99.7% of the data should be within 3 standard deviations, otherwise it is considered as an anomaly
            levels.append("Moderate Anomaly")
        else:
            levels.append("Severe Anomaly")
    return levels

def visualize_anomaly_levels(timestamps, anomaly_scores, levels, output_path):
    level_colors = {
        "Normal": "blue",
        "Mild Anomaly": "yellow",
        "Moderate Anomaly": "orange",
        "Severe Anomaly": "red"
    }

    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, anomaly_scores, label="Anomaly Scores", color='black', linestyle='-', linewidth=1)

    for i in range(len(timestamps) - 1):
        plt.plot(
            timestamps[i:i + 2],
            anomaly_scores[i:i + 2],
            color=level_colors[levels[i]],
            linewidth=2
        )
    
    legend_patches = [
        plt.Line2D([0], [0], color=color, lw=4, label=level)
        for level, color in level_colors.items()
    ]
    plt.legend(handles=legend_patches, loc="upper left", fontsize=12)
    plt.title("Anomaly Levels Over Time", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Anomaly Score", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Anomaly levels plot saved to {output_path}")



def main():
    
    city = "Shenzhen"  # you need to ADJUST this part !!!
    
    checkpoint_path = f"experiment/{city}_port/best_model.ckpt"
    data_path = f"data/smoothed_data/smoothed_{city}_data.csv"
    output_path = f"experiment/{city}_port/full_dataset_anomaly_plot.png"
    anomaly_timestamps_path = f"experiment/{city}_port/full_dataset_anomaly_timestamps.json"
    anomaly_levels_path = f"experiment/{city}_port/full_dataset_anomaly_levels.json"
    anomaly_levels_plot_path = f"experiment/{city}_port/full_dataset_anomaly_levels_plot.png"
    output_excel_path = f"experiment/{city}_port/anomaly_results_summation.xlsx"
    window_size = 45  # make sure this is the SAME as the window_size used in training
    threshold_std_rate = 3  

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, timestamps = load_full_dataset(data_path, window_size, device)
    
    model = AnomalyDetector.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=DeepAntPredictor(feature_dim=loader.dataset.data_x.shape[-1], window_size=window_size).to(device),
        lr=1e-3
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = []
        for batch in loader:
            x, _ = batch
            predictions.append(model(x).cpu().numpy())
        predictions = np.concatenate(predictions)

    # Calculate anomaly scores
    ground_truth = loader.dataset.data_y
    anomaly_scores = calculate_anomaly_scores(predictions, ground_truth)
    mean_error = np.mean(anomaly_scores)
    std_error = np.std(anomaly_scores)
    threshold = mean_error + threshold_std_rate * std_error
    print(f"Threshold is set to: {threshold:.4f}")
    anomalies = [i for i, score in enumerate(anomaly_scores) if score > threshold]
    anomaly_timestamps = [str(timestamps[i]) for i in anomalies]  # Convert timestamps to string for JSON compatibility
    # print("Anomalies detected at timestamps:", [timestamps[i] for i in anomalies])
    
    anomaly_levels = classify_anomaly_levels(anomaly_scores, mean_error, std_error)

    with open(anomaly_timestamps_path, "w") as json_file:
        json.dump({"anomaly_timestamps": anomaly_timestamps}, json_file)

    visualize_anomalies(timestamps, anomaly_scores, threshold, anomalies, output_path)

    visualize_anomaly_levels(timestamps, anomaly_scores, anomaly_levels, anomaly_levels_plot_path)
    
    timestamps = timestamps.strftime('%Y-%m-%d %H:%M:%S')
    anomaly_results = pd.DataFrame({
        "timestamp": timestamps,
        "anomaly_score": anomaly_scores,
        "anomaly_level": anomaly_levels
    })
    anomaly_results.to_excel(output_excel_path, index=False, header=True)
    print(f"Anomaly results saved to {output_excel_path}")
    


if __name__ == "__main__":
    main()
