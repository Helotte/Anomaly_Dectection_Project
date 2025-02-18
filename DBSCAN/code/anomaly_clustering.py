import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# DBSCAN参数选择：最大化silhouette score
# eps = [0.05, 0.5] step_size = 0.05
# min_points = [2, 30] step_size = 2
def DBSCAN_para(data):
    features = ['total_dwt', 'SpComplexity', 'SpDensity', 'TmCriticality']
    eps_range = np.arange(0.05, 0.51, 0.05)
    min_samples_range = np.arange(2, 31, 2)

    best_score = -1
    best_eps = 0
    best_min_samples = 0

    for eps in eps_range:
        for min_samples in min_samples_range:

            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(data[features])

            # Check if the clustering has at least 2 clusters
            if len(set(clusters)) > 1:
                score = silhouette_score(data[features], clusters)
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples

    # Print the best parameters and silhouette score
    print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}, Best silhouette score: {best_score}")
    return best_eps, best_min_samples


def dbscan(data, best_eps, best_min_samples):
    # 使用DBSCAN进行异常检测
    features = ['total_dwt', 'SpComplexity', 'SpDensity', 'TmCriticality']
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    anomaly_labels = dbscan.fit_predict(data[features])

    # 由于只关注噪点，先将所有点的聚类标记为0(正常类)
    data['Clusters'] = 0

    # 记录异常点, -1代表异常
    data['Clusters'][anomaly_labels == -1] = -1

    # 分离正常和异常数据
    clustered_data = data[data['Clusters'] != -1]
    anomaly_data = data[data['Clusters'] == -1]

    # 打印信息
    print(f"DBSCAN 发现异常点数量: {len(anomaly_data)}")
    print(anomaly_data)

    return data, clustered_data, anomaly_data


def evaluate_and_visualize_anomaly(data):
    """
    使用各种评估指标评估DBSCAN聚类结果, 并使用PCA可视化聚类。

    参数:
        data (pd.DataFrame): Clusters中分为0, -1的数据。
        (0 是正常类, -1是异常类)
    """
    features = ['total_dwt', 'SpComplexity', 'SpDensity', 'TmCriticality']

    if len(data[data['Clusters'] == -1]) > 1:
        # Silhouette Score
        sil_score = silhouette_score(data[features], data['Clusters'])
        print(f"\nSilhouette Score: {sil_score:.4f}")

        # Davies-Bouldin Index
        db_index = davies_bouldin_score(data[features], data['Clusters'])
        print(f"Davies-Bouldin Index: {db_index:.4f}")

        # Calinski-Harabasz Index
        ch_index = calinski_harabasz_score(data[features], data['Clusters'])
        print(f"Calinski-Harabasz Index: {ch_index:.4f}")

    else:
        print("\nThere is no anomaly data.")

    # PCA (2d)可视化
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data[features])

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['Clusters'], cmap='viridis')
    plt.title("PCA of DBSCAN")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # Create legend manually
    cluster_labels = data['Clusters'].unique()
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10)
        for label in cluster_labels]
    plt.legend(handles, [f"Cluster {label}" for label in cluster_labels], title="Clusters")
    plt.show()

    # PCA (3d)可视化
    pca = PCA(n_components=3)
    reduced_data2 = pca.fit_transform(data[features])

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维散点图
    scatter = ax.scatter(reduced_data2[:, 0], reduced_data2[:, 1], reduced_data2[:, 2],
                         c=data['Clusters'], cmap='viridis')

    # 设置标题和坐标轴标签
    ax.set_title("PCA of DBSCAN Clusters")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")

    # 创建图例
    cluster_labels = data['Clusters'].unique()
    handles = [ax.scatter([], [], [], color=scatter.cmap(scatter.norm(label)), label=f"Cluster {label}")
               for label in cluster_labels]
    ax.legend(handles, [f"Cluster {label}" for label in cluster_labels], title="Clusters")

    plt.show()

    # cluster summary
    cluster_summary = data[['total_dwt', 'SpComplexity', 'SpDensity', 'TmCriticality', 'Clusters']].groupby(
        'Clusters').mean()
    cluster_summary['Count'] = data['Clusters'].value_counts()
    print(cluster_summary)


def k_means(data, clustered_data, n_clusters=3):
    """
    使用KMeans进行聚类（假设有3个拥堵等级）并返回更新后的数据框。

    参数：
        data (pd.DataFrame): 包含所有数据的原始数据框。
        clustered_data (pd.DataFrame): 只包含正常数据（即不包含异常点）的数据框。
        n_clusters (int): 聚类数，表示拥堵等级数。

    返回：
        data (pd.DataFrame): 包含KMeans聚类标签的更新后的数据框。
    """
    features = ['total_dwt', 'SpComplexity', 'SpDensity', 'TmCriticality']

    # 使用KMeans进行聚类（假设有3个拥堵等级）
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(clustered_data[features])

    # 将KMeans聚类标签添加到正常数据中
    clustered_data['Cluster_Label'] = kmeans_labels

    # 确保原始数据框data中有'Clusters'列
    if 'Clusters' not in data.columns:
        data['Clusters'] = 0  # 初始化'Clusters'列，如果不存在

    # 将KMeans标签回传到原始数据集
    data.loc[clustered_data.index, 'Clusters'] = clustered_data['Cluster_Label']

    return data


def evaluate_and_visualize_clusters(data):
    """
    使用各种评估指标评估K-Means聚类结果, 并使用PCA可视化聚类。

    参数:
        data (pd.DataFrame)
    """

    features = ['total_dwt', 'SpComplexity', 'SpDensity', 'TmCriticality']
    clustered_data = data[data['Clusters'] != -1]

    if len(clustered_data['Clusters'].unique()) > 1:
        # Silhouette Score
        sil_score = silhouette_score(clustered_data[features], clustered_data['Clusters'])
        print(f"\nSilhouette Score: {sil_score:.4f}")

        # Davies-Bouldin Index
        db_index = davies_bouldin_score(clustered_data[features], clustered_data['Clusters'])
        print(f"Davies-Bouldin Index: {db_index:.4f}")

        # Calinski-Harabasz Index
        ch_index = calinski_harabasz_score(clustered_data[features], clustered_data['Clusters'])
        print(f"Calinski-Harabasz Index: {ch_index:.4f}")

    else:
        print("\nThere is only one cluster.")

    # PCA (2d)可视化
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(clustered_data[features])

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clustered_data['Clusters'], cmap='viridis')
    plt.title("PCA of clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # Create legend manually
    cluster_labels = clustered_data['Clusters'].unique()
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10)
        for label in cluster_labels]
    plt.legend(handles, [f"Cluster {label}" for label in cluster_labels], title="Clusters")
    plt.show()

    # PCA (3d)可视化
    pca = PCA(n_components=3)
    reduced_data2 = pca.fit_transform(clustered_data[features])

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维散点图
    scatter = ax.scatter(reduced_data2[:, 0], reduced_data2[:, 1], reduced_data2[:, 2],
                         c=clustered_data['Clusters'], cmap='viridis')

    # 设置标题和坐标轴标签
    ax.set_title("PCA of DBSCAN Clusters")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")

    # 创建图例
    cluster_labels = clustered_data['Clusters'].unique()
    handles = [ax.scatter([], [], [], color=scatter.cmap(scatter.norm(label)), label=f"Cluster {label}")
               for label in cluster_labels]
    ax.legend(handles, [f"Cluster {label}" for label in cluster_labels], title="Clusters")

    plt.show()

    # cluster summary
    cluster_summary = clustered_data[['total_dwt', 'SpComplexity', 'SpDensity', 'TmCriticality', 'Clusters']].groupby(
        'Clusters').mean()
    cluster_summary['Count'] = clustered_data['Clusters'].value_counts()
    print(cluster_summary)


def main():
    # 读取数据文件
    data = pd.read_excel(r"香港3个月聚类指标.xlsx")

    data = data.drop(columns=['Unnamed: 0'])

    # Step 1: DBSCAN 进行异常检测
    # 最大化silhouette得到DBSCAN参数
    best_eps, best_min_samples = DBSCAN_para(data)

    data, clustered_data, anomaly_data = dbscan(data, best_eps, best_min_samples)
    # DBSCAN 评估与可视化
    evaluate_and_visualize_anomaly(data)


    # Step 2: 对正常数据进行分级
    data = k_means(data, clustered_data)
    evaluate_and_visualize_clusters(data)

    return

if __name__ == "__main__":
    main()