import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from scipy.spatial import ConvexHull
import geohash2
from geopy.distance import geodesic


def Preprocess(df):
    # print(f"Original length: {len(df)}")
    
    # 去掉 dwt 和 vessel_type 空值的异常数据
    df = df.dropna(subset=['dwt','vessel_type'])
    
    # 可能有拖轮或者渔船的 parent_code=60000， 需要筛选掉 vessel_sub_type >= 70000 的船只数据
    df = df[df['vessel_sub_type'] < 70000]
    
    # 提取有用的列
    df = df[['mmsi', 'postime', 'lon', 'lat', 'status', 'dwt']]
    # print(f"length: {len(df)}")
    # print(df.isna().sum())
    # print("------------------------------------------------------------------")

    return df



# 可能需要针对实际文件名进行格式修改
def extract_time_from_file_name(file_name):
    """
    从文件名中提取起始时间和结束时间。
    假设文件名格式为 '2024-05-28 00_00_00-2024-05-31 00_00_00.csv'

    参数:
    file_name (str): 文件名

    返回:
    tuple: (start_time, end_time) 的 datetime 对象
    """
    # 正则表达式匹配时间段
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2})-(\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2})'
    match = re.search(pattern, file_name)
    
    if not match:
        raise ValueError(f"文件名格式错误，无法提取时间: {file_name}")

    # 提取起始时间和结束时间字符串
    start_time_str, end_time_str = match.groups()

    # 转换为 datetime 对象
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H_%M_%S").date()
    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H_%M_%S").date()
    
    return start_time, end_time



def split_data_into_segments(df):
    # 确保 'postime' 是 datetime 类型
    df['postime'] = pd.to_datetime(df['postime'])

    # 创建一个字典来存储每4小时段的数据
    segments = {}

    # 对数据按日期进行分组
    df_date = df.groupby(df['postime'].dt.date)
    for date, group in df_date:  # 按天分组
        # 获取当天的零点时间
        start_time = group['postime'].dt.normalize().min()  # 获取当天的零点时间
        
        # 按 4 小时划分当天的时间段
        for i in range(6):  # 一天分为6个4小时段
            segment_start = start_time + pd.Timedelta(hours=i*4)
            segment_end = start_time + pd.Timedelta(hours=(i+1)*4)

            # 选取当前时间段内的数据
            segment_df = group[(group['postime'] >= segment_start) & (group['postime'] < segment_end)]

            # 如果 segment_df 不是空的，加入字典
            if not segment_df.empty:
                # 使用 segment_start 格式化段名称
                segment_name = segment_start.strftime('%Y-%m-%d %H:%M:%S')
                segments[segment_name] = segment_df

    return segments




def process_ships_in_segment(segment_df, min_berth_duration=0.3):
    """
    处理一个时间段内的数据，计算每艘船的靠泊周期等信息。

    参数:
    segment_df (pd.DataFrame): 该时间段内的船舶数据，包含 postime, lon, lat, status, dwt 等字段。
    min_berth_duration (float): 最小靠泊时长，单位为小时。低于该时长的靠泊记录将被删除。（为了排除认为导致的状态标记异常）
    
    返回:
    dict: 该时间段内每艘船的处理结果，包括 MMSI 和靠泊周期等信息。
    """
    result = {}

    if segment_df.empty:
        return result  # 如果该时间段内没有数据，直接返回空结果

    # 按 'mmsi' 进行分组
    grouped = segment_df.groupby('mmsi')

    # 遍历每一组
    for mmsi, group in grouped:
        group['postime'] = pd.to_datetime(group['postime'])

        # 初始化船舶数据
        ship_data = {
            'mmsi': mmsi,
            # 对 postime 进行排序
            'data': sorted(group[['postime', 'lon', 'lat', 'status', 'dwt']].to_dict(orient='records'), key=lambda x: x['postime']),
            # 初始化靠泊数据，用来存储靠泊的开始时间/结束时间/靠泊时长
            'berth_periods': [] 
        }

        previous_status = None
        berthing_start_postime = None

        if ship_data['data'][0]['status'] == 5:
            berthing_start_postime = ship_data['data'][0]['postime']

        # 遍历船只的每条记录
        for index, row in enumerate(ship_data['data']):
            current_status = row['status']
            current_time = row['postime']

            # 状态变化检查
            if previous_status is not None:
                # 检查是否从 0 -> 5（开始靠泊）
                if previous_status == 0 and current_status == 5:
                    berthing_start_postime = current_time

                # 检查是否从 5 -> 0（结束靠泊）
                if previous_status == 5 and current_status == 0 and berthing_start_postime is not None:
                    berthing_end_postime = current_time
                    # 记录整个靠泊周期（从开始到结束）
                    duration = (berthing_end_postime - berthing_start_postime).total_seconds() / 3600
                    if duration >= min_berth_duration:
                        ship_data['berth_periods'].append({
                            'start_postime': berthing_start_postime,
                            'end_postime': berthing_end_postime,
                            'duration': duration
                        })
                    berthing_start_postime = None

            # 更新 previous_status 为当前状态
            previous_status = current_status

        # 如果最后一条记录状态是 5，则记录结束时间
        if previous_status == 5 and berthing_start_postime is not None:
            berthing_end_postime = ship_data['data'][-1]['postime']
            duration = (berthing_end_postime - berthing_start_postime).total_seconds() / 3600
            if duration >= min_berth_duration:
                ship_data['berth_periods'].append({
                    'start_postime': berthing_start_postime,
                    'end_postime': berthing_end_postime,
                    'duration': (berthing_end_postime - berthing_start_postime).total_seconds() / 3600
                })

        # 将每艘船的状态变化和靠泊周期信息存储到结果字典
        result[mmsi] = ship_data

    return result




def process_segments(segments_dict):
    """
    处理所有时间段的数据，并将结果存储在一个大字典中。
    
    参数:
    segments_dict (dict): 包含 6 个时间段数据的字典，每个键为 'segment_1' 至 'segment_6'，值为对应的 DataFrame。
    
    返回:
    dict: 包含所有时间段的处理结果，每个键为 'segment_1' 至 'segment_6'，值为该时间段内处理后的数据。
    """
    processed_data = {}

    # 对每个时间段进行处理
    for segment_name, segment_df in segments_dict.items():
        # print(f"Preprocessing {segment_name}...")  # 打印正在处理的时间段
        processed_data[segment_name] = process_ships_in_segment(segment_df)

    return processed_data
    # 输出整个处理之后的数据字典



def calculate_total_dwt_in_segment(segment_df):
    """
    计算每个时间段(segment)中唯一 MMSI 的 dwt 总和。

    参数:
    segment_df (dict): 包含六个时间段（如 segment_1, segment_2, ...）的字典。

    返回:
    total_dwt_in_segment: 时间段内的 dwt 总和。
    """
    
    total_dwt = 0
    # 遍历每只船的数据，计算该时间段内的总dwt
    for info in segment_df.values():
        total_dwt += info["data"][0]["dwt"] # 这里只取data列表中的第一个dwt值

    return total_dwt



def calculate_ConvArea_in_segment(segment_df):
    """
    计算给定时间段内所有船只的经纬度的 Convex Hull 面积。

    参数:
    segment_dict : 包含该时间段所有船只信息的字典，每个船只包括其经纬度数据。

    返回:
    convex_hull_area (float): 计算出的凸包面积。
    """
    # 用于存储所有船只的经纬度
    all_coordinates = []

    # 遍历所有船只，提取经纬度
    for info in segment_df.values():
        for entry in info["data"]:
            lat = entry["lat"]
            lon = entry["lon"]
            all_coordinates.append([lat, lon])

    # 将所有经纬度转换为 numpy 数组
    all_coordinates = np.array(all_coordinates)
    
    # 计算 Convex Hull
    if len(all_coordinates) >= 3:  # 至少需要3个点才能形成凸包
        convex_hull = ConvexHull(all_coordinates)
        convex_hull_area = convex_hull.volume  # ConvexHull.area 在 2D 中表示面积
    else:
        convex_hull_area = 1e-05
        # convex_hull_area = 0  # 如果点数少于3，无法形成凸包，面积为0
        print(f"Not enough points to compute convex hull")

    return convex_hull_area


def calculate_all_dwt_and_ConvArea(processed_data):
    """
    计算每个时间段(segment)中的总dwt和 Convex Hull Area, 并将其添加到 features 中。

    参数:
    processed_data (dict): 包含六个时间段(如 segment_1, segment_2, ...)的字典，每个时间段是一个 dict。

    返回:
    features: 用来储存指标的字典，包含每个时间段的 dwt 总和以及 ConvArea。
    """
    features = {}

    # 计算所有时间段的 total_dwt
    total_dwt_list = []
    convex_hull_area_list = []
    for segment_name, segment_df in processed_data.items():
        # 将计算的总 dwt 添加到 features 字典中
        total_dwt = calculate_total_dwt_in_segment(segment_df)
        total_dwt_list.append(total_dwt)
        convex_hull_area = calculate_ConvArea_in_segment(segment_df)
        convex_hull_area_list.append(convex_hull_area)
        features[segment_name] = {'total_dwt': total_dwt, 'ConvArea': convex_hull_area}

    max_dwt = max(total_dwt_list)
    max_ConvArea = max(convex_hull_area_list)
    for segment_name in features:
        features[segment_name]['total_dwt'] = features[segment_name]['total_dwt'] / max_dwt
        features[segment_name]['ConvArea'] = features[segment_name]['ConvArea'] / max_ConvArea

    return features




def MAP_area(geohash_value):
    """
    使用 Geohash 边界计算单个区域精确面积
    """
    # 获取 Geohash 边界：纬度范围和经度范围
    lat, lon, lat_err, lon_err = geohash2.decode_exactly(geohash_value)
    lat_min = lat - lat_err
    lat_max = lat + lat_err
    lon_min = lon - lon_err
    lon_max = lon + lon_err

    # 使用 geopy 计算经纬度边界的距离（km）
    lat_span = geodesic((lat_min, lon_min), (lat_max, lon_min)).km  # 纬度跨度
    lon_span = geodesic((lat_min, lon_min), (lat_min, lon_max)).km  # 经度跨度

    # 返回面积（单位：平方公里）
    return lat_span * lon_span



# 假设每个 Geohash 区域的面积是 25 平方公里（精度为7）
def calculate_geohash_area_in_segment(segment_df, geohash_precision=7):
    """
    计算给定时间段内所有船只的 Geohash 区域面积。
    
    参数:
    segment_df (dict): 包含所有船只信息的字典，每个船只包括其经纬度数据。
    geohash_precision (int): Geohash 的精度，决定了区域的大小（默认为7）。
    
    返回:
    geohash_area (float): 计算出的 Geohash 区域面积，单位为平方公里。

    注意：
    由于本身数据量太大，一艘船可能在四小时内多次更新ais数据，
    只取该时间段内该船只的最后一条数据进行geohash的计算。
    """

    # 用于存储所有船只的经纬度的 Geohash 值
    geohash_set = set() # 存储唯一 Geohash
    geohash_area = 0 # 总 Geohash 区域面积初始化为 0
    
    # 遍历每只船只，提取经纬度并计算 Geohash
    for info in segment_df.values():
        lat = info['data'][-1]['lat']
        lon = info['data'][-1]['lon']
        
        # 将经纬度转换为 Geohash，并根据精度进行截取
        geohash_value = geohash2.encode(lat, lon, precision=geohash_precision)
        
        # 如果 Geohash 是新的，计算其区域面积
        if geohash_value not in geohash_set:
            geohash_set.add(geohash_value)
            geohash_area += MAP_area(geohash_value)
                
    return geohash_area, geohash_set



def calculate_all_GeoArea(processed_data, features):
    """
    计算每个时间段（segment）中的总 GeoArea，并将其添加到 features 中。

    参数:
    processed_data (dict): 包含六个时间段（如 segment_1, segment_2, ...）的字典，每个时间段是一个 dict。
    features (dict): 包含每个时间段的每个时间段的 dwt 总和以及 ConvArea。

    返回:
    features: 用来储存指标的字典，包含每个时间段的 dwt 总和，ConvArea，GeoArea。
    """
    # 遍历每个时间段
    geohash_area_list = []
    for segment_name, segment_df in processed_data.items():
        # 将计算的总 dwt 添加到 features 字典中
        geohash_area, geohash_set = calculate_geohash_area_in_segment(segment_df)
        geohash_area_list.append(geohash_area)
        features[segment_name]['GeoArea'] = geohash_area
        features[segment_name]['GeohashSet'] = geohash_set

    max_GeoArea = max(geohash_area_list)
    for segment_name in features:
        features[segment_name]['GeoArea'] = features[segment_name]['GeoArea'] / max_GeoArea
        
    return features




def calculate_average_proximity_in_segment(geohash_set):
    
    """
    计算 Geohash 中心点之间的平均接近度 (Average Proximity, ∆)。
    
    参数:
    geohash_set (set): 唯一 Geohash 值的集合。
    
    返回:
    average_proximity (float): 平均接近度，单位为公里。
    """
    
    # 提取 Geohash 的中心点
    centers = []
    for geohash_value in geohash_set:
        lat, lon, _, _ = geohash2.decode_exactly(geohash_value)
        centers.append((lat, lon))

    # 如果中心点少于 2 个，无法计算距离
    if len(centers) < 2:
        return 1e-05
    
    # 计算所有中心点之间的地理距离
    distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            distance = geodesic(centers[i], centers[j]).km
            distances.append(distance)

    # 转换为 numpy 数组
    distances = np.array(distances)

    # 去除 95% 分位数以上的异常值
    threshold = np.percentile(distances, 95)
    filtered_distances = distances[distances <= threshold]
    
    if len(filtered_distances) == 0:
        return 1e-05  # 如果所有距离都是异常值，返回 0
        
    # 计算平均接近度
    average_proximity = np.mean(filtered_distances)

    return average_proximity


def calculate_all_average_proximity(processed_data, features):
    """
    计算每个时间段的平均接近度 (∆)，并将其添加到 features 中。

    参数:
    processed_data (dict): 包含多个时间段的字典，每个时间段是一个 dict。
    features (dict): 包含每个时间段的指标字典。

    返回:
    features: 更新后的 features，包含每个时间段的平均接近度。
    """
    average_proximity_list = []
    for segment_name, segment_df in processed_data.items():
        # 提取每个时间段的 Geohash 集合
        geohash_set = features[segment_name]['GeohashSet']
        average_proximity = calculate_average_proximity_in_segment(geohash_set)
        average_proximity_list.append(average_proximity)
        features[segment_name]['AvgProximity'] = average_proximity

    max_average_proximity = max(average_proximity_list)
    for segment_name in features:
        features[segment_name]['AvgProximity'] /= max_average_proximity

    return features



def calculate_spatial_complexity(features):
    """
    根据已有的 features 字典计算每个时间段的空间复杂度 (SpComplexity)。

    参数:
    features (dict): 包含每个时间段的Total dwt, ConvArea, GeoArea, GeohashSet, AvgProximity。

    返回:
    features (dict): 更新后的 features，包含每个时间段归一化后的 SpComplexity。

    注意：
    归一化中取的最大 SpComplexity 是在所有时间段数据之中的最大值。
    这里是所有时间段中的最大值。如果数据是一年的，将是一年中的最大值。
    """
    # 初始化存储所有时间段的复杂度，用于归一化
    spatial_complexities = []

    # 计算每个时间段的空间复杂度
    for segment_name, segment_features in features.items():
        # 提取 ConvArea 和 AvgProximity
        conv_area = segment_features.get("ConvArea", 0)
        avg_proximity = segment_features.get("AvgProximity", 0)
        
        # 如果 ConvArea 或 AvgProximity 为 0，跳过该时间段
        if conv_area == 0 or avg_proximity == 0:
            segment_features["SpComplexity"] = 0.0
            continue

        # 计算空间复杂度
        sp_complexity = 1 / (avg_proximity * conv_area)
        segment_features["SpComplexity"] = sp_complexity

        # 将复杂度添加到列表中，用于归一化
        spatial_complexities.append(sp_complexity)

    # 归一化空间复杂度
    max_complexity = max(spatial_complexities) if spatial_complexities else 1  # 防止除以 0
    for segment_name, segment_features in features.items():
        if "SpComplexity" in segment_features:
            segment_features["SpComplexity"] /= max_complexity

    return features


def calculate_spatial_density(features):
    """
    根据已有的 features 字典计算每个时间段的空间密度 (SpDensity)。

    参数:
    features (dict): 包含每个时间段的 total_dwt, ConvArea, GeoArea, GeohashSet, AvgProximity, SpComplexity。

    返回:
    features (dict): 更新后的 features，包含每个时间段的 SpDensity。
    """
    # 初始化存储所有时间段的未归一化空间密度
    spatial_densities = []

    # 计算每个时间段的未归一化空间密度
    for segment_name, segment_features in features.items():
        # 提取 GeoArea 和 ConvArea
        geo_area = segment_features.get("GeoArea", 0)
        conv_area = segment_features.get("ConvArea", 0)

        # 如果 GeoArea 或 ConvArea 为 0，跳过该时间段
        if geo_area == 0 or conv_area == 0:
            segment_features["SpDensity"] = 0.0
            continue

        # 计算未归一化的空间密度
        sp_density_raw = 1 / (geo_area * conv_area)
        segment_features["SpDensity"] = sp_density_raw

        # 将未归一化的值添加到列表中，用于归一化
        spatial_densities.append(sp_density_raw)

    # 归一化空间密度
    max_density = max(spatial_densities) if spatial_densities else 1  # 防止除以 0
    for segment_name, segment_features in features.items():
        if "SpDensity" in segment_features:
            segment_features["SpDensity"] /= max_density

    return features


def calculate_time_criticality(features, processed_data):
    """
    根据 processed_data 中的泊位信息 (berth_periods) 计算每个时间段的时间关键性 (TmCriticality)。

    参数:
    features (dict): 包含每个时间段的基础数据。
    processed_data (dict): 每个时间段的船舶泊位信息。

    返回:
    features (dict): 更新后的 features，包含每个时间段的 TmCriticality。
    """
    # 初始化存储每个时间段的未归一化时间关键性
    time_criticalities = []

    for segment_name, segment_df in processed_data.items():
        # 初始化当前时间段的泊位时长和唯一 MMSI 集合
        service_times = []
        unique_mmsi = set()

        # 遍历每个船只的泊位信息
        for mmsi, info in segment_df.items():
            # 提取泊位信息
            berth_periods = info.get("berth_periods", [])
            if not berth_periods:
                continue

            # 遍历泊位时间段
            for period in berth_periods:
                duration = period.get("duration", 0)  # 泊位时长 (单位：小时)

                # 跳过无效记录
                if duration <= 0:
                    continue

                # 记录泊位时长和唯一 MMSI
                service_times.append(duration)
                unique_mmsi.add(mmsi)

        # 计算平均服务时间
        n_i = len(unique_mmsi)  # N(i)
        if n_i == 0:
            print(f"Skipping segment {segment_name}: No valid vessels")
            features[segment_name]["TmCriticality"] = 0.0
            continue

        avg_service_time = sum(service_times) / n_i
        time_criticalities.append(avg_service_time)
        
        # 记录未归一化的时间关键性
        features[segment_name]["TmCriticality"] = avg_service_time
        

    # 归一化时间关键性
    max_criticality = max(time_criticalities) if time_criticalities else 1  # 防止除以 0
    for segment_name, segment_features in features.items():
        if "TmCriticality" in segment_features:
            segment_features["TmCriticality"] /= max_criticality

    return features


def export_features_to_dataframe(features):
    """
    将 features 字典中的每个 segment 的 total_dwt, SpComplexity, SpDensity, TmCriticality
    导出为一个 DataFrame。

    参数:
    features (dict): 包含每个时间段的特征数据。

    返回:
    pd.DataFrame: 一个包含 total_dwt, SpComplexity, SpDensity, TmCriticality 的 DataFrame。
    """
    # 提取数据
    data = []
    for segment_name, segment_features in features.items():
        row = {
            "segment": segment_name,
            "total_dwt": segment_features.get("total_dwt", 0),
            "SpComplexity": segment_features.get("SpComplexity", 0),
            "SpDensity": segment_features.get("SpDensity", 0),
            "TmCriticality": segment_features.get("TmCriticality", 0),
        }
        data.append(row)
    
    # 转换为 DataFrame
    df = pd.DataFrame(data)
    return df



def main():

    # 设置文件夹路径
    folder_path = r"香港3个月数据"

    # 获取文件夹中所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 用一个字典存储读取的所有数据框
    processed_data = {}

    # 遍历所有CSV文件，读取数据并存入列表
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        # file_start_time, file_end_time = extract_time_from_file_name(csv_file)
        df = pd.read_csv(file_path)
        df = Preprocess(df)

        segments = split_data_into_segments(df)
        processed_segments = process_segments(segments)
        processed_data.update(processed_segments)

    print("Preprocessing is finished.")

    # 计算所需特征
    features = calculate_all_dwt_and_ConvArea(processed_data)
    features = calculate_all_GeoArea(processed_data, features)
    features = calculate_all_average_proximity(processed_data, features)

    # 计算聚类指标
    features = calculate_spatial_complexity(features)
    features = calculate_spatial_density(features)
    features = calculate_time_criticality(features, processed_data)

    # 导出聚类指标为dataframe
    features_df = export_features_to_dataframe(features)

    features_df.to_excel(r"香港3个月聚类指标.xlsx")
    return


if __name__ == "__main__":
    main()