import os
import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm # 使用tqdm来显示进度
import json
import argparse
import re
from datetime import datetime # 导入 datetime

def _parse_timestamp_to_datetime(timestamp_str):
    """将文件名中的时间字符串（假定格式为 MMDDHHMM）转换为datetime对象。"""
    # 由于年份缺失，我们为所有数据假设一个相同的年份（如2000年），
    # 这对于计算相对时间差进行插值是没有问题的。
    # 后续数据集出来了再改
    return datetime.strptime(f"2000{timestamp_str}", "%Y%m%d%H%M")

def parse_format_file(file_path):
    """
    解析数据集说明文档 (.txt)，提取起始经纬度。
    具体不知道他把经纬度放哪里了
    这里假设有个文件
    实际得到数据机放出来了再改
    """
    metadata = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lon_match = re.search(r'(start_lon|起始经度)\s*[:：]\s*(\d+\.?\d*)', content)
        lat_match = re.search(r'(start_lat|起始纬度)\s*[:：]\s*(\d+\.?\d*)', content)
        
        if lon_match and lat_match:
            metadata['lon_min'] = float(lon_match.group(2))
            metadata['lat_min'] = float(lat_match.group(2))
            return metadata
        else:
            raise ValueError("无法解析出经纬度。")
            
    except FileNotFoundError:
        raise FileNotFoundError(f"元数据文件未找到: {file_path}")
    except Exception as e:
        raise RuntimeError(f"解析文件 {file_path} 时出错: {e}")

# 加载数据函数

def collect_case_data(case_dir, radar_feats, nwp_feats):
    
    case_data = {"radar": {}, "nwp": {}, "label": {}}

    # 标签
    label_dir = os.path.join(case_dir, "LABEL")
    if os.path.exists(label_dir):
        for f in os.listdir(label_dir):
            if f.endswith(".npy"):
                # 获取时间索引
                t = f.split("_")[-1].replace(".npy", "")
                case_data["label"][t] = np.load(os.path.join(label_dir, f))

    # 雷达
    for feat in radar_feats:
        feat_dir = os.path.join(case_dir, "RADAR", feat)
        case_data["radar"][feat] = {}
        if os.path.exists(feat_dir):
            for f in os.listdir(feat_dir):
                if f.endswith(".npy"):
                    t = f.split("_")[-1].replace(".npy", "")
                    case_data["radar"][feat][t] = np.load(os.path.join(feat_dir, f))

    # NWP
    for feat in nwp_feats:
        feat_dir = os.path.join(case_dir, "NWP", feat)
        case_data["nwp"][feat] = {}
        if os.path.exists(feat_dir):
            for f in os.listdir(feat_dir):
                if f.endswith(".npy"):
                    t = f.split("_")[-1].replace(".npy", "")
                    case_data["nwp"][feat][t] = np.load(os.path.join(feat_dir, f))
    return case_data


# 按照case处理数据

def process_one_case(case_dir, radar_feats, nwp_feats, doc_root):
    """
    处理单个 case 的完整流程。
    """
    # 1. 加载所有 .npy 数据
    case_data = collect_case_data(case_dir, radar_feats, nwp_feats)
    
    if not case_data["label"] or not any(case_data["radar"][f] for f in radar_feats) or not any(case_data["nwp"][f] for f in nwp_feats):
        print(f"Skipping {case_dir} due to missing essential data.")
        return None

    # 2. 解析元数据以获取坐标信息
    match = re.search(r'Z\d{4}', case_dir)
    if not match:
        print(f"警告: 无法从路径 {case_dir} 中解析出雷达站ID。跳过此case。")
        return None
    station_id = match.group(0)
    
    try:
        radar_doc_path = os.path.join(doc_root, 'Radar_Format_DOC', f'Radar_Format_{station_id}.txt')
        nwp_doc_path = os.path.join(doc_root, 'NWP_Format_DOC', f'NWP_Format_{station_id}.txt')
        radar_meta = parse_format_file(radar_doc_path)
        nwp_meta = parse_format_file(nwp_doc_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"处理 {case_dir} 时元数据出错: {e}")
        return None

    # 3. 创建坐标和时间信息
    H_r, W_r, res_r = 301, 301, 0.01
    lat_r_1d = np.arange(radar_meta['lat_min'], radar_meta['lat_min'] + H_r * res_r, res_r, dtype=np.float32)[:H_r]
    lon_r_1d = np.arange(radar_meta['lon_min'], radar_meta['lon_min'] + W_r * res_r, res_r, dtype=np.float32)[:W_r]

    H_n, W_n, res_n = 167, 167, 0.03
    lat_n_1d = np.arange(nwp_meta['lat_min'], nwp_meta['lat_min'] + H_n * res_n, res_n, dtype=np.float32)[:H_n]
    lon_n_1d = np.arange(nwp_meta['lon_min'], nwp_meta['lon_min'] + W_n * res_n, res_n, dtype=np.float32)[:W_n]

    # 创建高分辨率 (6min) 和低分辨率 (1h) 的时间坐标
    target_times_str = sorted(case_data["label"].keys())
    target_times_dt = [_parse_timestamp_to_datetime(t) for t in target_times_str]
    
    nwp_times_str = sorted(list(set(t for feat_data in case_data["nwp"].values() for t in feat_data.keys())))
    nwp_times_dt = [_parse_timestamp_to_datetime(t) for t in nwp_times_str]

    # 4. 时间插值
    nwp_hourly_arrays = {feat: np.stack([case_data["nwp"][feat].get(t, np.full((H_n, W_n), np.nan)) for t in nwp_times_str]) for feat in nwp_feats}
    ds_nwp_hourly = xr.Dataset(
        {feat: (("time", "lat", "lon"), arr) for feat, arr in nwp_hourly_arrays.items()},
        coords={"time": nwp_times_dt, "lat": lat_n_1d, "lon": lon_n_1d}
    )
    # 沿着时间轴将1小时数据插值为6分钟数据
    ds_nwp_6min = ds_nwp_hourly.interp(time=target_times_dt, method="linear", kwargs={"fill_value": "extrapolate"})

    # 5. 空间插值
    target_grid = xr.Dataset(coords={"lat": lat_r_1d, "lon": lon_r_1d})
    ds_nwp_resampled = ds_nwp_6min.interp_like(target_grid, method="nearest")

    # 6. 整合所有数据
    env_case = np.stack([ds_nwp_resampled[feat].values for feat in nwp_feats], axis=1)
    radar_stack = [np.stack([case_data["radar"][feat].get(t, np.full((H_r, W_r), np.nan)) for feat in radar_feats]) for t in target_times_str]
    radar_case = np.stack(radar_stack, axis=0)
    label_stack = [case_data["label"].get(t, np.full((H_r, W_r), np.nan)) for t in target_times_str]
    precip_case = np.stack(label_stack, axis=0)

    # 7. 对因文件缺失产生的NaN进行最终插值
    radar_case = handle_nan_with_temporal_interp(radar_case)
    precip_case = handle_nan_with_temporal_interp(precip_case)

    # 8. 按 case 进行标准化和归一化
    def normalize_in_case(data):
        min_val, max_val = np.nanmin(data), np.nanmax(data)
        return (data - min_val) / (max_val - min_val + 1e-6) if (max_val - min_val) > 1e-6 else np.zeros_like(data)

    radar_norm = normalize_in_case(radar_case)
    env_norm = normalize_in_case(env_case)
    precip_log_norm = normalize_in_case(np.log1p(precip_case))

    # 9. 创建未经改变的原始坐标网格
    lon_grid_r, lat_grid_r = np.meshgrid(lon_r_1d, lat_r_1d)
    coords_raw = np.stack([lat_grid_r, lat_grid_r], axis=0)

    return {"radar": radar_norm, "env": env_norm, "precip": precip_log_norm, "coords": coords_raw}

def handle_nan_with_temporal_interp(data_array):
    """
    对时间维度上的NaN值进行线性插值。
    """
    if data_array.ndim == 4:
        dims = ('time', 'feature', 'lat', 'lon')
    elif data_array.ndim == 3:
        dims = ('time', 'lat', 'lon')
    else:
        raise ValueError("不支持的数组维度")
    
    da = xr.DataArray(data_array, dims=dims)
    da_interp = da.interpolate_na(dim='time', method='linear', fill_value="extrapolate")
    return da_interp.values




def main(params):
    # 加载路径和超参数
    data_root = params['raw_data_root']
    output_dir = params['processed_data_dir']
    doc_root = params['doc_root']
    radar_feats = params['radar_feats']
    nwp_feats = params['nwp_feats']

    # 1. 收集所有 case 路径
    all_case_paths = [os.path.join(dp, d) for dp, dn, _ in os.walk(data_root) for d in dn if d.startswith('Case')]
    print(f"Found {len(all_case_paths)} total cases.")

    # 2. 划分 case
    np.random.seed(42)
    np.random.shuffle(all_case_paths)
    n_total = len(all_case_paths)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    train_paths = all_case_paths[:n_train]
    val_paths = all_case_paths[n_train : n_train + n_val]
    test_paths = all_case_paths[n_train + n_val:]
    print(f"Splitting into {len(train_paths)} train, {len(val_paths)} validation, {len(test_paths)} test cases.")

    # 3. 按数据集（训练/验证/测试）进行处理
    for name, paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
        output_path = os.path.join(output_dir, name)
        os.makedirs(output_path, exist_ok=True)
        print(f"\nProcessing and saving {name} set...")

        for path in tqdm(paths):
            # 对每个 case 调用核心处理函数
            case_data = process_one_case(path, radar_feats, nwp_feats, doc_root)
            if not case_data: 
                continue

            # 保存为独立的 zarr 文件
            case_name = os.path.basename(path)
            ds = xr.Dataset(
                {
                    "radar": (("time", "feature", "lat", "lon"), case_data['radar']),
                    "env": (("time", "feature", "lat", "lon"), case_data['env']),
                    "precip": (("time", "lat", "lon"), case_data['precip']),
                    "coords": (("feature", "lat", "lon"), case_data['coords']), # 保存原始坐标
                }
            )
            ds.to_zarr(os.path.join(output_path, f"{case_name}.zarr"), mode='w')

if __name__ == '__main__':
    PARAMS_PATH = "params.json"
    try:
        with open(PARAMS_PATH) as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 '{PARAMS_PATH}'。")
        print("请确保该文件与 training.py 存放在同一目录下。")
        exit() # 找不到配置，直接退出
    except json.JSONDecodeError:
        print(f"错误: 配置文件 '{PARAMS_PATH}' 格式不正确，请检查 JSON 语法。")
        exit() # JSON格式错误，直接退出

    main(params)