import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import os
import random
import glob

class ZarrPatchDataset(Dataset):
    def __init__(self, zarr_dir, input_steps=6, prediction_steps=20, patch_size=128):
        """
        zarr_dir: 包含多个独立 zarr 事件文件的目录 (e.g., "path/to/processed_data/train")
        input_steps: 输入时间步数
        patch_size: 训练时随机裁剪的图块大小
        prediction_window_hours: 需要预测的未来时长（小时）
        """
        super().__init__()
        # 输入的时间步长（如用6h的数据预测未来）
        self.input_steps = input_steps
        # 随即裁剪的尺寸大小，防止过拟合以及OOM
        self.patch_size = patch_size

        self.prediction_steps = prediction_steps
        
        # 计算需要预测的未来时间步数
        self.prediction_steps = int((prediction_window_hours * 60) / time_resolution_minutes)

        self.event_files = sorted(glob.glob(os.path.join(zarr_dir, "*.zarr")))
        
        self.sample_map = [] 
        total_samples = 0
        print(f"Loading dataset from {zarr_dir}...")
        for zarr_path in self.event_files:
            with xr.open_zarr(zarr_path) as ds:
                time_len = ds.dims['time']
                h, w = ds.dims['lat'], ds.dims['lon']
                
                # 样本太小就跳过，无法被裁剪
                if h < patch_size or w < patch_size:
                    print(f"Skipping {zarr_path}, its dimensions ({h}x{w}) are smaller than patch_size ({patch_size}).")
                    continue
            
            # 一个样本需要 input_steps + target_offset 的长度
            # 计算这个事件可以生成多少样本
            # 并记录该zarr文件路径、事件的样本在全局中的起始位置、该事件样本数
            required_len = self.input_steps + self.prediction_steps
            if time_len >= required_len:
                num_samples_in_event = time_len - required_len + 1
                self.sample_map.append({
                    "path": zarr_path,
                    "start_index": total_samples,
                    "num_samples": num_samples_in_event
                })
                total_samples += num_samples_in_event
        
        self.total_length = total_samples
        print(f"Found {len(self.event_files)} events, with a total of {self.total_length} samples.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # 1. 根据全局索引 idx 找到对应的事件文件和内部的时间索引
        event_info = None
        for info in self.sample_map:
            # 若该索引属于某个zarr文件（在该文件的起始索引+样本长度的范围内）
            if idx < info['start_index'] + info['num_samples']:
                event_info = info
                break
        
        if event_info is None:
            raise IndexError("Index out of bounds")

        # 某样本在具体文件中的索引=该样本的全局索引-该zarr文件中第一个样本的全局索引
        local_t_idx = idx - event_info['start_index']
        
        # 2. 读取对应的数据切片 (只读取需要的部分，I/O高效)
        with xr.open_zarr(event_info['path']) as ds:
            H, W = ds.dims['lat'], ds.dims['lon']
            
            # 3. 随机裁剪一个patch
            h_start = random.randint(0, H - self.patch_size)
            w_start = random.randint(0, W - self.patch_size)
            h_end = h_start + self.patch_size
            w_end = w_start + self.patch_size
            
            # 雷达观测、环境变量序列(T, F, H_patch, W_patch)
            radar_seq = ds['radar'][local_t_idx : local_t_idx + self.input_steps, :, h_start:h_end, w_start:w_end].values
            env_seq = ds['env'][local_t_idx : local_t_idx + self.input_steps, :, h_start:h_end, w_start:w_end].values
            
            # 降水场序列(T, H_patch, W_patch)
            precip_seq = ds['precip'][local_t_idx : local_t_idx + self.input_steps, h_start:h_end, w_start:w_end].values
            
            coords_patch = ds['coords'][:, h_start:h_end, w_start:w_end].values

            # 目标序列 Y: (T_out, H_patch, W_patch)
            target_start_idx = local_t_idx + self.input_steps
            target_end_idx = target_start_idx + self.prediction_steps
            y_raw = ds['precip'][target_start_idx:target_end_idx, h_start:h_end, w_start:w_end].values

        # 4. 组装输入 X
        precip_seq = precip_seq[:, np.newaxis, :, :] # (T, 1, H_patch, W_patch)
        coords_seq = np.tile(coords_patch[np.newaxis, :, :, :], (self.input_steps, 1, 1, 1)) # (2, H, W) -> (T, 2, H, W)

        # 将 radar(F1), env(F2), precip(1) 在特征维度拼接
        x_data = np.concatenate([radar_seq, env_seq, precip_seq, coords_seq], axis=1) # (T, F1+F2+1, H_patch, W_patch)
        x = torch.from_numpy(x_data).float()

        # 5. 组装目标 Y, 增加通道维度
        y = torch.from_numpy(y_raw).float().unsqueeze(1) # Shape: (T_out, 1, H_patch, W_patch)

        return x, y


def load_datasets(processed_data_dir, input_steps=6, batch_size=8, patch_size=128, num_workers=4, 
                  time_resolution_minutes=60, prediction_window_hours=0):
    train_dir = os.path.join(processed_data_dir, "train")
    val_dir = os.path.join(processed_data_dir, "val")
    test_dir = os.path.join(processed_data_dir, "test")

     # 计算预测步数
    prediction_steps = int((prediction_window_hours * 60) / time_resolution_minutes)

    dataset_args = {
        "input_steps": input_steps,
        "prediction_steps": prediction_steps,
        "patch_size": patch_size
    }

    train_ds = ZarrPatchDataset(train_dir, **dataset_args)
    val_ds   = ZarrPatchDataset(val_dir, **dataset_args)
    test_ds  = ZarrPatchDataset(test_dir, **dataset_args)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
