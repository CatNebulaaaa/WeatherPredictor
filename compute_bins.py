import os
import json
from dataset import ZarrPatchDataset 
from loss import compute_and_save_bins

def run_computation():
    """
    独立运行此脚本来为 IBLoss 预计算和保存数据分布。
    """
    # 加载配置
    PARAMS_PATH = "params.json"
    try:
        with open(PARAMS_PATH) as f:
            params = json.load(f)
    except Exception as e:
        print(f"错误: 无法加载 '{PARAMS_PATH}': {e}")
        exit()

    train_dir = os.path.join(params["processed_data_dir"], "train")
    if not os.path.exists(train_dir):
        print(f"错误: 训练数据目录不存在: '{train_dir}'")
        print("请先运行 preproc.py 生成预处理数据。")
        exit()

    # 初始化训练数据集
    print("Initializing training dataset to compute pixel distribution...")
    
    # 从 params.json 中获取 prediction_steps 的计算所需参数
    prediction_window_hours = params["prediction_window_hours"]
    time_resolution_minutes = params["time_resolution_minutes"]
    
    train_ds = ZarrPatchDataset(
        zarr_dir=train_dir,
        input_steps=params["input_steps"],
        # 传递小时和分钟，让 Dataset 类自己计算步数
        prediction_window_hours=prediction_window_hours,
        time_resolution_minutes=time_resolution_minutes,
        patch_size=params["patch_size"]
    )
    
    # 检查并运行计算
    if len(train_ds) == 0:
        print("错误: 训练数据集为空，无法计算像素分布。请检查数据路径和文件。")
    else:
        # 调用辅助函数，计算并保存分桶信息
        compute_and_save_bins(train_ds, num_bins=92, save_path="bins.pt")
        print("\n计算完成!'bins.pt' 文件已生成。")
        print("现在你可以运行 training.py 来开始训练了。")

if __name__ == '__main__':
    run_computation()