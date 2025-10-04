import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import json
from dataset import load_datasets
from model import ConvLSTM_UNet
import os


def plot_predictions(
    x, y_true, y_pred, params, save_path="prediction_visualization.png"
):
    """
    可视化输入、真实标签和模型预测。
    x: (T_in, C, H, W)
    y_true: (T_out, 1, H, W)
    y_pred: (T_out, 1, H, W)
    """
    # 将 Tensor 转换为 numpy 数组，并移动到 CPU
    x = x.cpu().numpy()
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    input_steps = params["input_steps"]

    # 预测步长由 y_true 的形状决定
    prediction_steps = y_true.shape[0]

    # 我们只可视化 precip 通道 (假设它是第 F1+F2 个通道, 索引为 F1+F2)
    # 根据你的代码, radar_seq, env_seq, precip_seq, coords_seq
    # precip_seq 在第三个位置, 但它是单通道, radar/env可能多通道, 需要精确计算索引
    # 假设 radar 2 通道, env 2 通道, 那么 precip 在索引 4
    # 更稳妥的方式是从 params.json 读取 radar/env 特征数
    # 这里为了简单，我们可视化输入的最后一帧的降水场
    num_radar_feats = len(params.get("radar_feats", []))
    num_nwp_feats = len(params.get("nwp_feats", []))
    precip_channel_idx = num_radar_feats + num_nwp_feats

    input_precip_last_frame = x[-1, precip_channel_idx, :, :]

    # 设置子图
    num_cols = prediction_steps
    fig, axes = plt.subplots(3, num_cols, figsize=(num_cols * 3, 9))

    # 定义颜色映射和范围 (0-1 的归一化范围)
    cmap = "jet"
    norm = colors.Normalize(vmin=0, vmax=1.0)

    for i in range(num_cols):
        # 1. 绘制输入 (只绘制最后一帧输入作为参考)
        ax = axes[0, i]
        if i == 0:
            ax.imshow(input_precip_last_frame, cmap=cmap, norm=norm)
            ax.set_title(f"Input (t=-1)")
        ax.axis("off")

        # 2. 绘制真实值
        ax = axes[1, i]
        im = ax.imshow(y_true[i, 0, :, :], cmap=cmap, norm=norm)
        ax.set_title(f"True (t=+{i+1})")
        ax.axis("off")

        # 3. 绘制预测值
        ax = axes[2, i]
        im = ax.imshow(y_pred[i, 0, :, :], cmap=cmap, norm=norm)
        ax.set_title(f"Pred (t=+{i+1})")
        ax.axis("off")

    # 添加一个共享的颜色条
    fig.colorbar(
        im, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.02, pad=0.04
    )
    fig.suptitle("Precipitation Forecast Visualization", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    print(f"Saving visualization to {save_path}")
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 计算预测步数
    prediction_steps = int(
        (params["prediction_window_hours"] * 60) / params["time_resolution_minutes"]
    )

    # 加载测试数据集
    # 注意：这里我们设置 batch_size=1, shuffle=False 以便稳定地获取一个样本
    _, _, test_loader = load_datasets(
        processed_data_dir=params["processed_data_dir"],
        input_steps=params["input_steps"],
        batch_size=1,  # 只取一个样本
        patch_size=params["patch_size"],
        num_workers=0,  # 可视化时设为0
        time_resolution_minutes=params["time_resolution_minutes"],
        prediction_window_hours=params["prediction_window_hours"],
    )

    # 初始化模型
    model = ConvLSTM_UNet(
        in_ch=params["in_channels"],
        out_ch=1,
        base_ch=params["base_ch"],
        out_steps=prediction_steps,
    ).to(device)

    # 加载训练好的模型权重
    model_path = params["model_save_path"]
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 获取一个测试样本
    with torch.no_grad():
        x_sample, y_true_sample = next(iter(test_loader))
        x_sample, y_true_sample = x_sample.to(device), y_true_sample.to(device)

        # 进行预测
        y_pred_sample = model(x_sample)

    # 从批次中取出第一个样本进行绘图
    # (B, T, C, H, W) -> (T, C, H, W)
    x_single = x_sample[0]
    y_true_single = y_true_sample[0]
    y_pred_single = y_pred_sample[0]

    # 调用绘图函数
    plot_predictions(x_single, y_true_single, y_pred_single, params)


if __name__ == "__main__":
    PARAMS_PATH = "params.json"
    try:
        with open(PARAMS_PATH) as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 '{PARAMS_PATH}'。")
        exit()
    except json.JSONDecodeError:
        print(f"错误: 配置文件 '{PARAMS_PATH}' 格式不正确。")
        exit()

    main(params)
