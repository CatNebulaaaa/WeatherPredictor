import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from dataset import load_datasets
from model import ConvLSTM_UNet
import torch.nn.functional as F
import json


def weighted_mse_loss(output, target, weights):
    # output: (B, 1, H, W)
    # target: (B, 1, H, W)
    # weights: (B, 1, H, W)
    loss = weights * F.mse_loss(output, target, reduction="none")
    return torch.mean(loss)


# 定义生成权重图的函数
def get_pixel_weights(target_seq, thresholds, weight_values):
    """
    根据目标降水强度生成权重图。
    target: 归一化后的目标张量 (B, 1, H, W)
    thresholds: 归一化后的强度阈值列表, e.g., [norm(0.1), norm(5), norm(20)]
    weight_values: 对应的权重列表, e.g., [1, 5, 10, 20] (长度必须比thresholds多1)
    """
    B, _, H, W = target_seq.shape
    weights_seq = torch.ones_like(target_seq) * weight_values[0]  # 默认为基础权重

    for i in range(len(thresholds)):
        weights_seq[target_seq >= thresholds[i]] = weight_values[i + 1]

    return weights_seq.to(target_seq.device)


#  训练函数
def train_one_epoch(model, loader, optimizer, params, device):
    model.train()
    total_loss = 0

    thresholds = params.get("loss_thresholds_norm", [])
    weights = params.get("loss_weights", [1.0])

    for x, y_seq in loader:
        # x: (B, T_in, C, H, W), y_seq: (B, T_out, 1, H, W)
        x, y_seq = x.to(device), y_seq.to(device)

        optimizer.zero_grad()
        output_seq = model(x) # output_seq: (B, T_out, 1, H, W)

        # --- 计算带权重的序列损失 ---
        pixel_weights_seq = get_pixel_weights(y_seq, thresholds, weights)
        
        loss = 0
        # 遍历每一个预测时间步计算损失
        for t in range(output_seq.shape[1]):
            loss += weighted_mse_loss(output_seq[:, t], y_seq[:, t], pixel_weights_seq[:, t])
        
        loss = loss / output_seq.shape[1] # 对时间步取平均

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


#  验证函数
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for x, y_seq in loader:
        x, y_seq = x.to(device), y_seq.to(device)
        output_seq = model(x)
        
        # 计算整个序列的 MSE 损失
        loss = criterion(output_seq, y_seq)
        total_loss += loss.item()
    return total_loss / len(loader)


def main(params):

    #  配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 从 params 加载超参数
    batch_size = params["batch_size"]
    input_steps = params["input_steps"]
    epochs = params["epochs"]
    lr = params["lr"]
    patch_size = params["patch_size"]
    prediction_window_hours = params["prediction_window_hours"]
    time_resolution_minutes = params["time_resolution_minutes"]
    # 计算预测步数
    prediction_steps = int((params["prediction_window_hours"] * 60) / params["time_resolution_minutes"])

    #  数据
    print("Loading datasets...")
    train_loader, val_loader, test_loader = load_datasets(
        processed_data_dir=params["processed_data_dir"],
        input_steps=params["input_steps"],
        batch_size=params["batch_size"],
        patch_size=params["patch_size"],
        num_workers=params["num_workers"],
        time_resolution_minutes=time_resolution_minutes,
        prediction_window_hours=prediction_window_hours,
    )

    #  模型
    print("Initializing model...")
    model = ConvLSTM_UNet(
        in_ch=params["in_channels"], 
        out_ch=1, 
        base_ch=params["base_ch"],
        out_steps=prediction_steps  # 传递输出步数
    ).to(device)

    val_criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=2, verbose=True
    )

    best_val_loss = float('inf') 

    #  主训练循环
    for epoch in range(params["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, params, device)
        val_loss = evaluate(model, val_loader, val_criterion, device) 

        print(
            f"Epoch [{epoch+1}/{params['epochs']}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), params["model_save_path"])
            print(f"   -> Best model saved to {params['model_save_path']}.")

    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load(params["model_save_path"]))
    test_loss = evaluate(model, test_loader, val_criterion, device)
    print(f"Final Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    PARAMS_PATH = "params.json"
    try:
        with open(PARAMS_PATH) as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 '{PARAMS_PATH}'。")
        print("请确保该文件与 training.py 存放在同一目录下。")
        exit()  # 找不到配置，直接退出
    except json.JSONDecodeError:
        print(f"错误: 配置文件 '{PARAMS_PATH}' 格式不正确，请检查 JSON 语法。")
        exit()  # JSON格式错误，直接退出

    main(params)
