import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from dataset import load_datasets
from model import ConvLSTM_UNet
from loss import IBLoss
import json
import os


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    使用新的 IBLoss 进行单轮训练。
    """
    model.train()
    total_loss = 0.0

    # 数据加载器返回x和y_seq
    for x, y_seq in loader:
        # x: (B, T_in, C, H, W), y_seq: (B, T_out, 1, H, W)
        x, y_seq = x.to(device), y_seq.to(device)

        optimizer.zero_grad()

        # 模型输出一个序列
        output_seq = model(x)  # output_seq: (B, T_out, 1, H, W)

        # 使用论文中的IBLoss
        loss = loss_fn(output_seq, y_seq)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# 验证函数
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x, y_seq in loader:
        x, y_seq = x.to(device), y_seq.to(device)
        output_seq = model(x)

        # 验证时仍然使用标准的MSE损失
        loss = criterion(output_seq, y_seq)
        total_loss += loss.item()

    return total_loss / len(loader)


def main(params):

    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 从 params 加载超参数
    prediction_steps = int(
        (params["prediction_window_hours"] * 60) / params["time_resolution_minutes"]
    )

    # 数据加载器
    print("Loading datasets...")
    train_loader, val_loader, test_loader = load_datasets(
        processed_data_dir=params["processed_data_dir"],
        input_steps=params["input_steps"],
        batch_size=params["batch_size"],
        patch_size=params["patch_size"],
        num_workers=params["num_workers"],
        time_resolution_minutes=params["time_resolution_minutes"],
        prediction_window_hours=params["prediction_window_hours"],
    )

    # 初始化IBLoss
    bins_path = "bins.pt"
    if not os.path.exists(bins_path):
        print(f"错误: 找不到分桶信息文件 '{bins_path}'。")
        print("请先独立运行 'python compute_bins.py' 来生成该文件。")
        exit()

    bin_data = torch.load(bins_path, map_location=device)
    bin_edges = bin_data["bin_edges"]
    bin_probs = bin_data["bin_probs"]

    # 实例化IB损失函数
    ib_loss_fn = IBLoss(bin_edges, bin_probs, tau=2.0).to(device)

    # 模型
    print("Initializing model...")
    model = ConvLSTM_UNet(
        in_ch=params["in_channels"],
        out_ch=1,
        base_ch=params["base_ch"],
        out_steps=prediction_steps,
    ).to(device)

    # 验证时仍然使用标准的MSE
    val_criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=2, verbose=True
    )

    best_val_loss = float("inf")

    # 主训练循环
    print("Starting training...")
    for epoch in range(params["epochs"]):
        # 将新的ib_loss_fn传递给训练函数
        train_loss = train_one_epoch(model, train_loader, optimizer, ib_loss_fn, device)
        val_loss = evaluate(model, val_loader, val_criterion, device)

        print(
            f"Epoch [{epoch+1}/{params['epochs']}] | "
            f"Train Loss (IBLoss): {train_loss:.6f} | "
            f"Val Loss (MSE): {val_loss:.6f}"
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
        exit()
    except json.JSONDecodeError:
        print(f"错误: 配置文件 '{PARAMS_PATH}' 格式不正确。")
        exit()

    main(params)
