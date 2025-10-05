import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm # 用于显示进度条

class IBLoss(nn.Module):
    """
    Information Balance (IB) Loss from Leadsee-Precip paper.
    使用论文中的加权损失函数
    """
    def __init__(self, bin_edges, bin_probs, tau=2.0, reduction='mean'):
        """
        Args:
            bin_edges (torch.Tensor): 定义分桶边界的张量, shape: (num_bins + 1,).
            bin_probs (torch.Tensor): 每个桶的预计算概率, shape: (num_bins,).
            tau (float): 论文中的温度系数，用于调整权重。论文设置为 2.0。
            reduction (str): 'mean' 或 'sum'。
        """
        super().__init__()
        if not isinstance(bin_edges, torch.Tensor) or not isinstance(bin_probs, torch.Tensor):
            raise TypeError("bin_edges and bin_probs must be torch.Tensors.")
            
        # 注册为 buffer，这样在模型 to(device) 时，它们也会被移动到相应设备
        self.register_buffer('bin_edges', bin_edges)
        
        # 加上一个很小的 epsilon 防止 log(0)
        safe_bin_probs = bin_probs + 1e-9
        
        # 计算信息熵: -log(P(y))
        info_content = -torch.log(safe_bin_probs)
        
        # 根据 IB 方案的公式计算权重
        # W_i = [-log P(y_i)]^tau
        weights = torch.pow(info_content, tau)
        
        self.register_buffer('weights', weights)
        self.reduction = reduction
        print("IBLoss initialized.")
        print(f"Bin edges min: {self.bin_edges.min():.4f}, max: {self.bin_edges.max():.4f}")
        print(f"Weights min: {self.weights.min():.4f}, max: {self.weights.max():.4f}")


    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): 模型预测值。
            y_true (torch.Tensor): 真实值。
        """
        # 将 y_true 和 y_pred 展平，以便按像素处理
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)

        # 对 y_true 中的每个像素，找到它所属的桶的索引
        # torch.bucketize 非常适合这个任务。right=True 表示 (edge1, edge2] 区间
        bin_indices = torch.bucketize(y_true_flat, self.bin_edges, right=True) - 1
        
        # 确保索引在有效范围内 [0, num_bins-1]
        bin_indices.clamp_(0, len(self.weights) - 1)

        # 根据桶索引获取每个像素对应的权重
        pixel_weights = self.weights[bin_indices]
        
        # 计算每个像素的标准 MSE
        pixel_mse = F.mse_loss(y_pred_flat, y_true_flat, reduction='none')
        
        # 应用权重
        weighted_mse = pixel_mse * pixel_weights
        
        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:
            return weighted_mse

def compute_and_save_bins(dataset, num_bins=92, save_path="bins.pt"):
    """
    遍历数据集以计算 IBLoss 所需的像素分布，并将结果保存到文件。
    Args:
        dataset: 你的 PyTorch Dataset 对象。
        num_bins (int): 直方图的分桶数量。论文中使用了 92。
        save_path (str): 保存计算结果的文件路径。
    """
    if os.path.exists(save_path):
        print(f"Loading pre-computed bins from {save_path}...")
        return torch.load(save_path)

    print(f"Computing pixel distribution for IBLoss with {num_bins} bins...")
    
    # 为了避免内存爆炸，我们采用迭代方式计算直方图
    # 首先需要确定全局的最大值和最小值
    print("Step 1: Finding global min and max pixel values...")
    global_min = float('inf')
    global_max = float('-inf')
    # 注意：这里的 `__getitem__` 返回 (x, y)，所以 y 是第二个元素
    for i in tqdm(range(len(dataset)), desc="Finding min/max"):
        _, target_sequence = dataset[i] 
        if target_sequence.num_el() > 0:
            global_min = min(global_min, target_sequence.min())
            global_max = max(global_max, target_sequence.max())
    
    # 确保有有效的值范围
    if global_min == float('inf') or global_max == float('-inf'):
        raise ValueError("Could not determine valid min/max from the dataset. Is the dataset empty or all NaNs?")

    print(f"Global min: {global_min:.4f}, Global max: {global_max:.4f}")

    # 创建直方图的桶边界
    bin_edges = torch.linspace(global_min, global_max, steps=num_bins + 1)
    counts = torch.zeros(num_bins, dtype=torch.long)
    
    print("Step 2: Computing histogram...")
    for i in tqdm(range(len(dataset)), desc="Computing histogram"):
        _, target_sequence = dataset[i]
        target_flat = target_sequence.view(-1)
        
        # 使用 torch.histc 计算当前样本的直方图并累加
        counts += torch.histc(target_flat, bins=num_bins, min=global_min, max=global_max)

    # 计算概率
    total_pixels = counts.sum()
    if total_pixels == 0:
        raise ValueError("Dataset contains no pixels to analyze.")
    probs = counts.float() / total_pixels
    
    print("Bin computation complete.")
    
    result = {"bin_edges": bin_edges, "bin_probs": probs}
    torch.save(result, save_path)
    print(f"Bin information saved to {save_path}")
    
    return result