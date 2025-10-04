# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ConvLSTM cell
class ConvLSTMCell(nn.Module):
    """
    Basic ConvLSTM cell.
    x: (B, C_in, H, W)
    h_prev, c_prev: (B, C_hidden, H, W)
    returns h_next, c_next
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # 卷积代替全连接
        self.conv = nn.Conv2d(in_channels + hidden_channels,
                              4 * hidden_channels,
                              kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, x, h_prev, c_prev):
        # x: (B, C_in, H, W)
        combined = torch.cat([x, h_prev], dim=1)  # (B, C_in + C_h, H, W)
        conv_out = self.conv(combined)
        # split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# ConvLSTM wrapper for sequences

class ConvLSTM(nn.Module):
    """
    Runs a ConvLSTMCell over a sequence.
    Input: x_seq: (B, T, C, H, W)
    Returns:
      out_seq: (B, T, C_hidden, H, W)
      (h_last, c_last) each: (B, C_hidden, H, W)
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)

    def forward(self, x_seq, hidden=None):
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        if hidden is None:
            h = torch.zeros(B, self.cell.hidden_channels, H, W, device=device)
            c = torch.zeros(B, self.cell.hidden_channels, H, W, device=device)
        else:
            h, c = hidden
        outputs = []
        for t in range(T):
            x_t = x_seq[:, t]           # (B, C, H, W)
            h, c = self.cell(x_t, h, c)
            outputs.append(h)
        out_seq = torch.stack(outputs, dim=1)  # (B, T, C_hidden, H, W)
        return out_seq, (h, c)


# Standard conv block

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# Encoder block: per-frame conv -> ConvLSTM over time

class EncBlockWithConvLSTM(nn.Module):
    """
    For a block: apply the same conv_block to each frame (shared weights),
    then run ConvLSTM on the resulting feature sequence.
    Input: x_seq: (B, T, C_in, H, W)
    Output: out_seq: (B, T, C_out, H, W) and last_hidden: (B, C_out, H, W)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.shared_conv = conv_block(in_ch, out_ch)
        self.temporal = ConvLSTM(out_ch, out_ch)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        # apply shared conv to each frame efficiently
        x_flat = x_seq.view(B * T, C, H, W)                # (B*T, C, H, W)
        feat_flat = self.shared_conv(x_flat)               # (B*T, out_ch, H, W)
        feat_seq = feat_flat.view(B, T, feat_flat.size(1), H, W)  # (B, T, out_ch, H, W)
        out_seq, (h_last, c_last) = self.temporal(feat_seq)
        return out_seq, h_last  # return sequence + last hidden for skip


# Decoder block (standard 2D)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x, skip):
        """
        x: (B, C, H, W) - lower resolution
        skip: (B, C_skip, H*2, W*2)
        """
        x = self.up(x)
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


# Full ConvLSTM U-Net

class ConvLSTM_UNet(nn.Module):
    """
    U-Net where encoder blocks contain ConvLSTM temporal modeling.
    Input: x_seq: (B, T, C, H, W)
    Output: next frame prediction: (B, 1, H, W)
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=32, out_steps=1):
        super().__init__()
        f = base_ch
        self.out_ch = out_ch
        self.out_steps = out_steps
        # Encoder
        self.enc1 = EncBlockWithConvLSTM(in_ch, f)       # out: f
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = EncBlockWithConvLSTM(f, f*2)         # out: 2f
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = EncBlockWithConvLSTM(f*2, f*4)       # out: 4f
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = EncBlockWithConvLSTM(f*4, f*8)       # out: 8f
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck - do an extra conv (could be ConvLSTM if you want)
        self.bottleneck_conv = conv_block(f*8, f*16)

        # Decoder
        self.up4 = DecoderBlock(f*16, f*8)
        self.up3 = DecoderBlock(f*8, f*4)
        self.up2 = DecoderBlock(f*4, f*2)
        self.up1 = DecoderBlock(f*2, f)

        self.final = nn.Conv2d(f, out_ch * self.out_steps, kernel_size=1)

    def forward(self, x_seq):
        """
        x_seq: (B, T, C, H, W)
        returns: out: (B, out_ch, H, W)
        """
        B, T, C, H, W = x_seq.shape

        # Encoder level 1
        e1_seq, e1_last = self.enc1(x_seq)   # e1_seq (B,T,f,H,W), e1_last (B,f,H,W)
        # Downsample the sequence for next level
        e1_seq_pooled = self.pool1(e1_seq.view(B*T, *e1_seq.shape[2:])).view(B, T, e1_seq.shape[2], H//2, W//2)

        # Encoder level 2
        e2_seq, e2_last = self.enc2(e1_seq_pooled)
        e2_seq_pooled = self.pool2(e2_seq.view(B*T, *e2_seq.shape[2:])).view(B, T, e2_seq.shape[2], H//4, W//4)

        # Encoder level 3
        e3_seq, e3_last = self.enc3(e2_seq_pooled)
        e3_seq_pooled = self.pool3(e3_seq.view(B*T, *e3_seq.shape[2:])).view(B, T, e3_seq.shape[2], H//8, W//8)

        # Encoder level 4
        e4_seq, e4_last = self.enc4(e3_seq_pooled)
        e4_seq_pooled = self.pool4(e4_seq.view(B*T, *e4_seq.shape[2:])).view(B, T, e4_seq.shape[2], H//16, W//16)

        # Bottleneck: use last hidden of e4 as spatial feature
        b = e4_last  # (B, f*8, H/16, W/16)
        b = self.bottleneck_conv(b)  # (B, f*16, H/16, W/16)

        # Decoder: use last hidden states as skips
        d4 = self.up4(b, e4_last)  # (B, f*8, H/8, W/8)
        d3 = self.up3(d4, e3_last)  # (B, f*4, H/4, W/4)
        d2 = self.up2(d3, e2_last)  # (B, f*2, H/2, W/2)
        d1 = self.up1(d2, e1_last)  # (B, f, H, W)

        out = self.final(d1)        # Shape: (B, T_out * out_ch, H, W)

        # 重塑输出张量
        out = out.view(B, self.out_steps, self.out_ch, H, W) # Shape: (B, T_out, out_ch, H, W)
        return out


# quick smoke test

if __name__ == "__main__":
    B, T, C, H, W = 16, 6, 38, 128, 128
    x = torch.randn(B, T, C, H, W)
    model = ConvLSTM_UNet(in_ch=C, out_ch=1, base_ch=16)
    out = model(x)
    print("output shape:", out.shape)  # expected (B,1,H,W)
