import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积块：Conv2d + GN + ReLU + Conv2d + GN + ReLU"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=min(
            8, out_channels), num_channels=out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=min(
            8, out_channels), num_channels=out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu2(x)
        return x


class Down(nn.Module):
    """下采样模块：MaxPool + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.double_conv(x)
        return x


class Up(nn.Module):
    """上采样模块：转置卷积 + 跳跃连接拼接 + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x


class BandWeight(nn.Module):
    """
    波段加权模块
    用 1×1 卷积对 6 个输入波段加权，初始化为单位矩阵
    训练后可读取列和，作为各波段的重要性
    """

    def __init__(self, in_channels=6):
        super(BandWeight, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=1, bias=False)
        # 初始化为单位矩阵（等价于恒等映射）
        nn.init.eye_(self.conv.weight[:, :, 0, 0])

    def forward(self, x):
        return self.conv(x)

    def get_weights(self):
        """返回每个输入波段的权重（1×1 卷积权重的列和）"""
        with torch.no_grad():
            w = self.conv.weight.squeeze()  # [6, 6]
            return w.abs().sum(dim=0).cpu().numpy()  # [6]


class UNetBinary(nn.Module):
    """
    6 通道二分类 UNet + 波段加权（1×1 卷积）
    - 4 层编码，瓶颈 32×32
    - 输入前先过 BandWeight
    - 输出 2 类 logits（0=栖息地，1=背景）
    """

    def __init__(self, in_channels=6, num_classes=2,
                 features=[64, 128, 256, 512]):
        super(UNetBinary, self).__init__()

        # ========== 波段加权 ==========
        self.band_weight = BandWeight(in_channels)

        # ========== 编码器（4 层） ==========
        self.enc1 = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[3] * 2)

        # ========== 解码器（4 层） ==========
        self.up1 = Up(features[3] * 2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])

        self.out_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

        self._initialize_weights()

    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, 6, H, W]
        Returns:
            logits: [B, 2, H, W]
        """
        # ===== 波段加权 =====
        x = self.band_weight(x)

        # ===== 编码 =====
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        # ===== 解码 =====
        d1 = self.up1(e5, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)

        logits = self.out_conv(d4)

        if return_features:
            features = {
                'enc1': e1, 'down1': e2, 'down2': e3,
                'down3': e4, 'down4': e5,
                'up1': d1, 'up2': d2, 'up3': d3, 'up4': d4,
                'logits': logits,
            }
            return logits, features
        return logits

    def get_band_weights(self):
        """返回 6 个波段的重要性"""
        return self.band_weight.get_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class UNetThree(nn.Module):
    """
    6 通道三分类 UNet + 波段加权（1×1 卷积）
    - 4 层编码，瓶颈 32×32
    - 输入前先过 BandWeight
    - 输出 3 类 logits（0=栖息地，1=背景，2=空类）
    """

    def __init__(self, in_channels=6, num_classes=3,
                 features=[64, 128, 256, 512]):
        super(UNetThree, self).__init__()

        # ========== 波段加权 ==========
        self.band_weight = BandWeight(in_channels)

        # ========== 编码器（4 层） ==========
        self.enc1 = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[3] * 2)

        # ========== 解码器（4 层） ==========
        self.up1 = Up(features[3] * 2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])

        self.out_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

        self._initialize_weights()

    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, 6, H, W]
        Returns:
            logits: [B, 3, H, W]
        """
        # ===== 波段加权 =====
        x = self.band_weight(x)

        # ===== 编码 =====
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        # ===== 解码 =====
        d1 = self.up1(e5, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)

        logits = self.out_conv(d4)

        if return_features:
            features = {
                'enc1': e1, 'down1': e2, 'down2': e3,
                'down3': e4, 'down4': e5,
                'up1': d1, 'up2': d2, 'up3': d3, 'up4': d4,
                'logits': logits,
            }
            return logits, features
        return logits

    def get_band_weights(self):
        """返回 6 个波段的重要性"""
        return self.band_weight.get_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ========== 快速测试 ==========
if __name__ == "__main__":
    x = torch.randn(4, 6, 128, 128)

    # 二分类
    model_binary = UNetBinary(in_channels=6, num_classes=2)
    out_binary = model_binary(x)
    print(f"UNetBinary - Input: {x.shape}, Output: {out_binary.shape}")
    print(f"  波段权重: {model_binary.get_band_weights()}")

    # 三分类
    model_three = UNetThree(in_channels=6, num_classes=3)
    out_three = model_three(x)
    print(f"UNetThree - Input: {x.shape}, Output: {out_three.shape}")
    print(f"  波段权重: {model_three.get_band_weights()}")

    # 参数量
    params_binary = sum(p.numel() for p in model_binary.parameters())
    params_three = sum(p.numel() for p in model_three.parameters())
    print(f"UNetBinary 参数量: {params_binary:,}")
    print(f"UNetThree 参数量: {params_three:,}")
