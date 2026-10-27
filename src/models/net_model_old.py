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


class UNetBinary(nn.Module):
    """
    最简单的 6 通道二分类 UNet
    - 4 层编码，瓶颈 32×32
    - 输出 2 类 logits（0=栖息地，1=背景）
    """

    def __init__(self, in_channels=6, num_classes=2,
                 features=[64, 128, 256, 512]):
        super(UNetBinary, self).__init__()

        # ========== 编码器（4 层，512 -> 256 -> 128 -> 64 -> 32） ==========
        self.enc1 = DoubleConv(in_channels, features[0])       # 512
        self.down1 = Down(features[0], features[1])            # 256
        self.down2 = Down(features[1], features[2])            # 128
        self.down3 = Down(features[2], features[3])            # 64
        self.down4 = Down(features[3], features[3] * 2)        # 32（瓶颈）

        # ========== 解码器（4 层，32 -> 64 -> 128 -> 256 -> 512） ==========
        self.up1 = Up(features[3] * 2, features[3])            # 64
        self.up2 = Up(features[3], features[2])                # 128
        self.up3 = Up(features[2], features[1])                # 256
        self.up4 = Up(features[1], features[0])                # 512

        self.out_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

        self._initialize_weights()

    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, 6, H, W]
        Returns:
            logits: [B, 2, H, W]
        """
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
    最简单的 6 通道三分类 UNet
    - 4 层编码，瓶颈 32×32
    - 输出 3 类 logits（0=栖息地，1=背景，2=空类）
    - 空类不参与 loss（ignore_index=2）
    """

    def __init__(self, in_channels=6, num_classes=3,
                 features=[64, 128, 256, 512]):
        super(UNetThree, self).__init__()

        # ========== 编码器（4 层，512 -> 256 -> 128 -> 64 -> 32） ==========
        self.enc1 = DoubleConv(in_channels, features[0])       # 512
        self.down1 = Down(features[0], features[1])            # 256
        self.down2 = Down(features[1], features[2])            # 128
        self.down3 = Down(features[2], features[3])            # 64
        self.down4 = Down(features[3], features[3] * 2)        # 32（瓶颈）

        # ========== 解码器（4 层，32 -> 64 -> 128 -> 256 -> 512） ==========
        self.up1 = Up(features[3] * 2, features[3])            # 64
        self.up2 = Up(features[3], features[2])                # 128
        self.up3 = Up(features[2], features[1])                # 256
        self.up4 = Up(features[1], features[0])                # 512

        self.out_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

        self._initialize_weights()

    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, 6, H, W]
        Returns:
            logits: [B, 3, H, W]
        """
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
    # 二分类
    model_binary = UNetBinary(in_channels=6, num_classes=2)
    x = torch.randn(4, 6, 512, 512)
    out_binary = model_binary(x)
    print(f"UNetBinary - Input: {x.shape}, Output: {out_binary.shape}")

    # 三分类
    model_three = UNetThree(in_channels=6, num_classes=3)
    out_three = model_three(x)
    print(f"UNetThree - Input: {x.shape}, Output: {out_three.shape}")

    # 参数量
    params_binary = sum(p.numel() for p in model_binary.parameters())
    params_three = sum(p.numel() for p in model_three.parameters())
    print(f"UNetBinary 参数量: {params_binary:,}")
    print(f"UNetThree 参数量: {params_three:,}")
