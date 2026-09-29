import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 模块，用于自动学习通道间的权重关系"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SEResidualBlock(nn.Module):
    """集成注意力机制的增强残差块"""
    def __init__(self, channels):
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) # 注入注意力
        out += residual
        return F.relu(out)

class MahjongConstraintNet(nn.Module):
    def __init__(self, res_blocks=10, channels=256):
        super(MahjongConstraintNet, self).__init__()
        # 输入: [B, 15, 34, 4] -> [B, 60, 34]
        self.initial_conv = nn.Conv1d(60, channels, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm1d(channels)

        # 深层残差主干 (10个块)
        self.res_layers = nn.ModuleList([
            SEResidualBlock(channels) for _ in range(res_blocks)
        ])

        # 强化预测头
        self.prediction_head = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 16, kernel_size=1) # 4位置 * 4实例
        )

    def forward(self, state):
        batch_size = state.shape[0]
        # [B, 15, 34, 4] -> [B, 60, 34]
        x = state.permute(0, 1, 3, 2).contiguous().reshape(batch_size, 60, 34)

        x = F.relu(self.initial_bn(self.initial_conv(x)))
        for res_block in self.res_layers:
            x = res_block(x)

        out = self.prediction_head(x) # [B, 16, 34]
        out = out.view(batch_size, 4, 4, 34).permute(0, 1, 3, 2).contiguous()
        return F.softmax(out, dim=1) # 在位置维度归一化