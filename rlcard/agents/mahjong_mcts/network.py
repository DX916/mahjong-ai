# 路径: rlcard/agents/mahjong_mcts/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MahjongResidualBlock(nn.Module):
    def __init__(self, channels):
        super(MahjongResidualBlock, self).__init__()
        # 保持 1D 卷积，但通道数已扩充
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class MahjongBrain(nn.Module):
    def __init__(self, num_actions=38, res_blocks=8):  # 默认增加到 8 个 ResBlocks
        super(MahjongBrain, self).__init__()

        # 输入: [batch, 15, 34, 4] -> 维度变换后处理为 [batch, 60, 34]
        # 初始卷积扩充到 256 通道，以承载 15 层输入的高密度信息
        self.initial_conv = nn.Conv1d(60, 256, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm1d(256)

        # 骨干网络深度增加，提升对二阶特征（如 Plane 14）的理解能力
        self.res_layers = nn.ModuleList([
            MahjongResidualBlock(256) for _ in range(res_blocks)
        ])

        # --- 策略头 (保持原有逻辑，仅调整输入通道) ---
        self.policy_head = nn.Sequential(
            nn.Conv1d(256, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 34, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        # --- 价值头 (保持原有逻辑，仅调整输入通道) ---
        self.value_head = nn.Sequential(
            nn.Conv1d(256, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # 保持 34 个位置展开，有助于理解牌池消耗对价值的影响
            nn.Flatten(),
            nn.Linear(16 * 34, 256),
            nn.ReLU(),
            # 依然输出 4 维向量，对应 4 个玩家视角
            nn.Linear(256, 4)
        )

    def forward(self, state, action_mask=None):
        batch_size = state.shape[0]
        # state 原本是 [B, 15, 34, 4]
        # 变换为 [B, 60, 34] 以适配 1D 卷积
        x = state.permute(0, 1, 3, 2).contiguous().reshape(batch_size, 60, 34)

        # 初始特征提取
        x = F.relu(self.initial_bn(self.initial_conv(x)))

        # 穿过深度残差骨干网
        for res_block in self.res_layers:
            x = res_block(x)

        # 策略预测
        policy_logits = self.policy_head(x)
        if action_mask is not None:
            # 执行 Action Masking 逻辑
            policy_logits = policy_logits.masked_fill(~action_mask, -1e9)

        # 输出概率分布 (Softmax)
        probs = F.softmax(policy_logits, dim=-1)

        # 价值评估 (线性输出，未限制在 [-1, 1])
        value = self.value_head(x)

        return probs, value