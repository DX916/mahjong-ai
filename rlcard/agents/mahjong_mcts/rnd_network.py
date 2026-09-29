# 新文件: rlcard/agents/mahjong_mcts/rnd_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class RNDTarget(nn.Module):
    """固定的随机网络（不训练）"""

    def __init__(self, input_shape=(15, 34, 4), output_dim=128):
        super(RNDTarget, self).__init__()

        # 简单的卷积特征提取器
        self.conv1 = nn.Conv2d(15, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # 全局池化 + 全连接
        self.fc = nn.Linear(64, output_dim)

        # ⚠️ 初始化后冻结参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, state):
        # state: [B, 15, 34, 4]
        x = state.permute(0, 1, 3, 2)  # [B, 15, 4, 34]
        x = F.relu(self.conv1(x))  # [B, 32, 4, 34]
        x = F.relu(self.conv2(x))  # [B, 64, 4, 34]
        x = F.relu(self.conv3(x))  # [B, 64, 4, 34]

        # 全局平均池化
        x = x.mean(dim=[2, 3])  # [B, 64]
        x = self.fc(x)  # [B, 128]

        return x


class RNDPredictor(nn.Module):
    """可训练的预测网络"""

    def __init__(self, input_shape=(15, 34, 4), output_dim=128):
        super(RNDPredictor, self).__init__()

        # 与Target相同的结构
        self.conv1 = nn.Conv2d(15, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc = nn.Linear(64, output_dim)

        # ✅ 参数可训练

    def forward(self, state):
        x = state.permute(0, 1, 3, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.mean(dim=[2, 3])
        x = self.fc(x)

        return x


class RNDModule:
    """RND模块：管理target和predictor"""

    def __init__(self, device='cuda', output_dim=128, learning_rate=0.0001):
        self.target = RNDTarget(output_dim=output_dim).to(device)
        self.predictor = RNDPredictor(output_dim=output_dim).to(device)
        self.device = device
        self.output_dim = output_dim

        # 预测器的优化器
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=learning_rate
        )

        # 运行统计（用于归一化内在奖励）
        self.intrinsic_reward_mean = 0.0
        self.intrinsic_reward_std = 1.0
        self.update_count = 0

    def compute_intrinsic_reward(self, state):
        """
        计算内在奖励
        Args:
            state: numpy array [B, 15, 34, 4] or [15, 34, 4]
        Returns:
            intrinsic_reward: numpy array [B] or scalar
        """
        is_single = (state.ndim == 3)
        if is_single:
            state = state[np.newaxis, ...]

        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            target_feat = self.target(state_tensor)  # [B, 128]
            predictor_feat = self.predictor(state_tensor)  # [B, 128]

            # MSE作为新奇度
            mse = torch.mean((target_feat - predictor_feat) ** 2, dim=1)
            intrinsic_reward = mse.cpu().numpy()

        if is_single:
            return intrinsic_reward[0]
        return intrinsic_reward

    def update(self, state_batch):
        """
        更新预测网络
        Args:
            state_batch: numpy array [B, 15, 34, 4]
        """
        state_tensor = torch.FloatTensor(state_batch).to(self.device)

        # 前向传播
        target_feat = self.target(state_tensor).detach()
        predictor_feat = self.predictor(state_tensor)

        # 损失 = MSE
        loss = F.mse_loss(predictor_feat, target_feat)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def normalize_intrinsic_reward(self, intrinsic_reward):
        """
        归一化内在奖励（避免尺度过大）
        使用运行平均和标准差
        """
        self.update_count += 1
        alpha = 1.0 / min(self.update_count, 1000)

        # 更新统计
        self.intrinsic_reward_mean = (1 - alpha) * self.intrinsic_reward_mean + \
                                     alpha * np.mean(intrinsic_reward)
        self.intrinsic_reward_std = (1 - alpha) * self.intrinsic_reward_std + \
                                    alpha * np.std(intrinsic_reward)

        # 归一化
        normalized = (intrinsic_reward - self.intrinsic_reward_mean) / \
                     (self.intrinsic_reward_std + 1e-8)

        return normalized

    def save_checkpoint(self, save_path):
        """
        保存RND checkpoint
        Args:
            save_path: 保存路径（例如 'mahjong_checkpoints/rnd_ep120000.pth'）
        """
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        checkpoint = {
            'target_state_dict': self.target.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'intrinsic_reward_mean': self.intrinsic_reward_mean,
            'intrinsic_reward_std': self.intrinsic_reward_std,
            'update_count': self.update_count,
            'output_dim': self.output_dim,
            'device': str(self.device)
        }

        torch.save(checkpoint, save_path)
        print(f"✅ RND checkpoint 已保存到: {save_path}")

    def load_checkpoint(self, load_path):
        """
        加载RND checkpoint
        Args:
            load_path: 加载路径
        Returns:
            bool: 是否加载成功
        """
        if not os.path.exists(load_path):
            print(f"⚠️  RND checkpoint 不存在: {load_path}")
            return False

        try:
            # 🔧 修改这一行：添加 weights_only=False
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)

            # 加载网络参数
            self.target.load_state_dict(checkpoint['target_state_dict'])
            self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载统计量
            self.intrinsic_reward_mean = checkpoint['intrinsic_reward_mean']
            self.intrinsic_reward_std = checkpoint['intrinsic_reward_std']
            self.update_count = checkpoint['update_count']

            print(f"✅ RND checkpoint 已加载: {load_path}")
            print(f"   - 内在奖励均值: {self.intrinsic_reward_mean:.4f}")
            print(f"   - 内在奖励标准差: {self.intrinsic_reward_std:.4f}")
            print(f"   - 更新次数: {self.update_count}")

            return True

        except Exception as e:
            print(f"❌ RND checkpoint 加载失败: {e}")
            return False

    def get_state(self):
        """
        获取当前状态（用于调试）
        Returns:
            dict: 当前状态信息
        """
        return {
            'intrinsic_reward_mean': self.intrinsic_reward_mean,
            'intrinsic_reward_std': self.intrinsic_reward_std,
            'update_count': self.update_count,
            'output_dim': self.output_dim,
            'device': str(self.device)
        }