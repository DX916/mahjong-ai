# 路径: rlcard/agents/mahjong_mcts/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np


class MahjongTrainer:
    def __init__(self, model, config):
        """
        麻将智能体训练器 (支持 A2C 与 TensorBoard 可视化)
        """
        self.model = model
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['train']['learning_rate']
        )

        # 初始化 TensorBoard
        log_dir = config.get('log_dir', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.step_count = 0

    def log_custom_scalar(self, tag, value, step):
        """
        核心修正：添加缺失的自定义指标记录方法
        """
        self.writer.add_scalar(tag, value, step)

    def train_step(self, buffer):
        """
        执行一步 A2C 策略梯度更新
        """
        if len(buffer) < self.config['train']['batch_size']:
            return 0.0, 0.0, 0.0

        self.model.train()
        total_p_loss, total_v_loss, total_entropy = 0, 0, 0
        epochs = self.config['train'].get('epochs', 1)

        for _ in range(epochs):
            # 1. 采样
            s_batch, pi_batch, z_batch = buffer.sample(self.config['train']['batch_size'])

            # 2. 转换为 Tensor
            s_tensor = torch.FloatTensor(s_batch).to(self.device)  # [Batch, 15, 34, 4]
            pi_target = torch.FloatTensor(pi_batch).to(self.device)  # [Batch, 38]
            z_target = torch.FloatTensor(z_batch).to(self.device)  # [Batch, 4] (Rotated)

            # 3. 前向传播
            probs, value = self.model(s_tensor)
            log_probs = torch.log(probs + 1e-8)

            # 4. A2C 策略损失计算
            # advantage = 实际收益 - 价值预测 (仅针对决策者，即索引 0)
            advantage = (z_target[:, 0] - value[:, 0]).detach()

            # 提取执行动作的对数概率
            log_prob_actions = torch.sum(pi_target * log_probs, dim=1)
            policy_loss = -torch.mean(advantage * log_prob_actions)

            # 5. 价值损失 (MSE)
            value_loss = F.mse_loss(value, z_target)

            # 6. 策略熵 (用于维持探索)
            entropy = -torch.mean(torch.sum(probs * log_probs, dim=1))

            # 7. 合并总损失
            loss = (self.config['train']['policy_loss_weight'] * policy_loss +
                    self.config['train']['value_loss_weight'] * value_loss -
                    self.config['train']['entropy_weight'] * entropy)

            # 8. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_p_loss += policy_loss.item()
            total_v_loss += value_loss.item()
            total_entropy += entropy.item()

        # 9. 记录基本指标
        self.step_count += 1
        avg_p, avg_v, avg_e = total_p_loss / epochs, total_v_loss / epochs, total_entropy / epochs

        self.writer.add_scalar('Loss/Policy', avg_p, self.step_count)
        self.writer.add_scalar('Loss/Value', avg_v, self.step_count)
        self.writer.add_scalar('Metric/Entropy', avg_e, self.step_count)

        return avg_p, avg_v, avg_e

    def close(self):
        self.writer.close()