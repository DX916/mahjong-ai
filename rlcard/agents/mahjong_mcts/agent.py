# rlcard/agents/mahjong_mcts/agent.py
import torch
import numpy as np
import os
from .network import MahjongBrain
from .mcts import ISMCTS


class MahjongMCTSAgent:
    def __init__(self, env, device='cpu', n_simulations=100):
        self.use_raw = False  # RLCard 强制要求
        self.device = device
        self.env = env

        # 1. 实例化大脑 (剪枝网络 - 用于 AC 训练)
        self.brain = MahjongBrain(num_actions=38).to(device)

        # 2. 实例化搜索器 (修复接口不匹配问题)
        # 在训练 AC 网络阶段，我们将 constraint_net 设为 None，并关闭 cgi_on
        self.mcts = ISMCTS(
            pruning_net=self.brain,
            constraint_net=None,
            env=env,
            n_simulations=n_simulations,
            cgi_on=False  # 训练 AC 时不需要 CGI 辅助
        )

    def step(self, state):
        """
        RLCard 环境调用的动作选择接口
        """
        # 注意：这里调用 get_action_probs 或 get_action
        # 根据我们最新的 mcts.py 实现，使用 get_action_probs
        action, _ = self.mcts.get_action_probs(state)
        return action

    def eval_step(self, state):
        """
        评估时调用的动作选择接口
        """
        action, probs = self.mcts.get_action_probs(state)
        info = {'probs': probs}
        return action, info

    def save_checkpoint(self, path, optimizer, episode):
        """
        保存模型、优化器状态以及当前局数
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.brain.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, optimizer=None):
        """
        加载检查点，如果提供了 optimizer，则恢复其状态
        """
        if not os.path.exists(path):
            print(f"⚠️ 未找到 Checkpoint: {path}")
            return 0

        checkpoint = torch.load(path, map_location=self.device)
        # 兼容处理不同的保存键名
        s_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.brain.load_state_dict(s_dict)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint.get('episode', 0)