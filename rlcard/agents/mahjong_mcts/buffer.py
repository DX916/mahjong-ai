# 路径: rlcard/agents/mahjong_mcts/buffer.py
import random
from collections import deque
import numpy as np
import pickle
import os

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def save(self, state, pi, z):
        """
        Args:
            state: [10, 34, 4] numpy array
            pi: [38] 动作概率分布
            z: 旋转后的回报向量 [4]
        """
        self.buffer.append((state, pi, z))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, pis, zs = zip(*batch)
        return np.array(states), np.array(pis), np.array(zs)

    def save_to_disk(self, path):
        """将经验池保存到本地"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                # 转换成 list 方便 pickle
                pickle.dump(list(self.buffer), f)
            print(f"✅ 经验池已保存至: {path} (Size: {len(self.buffer)})")
        except Exception as e:
            print(f"❌ 经验池保存失败: {e}")

    def load_from_disk(self, path):
        """
        从本地加载经验池，支持动态容量调整：
        1. 如果本地数据量 > 当前 capacity，则从后往前截取最新的数据。
        2. 如果本地数据量 <= 当前 capacity，则全部纳入。
        """
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)

                if isinstance(data, list):
                    # 获取当前 deque 设定的最大容量
                    current_capacity = self.buffer.maxlen

                    # 核心逻辑：如果读取的数据比当前容量大，执行从后往前的切片
                    if len(data) > current_capacity:
                        print(f"⚠️ 本地经验池大小({len(data)})超过当前配置({current_capacity})，将截取最新的数据。")
                        # 仅保留最后面（最新）的 capacity 个样本
                        data = data[-current_capacity:]

                    # 清空当前 buffer（防止重复加载）并注入
                    self.buffer.clear()
                    self.buffer.extend(data)

                    print(f"✅ 经验池加载成功: {path} (Current Size: {len(self.buffer)})")
                    return True
                else:
                    print(f"❌ 经验池格式错误: 期望 list，实际得到 {type(data)}")
            except Exception as e:
                print(f"❌ 经验池加载失败 (可能文件损坏或容量冲突): {e}")
        else:
            print("ℹ️ 未发现本地经验池，将从空池开始训练。")
        return False

    def __len__(self):
        return len(self.buffer)