import torch
import torch.optim as optim
import yaml
import numpy as np
import os
import rlcard
import random
from collections import deque

# 从项目中导入网络结构
from rlcard.agents.mahjong_mcts.network import MahjongBrain
from rlcard.agents.mahjong_mcts.constraint_network import MahjongConstraintNet
from rlcard.games.mahjong.utils import card_encoding_dict


def set_seed(seed):
    """设置全局随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"✅ 随机种子已设置为: {seed}")


class ConstraintDataPool:
    """基于队列逻辑的经验池 (FIFO)，用于动态存储博弈轨迹数据"""

    def __init__(self, capacity):
        self.pool = deque(maxlen=capacity)

    def add(self, obs, label):
        self.pool.append((obs, label))

    def sample(self, batch_size):
        """随机独立采样 batch_size 个样本"""
        samples = random.sample(self.pool, batch_size)
        obs_batch = np.array([s[0] for s in samples])
        label_batch = np.array([s[1] for s in samples])
        return obs_batch, label_batch

    def __len__(self):
        return len(self.pool)


def extract_view_rotated_labels(game, current_player_id):
    """
    提取与输入 X 视角一致的 4*34*4 标签 (0-1 编码)
    视角索引: 0:牌墙, 1:下家, 2:对家, 3:上家
    """
    label = np.zeros((4, 34, 4), dtype=np.float32)
    type_instance_counters = np.zeros(34, dtype=int)

    def fill_label(card_obj, pos_idx):
        """辅助函数：将一张 Card 对象填入对应相对位置的张量中"""
        card_str = card_obj.get_str()
        if card_str in card_encoding_dict:
            c_idx = card_encoding_dict[card_str]
            # 仅处理 34 种基础牌型，且确保每个牌型实例不超过 4 个
            if c_idx < 34 and type_instance_counters[c_idx] < 4:
                label[pos_idx, c_idx, type_instance_counters[c_idx]] = 1.0
                type_instance_counters[c_idx] += 1

    # 1. 提取牌墙信息 (映射到相对视角 0)
    for card in game.dealer.deck:
        fill_label(card, 0)

    # 2. 提取对手手牌信息 (映射到相对视角 1:下家, 2:对家, 3:上家)
    for i in range(1, 4):
        relative_id = (current_player_id + i) % 4
        target_player = game.players[relative_id]
        for card in target_player.hand:
            fill_label(card, i)

    return label


def main():
    # 1. 加载配置 (显式指定 UTF-8 编码)
    config_path = 'rlcard/agents/mahjong_mcts/constraint.yaml'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return

    set_seed(cfg.get('seed', 42))
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    env = rlcard.make('mahjong')

    # 2. 初始化专家模型 (Teacher - 剪枝网络)
    teacher = MahjongBrain(num_actions=38).to(device)
    if os.path.exists(cfg['teacher_model_path']):
        # 加载本地模型快照
        ckpt = torch.load(cfg['teacher_model_path'], map_location=device)
        # 注意：这里适配你 main.py 中保存的 'model_state_dict' 键名
        teacher.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        print(f"✅ 专家模型已加载并进入评估模式: {cfg['teacher_model_path']}")

    teacher.eval()  # 核心：锁定老师参数，不构建计算图

    # 3. 初始化约束网络 (Student)
    student = MahjongConstraintNet(res_blocks=cfg['train']['res_blocks']).to(device)
    optimizer = optim.Adam(student.parameters(), lr=cfg['train']['learning_rate'])
    criterion = torch.nn.BCELoss()  # 监督学习常用损失函数

    data_pool = ConstraintDataPool(cfg['data_pool_capacity'])

    print(f"🚀 开始采集与训练：共 {cfg['num_round']} 轮，每轮博弈 {cfg['round_games']} 局")

    # --- 主训练循环 ---
    for r in range(1, cfg['num_round'] + 1):

        # A阶段：数据采集 (由专家模型主导博弈)
        with torch.no_grad():  # 核心：关闭梯度计算，节省资源
            for _ in range(cfg['round_games']):
                state, player_id = env.reset()
                while not env.is_over():
                    obs = state['obs']
                    legal_actions = list(state['legal_actions'].keys())

                    # 1. 提取与当前玩家视角一致的 0-1 标签
                    label = extract_view_rotated_labels(env.game, player_id)
                    data_pool.add(obs, label)

                    # 2. 专家决策推理 (应用动作掩码防止非法动作)
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    probs, _ = teacher(obs_t)
                    probs = probs[0].cpu().numpy()

                    # 处理动作掩码，确保博弈引擎不崩溃
                    masked_probs = np.zeros(38)
                    masked_probs[legal_actions] = probs[legal_actions]
                    if masked_probs.sum() <= 0:
                        masked_probs[legal_actions] = 1.0 / len(legal_actions)
                    else:
                        masked_probs /= masked_probs.sum()

                    action = np.random.choice(38, p=masked_probs)
                    state, player_id = env.step(action)

        # B阶段：约束网络训练 (复盘学习阶段)
        if len(data_pool) >= cfg['train']['batch_size']:
            losses = []
            for _ in range(cfg['train']['train_step']):
                # 从数据池随机抽取批量数据
                obs_b, label_b = data_pool.sample(cfg['train']['batch_size'])

                obs_b = torch.from_numpy(obs_b).float().to(device)
                label_b = torch.from_numpy(label_b).float().to(device)

                # 梯度更新
                optimizer.zero_grad()
                output = student(obs_b)  # 输出已在 forward 中经过 dim=1 的 Softmax
                loss = criterion(output, label_b)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # 实时监控训练进度
            print(f"Round {r:3d}/{cfg['num_round']} | Pool: {len(data_pool):5d} | Loss: {np.mean(losses):.6f}")

    # 4. 模型持久化
    os.makedirs(os.path.dirname(cfg['save_path']), exist_ok=True)
    torch.save(student.state_dict(), cfg['save_path'])
    print(f"💾 约束网络训练完成，模型已保存至: {cfg['save_path']}")


if __name__ == "__main__":
    main()