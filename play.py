import torch
import yaml
import json
import os
import rlcard
import numpy as np
from rlcard.agents.mahjong_mcts.network import MahjongBrain
from rlcard.agents.mahjong_mcts.constraint_network import MahjongConstraintNet
from rlcard.agents.mahjong_mcts.mcts import ISMCTS
from datetime import datetime


def normalize_card(card):
    """统一将牌对象转成小写、去空格的标准名称"""
    if hasattr(card, 'get_str'):
        raw = card.get_str()
    else:
        raw = str(card)
    return raw.strip().lower().replace(" ", "").replace("\n", "")


def get_action_name(action_id):
    """将动作ID转换为可读名称"""
    action_names = {
        **{i: f"打出{i}" for i in range(34)},
        34: "碰牌",
        35: "吃牌",
        36: "杠牌",
        37: "过"
    }
    return action_names.get(action_id, f"动作{action_id}")


def get_state_value(agent, state, player_id):
    """
    获取当前状态下指定玩家的价值评估
    返回归一化到[0,1]的价值（用于绘制胜率曲线）
    """
    try:
        if hasattr(agent, 'pruning_net') and agent.pruning_net is not None:
            device = next(agent.pruning_net.parameters()).device

            if 'obs' in state:
                obs = state['obs']
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                else:
                    obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs, value = agent.pruning_net(obs_tensor)
                    player_value = value[0, 0].item()
                    normalized_value = 1.0 / (1.0 + np.exp(-player_value))
                    return float(normalized_value)

        return None
    except Exception as e:
        print(f"⚠️  获取状态价值失败: {e}")
        return None


def load_mcts_agent(p_cfg, device):
    pruning_net = MahjongBrain(num_actions=38).to(device)
    if p_cfg.get('pruning_model') and os.path.exists(p_cfg['pruning_model']):
        ckpt = torch.load(p_cfg['pruning_model'], map_location=device)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        pruning_net.load_state_dict(state_dict)
        print(f"✅ Player {p_cfg['id']} 剪枝网络加载成功")
    pruning_net.eval()

    constraint_net = None
    if p_cfg.get('cgi_on') and p_cfg.get('constraint_model'):
        target_res_blocks = 4
        constraint_net = MahjongConstraintNet(res_blocks=target_res_blocks).to(device)
        if os.path.exists(p_cfg['constraint_model']):
            try:
                ckpt_c = torch.load(p_cfg['constraint_model'], map_location=device)
                s_dict = ckpt_c['model_state_dict'] if isinstance(ckpt_c,
                                                                  dict) and 'model_state_dict' in ckpt_c else ckpt_c
                constraint_net.load_state_dict(s_dict)
                constraint_net.eval()
                print(f"✅ Player {p_cfg['id']} 约束网络加载成功 (ResBlocks={target_res_blocks})")
            except RuntimeError as e:
                print(f"❌ Constraint load failed: {e}")
                p_cfg['cgi_on'] = False
                constraint_net = None

    return ISMCTS(
        pruning_net=pruning_net,
        constraint_net=constraint_net,
        env=None,
        n_simulations=p_cfg.get('n_simulations', 100),
        cgi_on=p_cfg.get('cgi_on', False)
    )


def main():
    with open('play.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    env = rlcard.make('mahjong')

    agents = []
    for p_cfg in cfg['players']:
        if p_cfg['type'] == 'mcts':
            agents.append(load_mcts_agent(p_cfg, device))
        else:
            agents.append(None)

    history = []
    state, player_id = env.reset()
    print("🀄 对局开始!")
    step = 0

    # 初始状态记录
    initial_values = []
    for i in range(4):
        if agents[i] is not None:
            value = get_state_value(agents[i], state, i)
            initial_values.append(value)
        else:
            initial_values.append(None)

    initial_state = {
        'step': 0,
        'player_id': int(player_id),
        'action': None,
        'action_name': "开局",
        'discards': [[] for _ in range(4)],
        'all_hands': [[normalize_card(c) for c in p.hand] for p in env.game.players],
        'all_piles': [
            [normalize_card(c) for meld in p.pile for c in meld]
            for p in env.game.players
        ],
        'timestamp': datetime.now().isoformat(),
        'legal_actions': list(state['legal_actions'].keys()),
        'action_probs': None,
        'state_values': initial_values
    }
    history.append(initial_state)

    while not env.is_over():
        agent = agents[player_id]
        if agent is not None:
            agent.env = env
            action, probs = agent.get_action_probs(state)
        else:
            action = np.random.choice(list(state['legal_actions'].keys()))
            probs = np.zeros(38)

        def clean(card):
            return normalize_card(card)

        # 执行动作前获取所有玩家的状态价值
        current_values = []
        for i in range(4):
            if agents[i] is not None:
                value = get_state_value(agents[i], state, i)
                current_values.append(value)
            else:
                current_values.append(None)

        # 记录当前步骤的详细信息
        step_data = {
            'step': step + 1,
            'player_id': int(player_id),
            'action': int(action),
            'action_name': get_action_name(action),
            'discards': [
                [clean(card) for p_idx, card in env.action_recorder
                 if p_idx == i and hasattr(card, 'get_str')]
                for i in range(4)
            ],
            'all_hands': [[clean(c) for c in p.hand] for p in env.game.players],
            'all_piles': [
                [clean(c) for meld in p.pile for c in meld]
                for p in env.game.players
            ],
            'timestamp': datetime.now().isoformat(),
            'legal_actions': list(state['legal_actions'].keys()),
            'action_probs': probs.tolist() if isinstance(probs, np.ndarray) else None,
            'state_values': current_values
        }
        history.append(step_data)

        step += 1
        action_desc = get_action_name(action)
        print(f"Step {step}: Player {player_id} ({cfg['players'][player_id]['name']}) -> {action_desc}")

        state, player_id = env.step(action)

    # 🔧 新增：记录游戏结束后的最终状态
    def clean(card):
        return normalize_card(card)

    # 获取最终状态的价值
    final_values = []
    for i in range(4):
        if agents[i] is not None:
            try:
                value = get_state_value(agents[i], state, i)
                final_values.append(value)
            except:
                final_values.append(None)
        else:
            final_values.append(None)

    # 记录最终状态
    final_state = {
        'step': step + 1,
        'player_id': int(env.game.winner) if hasattr(env.game, 'winner') else -1,
        'action': None,
        'action_name': "游戏结束",
        'discards': [
            [clean(card) for p_idx, card in env.action_recorder
             if p_idx == i and hasattr(card, 'get_str')]
            for i in range(4)
        ],
        'all_hands': [[clean(c) for c in p.hand] for p in env.game.players],
        'all_piles': [
            [clean(c) for meld in p.pile for c in meld]
            for p in env.game.players
        ],
        'timestamp': datetime.now().isoformat(),
        'legal_actions': [],
        'action_probs': None,
        'state_values': final_values
    }
    history.append(final_state)
    print(f"Step {step + 1}: 游戏结束 - Winner: Player {env.game.winner if hasattr(env.game, 'winner') else -1}")

    payoffs = env.get_payoffs()
    print("\n🏁 对局结束!")
    for i, p in enumerate(cfg['players']):
        print(f"{p['name']}: {payoffs[i]:.2f}")

    # 保存完整的replay数据
    replay_data = {
        'metadata': {
            'version': '2.1',
            'timestamp': datetime.now().isoformat(),
            'total_steps': len(history) - 1,
            'seed': cfg.get('seed', None)
        },
        'config': cfg,
        'payoffs': payoffs.tolist(),
        'history': history
    }

    os.makedirs(os.path.dirname(cfg['replay_save_path']), exist_ok=True)
    with open(cfg['replay_save_path'], 'w', encoding='utf-8') as f:
        json.dump(replay_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Replay saved to {cfg['replay_save_path']}")


if __name__ == "__main__":
    main()