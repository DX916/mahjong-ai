# 路径: main.py
import yaml
import rlcard
import torch
import numpy as np
import os
import sys
import copy
import random
from collections import deque
from rlcard.agents.mahjong_mcts.network import MahjongBrain
from rlcard.agents.mahjong_mcts.agent import MahjongMCTSAgent
from rlcard.agents.mahjong_mcts.buffer import ReplayBuffer
from rlcard.agents.mahjong_mcts.trainer import MahjongTrainer
from rlcard.agents.mahjong_mcts.rnd_network import RNDModule


def main():
    # 1. 加载配置与奖励设定
    config_path = 'rlcard/agents/mahjong_mcts/config.yaml'
    reward_path = 'rlcard/agents/mahjong_mcts/reward.yaml'

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        with open(reward_path, 'r', encoding='utf-8') as f:
            reward_cfg = yaml.safe_load(f)
        print("✅ 配置文件与奖励规则加载成功")
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return

    # 2. 硬件与环境初始化
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"🚀 GPU 驱动加速中: {torch.cuda.get_device_name(0)}")

    # RND 配置加载
    use_rnd = config.get('rnd', {}).get('use_rnd', False)
    intrinsic_coeff = config.get('rnd', {}).get('intrinsic_reward_coeff', 0.5)
    rnd_normalize = config.get('rnd', {}).get('normalize_intrinsic', True)
    save_rnd_with_model = config.get('rnd', {}).get('save_rnd_with_model', True)
    rnd_checkpoint_dir = config.get('rnd', {}).get('rnd_checkpoint_dir', 'mahjong_checkpoints/rnd')
    print(f"🔬 RND内在激励: {'启用' if use_rnd else '禁用'} | 系数: {intrinsic_coeff}")

    env = rlcard.make('mahjong', config=config['env'])

    # 初始化训练主模型
    model = MahjongBrain(num_actions=38).to(device)

    # --- [核心：模型池初始化] ---
    pool_size = config['train'].get('model_pool_capacity', 50)
    model_pool = deque(maxlen=pool_size)
    model_pool.append(copy.deepcopy(model.state_dict()))

    # 预创建推理网络，减少循环内开销
    inference_nets = [MahjongBrain(num_actions=38).to(device) for _ in range(4)]
    for net in inference_nets:
        net.eval()
    # ---------------------------

    buffer = ReplayBuffer(config['train']['buffer_capacity'])
    buffer_save_path = config['train'].get('buffer_path', 'mahjong_checkpoints/replay_buffer.pkl')
    buffer.load_from_disk(buffer_save_path)

    trainer = MahjongTrainer(model, config)

    # 初始化 RND 模块（如果启用）
    rnd_module = None
    if use_rnd:
        rnd_config = config.get('rnd', {})
        rnd_output_dim = rnd_config.get('feature_dim', 128)
        rnd_learning_rate = rnd_config.get('rnd_learning_rate', 0.0001)

        rnd_module = RNDModule(
            device=str(device),
            output_dim=rnd_output_dim,
            learning_rate=rnd_learning_rate
        )
        print(f"✅ RND模块初始化完成 | 特征维度: {rnd_output_dim} | 学习率: {rnd_learning_rate}")

        # 尝试加载RND checkpoint
        load_rnd_path = rnd_config.get('load_rnd_checkpoint', '')
        if load_rnd_path and os.path.exists(load_rnd_path):
            rnd_module.load_checkpoint(load_rnd_path)
        else:
            if load_rnd_path:
                print(f"⚠️  指定的RND checkpoint不存在，将从头开始: {load_rnd_path}")
            print("🆕 RND从随机初始化开始训练")

    mcts_agent = MahjongMCTSAgent(env, device=device, n_simulations=config['mcts']['n_simulations'])
    mcts_agent.brain = model

    # 进度恢复
    start_episode = 1
    load_path = config.get('load_path', '')
    if load_path and os.path.exists(load_path):
        start_episode = mcts_agent.load_checkpoint(load_path, optimizer=trainer.optimizer)
        print(f"--- 已恢复进度，从 Episode {start_episode + 1} 开始 ---")
        start_episode += 1
        model_pool.append(copy.deepcopy(model.state_dict()))

    # 3. 训练参数预存
    total_eps = config['train']['total_episodes']
    round_size = config['train'].get('round_game_num', 20)
    gamma = config['train'].get('gamma', 0.98)
    use_mcts = config['train'].get('use_mcts_in_train', False)

    # 奖励数值缓存 (从 reward.yaml 读取)
    r_win = reward_cfg['final']['win']
    r_lose = reward_cfg['final']['lose']
    r_draw = reward_cfg['final']['draw']
    r_punish_not_ting = reward_cfg['final'].get('extra_punish_not_ting_on_draw', -3.0)
    r_punish_ting_no_win = reward_cfg['final'].get('punish_ting_but_not_win', -0.5)

    r_deal_in = reward_cfg['intermediate'].get('deal_in', -3.5)
    r_pong = reward_cfg['intermediate']['pong']
    r_chow = reward_cfg['intermediate']['chow']
    r_gong = reward_cfg['intermediate']['gong']

    # 特殊听牌奖励参数
    ting_decay = reward_cfg.get('special_tingpai', {}).get('decay', 0.90)
    ting_phases = reward_cfg.get('special_tingpai', {}).get('phases', [])

    # 动态步数惩罚配置
    step_phases = reward_cfg.get('penalty', {}).get('phases', [])

    # 统计量初始化
    total_draw = 0
    total_win = 0
    round_draw = 0
    round_win = 0
    round_steps_total = []

    # --- 听牌相关统计量 ---
    round_ting_count_total = 0
    round_ting_steps = []
    # ----------------------

    # --- 胡牌类型统计 ---
    round_hu_standard = 0  # 标准型（4面子+1对将）
    round_hu_seven_pairs = 0  # 七对子
    round_hu_thirteen = 0  # 十三幺
    total_hu_standard = 0
    total_hu_seven_pairs = 0
    total_hu_thirteen = 0
    # ----------------------

    # --- RND 相关统计 ---
    round_intrinsic_rewards = []  # 存储本Round的内在奖励
    # ----------------------

    print(f"--- 模式: 联赛训练 (2 Latest + 2 Pool) | Round: {round_size} | 池深度: {pool_size} ---")

    try:
        for episode in range(start_episode, total_eps + 1):

            # --- [对局前：分配模型] ---
            latest_players = random.sample([0, 1, 2, 3], 2)
            for p_id in range(4):
                if p_id in latest_players:
                    inference_nets[p_id].load_state_dict(model.state_dict())
                else:
                    inference_nets[p_id].load_state_dict(random.choice(model_pool))

            state, player_id = env.reset()
            trajectories = [[] for _ in range(4)]
            has_tinged = [False] * 4
            last_acting_player = None

            # --- [对局中：步进循环] ---
            step_count = 0
            while not env.is_over():
                current_obs = state['obs']
                current_player = player_id
                legal_actions = list(state['legal_actions'].keys())

                # A. 动作决策
                if use_mcts and current_player in latest_players:
                    action, pi = mcts_agent.mcts.get_action_probs(state)
                else:
                    obs_tensor = torch.from_numpy(current_obs).float().unsqueeze(0).to(device)
                    mask = torch.zeros(38, dtype=torch.bool).to(device)
                    mask[legal_actions] = True

                    with torch.no_grad():
                        probs, _ = inference_nets[current_player](obs_tensor, action_mask=mask)

                    pi_dist = probs[0].cpu().numpy()
                    action = np.random.choice(len(pi_dist), p=pi_dist)
                    pi = np.zeros(38)
                    pi = pi_dist.copy()

                # B. 执行动作
                last_acting_player = current_player
                state, player_id = env.step(action)
                step_count += 1

                # C. 计算中间奖励与听牌状态
                r_inter = 0.0
                is_first_ting = False
                r_intrinsic = 0.0  # 内在奖励

                if last_acting_player in latest_players:
                    if action == 34:
                        r_inter = r_pong
                    elif action == 35:
                        r_inter = r_chow
                    elif action == 36:
                        r_inter = r_gong

                    if reward_cfg['settings']['use_tingpai_reward'] and not has_tinged[last_acting_player]:
                        p_obj = env.game.players[last_acting_player]
                        if env._check_ting(p_obj.hand, p_obj.pile):
                            has_tinged[last_acting_player] = True
                            is_first_ting = True
                            # 统计听牌时的单人轨迹长度
                            round_ting_steps.append(len(trajectories[last_acting_player]) + 1)

                    # 计算内在奖励
                    if use_rnd and rnd_module is not None:
                        r_intrinsic_raw = rnd_module.compute_intrinsic_reward(current_obs)
                        if rnd_normalize:
                            r_intrinsic = rnd_module.normalize_intrinsic_reward(np.array([r_intrinsic_raw]))[0]
                        else:
                            r_intrinsic = r_intrinsic_raw
                        round_intrinsic_rewards.append(r_intrinsic)

                    trajectories[last_acting_player].append({
                        'obs': current_obs,
                        'pi': pi,
                        'r_inter': r_inter,
                        'r_intrinsic': r_intrinsic,
                        'is_first_ting': is_first_ting
                    })

                elif not has_tinged[last_acting_player]:
                    p_obj = env.game.players[last_acting_player]
                    if env._check_ting(p_obj.hand, p_obj.pile):
                        has_tinged[last_acting_player] = True
                        round_ting_steps.append(step_count // 4)

            # --- [对局后：结算与计算 Z] ---
            payoffs = env.get_payoffs()
            is_draw = np.all(payoffs == 0)

            # 累计听牌总人数
            round_ting_count_total += sum(has_tinged)

            if is_draw:
                total_draw += 1
                round_draw += 1
                final_rewards = np.array([r_draw] * 4)
            else:
                total_win += 1
                round_win += 1
                winner_id = np.argmax(payoffs)
                final_rewards = np.array([r_lose] * 4)
                final_rewards[winner_id] = r_win

                # 点炮惩罚
                if reward_cfg['settings']['use_deal_in_penalty'] and winner_id != last_acting_player:
                    if last_acting_player in latest_players and len(trajectories[last_acting_player]) > 0:
                        trajectories[last_acting_player][-1]['r_inter'] += r_deal_in

            round_steps_total.append(step_count)

            # --- 检测胡牌类型 ---
            if not is_draw:
                winner_id = np.argmax(payoffs)
                winner_player = env.game.players[winner_id]

                # 调用judge_hu获取胡牌类型（通过返回值判断）
                win, val = env.game.judger.judge_hu(winner_player)

                if win:
                    if val == 13:
                        # 十三幺
                        round_hu_thirteen += 1
                        total_hu_thirteen += 1
                    elif val == 7:
                        # 七对子
                        round_hu_seven_pairs += 1
                        total_hu_seven_pairs += 1
                    else:
                        # 标准型（val通常是4或其他值）
                        round_hu_standard += 1
                        total_hu_standard += 1
            # ----------------------

            # 5. 回报计算 (4个轨迹独立)
            for i in latest_players:
                traj_len = len(trajectories[i])

                # A. 确定该玩家专属的终局分
                final_payout_i = final_rewards.copy()
                if is_draw:
                    if not has_tinged[i]:
                        final_payout_i[i] += r_punish_not_ting
                else:
                    winner_id = np.argmax(payoffs)
                    if i != winner_id and has_tinged[i]:
                        final_payout_i[i] += r_punish_ting_no_win

                # B. 5 段分段步数惩罚（对所有玩家都计算）
                cum_step_penalties = []
                temp_cum = 0.0
                for step_num in range(1, traj_len + 1):
                    p_val = 0.0
                    for ph in step_phases:
                        if step_num <= ph['end']:
                            p_val = ph['val']
                            break
                    temp_cum += p_val
                    cum_step_penalties.append(temp_cum)

                # C. 听牌阶梯奖励与衰减
                ting_step_idx = -1
                ting_base_reward = 0.0
                for idx, data in enumerate(trajectories[i]):
                    if data.get('is_first_ting', False):
                        ting_step_idx = idx
                        for ph in ting_phases:
                            if (idx + 1) <= ph['end']:
                                ting_base_reward = ph['reward']
                                break
                        break

                # D. 生成 Z 并保存
                for t, data in enumerate(trajectories[i]):
                    steps_to_end = traj_len - 1 - t
                    z_vector = (final_payout_i * (gamma ** steps_to_end)).copy()

                    # 叠加步数罚和中间奖
                    z_vector[i] += cum_step_penalties[t]
                    z_vector[i] += data['r_inter']

                    # 叠加特殊听牌奖励衰减
                    if ting_step_idx != -1 and t <= ting_step_idx:
                        steps_back = ting_step_idx - t
                        z_vector[i] += ting_base_reward * (ting_decay ** steps_back)

                    # 叠加内在奖励
                    if use_rnd:
                        z_vector[i] += data.get('r_intrinsic', 0.0) * intrinsic_coeff

                    buffer.save(data['obs'], data['pi'], np.roll(z_vector, -i))

            # --- [6. Round 模型更新与打印] ---
            if episode % round_size == 0:
                p_avg, v_avg, e_avg = 0.0, 0.0, 0.0
                rnd_loss_avg = 0.0
                if len(buffer) > config['train']['batch_size']:
                    p_ls, v_ls, ents = [], [], []
                    rnd_losses = []
                    for _ in range(config['train']['train_steps_per_episode']):
                        pl, vl, ent = trainer.train_step(buffer)
                        p_ls.append(pl)
                        v_ls.append(vl)
                        ents.append(ent)

                        # 更新 RND 预测网络
                        if use_rnd and rnd_module is not None:
                            s_batch, _, _ = buffer.sample(config['train']['batch_size'])
                            rnd_loss = rnd_module.update(s_batch)
                            rnd_losses.append(rnd_loss)

                    p_avg = np.mean(p_ls)
                    v_avg = np.mean(v_ls)
                    e_avg = np.mean(ents)
                    if rnd_losses:
                        rnd_loss_avg = np.mean(rnd_losses)

                # 更新模型池
                model_pool.append(copy.deepcopy(model.state_dict()))

                # 统计计算
                total_count = total_win + total_draw
                total_dr = total_draw / total_count if total_count > 0 else 0
                round_dr = round_draw / round_size
                avg_steps = np.mean(round_steps_total)
                avg_ting_count = round_ting_count_total / round_size
                avg_ting_step = np.mean(round_ting_steps) if round_ting_steps else 0.0

                # RND 统计
                avg_intrinsic_reward = np.mean(round_intrinsic_rewards) if round_intrinsic_rewards else 0.0

                # 胡牌类型统计
                round_win_count = round_win  # 这个Round的胜局数
                if round_win_count > 0:
                    round_std_rate = round_hu_standard / round_win_count
                    round_7p_rate = round_hu_seven_pairs / round_win_count
                    round_13_rate = round_hu_thirteen / round_win_count
                else:
                    round_std_rate = round_7p_rate = round_13_rate = 0.0

                total_win_count = total_win
                if total_win_count > 0:
                    total_std_rate = total_hu_standard / total_win_count
                    total_7p_rate = total_hu_seven_pairs / total_win_count
                    total_13_rate = total_hu_thirteen / total_win_count
                else:
                    total_std_rate = total_7p_rate = total_13_rate = 0.0

                # 记录 TensorBoard
                trainer.log_custom_scalar('Metric/RoundDrawRate', round_dr, episode)
                trainer.log_custom_scalar('Metric/TotalDrawRate', total_dr, episode)
                trainer.log_custom_scalar('Metric/AvgSteps', avg_steps, episode)
                trainer.log_custom_scalar('Metric/AvgTingCount', avg_ting_count, episode)
                trainer.log_custom_scalar('Metric/AvgTingStep', avg_ting_step, episode)
                trainer.log_custom_scalar('HuType/Standard', round_std_rate, episode)
                trainer.log_custom_scalar('HuType/SevenPairs', round_7p_rate, episode)
                trainer.log_custom_scalar('HuType/Thirteen', round_13_rate, episode)

                # RND 指标记录
                if use_rnd:
                    trainer.log_custom_scalar('RND/IntrinsicReward', avg_intrinsic_reward, episode)
                    trainer.log_custom_scalar('RND/PredictorLoss', rnd_loss_avg, episode)
                    if rnd_module is not None:
                        trainer.log_custom_scalar('RND/IntrinsicMean', rnd_module.intrinsic_reward_mean, episode)
                        trainer.log_custom_scalar('RND/IntrinsicStd', rnd_module.intrinsic_reward_std, episode)

                # 终端打印 - 第一行
                rnd_str = f" | RND:{avg_intrinsic_reward:5.3f}" if use_rnd else ""
                print(f"E{episode:5d} | R-Draw:{round_dr:5.1%} | T-Draw:{total_dr:5.2%} | "
                      f"AvgTing:{avg_ting_count:4.2f} | T-Step:{avg_ting_step:4.1f} | "
                      f"Steps:{avg_steps:4.2f} | P-Loss:{p_avg:7.3f} | V-Loss:{v_avg:7.3f} | Entropy:{e_avg:7.3f}{rnd_str}")

                # 终端打印 - 第二行（胡牌类型）
                print(f"         | HuType - Std:{round_std_rate:5.1%}({round_hu_standard}/{round_win_count}) | "
                      f"7Pair:{round_7p_rate:5.1%}({round_hu_seven_pairs}/{round_win_count}) | "
                      f"13Orph:{round_13_rate:5.1%}({round_hu_thirteen}/{round_win_count}) | ")

                # 重置 Round 统计量
                round_draw = 0
                round_win = 0
                round_steps_total = []
                round_ting_count_total = 0
                round_ting_steps = []
                round_hu_standard = 0
                round_hu_seven_pairs = 0
                round_hu_thirteen = 0
                round_intrinsic_rewards = []

                # 定期保存 Checkpoint
                save_interval = config['train'].get('save_every', 5000)
                if episode % save_interval == 0:
                    # 保存主模型
                    ckpt_path = os.path.join(config['save_path'], f"checkpoint_ep{episode}.pth")
                    mcts_agent.save_checkpoint(ckpt_path, trainer.optimizer, episode)
                    buffer.save_to_disk(buffer_save_path)

                    # 保存RND checkpoint
                    if use_rnd and rnd_module is not None and save_rnd_with_model:
                        # 确保RND checkpoint目录存在
                        if not os.path.exists(rnd_checkpoint_dir):
                            os.makedirs(rnd_checkpoint_dir)

                        rnd_ckpt_path = os.path.join(rnd_checkpoint_dir, f"rnd_ep{episode}.pth")
                        rnd_module.save_checkpoint(rnd_ckpt_path)

    except KeyboardInterrupt:
        print("\n--- 训练被手动中断 ---")
        # 保存主模型
        mcts_agent.save_checkpoint(os.path.join(config['save_path'], "latest.pth"), trainer.optimizer, episode)
        buffer.save_to_disk(buffer_save_path)

        # 保存RND checkpoint
        if use_rnd and rnd_module is not None and save_rnd_with_model:
            if not os.path.exists(rnd_checkpoint_dir):
                os.makedirs(rnd_checkpoint_dir)
            rnd_ckpt_path = os.path.join(rnd_checkpoint_dir, "rnd_latest.pth")
            rnd_module.save_checkpoint(rnd_ckpt_path)

    finally:
        trainer.close()

if __name__ == "__main__":
    main()