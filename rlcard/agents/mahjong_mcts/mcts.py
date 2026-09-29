import numpy as np
import torch
import copy
import random
from collections import defaultdict
from rlcard.games.mahjong.utils import card_encoding_dict


class MCTSNode:
    def __init__(self, prior_prob, parent=None):
        self.parent = parent
        self.children = {}  # action_id -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior_prob = prior_prob
        self.player_id = None  # 该节点的行动玩家 ID (绝对 ID)

    def value(self):
        """计算节点的平均价值 (Q值)"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct, legal_actions):
        """
        基于 PUCT 公式从合法动作中选择子节点
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None

        total_visits = sum(self.children[a].visit_count for a in legal_actions if a in self.children)

        for action in legal_actions:
            # 如果动作未在当前树中展开
            if action not in self.children:
                return action, None

            child = self.children[action]
            # PUCT = Q + U
            u_score = c_puct * child.prior_prob * np.sqrt(total_visits) / (1 + child.visit_count)
            score = child.value() + u_score

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child


class ISMCTS:
    def __init__(self, pruning_net, constraint_net, env, n_simulations=100, c_puct=1.5, cgi_on=True):
        self.pruning_net = pruning_net
        self.constraint_net = constraint_net
        self.env = env
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.cgi_on = cgi_on
        self.device = next(pruning_net.parameters()).device

    def get_action_probs(self, raw_state):
        """
        执行搜索并返回动作概率。
        新增逻辑：如果 n_simulations 为 0，则退化为纯神经网络推理模式。
        """
        # --- 核心新增逻辑：纯网络直接博弈 ---
        if self.n_simulations <= 0:
            obs_tensor = torch.from_numpy(raw_state['obs']).float().unsqueeze(0).to(self.device)
            legal_actions = list(raw_state['legal_actions'].keys())
            mask = torch.zeros(38, dtype=torch.bool).to(self.device)
            mask[legal_actions] = True

            with torch.no_grad():
                probs, _ = self.pruning_net(obs_tensor, action_mask=mask)

            pi = probs[0].cpu().numpy()
            best_action = np.argmax(pi)
            return best_action, pi
        # -----------------------------------

        # 初始根节点
        root = MCTSNode(prior_prob=1.0)

        # 预加载约束网络预测的位置分布
        prob_dist = None
        if self.cgi_on and self.constraint_net is not None:
            obs_tensor = torch.from_numpy(raw_state['obs']).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                prob_dist = self.constraint_net(obs_tensor)[0].cpu().numpy()

        for _ in range(self.n_simulations):
            # 1. 确定化过程 (Determinization)
            sim_env = self._determinize(self.env, prob_dist)

            # 2. 搜索选择阶段 (Selection)
            node = root
            search_path = [node]

            while not sim_env.is_over():
                curr_p_id = sim_env.get_player_id()
                state = sim_env.get_state(curr_p_id)
                legal_actions = list(state['legal_actions'].keys())

                action, next_node = node.select_child(self.c_puct, legal_actions)

                if next_node is None:
                    # 3. 扩展与评估
                    value_vector, leaf_p_id = self._evaluate_and_expand(node, sim_env, action)
                    break

                sim_env.step(action)
                node = next_node
                search_path.append(node)
            else:
                value_vector, leaf_p_id = self._evaluate_and_expand(node, sim_env, action=None)

            # 4. 视角回溯 (Backpropagation)
            self._backpropagate(search_path, value_vector, leaf_p_id)

        # 基于访问量生成决策概率
        counts = [root.children[a].visit_count if a in root.children else 0 for a in range(38)]
        sum_counts = sum(counts)
        if sum_counts == 0:
            probs = np.zeros(38)
            probs[list(raw_state['legal_actions'].keys())] = 1.0 / len(raw_state['legal_actions'])
        else:
            probs = np.array(counts) / sum_counts

        best_action = np.argmax(probs)
        return best_action, probs

    def _determinize(self, env, prob_dist):
        """生成确定性的完全信息环境副本"""
        sim_env = copy.deepcopy(env)
        game = sim_env.game
        root_p_id = game.get_player_id()

        unseen_cards = []
        unseen_cards.extend(game.dealer.deck)
        for p_id, p in enumerate(game.players):
            if p_id != root_p_id:
                unseen_cards.extend(p.hand)

        if not self.cgi_on or prob_dist is None:
            random.shuffle(unseen_cards)
        else:
            unseen_cards = self._probabilistic_sampling(unseen_cards, prob_dist, game, root_p_id)

        for p_id, p in enumerate(game.players):
            if p_id != root_p_id:
                num_needed = len(p.hand)
                p.hand = [unseen_cards.pop() for _ in range(num_needed)]
        game.dealer.deck = unseen_cards

        return sim_env

    def _probabilistic_sampling(self, cards, prob_dist, game, root_id):
        """执行概率受限的启发式采样"""
        slot_caps = [len(game.dealer.deck)]
        for i in range(1, 4):
            rel_id = (root_id + i) % 4
            slot_caps.append(len(game.players[rel_id].hand))

        type_counters = defaultdict(int)
        assigned = [[] for _ in range(4)]

        random.shuffle(cards)
        for card in cards:
            c_name = card.get_str().strip().lower().replace(" ", "")
            if c_name in card_encoding_dict:
                c_idx = card_encoding_dict[c_name]
                if c_idx >= 34: c_idx = 0

                inst_idx = type_counters[c_idx]
                p_vec = prob_dist[:, c_idx, inst_idx % 4].copy()

                for i in range(4):
                    if slot_caps[i] <= 0: p_vec[i] = 0

                if p_vec.sum() <= 0:
                    dest = np.random.choice([i for i, c in enumerate(slot_caps) if c > 0])
                else:
                    p_vec /= p_vec.sum()
                    dest = np.random.choice(4, p=p_vec)

                assigned[dest].append(card)
                slot_caps[dest] -= 1
                type_counters[c_idx] += 1
            else:
                dest = np.random.choice([i for i, c in enumerate(slot_caps) if c > 0])
                assigned[dest].append(card)
                slot_caps[dest] -= 1

        return assigned[1] + assigned[2] + assigned[3] + assigned[0]

    def _evaluate_and_expand(self, parent_node, sim_env, action):
        """评估当前状态并扩展指定动作"""
        if action is not None:
            sim_env.step(action)

        curr_p_id = sim_env.get_player_id()

        if sim_env.is_over():
            payoffs = sim_env.get_payoffs()
            return np.sign(np.roll(payoffs, -curr_p_id)), curr_p_id

        state = sim_env.get_state(curr_p_id)
        obs_t = torch.from_numpy(state['obs']).float().unsqueeze(0).to(self.device)
        legal_actions = list(state['legal_actions'].keys())
        mask = torch.zeros(38, dtype=torch.bool).to(self.device)
        mask[legal_actions] = True

        with torch.no_grad():
            probs, value = self.pruning_net(obs_t, action_mask=mask)

        probs = probs[0].cpu().numpy()
        value_vector = value[0].cpu().numpy()

        if action is not None and action not in parent_node.children:
            new_node = MCTSNode(prior_prob=probs[action], parent=parent_node)
            new_node.player_id = curr_p_id
            parent_node.children[action] = new_node

        return value_vector, curr_p_id

    def _backpropagate(self, path, value_vector, leaf_p_id):
        """沿着搜索路径更新相对价值向量"""
        for node in reversed(path):
            node.visit_count += 1
            if node.player_id is not None:
                rel_idx = (node.player_id - leaf_p_id) % 4
                node.value_sum += value_vector[rel_idx]