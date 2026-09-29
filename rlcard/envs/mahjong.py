import numpy as np
from collections import OrderedDict
from rlcard.envs import Env
from rlcard.games.mahjong import Game
from rlcard.games.mahjong import Card
from rlcard.games.mahjong.utils import card_encoding_dict, encode_cards, pile2list


class MahjongEnv(Env):
    def __init__(self, config):
        self.name = 'mahjong'
        self.game = Game()
        super().__init__(config)
        self.action_id = card_encoding_dict
        self.de_action_id = {self.action_id[key]: key for key in self.action_id.keys()}
        # 更新 state_shape 为 [15, 34, 4]
        self.state_shape = [[15, 34, 4] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _extract_state(self, state):
        from rlcard.games.mahjong.utils import encode_cards, pile2list
        from rlcard.games.mahjong.card import MahjongCard as Card

        current_player_id = self.get_player_id()
        planes = []
        # --- Plane 0: 自己手里的牌 ---
        hand_rep = encode_cards(state['current_hand'])
        planes.append(hand_rep)
        # --- Planes 1-4: 自己、下家、对家、上家的副露 (视角旋转) ---
        players_pile = state['players_pile']
        for i in range(4):
            rel_id = (current_player_id + i) % 4
            if rel_id in players_pile:
                planes.append(encode_cards(pile2list(players_pile[rel_id])))
            else:
                planes.append(np.zeros((34, 4), dtype=int))
        # --- Planes 5-8: 自己、下家、对家、上家分别已打出的所有牌 (从历史记录提取) ---
        discards_per_player = [[] for _ in range(4)]
        for p_id, action in self.action_recorder:
            if isinstance(action, Card):  # 只有打牌动作是 Card 对象
                discards_per_player[p_id].append(action)

        for i in range(4):
            rel_id = (current_player_id + i) % 4
            planes.append(encode_cards(discards_per_player[rel_id]))
        # --- Planes 9-12: 自己、下家、对家、上家上一次出的牌 (视角旋转) ---
        last_discards = [np.zeros((34, 4), dtype=int) for _ in range(4)]
        found_players = set()
        for p_id, action in reversed(self.action_recorder):
            if len(found_players) == 4: break
            if p_id not in found_players and isinstance(action, Card):
                rel_pos = (p_id - current_player_id) % 4
                last_discards[rel_pos] = encode_cards([action])
                found_players.add(p_id)
        planes.extend(last_discards)
        # --- Plane 13: 场上所有已经打出的牌 (Table Discards) ---
        table_rep = encode_cards(state['table'])
        planes.append(table_rep)
        # --- Plane 14: 当前视角下不知道的牌 (反推逻辑) ---
        # 初始假设全场 34 种牌各 4 张
        all_cards_const = np.ones((34, 4), dtype=int)
        # 减去已经暴露的牌：自己的手牌 + 全场副露 + 全场弃牌
        exposed_melds_total = np.zeros((34, 4), dtype=int)
        for p_id in players_pile:
            exposed_melds_total += encode_cards(pile2list(players_pile[p_id]))

        unseen_rep = all_cards_const - hand_rep - table_rep - exposed_melds_total
        # 确保不会出现负数（防止环境逻辑溢出）
        unseen_rep = np.maximum(unseen_rep, 0)
        planes.append(unseen_rep)
        # 拼接 15 个平面
        obs = np.array(planes)  # [15, 34, 4]
        # 保持听牌判定逻辑
        is_ting = self._check_ting(state['current_hand'], players_pile[current_player_id])
        return {
            'obs': obs,
            'is_ting': is_ting,
            'legal_actions': self._get_legal_actions(),
            'raw_obs': state,
            'raw_legal_actions': [a for a in state['action_cards']],
            'action_record': self.action_recorder
        }
    def _check_ting(self, hand, pile):
        """
        辅助函数：判定当前手牌是否已听牌
        """
        from rlcard.games.mahjong.card import MahjongCard

        # 建立一个 MockPlayer 适配判定接口
        class MockPlayer:
            def __init__(self, h, p):
                self.hand = h
                self.pile = p
                self.player_id = 0

        # 尝试摸入 34 种牌中的任何一张看是否能胡
        for card_str, card_id in self.action_id.items():
            if card_id < 34:
                ctype, ctrait = card_str.split('-')
                test_card = MahjongCard(ctype, ctrait)

                test_hand = hand + [test_card]
                mock_p = MockPlayer(test_hand, pile)

                win, _ = self.game.judger.judge_hu(mock_p)
                if win:
                    return True
        return False

    def get_payoffs(self):
        ''' Get the payoffs of players.
        '''
        _, player, _ = self.game.judger.judge_game(self.game)
        if player == -1:
            payoffs = [0, 0, 0, 0]
        else:
            payoffs = [-1, -1, -1, -1]
            payoffs[player] = 1
        return np.array(payoffs)

    def _decode_action(self, action_id):
        ''' Action id -> the action in the game.
        '''
        action = self.de_action_id[action_id]
        if action_id < 34:
            candidates = self.game.get_legal_actions(self.game.get_state(self.game.round.current_player))
            for card in candidates:
                if card.get_str() == action:
                    action = card
                    break
        return action

    def _get_legal_actions(self):
        ''' Get all legal actions for current state
        '''
        legal_action_id = {}
        legal_actions = self.game.get_legal_actions(self.game.get_state(self.game.round.current_player))
        if legal_actions:
            for action in legal_actions:
                if isinstance(action, Card):
                    action = action.get_str()
                action_id = self.action_id[action]
                legal_action_id[action_id] = None
        else:
            # 这里的打印是你在调试中留下的重要信息，已全部保留
            print("##########################")
            print("No Legal Actions")
            print(self.game.judger.judge_game(self.game))
            print(self.game.is_over())
            print([len(p.pile) for p in self.game.players])
        return OrderedDict(legal_action_id)
