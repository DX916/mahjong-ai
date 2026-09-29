# -*- coding: utf-8 -*-
''' Implement Mahjong Judger class
'''
from collections import defaultdict
import numpy as np


class MahjongJudger:
    ''' Determine what cards a player can play
    '''

    def __init__(self, np_random):
        ''' Initilize the Judger class for Mahjong
        '''
        self.np_random = np_random

    @staticmethod
    def judge_pong_gong(dealer, players, last_player):
        ''' Judge which player has pong/gong
        Args:
            dealer (object): The dealer object.
            players (list): List of all players
            last_player (int): The player id of last player

        '''
        last_card = dealer.table[-1]
        last_card_str = last_card.get_str()
        for player in players:
            hand = [card.get_str() for card in player.hand]
            hand_dict = defaultdict(list)
            for card in hand:
                hand_dict[card.split("-")[0]].append(card.split("-")[1])
            # check gong
            if hand.count(last_card_str) == 3 and last_player != player.player_id:
                return 'gong', player, [last_card] * 4
            # check pong
            if hand.count(last_card_str) == 2 and last_player != player.player_id:
                return 'pong', player, [last_card] * 3
        return False, None, None

    def judge_chow(self, dealer, players, last_player):
        ''' Judge which player has chow
        Args:
            dealer (object): The dealer object.
            players (list): List of all players
            last_player (int): The player id of last player
        '''

        last_card = dealer.table[-1]
        last_card_str = last_card.get_str()
        last_card_type = last_card_str.split("-")[0]
        last_card_index = last_card.index_num
        for player in players:
            if last_card_type != "dragons" and last_card_type != "winds" and last_player == player.get_player_id() - 1:
                hand_list = np.zeros(9)

                for card in player.hand:
                    if card.get_str().split("-")[0] == last_card_type:
                        hand_list[card.index_num] = hand_list[card.index_num] + 1

                # check chow - 检查所有可能的吃牌组合
                test_cases = []

                # 修正：检查所有可能的吃牌组合
                if last_card_index >= 2:  # 可以作为顺子的第三张 (如 [1,2,3] 中的3)
                    if hand_list[last_card_index - 2] > 0 and hand_list[last_card_index - 1] > 0:
                        test_cases.append([last_card_index - 2, last_card_index - 1])

                if last_card_index >= 1 and last_card_index <= 7:  # 可以作为中间张
                    if hand_list[last_card_index - 1] > 0 and hand_list[last_card_index + 1] > 0:
                        test_cases.append([last_card_index - 1, last_card_index + 1])

                if last_card_index <= 6:  # 可以作为第一张
                    if hand_list[last_card_index + 1] > 0 and hand_list[last_card_index + 2] > 0:
                        test_cases.append([last_card_index + 1, last_card_index + 2])

                if not test_cases:
                    continue

                for l in test_cases:
                    cards = []
                    for i in l:
                        for card in player.hand:
                            if card.index_num == i and card.get_str().split("-")[0] == last_card_type:
                                cards.append(card)
                                break
                    cards.append(last_card)
                    return 'chow', player, cards
        return False, None, None

    def judge_game(self, game):
        ''' Judge which player has win the game
        Args:
            dealer (object): The dealer object.
            players (list): List of all players
            last_player (int): The player id of last player
        '''
        players_val = []
        win_player = -1
        for player in game.players:
            win, val = self.judge_hu(player)
            players_val.append(val)
            if win and win_player == -1:  # 只记录第一个赢家
                win_player = player.player_id

        # 【关键修复1】检查是否有玩家处于无法继续的状态（4个副露问题）
        for player in game.players:
            pile_count = len(player.pile)
            hand_count = len(player.hand)
            total_count = hand_count + sum(len(meld) for meld in player.pile)

            # 情况1：有4个副露但总牌数不是14张 → 异常状态，强制流局
            if pile_count >= 4 and total_count != 14:
                return True, -1, players_val

            # 情况2：有4个副露且总牌数是14张
            if pile_count >= 4 and total_count == 14:
                # 手牌应该是2张
                if hand_count != 2:
                    # 手牌不是2张 → 异常状态，强制流局
                    return True, -1, players_val
                elif hand_count == 2:
                    # 检查是否是一对
                    hand_strs = [c.get_str() for c in player.hand]
                    if len(set(hand_strs)) != 1:
                        # 不是一对 → 死锁状态，强制流局
                        return True, -1, players_val

            # 情况3：有4个副露但手牌数异常（任何<=1张的情况）
            if pile_count >= 4 and hand_count <= 1:
                return True, -1, players_val

        # 【关键修复2】检查牌数异常情况
        for player in game.players:
            total_tiles = len(player.hand) + sum(len(meld) for meld in player.pile)
            # 如果任何玩家的牌数不是13或14张，且牌库还有牌，说明有bug
            if total_tiles < 13 and len(game.dealer.deck) > 0:
                # 强制流局，避免崩溃
                return True, -1, players_val

        if win_player != -1 or len(game.dealer.deck) == 0:
            return True, win_player, players_val
        else:
            return False, win_player, players_val

    def judge_hu(self, player):
        ''' Judge whether the player has win the game
        Args:
            player (object): Target player
        Return:
            Result (bool): Win or not
            Maximum_score (int): Set count score of the player
        '''
        # 标准麻将胡牌：需要14张牌（手牌+副露）
        total_tiles = len(player.hand) + sum(len(meld) for meld in player.pile)

        # 如果总牌数不是14张，肯定不能胡
        if total_tiles != 14:
            return False, 0

        hand = [card.get_str() for card in player.hand]
        set_count = len(player.pile)

        # 【新增1】七对子检查（只能在无副露时）
        if set_count == 0 and len(hand) == 14:
            if self._check_seven_pairs(hand):
                return True, 7  # 返回7表示七对子

        # 【新增2】十三幺检查（只能在无副露时）
        if set_count == 0 and len(hand) == 14:
            if self._check_thirteen_orphans(hand):
                return True, 13  # 返回13表示十三幺

        # 【标准型】如果已经有4组副露，手牌应该是2张且是一对
        if set_count >= 4:
            # 标准情况：手牌恰好2张且是一对
            if len(hand) == 2 and len(set(hand)) == 1:
                return True, set_count
            else:
                # 任何其他情况都不能胡牌
                return False, set_count

        # 【标准型】正常情况：计算手牌能组成多少组面子
        count_dict = {card: hand.count(card) for card in hand}
        used = []
        maximum = 0
        for each in count_dict:
            if each in used:
                continue
            tmp_set_count = 0
            tmp_hand = hand.copy()
            if count_dict[each] >= 2:
                # 移除将牌
                tmp_hand.remove(each)
                tmp_hand.remove(each)
                # 计算剩余手牌能组成多少组面子
                tmp_set_count, _set = self.cal_set(tmp_hand)
                used.extend(_set)
                if tmp_set_count + set_count > maximum:
                    maximum = tmp_set_count + set_count
                # 标准胡牌：4组面子 + 1对将
                if tmp_set_count + set_count >= 4:
                    return True, maximum

        return False, maximum

    def _check_seven_pairs(self, hand):
        """
        检查七对子牌型
        Args:
            hand (list): 手牌字符串列表
        Return:
            bool: 是否为七对子
        """
        count_dict = {}
        for card in hand:
            count_dict[card] = count_dict.get(card, 0) + 1

        # 必须恰好7种牌，每种2张
        if len(count_dict) != 7:
            return False

        for count in count_dict.values():
            if count != 2:
                return False

        return True

    def _check_thirteen_orphans(self, hand):
        """
        检查十三幺（国士无双）牌型
        Args:
            hand (list): 手牌字符串列表
        Return:
            bool: 是否为十三幺
        """
        # 定义13种幺九牌
        orphans = [
            'characters-1', 'characters-9',
            'bamboo-1', 'bamboo-9',
            'dots-1', 'dots-9',
            'winds-east', 'winds-south', 'winds-west', 'winds-north',
            'dragons-red', 'dragons-green', 'dragons-white'
        ]

        count_dict = {}
        for card in hand:
            count_dict[card] = count_dict.get(card, 0) + 1

        # 检查是否只包含幺九牌
        for card in count_dict.keys():
            if card not in orphans:
                return False

        # 必须包含全部13种幺九牌
        for orphan in orphans:
            if orphan not in count_dict:
                return False

        # 其中一种有2张（作为将牌），其余各1张
        pair_count = sum(1 for c in count_dict.values() if c == 2)
        single_count = sum(1 for c in count_dict.values() if c == 1)

        return pair_count == 1 and single_count == 12

    @staticmethod
    def check_consecutive(_list):
        ''' Check if list is consecutive
        Args:
            _list (list): The target list

        Return:
            Result (bool): consecutive or not
        '''
        l = list(map(int, _list))
        if sorted(l) == list(range(min(l), max(l) + 1)):
            return True
        return False

    def cal_set(self, cards):
        ''' Calculate the set for given cards
        Args:
            Cards (list): List of cards.

        Return:
            Set_count (int):
            Sets (list): List of cards that has been pop from user's hand
        '''
        tmp_cards = cards.copy()
        sets = []
        set_count = 0
        _dict = {card: tmp_cards.count(card) for card in tmp_cards}

        # 1. 优先检查刻子/杠子（3或4张相同）
        for each in _dict:
            if _dict[each] == 3 or _dict[each] == 4:
                set_count += 1
                for _ in range(_dict[each]):
                    tmp_cards.pop(tmp_cards.index(each))

        # 2. 检查顺子（修正版：避免在循环中修改列表）
        _dict_by_type = defaultdict(list)
        for card in tmp_cards:
            _type = card.split("-")[0]
            _trait = card.split("-")[1]
            if _type == 'dragons' or _type == 'winds':
                continue
            else:
                _dict_by_type[_type].append(_trait)

        # 修正：对每种花色单独处理
        for _type in _dict_by_type.keys():
            values = sorted(_dict_by_type[_type])

            # 使用 while 循环，不断从头查找顺子
            while len(values) >= 3:
                found_sequence = False

                # 从头开始找第一个顺子
                for i in range(len(values) - 2):
                    test_case = [values[i], values[i + 1], values[i + 2]]
                    if self.check_consecutive(test_case):
                        set_count += 1
                        # 移除这3张牌
                        for each in test_case:
                            values.remove(each)
                            c = _type + "-" + str(each)
                            sets.append(c)
                        found_sequence = True
                        break  # 找到一个顺子后重新开始

                # 如果没找到任何顺子，退出循环
                if not found_sequence:
                    break

        return set_count, sets