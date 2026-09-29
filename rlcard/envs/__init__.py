''' Register environments (mahjong only)
'''
from rlcard.envs.env import Env
from rlcard.envs.registration import register, make

register(
    env_id='mahjong',
    entry_point='rlcard.envs.mahjong:MahjongEnv',
)

register(
    env_id='mahjong_org',
    entry_point='rlcard.envs.mahjong_org:MahjongOrgEnv',
)
