import rlcard
from rlcard.agents import MahjongMCTSAgent, RandomAgent
import torch

# 1. 创建环境 (记得开启 allow_step_back)
config = {'allow_step_back': True, 'seed': 42}
env = rlcard.make('mahjong', config=config)

# 2. 初始化你的 MCTS 智能体
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mcts_agent = MahjongMCTSAgent(env=env, device=device, n_simulations=50)

# 3. 设置其他对手 (初期可以用随机智能体，后期用自己训练出来的旧版本)
agents = [mcts_agent]
for _ in range(env.num_players - 1):
    agents.append(RandomAgent(num_actions=env.num_actions))

env.set_agents(agents)

# 4. 训练循环 (简化版专家迭代)
optimizer = torch.optim.Adam(mcts_agent.brain.parameters(), lr=1e-4)

for episode in range(1000):
    # 运行一局游戏获取轨迹
    trajectories, payoffs = env.run(is_training=True)

    # 这里你需要根据轨迹中的状态和最后的回报(payoffs)来训练价值头(Value Head)
    # 提示：毕设论文中这里叫 "Expert Iteration"
    # 你可以提取 trajectories 中的 obs 和对应的最终 payoff 进行 Backprop

    if episode % 10 == 0:
        print(f"Episode {episode} finished. Payoffs: {payoffs}")
        mcts_agent.save_checkpoint("mahjong_mcts.pth")