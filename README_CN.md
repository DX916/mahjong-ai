# 麻将 AI — CGI-ISMCTS

[English](README.md) | **中文文档**

基于 [RLCard](https://github.com/datamllab/rlcard) 开发的麻将深度强化学习系统。智能体的核心是信息集蒙特卡洛树搜索（ISMCTS）配合策略/价值神经网络，并引入约束指导推理（CGI）模块改进搜索过程中对信息集的采样方式。

这是我的本科毕业设计项目，从训练、推理到对局可视化回放的完整流程都在这个仓库里。

---

## 方法思路

麻将是不完全信息博弈，每个玩家看不到对手手牌，标准 MCTS 无法直接用。ISMCTS 的做法是在每个节点对可能的世界状态做采样（确定化），即对隐藏信息进行猜测，在这些采样上跑模拟，最后汇总统计结果。神经网络在这里起两个作用：策略头输出动作概率分布用于剪枝，价值头提供快速的局面估计，不需要把每局对弈玩到结束。

传统 ISMCTS 的一个核心局限在于：它从所有与可观测状态相容的可能世界中**均匀随机**采样隐藏信息。这与真实的人类推牌方式差距很大——有经验的玩家会根据弃牌记录、副露情况、出牌顺序等可见信息，对对手的手牌形成有根据的概率判断，而不是一视同仁地对待所有可能性。

CGI 模块针对的正是这个问题。约束网络独立训练，其任务是以当前可观测信息为条件，预测不可观测信息（对手手牌）的概率分布。搜索过程中，信息集的采样由这个预测分布引导，而不再均匀随机抽取。这样，输入 MCTS 的确定化世界状态更贴近实际的牌局分布，模拟质量随之提升，更接近人类推算牌局的思考方式。

训练阶段还支持 RND（随机网络蒸馏）作为内在奖励，在训练早期终局奖励稀疏的阶段有助于改善探索。

---

## 项目结构

```
├── main.py                          # 训练入口
├── play.py                          # 推理 / 生成回放
├── render.py                        # 生成 HTML 回放页面
├── play.yaml                        # 推理配置（填写模型路径后使用）
│
├── rlcard/
│   ├── agents/mahjong_mcts/         # 本项目新增/修改的核心模块
│   │   ├── mcts.py                  # ISMCTS 实现（含 CGI 引导采样）
│   │   ├── network.py               # 策略网络 + 价值网络
│   │   ├── constraint_network.py    # CGI 约束网络
│   │   ├── rnd_network.py           # RND 内在激励模块
│   │   ├── trainer.py               # 训练循环、优化器、TensorBoard 日志
│   │   ├── buffer.py                # 经验回放池
│   │   ├── agent.py                 # 智能体接口
│   │   ├── config.yaml              # 训练超参数
│   │   ├── reward.yaml              # 奖励函数配置
│   │   └── constraint.yaml          # CGI 约束规则
│   │
│   ├── envs/mahjong.py              # 在 RLCard 基础上扩展（观测空间、听牌检测）
│   └── games/mahjong/               # 在 RLCard 基础上扩展（新增七对子、十三幺判断）
│
├── replays/
│   └── sample_game.json             # 示例对局数据
└── riichi-mahjong-tiles-master/     # 牌面图片素材（CC BY 4.0）
```

`rlcard/` 下其余部分为 RLCard 原版代码。

---

## 快速上手

```bash
git clone https://github.com/YOUR_USERNAME/mahjong-ai.git
cd mahjong-ai
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**训练：**

```bash
python main.py
```

训练配置在 `rlcard/agents/mahjong_mcts/config.yaml`，以下为初始默认值，实际训练过程中会根据模型表现持续动态调整：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `train.total_episodes` | 140000 | 总训练局数 |
| `train.round_game_num` | 50 | 每轮更新对局数 |
| `train.learning_rate` | 0.000025 | 学习率 |
| `mcts.n_simulations` | 100 | 每步 MCTS 模拟次数 |
| `rnd.use_rnd` | false | 是否启用 RND 内在激励 |

TensorBoard 查看训练进度：

```bash
tensorboard --logdir logs/
```

**推理并生成回放：**

编辑 `play.yaml` 填写模型路径：

```yaml
players:
  - id: 0
    pruning_model: 'mahjong_checkpoints/checkpoint_epXXXXX.pth'
```

运行：

```bash
python play.py
```

对局数据保存到 `replays/latest_game.json`。

**查看回放：**

```bash
python render.py --replay replays/latest_game.json --output replay_viewer.html
```

自动在浏览器打开独立 HTML 页面，`←` / `→` 逐步浏览，空格键自动播放，下拉菜单切换玩家视角。

---

## 奖励函数设计

奖励函数调了很多轮，最终的方案：

- 终局胡牌/输牌奖励
- 点炮惩罚（自己出牌让对手胡牌）
- 听牌奖励，按阶梯衰减——越早听牌奖励越高
- 步数惩罚，随游戏进程加重，防止拖局
- 流局惩罚，未听牌时额外扣分

具体数值全部在 `reward.yaml` 里，调参不需要动代码。

---

## TensorBoard 训练指标

| 指标 | 说明 |
|---|---|
| `Metric/RoundDrawRate` | 当前轮流局率 |
| `Metric/AvgTingCount` | 每局平均听牌人数 |
| `Metric/AvgTingStep` | 平均听牌步数 |
| `HuType/Standard` | 标准型胡牌占比 |
| `HuType/SevenPairs` | 七对子胡牌占比 |
| `HuType/Thirteen` | 十三幺胡牌占比 |
| `RND/IntrinsicReward` | RND 内在奖励均值（启用时）|

---

## 致谢

本项目基于 [RLCard](https://github.com/datamllab/rlcard)（MIT License）开发：

> Zha, D., Lai, K. H., Cao, Y., Huang, S., Wei, R., Guo, J., & Hu, X. (2019). RLCard: A Toolkit for Reinforcement Learning in Card Games. *IJCAI 2019*.

麻将牌面图片来自 [riichi-mahjong-tiles](https://github.com/FluffyStuff/riichi-mahjong-tiles)（CC BY 4.0）。
