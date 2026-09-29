# Mahjong AI — CGI-ISMCTS

**[中文文档](README_CN.md)** | English

A deep reinforcement learning system for playing Riichi-style Mahjong, built on top of [RLCard](https://github.com/datamllab/rlcard). The core of the agent is Information Set Monte Carlo Tree Search (ISMCTS) guided by a neural network policy/value head, with an optional Constraint-Guided Inference (CGI) module that improves how the search samples from the information set.

This was developed as my undergraduate thesis project. The whole pipeline — training, inference, and replay visualization — is contained in this repo.

---

## How it works

Standard MCTS doesn't directly apply to Mahjong because players can't observe each other's hands. ISMCTS handles this by sampling possible world states (determinizations) at each node — i.e., guessing what the hidden tiles might be — and running simulations across that sample. The neural network guides search in two ways: the policy head prunes the action space, and the value head gives a faster terminal estimate without needing to play out full games.

The key limitation of vanilla ISMCTS is that it samples hidden information uniformly at random from all possibilities consistent with the observable state. This is quite different from how a human player reasons — a skilled player uses observable cues (discards, melds, game history) to form a much more informed estimate of what opponents are likely holding.

The CGI module addresses this. A separate constraint network is trained to predict the probability distribution over unobservable information (other players' hand tiles) conditioned on what can be observed. During search, world-state sampling is guided by this predicted distribution rather than drawn uniformly at random. The result is that the determinizations fed into MCTS are more realistic, leading to better-quality simulations.

RND (Random Network Distillation) is also supported as an optional intrinsic reward signal, which helped with exploration during early training when terminal rewards are sparse.

---

## Project layout

```
├── main.py                          # training entry point
├── play.py                          # run inference / generate a replay
├── render.py                        # generate an HTML replay viewer
├── play.yaml                        # inference config (fill in model paths)
│
├── rlcard/
│   ├── agents/mahjong_mcts/         # everything I wrote or significantly modified
│   │   ├── mcts.py                  # ISMCTS implementation with CGI-guided sampling
│   │   ├── network.py               # policy + value network
│   │   ├── constraint_network.py    # CGI constraint network
│   │   ├── rnd_network.py           # RND intrinsic reward module
│   │   ├── trainer.py               # training loop, optimizer, TensorBoard logging
│   │   ├── buffer.py                # replay buffer
│   │   ├── agent.py                 # agent interface
│   │   ├── config.yaml              # training hyperparameters
│   │   ├── reward.yaml              # reward shaping config
│   │   └── constraint.yaml          # CGI constraint rules
│   │
│   ├── envs/mahjong.py              # extended from RLCard (observation space, tenpai detection)
│   └── games/mahjong/               # extended from RLCard (added seven-pairs and kokushi rules)
│
├── replays/
│   └── sample_game.json             # example replay
└── riichi-mahjong-tiles-master/     # tile images (CC BY 4.0)
```

The rest of `rlcard/` is the original RLCard codebase.

---

## Getting started

```bash
git clone https://github.com/YOUR_USERNAME/mahjong-ai.git
cd mahjong-ai
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Train:**

```bash
python main.py
```

Key config options in `rlcard/agents/mahjong_mcts/config.yaml` (these are starting defaults — in practice most of these were adjusted throughout training based on observed behavior):

| Parameter | Default | Notes |
|---|---|---|
| `train.total_episodes` | 140000 | total self-play games |
| `train.round_game_num` | 50 | games per update round |
| `train.learning_rate` | 0.000025 | |
| `mcts.n_simulations` | 100 | simulations per move |
| `rnd.use_rnd` | false | enable intrinsic reward |

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs/
```

**Run inference and generate a replay:**

Edit `play.yaml` to point to your checkpoint:

```yaml
players:
  - id: 0
    pruning_model: 'mahjong_checkpoints/checkpoint_epXXXXX.pth'
```

Then:

```bash
python play.py
```

This saves a replay to `replays/latest_game.json`.

**View the replay:**

```bash
python render.py --replay replays/latest_game.json --output replay_viewer.html
```

Opens a self-contained HTML page in your browser. Use `←` / `→` to step through moves, `Space` to auto-play, and the dropdown to switch player perspectives.

---

## Reward shaping

Getting the reward function right took most of the iteration time. The final setup uses:

- Terminal win/loss reward
- Discard penalty when a player deals into another's winning hand
- Tenpai (ready hand) reward on a step-decay schedule — reaching tenpai earlier gives a larger bonus
- Per-step penalty that increases in later stages of the game to discourage stalling
- Draw penalty for players who aren't in tenpai when the game ends

The exact values are all in `reward.yaml` so they're easy to tune without touching the code.

---

## Training metrics (TensorBoard)

| Tag | Description |
|---|---|
| `Metric/RoundDrawRate` | draw rate over the last round |
| `Metric/AvgTingCount` | average number of players in tenpai per game |
| `Metric/AvgTingStep` | average step at which tenpai is reached |
| `HuType/Standard` | fraction of wins by standard hand |
| `HuType/SevenPairs` | fraction of wins by seven pairs |
| `HuType/Thirteen` | fraction of wins by kokushi musou |
| `RND/IntrinsicReward` | mean intrinsic reward (when RND is on) |

---

## Credits

Built on [RLCard](https://github.com/datamllab/rlcard) (MIT License):

> Zha, D., Lai, K. H., Cao, Y., Huang, S., Wei, R., Guo, J., & Hu, X. (2019). RLCard: A Toolkit for Reinforcement Learning in Card Games. *IJCAI 2019*.

Tile images from [riichi-mahjong-tiles](https://github.com/FluffyStuff/riichi-mahjong-tiles) (CC BY 4.0).
