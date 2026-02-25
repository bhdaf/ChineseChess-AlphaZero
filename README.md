# 中国象棋Zero（CCZero）

<a href="https://cczero.org">
  <img src="https://cczero.org/zero2.png" alt="App Icon" />
</a>

## About

Chinese Chess reinforcement learning by [AlphaZero](https://arxiv.org/abs/1712.01815) methods.

This project is based on these main resources:
1. DeepMind's Oct 19th publication: [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ).
2. The **great** Reversi/Chess/Chinese chess development of the DeepMind ideas that @mokemokechicken/@Akababa/@TDteach did in their repo: https://github.com/mokemokechicken/reversi-alpha-zero, https://github.com/Akababa/Chess-Zero, https://github.com/TDteach/AlphaZero_ChineseChess
3. A Chinese chess engine with gui: https://github.com/mm12432/MyChess


## Help to train

In order to build a strong chinese chess AI following the same type of techniques as AlphaZero, we need to do this with a distributed project, as it requires a huge amount of computations.

If you want to join us to build the best chinese chess AI in the world:

* For instructions, see [wiki](https://github.com/NeymarL/ChineseChess-AlphaZero/wiki)
* For live status, see https://cczero.org

![elo](elo.png)


## Environment

* Python 3.6.3
* tensorflow-gpu: 1.3.0
* Keras: 2.0.8


## Modules

### Reinforcement Learning

This AlphaZero implementation consists of two workers: `self` and  `opt`.

* `self` is Self-Play to generate training data by self-play using BestModel.
* `opt` is Trainer to train model, and generate new models.

For the sake of faster training, another two workers are involved:

* `sl` is Supervised learning to train data crawled from the Internet.
* `eval` is Evaluator to evaluate the NextGenerationModel with the current BestModel.

### Built-in GUI

Requirement: pygame

```bash
python cchess_alphazero/run.py play
```

**Screenshots**

![board](screenshots/board.png)

You can choose different board/piece styles and sides, see [play with human](#play-with-human).


## How to use

### Setup

### install libraries
```bash
pip install -r requirements.txt
```

If you want to use CPU only, replace `tensorflow-gpu` with `tensorflow` in `requirements.txt`.

Make sure Keras is using Tensorflow and you have Python 3.6.3+.

### Configuration

**PlayDataConfig**

* `nb_game_in_file, max_file_num`: The max game number of training data is `nb_game_in_file * max_file_num`.

**PlayConfig, PlayWithHumanConfig**

* `simulation_num_per_move` : MCTS number per move.
* `c_puct`: balance parameter of value network and policy network in MCTS.
* `search_threads`: balance parameter of speed and accuracy in MCTS.
* `dirichlet_alpha`: random parameter in self-play.

### Full Usage

```
usage: run.py [-h] [--new] [--type TYPE] [--total-step TOTAL_STEP]
              [--ai-move-first] [--cli] [--gpu GPU] [--onegreen] [--skip SKIP]
              [--ucci] [--piece-style {WOOD,POLISH,DELICATE}]
              [--bg-style {CANVAS,DROPS,GREEN,QIANHONG,SHEET,SKELETON,WHITE,WOOD}]
              [--random {none,small,medium,large}] [--distributed] [--elo]
              {self,opt,eval,play,eval,sl,ob}

positional arguments:
  {self,opt,eval,play,eval,sl,ob}
                        what to do

optional arguments:
  -h, --help            show this help message and exit
  --new                 run from new best model
  --type TYPE           use normal setting
  --total-step TOTAL_STEP
                        set TrainerConfig.start_total_steps
  --ai-move-first       set human or AI move first
  --cli                 play with AI with CLI, default with GUI
  --gpu GPU             device list
  --onegreen            train sl work with onegreen data
  --skip SKIP           skip games
  --ucci                play with ucci engine instead of self play
  --piece-style {WOOD,POLISH,DELICATE}
                        choose a style of piece
  --bg-style {CANVAS,DROPS,GREEN,QIANHONG,SHEET,SKELETON,WHITE,WOOD}
                        choose a style of board
  --random {none,small,medium,large}
                        choose a style of randomness
  --distributed         whether upload/download file from remote server
  --elo                 whether to compute elo score
```

### Self-Play

```
python cchess_alphazero/run.py self
```

When executed, self-play will start using BestModel. If the BestModel does not exist, new random model will be created and become BestModel. Self-play records will store in `data/play_record` and BestMode will store in `data/model`.

options

* `--new`: create new BestModel
* `--type mini`: use mini config, (see `cchess_alphazero/configs/mini.py`)
* `--gpu '1'`: specify which gpu to use
* `--ucci`: whether to play with ucci engine (rather than self play, see `cchess_alphazero/worker/play_with_ucci_engine.py`)
* `--distributed`: run self play in distributed mode which means it will upload the play data to the remote server and download latest model from it

**Note1**: To help training, you should run `python cchess_alphazero/run.py --type distribute --distributed self` (and do not change the configuration file `configs/distribute.py`), for more info, see [wiki](https://github.com/NeymarL/ChineseChess-AlphaZero/wiki/For-Developers).

**Note2**: If you want to view the self-play records in GUI, see [wiki](https://github.com/NeymarL/ChineseChess-AlphaZero/wiki/View-self-play-games-in-GUI).

### Trainer

```
python cchess_alphazero/run.py opt
```

When executed, Training will start. The current BestModel will be loaded. Trained model will be saved every epoch as new BestModel.

options

* `--type mini`: use mini config, (see `cchess_alphazero/configs/mini.py`)
* `--total-step TOTAL_STEP`: specify total step(mini-batch) numbers. The total step affects learning rate of training.
* `--gpu '1'`: specify which gpu to use

**View training log in Tensorboard**

```
tensorboard --logdir logs/
```

And access `http://<The Machine IP>:6006/`.

### Play with human

**Run with built-in GUI**

```
python cchess_alphazero/run.py play
```

When executed, the BestModel will be loaded to play against human.

options

* `--ai-move-first`: if set this option, AI will move first, otherwise human move first.
* `--type mini`: use mini config, (see `cchess_alphazero/configs/mini.py`)
* `--gpu '1'`: specify which gpu to use
* `--piece-style WOOD`: choose a piece style, default is `WOOD`
* `--bg-style CANVAS`: choose a board style, default is `CANVAS`
* `--cli`: if set this flag, play with AI in a cli environment rather than gui

**Note**: Before you start, you need to download/find a font file (`.ttc`) and rename it as `PingFang.ttc`, then put it into `cchess_alphazero/play_games`. I have removed the font file from this repo because it's too big, but you can download it from [here](http://alphazero.52coding.com.cn/PingFang.ttc).

You can also download Windows executable directly from [here](https://pan.baidu.com/s/1uE_zmkn0x9Be_olRL9U9cQ). For more information, see [wiki](https://github.com/NeymarL/ChineseChess-AlphaZero/wiki/For-Non-Developers#%E4%B8%8B%E6%A3%8B).

**UCI mode**

```
python cchess_alphazero/uci.py
```

If you want to play in general GUIs such as '冰河五四', you can download the Windows executable [here](https://share.weiyun.com/5cK50Z4). For more information, see [wiki](https://github.com/NeymarL/ChineseChess-AlphaZero/wiki/For-Non-Developers#%E4%B8%8B%E6%A3%8B).

### Evaluator

```
python cchess_alphazero/run.py eval
```

When executed, evaluate the NextGenerationModel with the current BestModel. If the NextGenerationModel does not exist, worker will wait until it exists and check every 5 minutes.

options

* `--type mini`: use mini config, (see `cchess_alphazero/configs/mini.py`)
* `--gpu '1'`: specify which gpu to use

### Supervised Learning

```
python cchess_alphazero/run.py sl
```

When executed, Training will start. The current SLBestModel will be loaded. Tranined model will be saved every epoch as new SLBestModel.

*About the data*

I have two data sources, one is downloaded from https://wx.jcloud.com/market/packet/10479 ; the other is crawled from http://game.onegreen.net/chess/Index.html (with option --onegreen).

options

* `--type mini`: use mini config, (see `cchess_alphazero/configs/mini.py`)
* `--gpu '1'`: specify which gpu to use
* `--onegreen`: if set the flag, `sl_onegreen` worker will start to train data crawled from `game.onegreen.net`
* `--skip SKIP`: if set this flag, games whoses index is less than `SKIP` would not be used to train (only valid when `onegreen` flag is set)

---

## Simple Chess AI (`simple_chess_ai`)

`simple_chess_ai` 是一个轻量级、自包含的中国象棋AI模块，基于PyTorch，无需Keras/TensorFlow即可独立运行。适合在普通CPU/GPU机器上快速训练与调试。

### 特性

- **完整象棋规则**：实现所有棋子移动规则，包括长将判负、**长捉判负**（perpetual chase）、三次重复局面判和等。
- **策略价值网络**：残差卷积网络（支持纯CNN或GNN后端），输入14通道特征平面，输出走法概率与局面价值。
- **MCTS搜索**：PUCT算法驱动，支持Dirichlet噪声、树复用（Tree Reuse）、局面缓存。
- **多种训练模式**：标准AlphaZero式自对弈训练、GRPO（Group Relative Policy Optimization）训练、FP16混合精度训练。
- **Gating评测**：定期用新模型与旧模型对局，胜率超过阈值后替换best模型。
- **数据导出**：自对弈记录（JSONL）、训练指标（CSV）、损失/胜率曲线（PNG）。
- **图形界面**：基于pygame的可交互棋盘（`python -m simple_chess_ai play`）。
- **命令行界面**：纯文本人机对弈（`python -m simple_chess_ai play_cli`）。
- **Reasoning模块**：链式推理（Chain-of-Thought）增强的走法分析。

### 快速开始

#### 安装依赖

```bash
pip install torch numpy
pip install pygame  # 仅图形界面需要
```

#### 训练模型

```bash
# 标准训练（50局自对弈）
python -m simple_chess_ai train --num_games 50 --num_simulations 100

# GRPO训练模式
python -m simple_chess_ai train --num_games 50 --use_grpo --grpo_group_size 8

# FP16混合精度训练（需要GPU）
python -m simple_chess_ai train --num_games 100 --use_fp16

# 快速验证训练流程
python -m simple_chess_ai train --quick
```

完整训练选项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_games` | 50 | 自对弈局数 |
| `--num_simulations` | 100 | 每步MCTS模拟次数 |
| `--num_epochs` | 5 | 每次训练轮数 |
| `--batch_size` | 256 | 批大小 |
| `--lr` | 0.001 | 学习率 |
| `--max_moves` | 200 | 每局最大步数（超过判和）|
| `--buffer_size` | 10000 | 训练数据缓冲区大小 |
| `--model_path` | — | 模型保存路径（默认 `simple_chess_ai/saved_model/model.pth`）|
| `--save_interval` | 10 | 每隔多少局保存模型 |
| `--use_grpo` | — | 使用GRPO训练模式 |
| `--grpo_group_size` | 8 | GRPO组采样大小 |
| `--use_fp16` | — | FP16混合精度训练 |
| `--gating_interval` | 20 | 每隔多少局进行gating评测（0=禁用）|
| `--gating_games` | 20 | gating对局数 |
| `--gating_winrate` | 0.55 | gating接受阈值（新模型最低胜率）|
| `--seed` | — | 随机种子（可复现）|
| `--deterministic` | — | cuDNN确定性模式（配合`--seed`）|
| `--runs_dir` | — | 日志导出目录（默认 `simple_chess_ai/runs/`）|
| `--quick` | — | 快速模式（1局+1次更新，验证流程）|

#### 图形界面对弈

```bash
python -m simple_chess_ai play [--model_path path/to/model.pth] [--num_simulations 200]
```

#### 命令行对弈

```bash
python -m simple_chess_ai play_cli [--model_path path/to/model.pth] [--num_simulations 200] [--human_color red|black]
```

走法格式：`x0 y0 x1 y1`，例如 `4 0 4 1` 表示帅从(4,0)向前走一步。

### 象棋规则说明

坐标系：x 为列(0–8)，y 为行(0–9)；红方在下方(y=0–4)，黑方在上方(y=5–9)。

棋子FEN编码：
- 大写=红方：`R`(车) `N`(马) `B`(象) `A`(仕) `K`(帅) `C`(炮) `P`(兵)
- 小写=黑方：`r`(车) `n`(马) `b`(象) `a`(仕) `k`(将) `c`(炮) `p`(卒)

重复局面处理：
- **长将**（perpetual check）：循环内一方连续将军维持循环 → 该方判负
- **长捉**（perpetual chase）：循环内一方始终用棋子威胁吃对方某一非将棋子，而对方无捉回 → 捉子方判负
- **三次重复且无长将/长捉**：判和

### 运行测试

```bash
python -m pytest simple_chess_ai/tests.py -v
```

### 模块结构

```
simple_chess_ai/
├── game.py           # 象棋规则、走法生成、长将/长捉判断
├── model.py          # 策略价值网络（CNN + 残差块）
├── mcts.py           # MCTS搜索（PUCT算法）
├── train.py          # 自对弈训练流程
├── grpo.py           # GRPO训练器
├── gnn_feature.py    # 图神经网络特征提取
├── reasoning.py      # 链式推理模块
├── reasoning_cli.py  # 推理CLI界面
├── action_encoding.py# 动作编码/解码
├── export.py         # 数据与图表导出
├── gui.py            # pygame图形界面
├── cli.py            # 命令行对弈界面
├── tests.py          # 单元测试
└── __main__.py       # 模块主入口
```
