"""
OneGreen棋谱监督学习训练模块

本模块实现了从OneGreen格式棋谱数据进行监督学习训练的功能，
使用PyTorch框架进行深度学习训练。

主要功能：
- 从JSON格式的OneGreen棋谱数据加载训练样本
- 使用Adam优化器训练策略价值网络
- 支持交叉熵损失和均方误差损失
"""

import os
import numpy as np
import json

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from time import sleep
from random import shuffle
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.config import Config
from cchess_alphazero.lib.data_helper import get_game_data_filenames, read_game_data_from_file
from cchess_alphazero.lib.model_helper import load_sl_best_model_weight, save_as_sl_best_model
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, flip_policy, flip_move
from cchess_alphazero.lib.tf_util import set_session_config
from cchess_alphazero.environment.lookup_tables import Winner

logger = getLogger(__name__)


def start(config: Config, skip):
    """
    启动OneGreen监督学习训练
    
    Args:
        config: 配置对象
        skip: 跳过的对局数量
    
    Returns:
        训练结果
    """
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    return SupervisedWorker(config).start(skip)


class SupervisedWorker:
    """
    OneGreen监督学习训练工作类
    
    从OneGreen格式的棋谱数据学习策略和价值函数。
    使用PyTorch实现训练循环。
    
    Attributes:
        config: 配置对象
        model: CChessModel实例
        optimizer: PyTorch优化器
        device: 计算设备
    """
    
    def __init__(self, config: Config):
        """
        初始化监督学习工作器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.model = None
        self.loaded_data = deque(maxlen=self.config.trainer.dataset_size)
        self.dataset = deque(), deque(), deque()
        self.filenames = []
        self.optimizer = None
        self.buffer = []
        self.games = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"OneGreen监督学习使用设备: {self.device}")

    def start(self, skip=0):
        """
        启动训练
        
        Args:
            skip: 跳过的对局数量
        """
        self.model = self.load_model()
        with open(self.config.resource.sl_onegreen, 'r') as f:
            self.games = json.load(f)
        self.training(skip)

    def training(self, skip=0):
        """
        主训练循环
        
        Args:
            skip: 跳过的对局数量
        """
        self.compile_model()
        total_steps = self.config.trainer.start_total_steps
        logger.info(f"开始训练，对局数量 = {len(self.games)}, 步长 = {self.config.trainer.sl_game_step} 局, 跳过 = {skip}")

        for i in range(skip, len(self.games), self.config.trainer.sl_game_step):
            games = self.games[i:i+self.config.trainer.sl_game_step]
            self.fill_queue(games)
            if len(self.dataset[0]) > self.config.trainer.batch_size:
                steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
                total_steps += steps
                self.save_current_model()
                a, b, c = self.dataset
                a.clear()
                b.clear()
                c.clear()
                logger.debug(f"总步数 = {total_steps}")

    def train_epoch(self, epochs):
        """
        执行训练epoch
        
        Args:
            epochs: 训练epoch数量
        
        Returns:
            int: 总训练步数
        """
        tc = self.config.trainer
        state_ary, policy_ary, value_ary = self.collect_all_loaded_data()
        
        # 将数据转换为PyTorch张量
        state_tensor = torch.FloatTensor(state_ary).to(self.device)
        policy_tensor = torch.FloatTensor(policy_ary).to(self.device)
        value_tensor = torch.FloatTensor(value_ary).to(self.device)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(state_tensor, policy_tensor, value_tensor)
        # 分离验证集（2%）
        val_size = int(len(dataset) * 0.02)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=tc.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=tc.batch_size, shuffle=False)
        
        # 设置模型为训练模式
        self.model.model.train()
        
        # 定义损失函数
        value_criterion = nn.MSELoss()
        
        total_steps = 0
        for epoch in range(epochs):
            # 训练阶段
            epoch_policy_loss = 0
            epoch_value_loss = 0
            num_batches = 0
            
            for batch_state, batch_policy, batch_value in train_loader:
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                pred_policy, pred_value = self.model.model(batch_state)
                
                # 计算策略损失（交叉熵）
                policy_loss = -torch.sum(batch_policy * torch.log(pred_policy + 1e-8)) / batch_policy.size(0)
                
                # 计算价值损失（MSE）
                value_loss = value_criterion(pred_value.squeeze(), batch_value)
                
                # 总损失
                total_loss = self.config.trainer.loss_weights[0] * policy_loss + \
                            self.config.trainer.loss_weights[1] * value_loss
                
                # 反向传播
                total_loss.backward()
                
                # 更新参数
                self.optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                num_batches += 1
                total_steps += 1
            
            # 验证阶段
            self.model.model.eval()
            val_policy_loss = 0
            val_value_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch_state, batch_policy, batch_value in val_loader:
                    pred_policy, pred_value = self.model.model(batch_state)
                    policy_loss = -torch.sum(batch_policy * torch.log(pred_policy + 1e-8)) / batch_policy.size(0)
                    value_loss = value_criterion(pred_value.squeeze(), batch_value)
                    val_policy_loss += policy_loss.item()
                    val_value_loss += value_loss.item()
                    val_batches += 1
            
            self.model.model.train()
            
            # 打印训练信息
            if num_batches > 0 and val_batches > 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, "
                          f"训练策略损失: {epoch_policy_loss / num_batches:.4f}, "
                          f"训练价值损失: {epoch_value_loss / num_batches:.4f}, "
                          f"验证策略损失: {val_policy_loss / val_batches:.4f}, "
                          f"验证价值损失: {val_value_loss / val_batches:.4f}")
        
        # 恢复为评估模式
        self.model.model.eval()
        
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        """
        编译模型（设置优化器）
        
        使用Adam优化器，带有L2正则化。
        """
        self.optimizer = optim.Adam(
            self.model.model.parameters(),
            lr=0.003,
            weight_decay=self.config.model.l2_reg
        )
        logger.info("优化器已配置: Adam, 学习率=0.003")

    def fill_queue(self, games):
        """
        填充数据队列
        
        Args:
            games: 对局数据列表
        """
        _tuple = self.generate_game_data(games)
        if _tuple is not None:
            for x, y in zip(self.dataset, _tuple):
                x.extend(y)

    def collect_all_loaded_data(self):
        """
        收集所有已加载的数据
        
        Returns:
            tuple: (状态数组, 策略数组, 价值数组)
        """
        state_ary, policy_ary, value_ary = self.dataset

        state_ary1 = np.asarray(state_ary, dtype=np.float32)
        policy_ary1 = np.asarray(policy_ary, dtype=np.float32)
        value_ary1 = np.asarray(value_ary, dtype=np.float32)
        return state_ary1, policy_ary1, value_ary1

    def load_model(self):
        """
        加载模型
        
        Returns:
            CChessModel: 加载的模型实例
        """
        model = CChessModel(self.config)
        if self.config.opts.new or not load_sl_best_model_weight(model):
            model.build()
            save_as_sl_best_model(model)
        return model

    def save_current_model(self):
        """保存当前模型"""
        logger.debug("保存最佳监督学习模型")
        save_as_sl_best_model(self.model)

    def generate_game_data(self, games):
        """
        生成训练数据
        
        Args:
            games: 对局数据列表
        
        Returns:
            tuple: 训练数据元组
        """
        self.buffer = []
        start_time = time()
        idx = 0
        cnt = 0
        for game in games:
            init = game['init']
            move_list = game['move_list']
            winner = Winner.draw
            if game['result'] == '红胜' or '胜' in game['title']:
                winner = Winner.red
            elif game['result'] == '黑胜' or '负' in game['title']:
                winner = Winner.black
            else:
                winner = Winner.draw
            v = self.load_game(init, move_list, winner, idx, game['title'], game['url'])
            if v == 1 or v == -1:
                cnt += 1
            idx += 1
        end_time = time()
        logger.debug(f"加载 {len(games)} 局对局，用时: {end_time - start_time}s, 结束对局 = {cnt}")
        return self.convert_to_trainging_data()

    def load_game(self, init, move_list, winner, idx, title, url):
        """
        加载单局对局数据
        
        Args:
            init: 初始局面
            move_list: 走法列表
            winner: 获胜方
            idx: 对局索引
            title: 对局标题
            url: 对局URL
        
        Returns:
            int: 红方胜负值
        """
        turns = 0
        env = CChessEnv(self.config).reset(init)
        red_moves = []
        black_moves = []
        moves = [move_list[i:i+4] for i in range(len(move_list)) if i % 4 == 0]

        for move in moves:
            action = senv.parse_onegreen_move(move)
            try:
                if turns % 2 == 0:
                    red_moves.append([env.observation, self.build_policy(action, flip=False)])
                else:
                    black_moves.append([env.observation, self.build_policy(action, flip=True)])
                env.step(action)
            except:
                logger.error(f"无效动作: idx = {idx}, action = {action}, turns = {turns}, moves = {moves}, "
                             f"winner = {winner}, init = {init}, title: {title}, url: {url}")
                return
            turns += 1

        if winner == Winner.red:
            red_win = 1
        elif winner == Winner.black:
            red_win = -1
        else:
            red_win = senv.evaluate(env.get_state())
            if not env.red_to_move:
                red_win = -red_win

        for move in red_moves:
            move += [red_win]
        for move in black_moves:
            move += [-red_win]

        data = []
        for i in range(len(red_moves)):
            data.append(red_moves[i])
            if i < len(black_moves):
                data.append(black_moves[i])
        self.buffer += data
        return red_win

    def build_policy(self, action, flip):
        """
        构建策略向量
        
        Args:
            action: 动作字符串
            flip: 是否翻转
        
        Returns:
            ndarray: 策略向量
        """
        labels_n = len(ActionLabelsRed)
        move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
        policy = np.zeros(labels_n)

        policy[move_lookup[action]] = 1

        if flip:
            policy = flip_policy(policy)
        return policy

    def convert_to_trainging_data(self):
        """
        转换为训练数据格式
        
        Returns:
            tuple: (状态数组, 策略数组, 价值数组)
        """
        data = self.buffer
        state_list = []
        policy_list = []
        value_list = []
        env = CChessEnv()

        for state_fen, policy, value in data:
            state_planes = env.fen_to_planes(state_fen)
            sl_value = value

            state_list.append(state_planes)
            policy_list.append(policy)
            value_list.append(sl_value)

        return np.asarray(state_list, dtype=np.float32), \
               np.asarray(policy_list, dtype=np.float32), \
               np.asarray(value_list, dtype=np.float32)



