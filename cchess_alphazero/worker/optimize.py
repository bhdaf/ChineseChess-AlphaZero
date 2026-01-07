"""
模型优化训练模块

本模块实现了策略价值网络的训练逻辑，
使用PyTorch框架进行深度学习训练。

主要功能：
- 从自我对弈数据中加载训练样本
- 使用Adam优化器训练策略价值网络
- 支持学习率调度和模型检查点保存
"""

import os
import time
import gc
import subprocess
import shutil
import numpy as np

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from time import sleep
from random import shuffle
from threading import Thread

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.config import Config
from cchess_alphazero.lib.data_helper import get_game_data_filenames, read_game_data_from_file
from cchess_alphazero.lib.model_helper import load_best_model_weight, save_as_best_model
from cchess_alphazero.lib.model_helper import need_to_reload_best_model_weight, save_as_next_generation_model, save_as_best_model
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_policy, flip_move
from cchess_alphazero.lib.tf_util import set_session_config
from cchess_alphazero.lib.web_helper import http_request

logger = getLogger(__name__)


def start(config: Config):
    """
    启动训练工作进程
    
    Args:
        config: 配置对象
    
    Returns:
        训练结果
    """
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    return OptimizeWorker(config).start()


class OptimizeWorker:
    """
    模型优化训练工作类
    
    负责加载训练数据、训练模型并保存检查点。
    使用PyTorch实现训练循环。
    
    Attributes:
        config: 配置对象
        model: CChessModel实例
        optimizer: PyTorch优化器
        device: 计算设备（CPU或GPU）
    """
    
    def __init__(self, config: Config):
        """
        初始化训练工作器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.model = None
        self.loaded_filenames = set()
        self.loaded_data = deque(maxlen=self.config.trainer.dataset_size)
        self.dataset = deque(), deque(), deque()
        self.executor = ProcessPoolExecutor(max_workers=config.trainer.cleaning_processes)
        self.filenames = []
        self.optimizer = None
        self.count = 0
        self.eva = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"训练使用设备: {self.device}")

    def start(self):
        """启动训练"""
        self.model = self.load_model()
        self.training()

    def training(self):
        """
        主训练循环
        
        从数据文件中加载训练样本，执行训练epoch，
        并定期保存模型检查点。
        """
        self.compile_model()
        total_steps = self.config.trainer.start_total_steps
        bef_files = []
        last_file = None

        while True:
            files = get_game_data_filenames(self.config.resource)
            offset = self.config.trainer.min_games_to_begin_learn
            if (len(files) < self.config.trainer.min_games_to_begin_learn \
              or ((last_file is not None and last_file in files) and files.index(last_file) + 1 + offset > len(files))):
                if last_file is not None:
                    self.save_current_model(send=True)
                break
            else:
                if last_file is not None and last_file in files:
                    idx = files.index(last_file) + 1
                    if len(files) - idx > self.config.trainer.load_step:
                        files = files[idx:idx + self.config.trainer.load_step]
                    else:
                        files = files[idx:]
                elif len(files) > self.config.trainer.load_step:
                    files = files[0:self.config.trainer.load_step]
                last_file = files[-1]
                logger.info(f"最后一个文件 = {last_file}")
                logger.debug(f"文件列表 = {files[0:-1:2000]}")
                self.filenames = deque(files)
                logger.debug(f"开始训练 {len(self.filenames)} 个文件")
                shuffle(self.filenames)
                self.fill_queue()
                self.update_learning_rate(total_steps)
                if len(self.dataset[0]) > self.config.trainer.batch_size:
                    steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
                    total_steps += steps
                    self.save_current_model(send=False)
                    self.update_learning_rate(total_steps)
                    self.count += 1
                    a, b, c = self.dataset
                    a.clear()
                    b.clear()
                    c.clear()
                    del self.dataset, a, b, c
                    gc.collect()
                    self.dataset = deque(), deque(), deque()
                    self.backup_play_data(files)

    def train_epoch(self, epochs):
        """
        执行训练epoch
        
        使用PyTorch进行模型训练，包括策略损失和价值损失的计算。
        
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
        dataloader = DataLoader(dataset, batch_size=tc.batch_size, shuffle=True)
        
        # 设置模型为训练模式
        self.model.model.train()
        
        # 定义损失函数
        # 策略损失：交叉熵损失（用于软标签/概率分布）
        # 注意：nn.CrossEntropyLoss期望硬标签（类别索引），
        # 但我们的策略目标是概率分布，所以使用手动计算的交叉熵
        # 价值损失：均方误差损失（MSE）
        value_criterion = nn.MSELoss()
        
        total_steps = 0
        for epoch in range(epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            num_batches = 0
            
            for batch_state, batch_policy, batch_value in dataloader:
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                pred_policy, pred_value = self.model.model(batch_state)
                
                # 计算策略损失（软标签交叉熵，等价于KL散度+常数）
                # 公式：-sum(target * log(pred)) / batch_size
                # 这里使用 +1e-8 避免 log(0) 产生数值问题
                policy_loss = -torch.sum(batch_policy * torch.log(pred_policy + 1e-8)) / batch_policy.size(0)
                
                # 计算价值损失（MSE）
                value_loss = value_criterion(pred_value.squeeze(), batch_value)
                
                # 总损失（加权求和）
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
            
            # 打印训练信息
            avg_policy_loss = epoch_policy_loss / num_batches
            avg_value_loss = epoch_value_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{epochs}, 策略损失: {avg_policy_loss:.4f}, 价值损失: {avg_value_loss:.4f}")
        
        # 恢复为评估模式
        self.model.model.eval()
        
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        """
        编译模型（设置优化器）
        
        使用Adam优化器，带有L2正则化（权重衰减）。
        注意：Adam优化器通常使用较低的学习率（0.001-0.01）
        """
        # 使用Adam优化器，并设置L2正则化（weight_decay）
        # Adam推荐学习率为0.001，比SGD的0.02低很多
        self.optimizer = optim.Adam(
            self.model.model.parameters(),
            lr=0.001,  # Adam推荐的学习率
            weight_decay=self.config.model.l2_reg  # L2正则化
        )
        logger.info(f"优化器已配置: Adam, 学习率=0.001, 权重衰减={self.config.model.l2_reg}")

    def update_learning_rate(self, total_steps):
        """
        更新学习率
        
        根据训练步数调整学习率，实现学习率调度。
        
        Args:
            total_steps: 当前总训练步数
        """
        # AlphaZero论文中的学习率调度:
        # ~400k: 1e-2
        # 400k~600k: 1e-3
        # 600k~: 1e-4

        lr = self.decide_learning_rate(total_steps)
        if lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            logger.debug(f"总步数={total_steps}, 设置学习率为 {lr}")

    def fill_queue(self):
        """
        填充数据队列
        
        从文件中加载训练数据并填充到数据队列中。
        """
        futures = deque()
        n = len(self.filenames)
        with ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes) as executor:
            for _ in range(self.config.trainer.cleaning_processes):
                if len(self.filenames) == 0:
                    break
                filename = self.filenames.pop()
                futures.append(executor.submit(load_data_from_file, filename, self.config.opts.has_history))
            while futures and len(self.dataset[0]) < self.config.trainer.dataset_size:
                _tuple = futures.popleft().result()
                if _tuple is not None:
                    for x, y in zip(self.dataset, _tuple):
                        x.extend(y)
                m = len(self.filenames)
                if m > 0:
                    if (n - m) % 1000 == 0:
                        logger.info(f"已读取 {n - m} 个文件")
                    filename = self.filenames.pop()
                    futures.append(executor.submit(load_data_from_file, filename, self.config.opts.has_history))

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
        if self.config.opts.new or not load_best_model_weight(model):
            model.build()
            save_as_best_model(model)
        return model

    def save_current_model(self, send=False):
        """
        保存当前模型
        
        Args:
            send: 是否作为下一代模型保存（用于评估）
        """
        logger.info("保存模型")
        if not send:
            save_as_best_model(self.model)
        else:
            save_as_next_generation_model(self.model)

    def decide_learning_rate(self, total_steps):
        """
        根据总步数决定学习率
        
        Args:
            total_steps: 当前总训练步数
        
        Returns:
            float: 新的学习率，如果不需要更新则返回None
        """
        ret = None

        for step, lr in self.config.trainer.lr_schedules:
            if total_steps >= step:
                ret = lr
        return ret

    def try_reload_model(self):
        """
        尝试重新加载模型
        
        Returns:
            bool: 是否重新加载了模型
        """
        logger.debug("检查模型更新")
        if need_to_reload_best_model_weight(self.model):
            # PyTorch不需要graph.as_default()上下文
            load_best_model_weight(self.model)
            return True
        return False

    def backup_play_data(self, files):
        """
        备份训练数据
        
        将已使用的训练数据文件移动到备份目录。
        
        Args:
            files: 要备份的文件列表
        """
        backup_folder = os.path.join(self.config.resource.data_dir, 'trained')
        cnt = 0
        if not os.path.exists(backup_folder):
            os.makedirs(backup_folder)
        for i in range(len(files)):
            try:
                shutil.move(files[i], backup_folder)
            except Exception as e:
                cnt = cnt + 1
        logger.info(f"备份 {len(files)} 个文件, {cnt} 个空文件")


def load_data_from_file(filename, use_history=False):
    """
    从文件加载数据
    
    Args:
        filename: 数据文件路径
        use_history: 是否使用历史信息
    
    Returns:
        tuple: 训练数据元组，如果加载失败则返回None
    """
    try:
        data = read_game_data_from_file(filename)
    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        os.remove(filename)
        return None
    if data is None:
        return None
    return expanding_data(data, use_history)


def expanding_data(data, use_history=False):
    """
    扩展数据
    
    将原始对局数据转换为训练样本格式。
    
    Args:
        data: 原始对局数据
        use_history: 是否使用历史信息
    
    Returns:
        tuple: 扩展后的训练数据
    """
    state = data[0]
    real_data = []
    action = None
    policy = None
    value = None
    if use_history:
        history = [state]
    else:
        history = None
    for item in data[1:]:
        action = item[0]
        value = item[1]
        try:
            policy = build_policy(action, flip=False)
        except Exception as e:
            logger.error(f"扩展数据错误 {e}, item = {item}, data = {data}, state = {state}")
            return None
        real_data.append([state, policy, value])
        state = senv.step(state, action)
        if use_history:
            history.append(action)
            history.append(state)
        
    return convert_to_trainging_data(real_data, history)


def convert_to_trainging_data(data, history):
    """
    转换为训练数据格式
    
    Args:
        data: 原始数据列表
        history: 历史记录（可选）
    
    Returns:
        tuple: (状态数组, 策略数组, 价值数组)
    """
    state_list = []
    policy_list = []
    value_list = []
    i = 0

    for state, policy, value in data:
        if history is None:
            state_planes = senv.state_to_planes(state)
        else:
            state_planes = senv.state_history_to_planes(state, history[0:i * 2 + 1])
        sl_value = value

        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(sl_value)
        i += 1

    return np.asarray(state_list, dtype=np.float32), \
           np.asarray(policy_list, dtype=np.float32), \
           np.asarray(value_list, dtype=np.float32)


def build_policy(action, flip):
    """
    构建策略向量
    
    Args:
        action: 动作字符串
        flip: 是否翻转
    
    Returns:
        list: 策略向量
    """
    labels_n = len(ActionLabelsRed)
    move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
    policy = np.zeros(labels_n)

    policy[move_lookup[action]] = 1

    if flip:
        policy = flip_policy(policy)
    return list(policy)



