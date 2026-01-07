"""
基于深度强化学习的中国象棋智能体 - 策略价值网络模型

本模块实现了AlphaZero风格的策略价值网络(Policy-Value Network)，
使用PyTorch框架构建，包含残差网络结构和双头输出。

主要组件：
- ResidualBlock: 残差块，包含两个卷积层和跳跃连接
- PolicyValueNet: 策略价值网络，输出走法概率和局面评估
- CChessModel: 模型管理类，处理模型的构建、保存和加载
"""

import hashlib
import json
import os
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, ActionLabelsBlack

logger = getLogger(__name__)


def get_device():
    """
    获取可用的计算设备（GPU或CPU）
    
    Returns:
        torch.device: 可用的计算设备
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class ResidualBlock(nn.Module):
    """
    残差块（Residual Block）
    
    实现了标准的残差网络结构，包含两个3x3卷积层、
    批归一化层和ReLU激活函数，以及跳跃连接。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # 第一个卷积层：3x3卷积，不使用偏置
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层：3x3卷积，不使用偏置
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, channels, height, width)
        
        Returns:
            输出张量，形状与输入相同
        """
        residual = x  # 保存输入用于跳跃连接
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 跳跃连接：将输入与输出相加
        out = out + residual
        out = F.relu(out)
        
        return out


class PolicyValueNet(nn.Module):
    """
    策略价值网络（Policy-Value Network）
    
    AlphaZero风格的深度神经网络，使用残差网络作为特征提取器，
    双头输出结构分别预测走法概率（策略头）和局面评估（价值头）。
    
    网络结构：
    1. 输入层：14x10x9的棋盘状态表示
    2. 初始卷积层：将输入映射到高维特征空间
    3. 残差塔：多个残差块堆叠
    4. 策略头：输出所有合法走法的概率分布
    5. 价值头：输出当前局面的评估值（-1到1）
    
    Args:
        config: 配置对象，包含网络结构参数
        n_labels: 动作标签数量（走法总数）
    """
    
    def __init__(self, config, n_labels):
        super(PolicyValueNet, self).__init__()
        
        mc = config.model
        self.n_labels = n_labels
        
        # 棋盘尺寸常量
        self.BOARD_HEIGHT = 10  # 中国象棋棋盘高度
        self.BOARD_WIDTH = 9    # 中国象棋棋盘宽度
        
        # 输入通道数：14个特征平面（7种棋子x2方）
        self.input_channels = mc.input_depth
        # 残差块的通道数
        self.cnn_filter_num = mc.cnn_filter_num
        # 残差块数量
        self.res_layer_num = mc.res_layer_num
        # 价值头全连接层大小
        self.value_fc_size = mc.value_fc_size
        
        # 策略头和价值头的卷积通道数
        self.policy_channels = 4
        self.value_channels = 2
        
        # 计算全连接层的输入维度
        self.policy_fc_input = self.policy_channels * self.BOARD_HEIGHT * self.BOARD_WIDTH
        self.value_fc_input = self.value_channels * self.BOARD_HEIGHT * self.BOARD_WIDTH
        
        # ========================
        # 初始卷积层（Input Convolution）
        # ========================
        # 将14通道输入映射到cnn_filter_num通道
        self.input_conv = nn.Conv2d(self.input_channels, self.cnn_filter_num,
                                    kernel_size=mc.cnn_first_filter_size,
                                    padding=mc.cnn_first_filter_size // 2,
                                    bias=False)
        self.input_bn = nn.BatchNorm2d(self.cnn_filter_num)
        
        # ========================
        # 残差塔（Residual Tower）
        # ========================
        # 堆叠多个残差块以提取深层特征
        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.cnn_filter_num, self.cnn_filter_num)
            for _ in range(self.res_layer_num)
        ])
        
        # ========================
        # 策略头（Policy Head）
        # ========================
        # 1x1卷积降维 + 全连接层输出走法概率
        self.policy_conv = nn.Conv2d(self.cnn_filter_num, self.policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(self.policy_channels)
        self.policy_fc = nn.Linear(self.policy_fc_input, self.n_labels)
        
        # ========================
        # 价值头（Value Head）
        # ========================
        # 1x1卷积降维 + 全连接层输出局面评估
        self.value_conv = nn.Conv2d(self.cnn_filter_num, self.value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(self.value_channels)
        self.value_fc1 = nn.Linear(self.value_fc_input, self.value_fc_size)
        self.value_fc2 = nn.Linear(self.value_fc_size, 1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, 14, 10, 9)
               14个特征平面表示棋盘状态
        
        Returns:
            policy: 走法概率分布，形状为 (batch, n_labels)
            value: 局面评估值，形状为 (batch, 1)，范围为[-1, 1]
        """
        # ========================
        # 初始卷积层
        # ========================
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)
        
        # ========================
        # 残差塔：通过所有残差块
        # ========================
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # ========================
        # 策略头：输出走法概率分布
        # ========================
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        # 展平：(batch, 4, 10, 9) -> (batch, 360)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        # 使用softmax归一化为概率分布
        policy = F.softmax(policy, dim=1)
        
        # ========================
        # 价值头：输出局面评估
        # ========================
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        # 展平：(batch, 2, 10, 9) -> (batch, 180)
        value = value.view(value.size(0), -1)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        # 使用tanh将输出限制在[-1, 1]范围内
        value = torch.tanh(value)
        
        return policy, value


class CChessModel:
    """
    中国象棋模型管理类
    
    负责模型的构建、保存、加载和推理API的管理。
    支持CPU和GPU自适应运行。
    
    Attributes:
        config: 配置对象
        model: PyTorch策略价值网络
        device: 计算设备（CPU或GPU）
        digest: 模型权重文件的哈希摘要
        api: 推理API对象
    """
    
    def __init__(self, config: Config):
        """
        初始化模型管理器
        
        Args:
            config: 配置对象，包含模型参数
        """
        self.config = config
        self.model = None  # PolicyValueNet实例
        self.digest = None  # 权重文件哈希摘要，用于版本控制
        self.n_labels = len(ActionLabelsRed)  # 动作空间大小
        self.device = get_device()  # 自适应选择CPU或GPU
        self.api = None  # 推理API
        
        logger.info(f"使用计算设备: {self.device}")
    
    def build(self):
        """
        构建策略价值网络
        
        创建PolicyValueNet实例并将其移动到指定设备上。
        """
        logger.debug("正在构建策略价值网络...")
        self.model = PolicyValueNet(self.config, self.n_labels)
        self.model = self.model.to(self.device)
        self.model.eval()  # 默认设置为评估模式
        logger.debug(f"模型已构建完成，参数数量: {sum(p.numel() for p in self.model.parameters())}")
    
    @staticmethod
    def fetch_digest(weight_path):
        """
        计算权重文件的SHA256哈希摘要
        
        用于检查模型权重是否已更新。
        
        Args:
            weight_path: 权重文件路径
        
        Returns:
            哈希摘要字符串，如果文件不存在则返回None
        """
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()
        return None
    
    def load(self, config_path, weight_path):
        """
        加载模型配置和权重
        
        从文件中加载模型配置和预训练权重。
        
        Args:
            config_path: 配置文件路径（JSON格式）
            weight_path: 权重文件路径（.pth格式）
        
        Returns:
            bool: 加载是否成功
        """
        # 检查权重文件是否存在
        if os.path.exists(weight_path):
            logger.debug(f"正在加载模型权重: {weight_path}")
            
            # 如果模型尚未构建，先构建模型
            if self.model is None:
                self.build()
            
            try:
                # 加载权重到指定设备
                state_dict = torch.load(weight_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()  # 设置为评估模式
                self.digest = self.fetch_digest(weight_path)
                logger.debug(f"模型加载成功，摘要: {self.digest}")
                return True
            except Exception as e:
                logger.error(f"加载模型权重失败: {e}")
                return False
        else:
            logger.debug(f"模型权重文件不存在: {weight_path}")
            return False
    
    def save(self, config_path, weight_path):
        """
        保存模型配置和权重
        
        将模型配置保存为JSON文件，权重保存为PyTorch格式。
        
        Args:
            config_path: 配置文件保存路径
            weight_path: 权重文件保存路径
        """
        logger.debug(f"正在保存模型到: {weight_path}")
        
        # 保存模型配置（用于记录网络结构参数）
        config_dict = {
            'input_channels': self.model.input_channels,
            'cnn_filter_num': self.model.cnn_filter_num,
            'res_layer_num': self.model.res_layer_num,
            'value_fc_size': self.model.value_fc_size,
            'n_labels': self.model.n_labels
        }
        with open(config_path, "wt") as f:
            json.dump(config_dict, f)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), weight_path)
        
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"模型保存成功，摘要: {self.digest}")
    
    def predict(self, state_planes):
        """
        使用模型进行预测
        
        Args:
            state_planes: 棋盘状态的特征平面，numpy数组
                         形状为 (batch, 14, 10, 9) 或 (14, 10, 9)
        
        Returns:
            policy: 走法概率分布，numpy数组
            value: 局面评估值，float
        """
        # 确保输入是批量形式
        if len(state_planes.shape) == 3:
            state_planes = np.expand_dims(state_planes, axis=0)
        
        # 转换为PyTorch张量并移动到设备上
        state_tensor = torch.FloatTensor(state_planes).to(self.device)
        
        # 使用no_grad()禁用梯度计算以提高推理性能
        with torch.no_grad():
            policy, value = self.model(state_tensor)
        
        # 转换回numpy数组
        policy = policy.cpu().numpy()
        value = value.cpu().numpy()
        
        return policy, value
    
    def get_pipes(self, num=1, api=None, need_reload=True):
        """
        获取推理管道
        
        创建用于多进程/多线程推理的通信管道。
        
        Args:
            num: 管道数量（已弃用，为兼容性保留）
            api: 现有API实例（已弃用，为兼容性保留）
            need_reload: 是否需要重新加载模型权重
        
        Returns:
            管道对象，用于发送/接收预测请求
        """
        # 延迟导入，避免循环依赖
        from cchess_alphazero.agent.api import CChessModelAPI
        
        if self.api is None:
            self.api = CChessModelAPI(self.config, self)
            self.api.start(need_reload)
        return self.api.get_pipe(need_reload)
    
    def close_pipes(self):
        """
        关闭推理管道
        
        释放API资源，停止后台预测线程。
        """
        if self.api is not None:
            self.api.close()
            self.api = None

