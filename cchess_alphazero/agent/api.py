"""
中国象棋模型推理API模块

本模块提供了多进程/多线程环境下的模型推理接口，
使用管道(Pipe)实现进程间通信，支持批量预测以提高效率。

主要功能：
- 异步批量推理
- 模型热重载
- 网络分布式权重同步
"""

from multiprocessing import connection, Pipe
from threading import Thread

import os
import numpy as np
import shutil
import torch

from cchess_alphazero.config import Config
from cchess_alphazero.lib.model_helper import load_best_model_weight, need_to_reload_best_model_weight
from cchess_alphazero.lib.web_helper import http_request, download_file
from time import time
from logging import getLogger

logger = getLogger(__name__)


class CChessModelAPI:
    """
    中国象棋模型推理API
    
    提供多进程环境下的异步批量推理功能。
    使用后台线程处理预测请求，通过管道与其他进程通信。
    
    Attributes:
        agent_model: CChessModel实例
        pipes: 通信管道列表
        config: 配置对象
        need_reload: 是否需要重新加载模型
        done: 是否已完成
    """

    def __init__(self, config: Config, agent_model):
        """
        初始化推理API
        
        Args:
            config: 配置对象
            agent_model: CChessModel实例
        """
        self.agent_model = agent_model  # CChessModel实例
        self.pipes = []     # 用于进程间通信的管道列表
        self.config = config
        self.need_reload = True
        self.done = False

    def start(self, need_reload=True):
        """
        启动后台预测线程
        
        Args:
            need_reload: 是否需要重新加载模型权重
        """
        self.need_reload = need_reload
        prediction_worker = Thread(target=self.predict_batch_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def get_pipe(self, need_reload=True):
        """
        获取通信管道
        
        创建一对管道用于与后台预测线程通信。
        
        Args:
            need_reload: 是否需要重新加载模型
        
        Returns:
            管道的一端，用于发送预测请求和接收结果
        """
        me, you = Pipe()
        self.pipes.append(me)
        self.need_reload = need_reload
        return you

    def predict_batch_worker(self):
        """
        后台批量预测工作线程
        
        持续监听管道，收集批量预测请求，
        使用PyTorch模型进行推理并返回结果。
        """
        # 如果启用分布式模式，尝试从网络加载最新权重
        if self.config.internet.distributed and self.need_reload:
            self.try_reload_model_from_internet()
        
        last_model_check_time = time()
        
        while not self.done:
            # 每10分钟检查一次是否需要重新加载模型
            if last_model_check_time + 600 < time() and self.need_reload:
                self.try_reload_model()
                last_model_check_time = time()
            
            # 等待管道中的数据，超时时间1毫秒
            ready = connection.wait(self.pipes, timeout=0.001)
            if not ready:
                continue
            
            # 收集所有准备好的预测请求
            data, result_pipes, data_len = [], [], []
            for pipe in ready:
                while pipe.poll():
                    try:
                        tmp = pipe.recv()
                    except EOFError as e:
                        logger.error(f"管道EOF错误: {e}")
                        pipe.close()
                    else:
                        data.extend(tmp)
                        data_len.append(len(tmp))
                        result_pipes.append(pipe)
            
            if not data:
                continue
            
            # 将数据转换为numpy数组
            data = np.asarray(data, dtype=np.float32)
            
            # 使用PyTorch进行批量预测
            # 使用torch.no_grad()禁用梯度计算以提高推理性能
            with torch.no_grad():
                # 转换为PyTorch张量并移动到设备上
                data_tensor = torch.FloatTensor(data).to(self.agent_model.device)
                # 执行前向传播
                policy_tensor, value_tensor = self.agent_model.model(data_tensor)
                # 转换回numpy数组
                policy_ary = policy_tensor.cpu().numpy()
                value_ary = value_tensor.cpu().numpy()
            
            # 将结果发送回各个管道
            buf = []
            k, i = 0, 0
            for p, v in zip(policy_ary, value_ary):
                buf.append((p, float(v)))
                k += 1
                if k >= data_len[i]:
                    result_pipes[i].send(buf)
                    buf = []
                    k = 0
                    i += 1

    def try_reload_model(self, config_file=None):
        """
        尝试重新加载模型
        
        检查本地权重文件是否已更新，如果更新则重新加载。
        
        Args:
            config_file: 指定的配置文件名（可选）
        """
        if config_file:
            config_path = os.path.join(self.config.resource.model_dir, config_file)
            shutil.copy(config_path, self.config.resource.model_best_config_path)
        try:
            if self.config.internet.distributed and not config_file:
                self.try_reload_model_from_internet()
            else:
                if self.need_reload and need_to_reload_best_model_weight(self.agent_model):
                    # PyTorch不需要graph.as_default()上下文
                    load_best_model_weight(self.agent_model)
        except Exception as e:
            logger.error(f"重新加载模型失败: {e}")

    def try_reload_model_from_internet(self, config_file=None):
        """
        尝试从网络加载最新模型
        
        检查服务器上是否有新版本的模型权重，
        如果有则下载并加载。
        
        Args:
            config_file: 指定的配置文件名（可选）
        """
        response = http_request(self.config.internet.get_latest_digest)
        if response is None:
            logger.error(f"无法连接到远程服务器！请检查网络连接，并重新打开客户端")
            return
        digest = response['data']['digest']

        if digest != self.agent_model.fetch_digest(self.config.resource.model_best_weight_path):
            logger.info(f"正在下载最新权重，请稍后...")
            if download_file(self.config.internet.download_url, self.config.resource.model_best_weight_path):
                logger.info(f"权重下载完毕！开始训练...")
                try:
                    # PyTorch不需要graph.as_default()上下文
                    load_best_model_weight(self.agent_model)
                except ValueError as e:
                    logger.error(f"权重架构不匹配，自动重新加载: {e}")
                    self.try_reload_model(config_file='model_192x10_config.json')
                except Exception as e:
                    logger.error(f"加载权重发生错误: {e}，稍后重新下载")
                    os.remove(self.config.resource.model_best_weight_path)
                    self.try_reload_model_from_internet()
            else:
                logger.error(f"权重下载失败！请检查网络连接，并重新打开客户端")
        else:
            logger.info(f"检查完毕，权重未更新")

    def close(self):
        """
        关闭API
        
        设置完成标志，停止后台预测线程。
        """
        self.done = True
