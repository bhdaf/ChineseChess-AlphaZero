"""
数据辅助模块

本模块提供了训练数据的读写功能，用于处理自我对弈产生的数据文件。

主要功能：
- 获取训练数据文件列表
- 读写JSON格式的对局数据
"""

import os
import json
from datetime import datetime
from glob import glob
from logging import getLogger

from cchess_alphazero.config import ResourceConfig

logger = getLogger(__name__)


def get_game_data_filenames(rc: ResourceConfig):
    """
    获取所有训练数据文件名
    
    Args:
        rc: 资源配置对象
    
    Returns:
        list: 排序后的文件路径列表
    """
    pattern = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % "*")
    files = list(sorted(glob(pattern)))
    return files


def write_game_data_to_file(path, data):
    """
    将对局数据写入文件
    
    Args:
        path: 文件路径
        data: 对局数据
    """
    with open(path, "wt") as f:
        json.dump(data, f)


def read_game_data_from_file(path):
    """
    从文件读取对局数据
    
    Args:
        path: 文件路径
    
    Returns:
        对局数据
    """
    with open(path, "rt") as f:
        return json.load(f)


def get_key(x):
    """
    获取文件创建时间作为排序键
    
    Args:
        x: 文件路径
    
    Returns:
        float: 文件创建时间戳
    """
    stat_x = os.stat(x) 
    return stat_x.st_ctime
