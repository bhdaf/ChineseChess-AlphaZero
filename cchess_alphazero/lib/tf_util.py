"""
PyTorch设备管理工具模块

本模块提供了PyTorch环境下的设备管理功能，
用于配置GPU内存使用和设备选择。

注意：从TensorFlow迁移到PyTorch后，此模块的功能已简化，
因为PyTorch的设备管理更加直观和自动化。
"""

import torch


def set_session_config(per_process_gpu_memory_fraction=None, allow_growth=None, device_list='0'):
    """
    设置PyTorch会话配置（GPU设备选择）
    
    注意：PyTorch不像TensorFlow那样需要显式设置GPU内存使用方式。
    PyTorch默认会根据需要动态分配GPU内存。
    
    此函数主要用于设置CUDA可见设备列表，以兼容原有代码接口。
    
    Args:
        per_process_gpu_memory_fraction: GPU内存使用比例（在PyTorch中已弃用，保留参数以兼容）
        allow_growth: 是否允许内存增长（在PyTorch中已弃用，保留参数以兼容）
        device_list: 可见GPU设备列表，如'0'或'0,1,2'
    """
    import os
    
    # 设置CUDA可见设备
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_list)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        # 获取当前可用的GPU数量
        device_count = torch.cuda.device_count()
        if device_count > 0:
            # 获取当前设备名称
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"PyTorch已配置使用GPU: {device_name} (设备列表: {device_list})")
        else:
            print(f"警告: CUDA可见设备列表'{device_list}'中没有可用的GPU")
    else:
        print("PyTorch将使用CPU进行计算（未检测到可用的CUDA设备）")


def get_device():
    """
    获取可用的计算设备
    
    Returns:
        torch.device: 可用的计算设备（GPU或CPU）
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
