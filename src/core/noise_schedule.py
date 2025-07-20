"""
噪声调度相关功能
"""

import torch
import numpy as np


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    """
    线性beta调度
    
    Args:
        T: 时间步数
        beta_start: 起始beta值
        beta_end: 结束beta值
    
    Returns:
        betas: beta序列
    """
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008):
    """
    余弦beta调度
    
    Args:
        T: 时间步数
        s: 偏移参数
    
    Returns:
        betas: beta序列
    """
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    """
    二次beta调度
    
    Args:
        T: 时间步数
        beta_start: 起始beta值
        beta_end: 结束beta值
    
    Returns:
        betas: beta序列
    """
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T) ** 2 