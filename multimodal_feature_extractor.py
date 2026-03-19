"""
多模态特征提取器
基于 Bot-DM 论文思想，提取流量的语义和图像特征，增强图对比学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import gc


class PayloadSemanticEncoder(nn.Module):
    """
    Payload 语义编码器
    将流量 payload 转换为隐式语义向量（类似 Bot-DM 的 Botflow-token）
    """
    def __init__(self, vocab_size: int = 256, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        
