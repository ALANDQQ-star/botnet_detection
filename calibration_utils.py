"""
分数校准工具

提供多种校准方法，无需重新训练即可改善分数分布
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


class TemperatureScaling:
    """
    温度缩放校准
    
    核心思想：学习一个温度参数 T，调整 logits 的尺度
    calibrated_prob = sigmoid(logit / T)
    
    - T > 1: 分数更集中在 0.5 附近（保守）
    - T < 1: 分数更分散到 0 和 1（激进）
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.fitted = False
    
    def fit(self, probs, labels, clip_range=(1e-7, 1-1e-7)):
        """
        在验证数据上拟合温度参数
        
        Args:
            probs: 原始概率分数 [0, 1]
            labels: 真实标签 {0, 1}
            clip_range: 概率裁剪范围
        """
        probs = np.asarray(probs).flatten()
        labels = np.asarray(labels).flatten()
        
        # 转换到 logit 空间
        probs_clipped = np.clip(probs, clip_range[0], clip_range[1])
        logits = np.log(probs_clipped / (1 - probs_clipped))
        
        # NLL 损失函数
        def nll_loss(logit_scale):
            scaled_logits = logits * logit_scale
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            scaled_probs = np.clip(scaled_probs, clip_range[0], clip_range[1])
            
            # 负对数似然
            nll = -np.mean(
                labels * np.log(scaled_probs) + 
                (1 - labels) * np.log(1 - scaled_probs)
            )
            return nll
        
        # 优化温度参数
        result = minimize_scalar(
            nll_loss,
            bounds=(0.01, 10.0),
            method='bounded'
        )
        
        # 温度是 logit 的缩放因子的倒数
        self.temperature = 1.0 / result.x
        self.fitted = True
        
        return self.temperature
    
    def transform(self, probs, clip_range=(1e-7, 1-1e-7)):
        """
        应用温度缩放变换
        
        Args:
            probs: 原始概率分数
            clip_range: 概率裁剪范围
        
        Returns:
            calibrated_probs: 校准后的概率
        """
        if not self.fitted:
            print("[Warning] 温度参数未拟合，使用默认值 T=1.0")
            return probs
        
        probs = np.asarray(probs).flatten()
        probs_clipped = np.clip(probs, clip_range[0], clip_range[1])
        
        # 转换到 logit 空间
        logits = np.log(probs_clipped / (1 - probs_clipped))
        
