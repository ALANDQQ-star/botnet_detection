"""
V3 Calibrated - 分数校准版本
核心改进：
1. 训练时增加分数分布损失，强制分数分散到[0,1]范围
2. 推理时使用分位数校准，确保分数有意义
3. 目标：保持AUC的同时，增加分离度
"""

import sys
import time
import os
import warnings
import argparse
import gc
import json
import numpy as np
import pandas as pd

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
import torch.cuda.amp as amp
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from scipy.stats import gaussian_kde
from torch_geometric.loader import NeighborLoader

from data_loader import CTU13Loader
from improved_heterogeneous_graph import ImprovedHeterogeneousGraphBuilder


class ScoreCalibrationLoss(nn.Module):
    """
    分数校准损失 - 强制分数分散到更广的范围
    目标：让正常样本分数集中在[0, 0.3]，僵尸样本分数集中在[0.7, 1]
    """
    def __init__(self, target_neg_mean=0.15, target_pos_mean=0.85, margin=0.5):
        super().__init__()
        self.target_neg_mean = target_neg_mean
        self.target_pos_mean = target_pos_mean
        self.margin = margin
    
    def forward(self, probs, labels):
        pos_mask = labels == 1
        neg_mask = labels == 0
        
