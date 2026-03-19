"""
改进的异构图构建模块 V3
核心改进：
1. 使用快速特征提取器
2. 修复数据泄露 - 归一化统计量仅从训练集计算
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
import gc
import warnings
warnings.filterwarnings('ignore')

from feature_extractor_fast import (
    FastBotnetFeatureExtractor,
    FastNodeSemanticFeatureExtractor,
    FastStructuralFeatureExtractor
)


class ImprovedHeterogeneousGraphBuilderV3:
    """
    改进的异构图构建器 V3
    
    整合三模态特征，修复数据泄露问题
    """
    
    def __init__(self):
        self.data = HeteroData()
        self.ip_map = {}
        self.norm_stats = {}  # 存储归一化统计量
        
    def build(self, df: pd.DataFrame, include_semantic: bool = True, include_struct: bool = True,
              fit_stats: dict = None):
        """
        构建包含三模态特征的异构图
        
        Args:
            df: 流数据 DataFrame
            include_semantic: 是否包含语义特征
            include_struct: 是否包含结构特征
            fit_stats: 归一化统计量（仅训练集计算，避免数据泄露）
            
        Returns:
            data: HeteroData 对象
            ip_map: IP 到节点索引的映射
            norm_stats: 归一化统计量
        """
        print("[ImprovedGraphV3] 构建三模态异构图...")
        
        # 1. 节点映射
        ips = pd.unique(np.concatenate([df['src_ip'].unique(), df['dst_ip'].unique()]))
        self.ip_map = {ip: i for i, ip in enumerate(ips)}
        num_nodes = len(self.ip_map)
        print(f"[ImprovedGraphV3] 节点数：{num_nodes}")
        
        # 2. 边构建 (聚合模式)
        print("[ImprovedGraphV3] 构建边...")
        edges = df.groupby(['src_ip', 'dst_ip']).size().reset_index(name='count')
        
        src_idx = [self.ip_map[ip] for ip in edges['src_ip']]
        dst_idx = [self.ip_map[ip] for ip in edges['dst_ip']]
        
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        edge_weight = torch.log1p(torch.tensor(edges['count'].values, dtype=torch.float)).unsqueeze(1)
        
        self.data['ip', 'flow', 'ip'].edge_index = edge_index
        self.data['ip', 'flow', 'ip'].edge_attr = edge_weight
        
        # 3. 提取统计特征 (模态 1) - 快速版本
        print("[ImprovedGraphV3] 提取统计特征 (快速版)...")
        stat_features, stat_stats = FastBotnetFeatureExtractor.extract_features(
            df, self.ip_map, fit_stats=fit_stats.get('stat') if fit_stats else None
        )
        self.data['ip'].x = stat_features
        
        # 4. 提取语义特征 (模态 2) - 快速版本
        if include_semantic:
            print("[ImprovedGraphV3] 提取语义特征 (快速版)...")
            semantic_features, semantic_stats = FastNodeSemanticFeatureExtractor.extract_semantic_features(
                df, self.ip_map, fit_stats=fit_stats.get('semantic') if fit_stats else None
            )
            self.data['ip'].semantic_x = semantic_features
        else:
            self.data['ip'].semantic_x = None
            semantic_stats = None
            
        # 5. 提取结构特征 (模态 3) - 快速版本
        if include_struct:
            print("[ImprovedGraphV3] 提取结构特征 (快速版)...")
            struct_features, struct_stats = FastStructuralFeatureExtractor.extract_structural_features(
                df, self.ip_map, fit_stats=fit_stats.get('struct') if fit_stats else None
            )
            self.data['ip'].struct_x = struct_features
        else:
            self.data['ip'].struct_x = None
            struct_stats = None
        
        # 保存归一化统计量
        self.norm_stats = {
            'stat': stat_stats,
            'semantic': semantic_stats,
            'struct': struct_stats
        }
        
        print(f"[ImprovedGraphV3] 图构建完成。")
        print(f"  - 统计特征维度：{stat_features.shape}")
        if include_semantic:
            print(f"  - 语义特征维度：{semantic_features.shape}")
        if include_struct:
            print(f"  - 结构特征维度：{struct_features.shape}")
        print(f"  - 边数：{edge_index.shape[1]}")
        
        return self.data, self.ip_map, self.norm_stats
    
    def get_norm_stats(self):
        """获取归一化统计量（用于验证/测试集）"""
        return self.norm_stats


class MiniBatchGraphBuilderV3:
    """
    支持 Mini-batch 训练的图构建器 V3
    修复数据泄露：使用简单随机采样而非分层采样
    """
    
    @staticmethod
    def prepare_for_training(data: HeteroData, labels: torch.Tensor, train_ratio: float = 0.8):
        """
        准备训练数据 - 使用简单随机采样，避免使用标签信息
        
        Args:
            data: HeteroData 对象
            labels: 节点标签
            train_ratio: 训练集比例
            
        Returns:
            train_mask, val_mask: 训练和验证掩码
        """
        num_nodes = labels.size(0)
        indices = np.arange(num_nodes)
        
        # 简单随机采样（不使用标签信息）
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_idx = int(num_nodes * train_ratio)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        
        data['ip'].train_mask = train_mask
        data['ip'].val_mask = val_mask
        data['ip'].y = labels
        
        return train_mask, val_mask