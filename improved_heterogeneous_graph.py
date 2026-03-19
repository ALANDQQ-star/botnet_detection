"""
改进的异构图构建模块
整合三模态特征：统计特征 + 语义特征 + 结构特征
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
import gc
import warnings

warnings.filterwarnings('ignore')

# 导入现有模块
from heterogeneous_graph import BotnetFeatureExtractor
from semantic_feature_extractor import NodeSemanticFeatureExtractor, StructuralFeatureExtractor


class ImprovedHeterogeneousGraphBuilder:
    """
    改进的异构图构建器
    
    整合三模态特征：
    1. 统计特征 (64维): 从现有的BotnetFeatureExtractor获取
    2. 语义特征 (48维): 节点行为语义
    3. 结构特征 (16维): 图结构特征
    """
    
    def __init__(self):
        self.data = HeteroData()
        self.ip_map = {}
        
    def build(self, df: pd.DataFrame, include_semantic: bool = True, include_struct: bool = True):
        """
        构建包含三模态特征的异构图
        
        Args:
            df: 流数据DataFrame
            include_semantic: 是否包含语义特征
            include_struct: 是否包含结构特征
            
        Returns:
            data: HeteroData对象
            ip_map: IP到节点索引的映射
        """
        print("[ImprovedGraph] 构建三模态异构图...")
        
        # 1. 节点映射
        ips = pd.unique(np.concatenate([df['src_ip'].unique(), df['dst_ip'].unique()]))
        self.ip_map = {ip: i for i, ip in enumerate(ips)}
        num_nodes = len(self.ip_map)
        print(f"[ImprovedGraph] 节点数: {num_nodes}")
        
        # 2. 边构建 (聚合模式)
        print("[ImprovedGraph] 构建边...")
        edges = df.groupby(['src_ip', 'dst_ip']).size().reset_index(name='count')
        
        src_idx = [self.ip_map[ip] for ip in edges['src_ip']]
        dst_idx = [self.ip_map[ip] for ip in edges['dst_ip']]
        
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        edge_weight = torch.log1p(torch.tensor(edges['count'].values, dtype=torch.float)).unsqueeze(1)
        
        self.data['ip', 'flow', 'ip'].edge_index = edge_index
        self.data['ip', 'flow', 'ip'].edge_attr = edge_weight
        
        # 3. 提取统计特征 (模态1)
        print("[ImprovedGraph] 提取统计特征...")
        stat_features = BotnetFeatureExtractor.extract_features(df, self.ip_map)
        self.data['ip'].x = stat_features  # 主特征
        
        # 4. 提取语义特征 (模态2)
        if include_semantic:
            print("[ImprovedGraph] 提取语义特征...")
            semantic_features = NodeSemanticFeatureExtractor.extract_semantic_features(df, self.ip_map)
            self.data['ip'].semantic_x = semantic_features
        else:
            self.data['ip'].semantic_x = None
            
        # 5. 提取结构特征 (模态3)
        if include_struct:
            print("[ImprovedGraph] 提取结构特征...")
            struct_features = StructuralFeatureExtractor.extract_structural_features(df, self.ip_map)
            self.data['ip'].struct_x = struct_features
        else:
            self.data['ip'].struct_x = None
        
        print(f"[ImprovedGraph] 图构建完成。")
        print(f"  - 统计特征维度: {stat_features.shape}")
        if include_semantic:
            print(f"  - 语义特征维度: {semantic_features.shape}")
        if include_struct:
            print(f"  - 结构特征维度: {struct_features.shape}")
        print(f"  - 边数: {edge_index.shape[1]}")
        
        return self.data, self.ip_map
    
    def build_with_labels(self, df: pd.DataFrame, label_col: str = 'label'):
        """
        构建图并添加标签
        
        Args:
            df: 流数据DataFrame
            label_col: 标签列名
            
        Returns:
            data: HeteroData对象（包含标签）
            ip_map: IP到节点索引的映射
            bot_ips: 僵尸网络IP列表
        """
        data, ip_map = self.build(df)
        
        # 添加标签
        labels, bot_ips = self._extract_labels(df, ip_map, label_col)
        data['ip'].y = labels
        
        return data, ip_map, bot_ips
    
    def _extract_labels(self, df: pd.DataFrame, ip_map: dict, label_col: str = 'label'):
        """提取节点标签"""
        bot_keywords = ['botnet', 'cc', 'c&c', 'neris', 'rbot', 'virut', 'menti', 'murlo', 'sogou']
        
        if label_col in df.columns:
            df['is_bot'] = df[label_col].apply(
                lambda x: any(k in str(x).lower() for k in bot_keywords)
            )
        else:
            # 如果没有标签列，返回全零标签
            return torch.zeros(len(ip_map), dtype=torch.float32), []
        
        bot_ips = set(df[df['is_bot']]['src_ip'].unique()) | set(df[df['is_bot']]['dst_ip'].unique())
        
        y = np.zeros(len(ip_map), dtype=np.float32)
        for ip, idx in ip_map.items():
            if ip in bot_ips:
                y[idx] = 1.0
                
        return torch.tensor(y), list(bot_ips)


class DynamicGraphAugmentor:
    """
    动态图数据增强器
    
    策略：
    1. 节点丢弃：根据节点重要性自适应丢弃
    2. 边扰动：保持语义结构的边修改
    3. 特征掩码：基于特征重要性的掩码
    """
    
    def __init__(self, aug_ratio: float = 0.2, adaptive: bool = True):
        self.aug_ratio = aug_ratio
        self.adaptive = adaptive
        
    def augment(self, data: HeteroData, node_importance: torch.Tensor = None) -> HeteroData:
        """
        生成增强视图
        
        Args:
            data: 原始HeteroData
            node_importance: 节点重要性分数 (可选)
            
        Returns:
            aug_data: 增强后的数据
        """
        aug_data = data.clone()
        num_nodes = aug_data['ip'].x.size(0)
        
        # 1. 边扰动
        edge_index = aug_data['ip', 'flow', 'ip'].edge_index
        num_edges = edge_index.size(1)
        
        # 随机保留部分边
        edge_mask = torch.rand(num_edges) > (self.aug_ratio / 2)
        aug_data['ip', 'flow', 'ip'].edge_index = edge_index[:, edge_mask]
        
        if hasattr(aug_data['ip', 'flow', 'ip'], 'edge_attr') and aug_data['ip', 'flow', 'ip'].edge_attr is not None:
            aug_data['ip', 'flow', 'ip'].edge_attr = aug_data['ip', 'flow', 'ip'].edge_attr[edge_mask]
        
        # 2. 特征掩码
        x = aug_data['ip'].x
        feat_dim = x.size(1)
        feat_mask = torch.rand(feat_dim) > self.aug_ratio
        aug_data['ip'].x = x * feat_mask.float()
        
        # 3. 语义特征掩码
        if hasattr(aug_data['ip'], 'semantic_x') and aug_data['ip'].semantic_x is not None:
            sem_x = aug_data['ip'].semantic_x
            sem_mask = torch.rand(sem_x.size(1)) > self.aug_ratio
            aug_data['ip'].semantic_x = sem_x * sem_mask.float()
        
        # 4. 结构特征掩码
        if hasattr(aug_data['ip'], 'struct_x') and aug_data['ip'].struct_x is not None:
            struct_x = aug_data['ip'].struct_x
            struct_mask = torch.rand(struct_x.size(1)) > self.aug_ratio
            aug_data['ip'].struct_x = struct_x * struct_mask.float()
        
        return aug_data
    
    def create_contrastive_views(self, data: HeteroData, node_importance: torch.Tensor = None):
        """
        创建两个对比视图
        
        Args:
            data: 原始数据
            node_importance: 节点重要性
            
        Returns:
            view1, view2: 两个增强视图
        """
        view1 = self.augment(data, node_importance)
        view2 = self.augment(data, node_importance)
        return view1, view2


class MiniBatchGraphBuilder:
    """
    支持Mini-batch训练的图构建器
    """
    
    @staticmethod
    def prepare_for_training(data: HeteroData, labels: torch.Tensor, train_ratio: float = 0.8):
        """
        准备训练数据
        
        Args:
            data: HeteroData对象
            labels: 节点标签
            train_ratio: 训练集比例
            
        Returns:
            train_mask, val_mask: 训练和验证掩码
        """
        from sklearn.model_selection import train_test_split
        
        num_nodes = labels.size(0)
        indices = np.arange(num_nodes)
        
        # 分层采样
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=1-train_ratio, 
            stratify=labels.numpy(),
            random_state=42
        )
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        
        data['ip'].train_mask = train_mask
        data['ip'].val_mask = val_mask
        data['ip'].y = labels
        
        return train_mask, val_mask


def build_improved_graph_from_df(df: pd.DataFrame, include_semantic: bool = True, include_struct: bool = True):
    """
    便捷函数：从DataFrame构建改进的异构图
    
    Args:
        df: 流数据DataFrame
        include_semantic: 是否包含语义特征
        include_struct: 是否包含结构特征
        
    Returns:
        data: HeteroData对象
        ip_map: IP到节点索引的映射
    """
    builder = ImprovedHeterogeneousGraphBuilder()
    return builder.build(df, include_semantic, include_struct)