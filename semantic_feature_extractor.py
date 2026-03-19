"""
语义特征提取器 - O(n) 优化版
借鉴 Bot-DM 的 Token 编码思想，创新性地应用于图节点级别的语义编码
使用纯向量化操作，避免 O(n²) 的重复遍历
"""

import torch
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')


class NodeSemanticFeatureExtractor:
    """
    【O(n)优化版】节点语义特征提取器
    
    时间复杂度：O(n)，其中 n 是流记录数
    空间复杂度：O(m)，其中 m 是节点数
    
    核心优化：
    1. 使用 groupby.agg 一次性聚合所有统计量
    2. 避免在循环中重复过滤 DataFrame
    3. 减少特征维度从 48 到 32，保留核心特征
    """
    
    SEMANTIC_DIM = 32  # 优化后的维度：32
    
    @staticmethod
    def extract_semantic_features(df: pd.DataFrame, node_map: dict) -> torch.Tensor:
        """
        【O(n)优化版】提取节点语义特征
        """
        print("[SemanticFeature] O(n) 优化版 - 提取节点语义特征 (32-Dim)...")
        num_nodes = len(node_map)
        
        if num_nodes == 0:
            return torch.zeros((1, NodeSemanticFeatureExtractor.SEMANTIC_DIM), dtype=torch.float32)
        
        # 初始化特征矩阵
        features = np.zeros((num_nodes, NodeSemanticFeatureExtractor.SEMANTIC_DIM), dtype=np.float32)
        
        if 'src_ip' not in df.columns or len(df) == 0:
            print("[SemanticFeature] 警告：缺少 src_ip 列或数据为空")
            return torch.from_numpy(features)
        
        # 预处理
        df = df.copy()
        df['dst_port'] = pd.to_numeric(df['dst_port'], errors='coerce').fillna(-1).astype(int)
        df['protocol'] = df['protocol'].fillna('unknown').str.lower()
        
        # ===== 关键优化：单次遍历完成所有聚合 =====
        print("[SemanticFeature] 使用向量化聚合...")
        grouped = df.groupby('src_ip')
        
        # 1. 基本计数
        flow_counts = grouped.size()
        
        # 2. 端口统计
        port_stats = grouped['dst_port'].agg(['nunique', 'mean', 'std', 'min', 'max'])
        
        # 3. 协议分布 (使用向量化交叉表 crosstab 或 groupby)
        proto_dist_df = pd.crosstab(df['src_ip'], df['protocol'])
        if not proto_dist_df.empty:
            proto_dist_df = proto_dist_df.div(proto_dist_df.sum(axis=1), axis=0) # 预先归一化，加快后面提取速度
            # 提前计算熵
            proto_probs = proto_dist_df.replace(0, np.nan)
            proto_entropy = -(proto_probs * np.log2(proto_probs)).sum(axis=1).fillna(0)
        else:
            proto_entropy = pd.Series()
        
        # 4. 字节统计
        if 'bytes' in df.columns:
            bytes_stats = grouped['bytes'].agg(['sum', 'mean', 'std', 'max'])
        else:
            bytes_stats = pd.DataFrame(0, index=port_stats.index, columns=['sum', 'mean', 'std', 'max'])
        
        # 5. 目标 IP 统计
        if 'dst_ip' in df.columns:
            dst_ip_stats = grouped['dst_ip'].agg(['nunique'])
        else:
            dst_ip_stats = pd.DataFrame(0, index=port_stats.index, columns=['nunique'])
        
        # 6. 端口范围统计 (向量化)
        # 用 pandas 向量化计算避免逐组 apply，极大地提高速度
        df['port_well_known'] = ((df['dst_port'] >= 0) & (df['dst_port'] <= 1023)).astype(int)
        df['port_registered'] = ((df['dst_port'] >= 1024) & (df['dst_port'] <= 49151)).astype(int)
        df['port_dynamic'] = ((df['dst_port'] >= 49152) & (df['dst_port'] <= 65535)).astype(int)
        
        port_range_stats = df.groupby('src_ip')[['port_well_known', 'port_registered', 'port_dynamic']].sum()
        port_range_stats.rename(columns={
            'port_well_known': 'well_known', 
            'port_registered': 'registered', 
            'port_dynamic': 'dynamic'
        }, inplace=True)
        
        # 7. 简化的转换统计 (向量化替代 apply)
        df = df.sort_values(by=['src_ip'])
        df['prev_port'] = df.groupby('src_ip')['dst_port'].shift(1)
        df['same_port_trans'] = (df['dst_port'] == df['prev_port']).astype(int)
        
        trans_sum = df.groupby('src_ip')['same_port_trans'].sum()
        trans_count = df.groupby('src_ip')['same_port_trans'].count() # count non-NA (which means shifted has values)
        
        # unique_ports is already in port_stats['nunique']
        # flow_counts is total length
        trans_result = pd.DataFrame(index=flow_counts.index)
        trans_result['trans_entropy'] = port_stats['nunique'] / flow_counts.replace(0, 1)
        # Handle same_ratio safely
        trans_result['same_ratio'] = np.where(trans_count > 0, trans_sum / trans_count, 1.0)
        
        # ===== 填充特征矩阵 (完全向量化) =====
        print("[SemanticFeature] 填充特征向量...")
        
        # 为了使用向量化，将所有的 index 对齐
        common_index = list(flow_counts.index)
        n_flows = flow_counts.values
        eps = 1e-6
        
        # 预先分配一个 DataFrame 存储特征，这样可以直接矩阵运算
        feat_df = pd.DataFrame(0.0, index=common_index, columns=range(NodeSemanticFeatureExtractor.SEMANTIC_DIM))
        
        # [0-2] 端口范围分布
        if not port_range_stats.empty:
            feat_df.loc[port_range_stats.index, 0] = port_range_stats['well_known'] / (flow_counts.loc[port_range_stats.index] + eps)
            feat_df.loc[port_range_stats.index, 1] = port_range_stats['registered'] / (flow_counts.loc[port_range_stats.index] + eps)
            feat_df.loc[port_range_stats.index, 2] = port_range_stats['dynamic'] / (flow_counts.loc[port_range_stats.index] + eps)
            
        # [3-4] 端口唯一性和熵
        if not port_stats.empty:
            feat_df.loc[port_stats.index, 3] = port_stats['nunique'] / (flow_counts.loc[port_stats.index] + eps)
            feat_df.loc[port_stats.index, 4] = (port_stats['nunique'] / 100).clip(upper=1.0)
            
        # [5-6] 转换模式
        if not trans_result.empty:
            feat_df.loc[trans_result.index, 5] = trans_result['trans_entropy']
            feat_df.loc[trans_result.index, 6] = trans_result['same_ratio']
            
        # [7-10] 协议分布
        if not proto_dist_df.empty:
            for i, proto in enumerate(['tcp', 'udp', 'icmp']):
                if proto in proto_dist_df.columns:
                    feat_df.loc[proto_dist_df.index, 7+i] = proto_dist_df[proto]
            feat_df.loc[proto_dist_df.index, 10] = (1 - feat_df.loc[proto_dist_df.index, [7,8,9]].sum(axis=1)).clip(lower=0)
            
        # [11] 协议熵
        if not proto_entropy.empty:
            feat_df.loc[proto_entropy.index, 11] = proto_entropy
            
        # [12-15] 字节统计
        if not bytes_stats.empty:
            feat_df.loc[bytes_stats.index, 12] = np.log1p(bytes_stats['sum'].abs()) / 20
            feat_df.loc[bytes_stats.index, 13] = np.log1p(bytes_stats['mean'].abs()) / 10
            feat_df.loc[bytes_stats.index, 14] = np.where(bytes_stats['std'] > 0, np.log1p(bytes_stats['std'].abs()) / 10, 0)
            feat_df.loc[bytes_stats.index, 15] = np.log1p(bytes_stats['max'].abs()) / 20
            
        # [16-17] 目标 IP 统计
        if not dst_ip_stats.empty:
            feat_df.loc[dst_ip_stats.index, 16] = dst_ip_stats['nunique'] / (flow_counts.loc[dst_ip_stats.index] + eps)
            feat_df.loc[dst_ip_stats.index, 17] = (dst_ip_stats['nunique'] / 100).clip(upper=1.0)
            
        # [18-19] 端口统计
        if not port_stats.empty:
            feat_df.loc[port_stats.index, 18] = np.where(port_stats['mean'] > 0, np.log1p(port_stats['mean'].abs()) / 15, 0)
            feat_df.loc[port_stats.index, 19] = np.where(port_stats['std'] > 0, np.log1p(port_stats['std'].abs()) / 15, 0)
            
        # [20-23] 流数量相关
        feat_df.loc[common_index, 20] = np.log1p(n_flows) / 15
        feat_df.loc[common_index, 21] = np.clip(n_flows / 1000, a_min=None, a_max=1.0)
        feat_df.loc[common_index, 22] = np.clip(n_flows / 10000, a_min=None, a_max=1.0)
        
        # [24-27] 端口范围变化
        if not port_stats.empty:
            port_range = port_stats['max'] - port_stats['min']
            feat_df.loc[port_stats.index, 24] = np.log1p(port_range) / 20
            feat_df.loc[port_stats.index, 25] = port_stats['min'] / 65535
            feat_df.loc[port_stats.index, 26] = port_stats['max'] / 65535

        # 将 feat_df 的值映射到 node_map 指定的索引中
        # 首先创建一个与 node_map 对应的索引数组
        ips = list(node_map.keys())
        idxs = list(node_map.values())
        
        # 获取有效在 feat_df 中的 ip
        valid_mask = [ip in feat_df.index for ip in ips]
        valid_ips = [ip for ip, m in zip(ips, valid_mask) if m]
        valid_idxs = [idx for idx, m in zip(idxs, valid_mask) if m]
        
        if valid_ips:
            features[valid_idxs, :] = feat_df.loc[valid_ips].values
        
        # 归一化
        print("[SemanticFeature] 归一化处理...")
        for col in range(features.shape[1]):
            col_mean = np.mean(features[:, col])
            col_std = np.std(features[:, col])
            if col_std > 1e-6:
                features[:, col] = (features[:, col] - col_mean) / col_std
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"[SemanticFeature] 语义特征提取完成。维度：{features.shape}")
        return torch.from_numpy(features)


class StructuralFeatureExtractor:
    """
    【O(n) 优化版】结构特征提取器
    
    时间复杂度：O(n)，其中 n 是流记录数
    空间复杂度：O(m)，其中 m 是节点数
    
    核心优化：
    1. 使用 groupby 避免 iterrows
    2. 使用集合操作计算邻居
    3. 避免在循环中重复过滤 DataFrame
    """
    
    STRUCT_DIM = 16  # 结构特征维度
    
    @staticmethod
    def extract_structural_features(df: pd.DataFrame, node_map: dict) -> torch.Tensor:
        """
        【O(n) 优化版】提取结构特征
        """
        print("[StructFeature] O(n) 优化版 - 提取结构特征 (16-Dim)...")
        num_nodes = len(node_map)
        features = np.zeros((num_nodes, StructuralFeatureExtractor.STRUCT_DIM), dtype=np.float32)
        
        if 'src_ip' not in df.columns or 'dst_ip' not in df.columns or len(df) == 0:
            print("[StructFeature] 警告：缺少必要的 IP 列或数据为空")
            return torch.from_numpy(features)
        
        # ===== 关键优化：使用 groupby 一次性构建邻接表 =====
        print("[StructFeature] 构建邻接表...")
        
        # 出邻居：src_ip -> set of dst_ip
        out_neighbors = df.groupby('src_ip')['dst_ip'].apply(lambda x: set(x.unique())).to_dict()
        
        # 入邻居：dst_ip -> set of src_ip
        in_neighbors = df.groupby('dst_ip')['src_ip'].apply(lambda x: set(x.unique())).to_dict()
        
        # 确保所有节点都有邻居（可能有些节点只有入边或出边）
        for ip in node_map.keys():
            if ip not in out_neighbors:
                out_neighbors[ip] = set()
            if ip not in in_neighbors:
                in_neighbors[ip] = set()
        
        # 计算度统计
        out_degrees = {ip: len(neighbors) for ip, neighbors in out_neighbors.items()}
        in_degrees = {ip: len(neighbors) for ip, neighbors in in_neighbors.items()}
        
        all_degrees = list(out_degrees.values()) + list(in_degrees.values())
        max_degree = max(all_degrees) if all_degrees else 1
        mean_degree = np.mean(all_degrees) if all_degrees else 0
        std_degree = np.std(all_degrees) if all_degrees else 1
        
        # 预计算每个 IP 的流统计（避免循环中重复过滤）
        print("[StructFeature] 预计算流统计...")
        flow_stats = df.groupby('src_ip').agg({
            'dst_ip': 'nunique',
        }).to_dict()['dst_ip']
        
        if 'dst_port' in df.columns:
            port_diversity = df.groupby('src_ip')['dst_port'].nunique().to_dict()
        else:
            port_diversity = {}
        
        if 'protocol' in df.columns:
            proto_diversity = df.groupby('src_ip')['protocol'].nunique().to_dict()
        else:
            proto_diversity = {}
        
        # ===== 填充特征矩阵 =====
        print("[StructFeature] 填充特征向量...")
        
        for ip, idx in node_map.items():
            out_degree = out_degrees.get(ip, 0)
            in_degree = in_degrees.get(ip, 0)
            total_degree = out_degree + in_degree
            
            if total_degree == 0:
                continue
            
            eps = 1e-6
            
            # [0-3] 度特征
            features[idx, 0] = out_degree / (max_degree + eps)
            features[idx, 1] = in_degree / (max_degree + eps)
            features[idx, 2] = total_degree / (max_degree * 2 + eps)
            features[idx, 3] = out_degree / (in_degree + 1)
            
            # [4-7] 邻居多样性
            out_neigh = out_neighbors.get(ip, set())
            in_neigh = in_neighbors.get(ip, set())
            
            overlap = len(out_neigh & in_neigh)
            union = len(out_neigh | in_neigh)
            features[idx, 4] = overlap / (union + eps)
            features[idx, 5] = len(out_neigh - in_neigh) / (len(out_neigh) + eps)
            features[idx, 6] = len(in_neigh - out_neigh) / (len(in_neigh) + eps)
            features[idx, 7] = overlap / (total_degree + eps)
            
            # [8-11] 中心性相关
            features[idx, 8] = total_degree / (num_nodes - 1 + eps)
            features[idx, 9] = total_degree / (mean_degree + eps)
            features[idx, 10] = 1.0 if total_degree > mean_degree else 0.0
            features[idx, 11] = (total_degree - mean_degree) / (std_degree + eps)
            
            # [12-15] 流特征相关的结构信息
            if ip in flow_stats:
                num_dst_ips = flow_stats[ip]
                features[idx, 12] = min(1.0, num_dst_ips / 100)
                
                if ip in port_diversity:
                    features[idx, 13] = min(1.0, port_diversity[ip] / 1000)
                
                if ip in proto_diversity:
                    features[idx, 14] = proto_diversity[ip] / 3.0
        
        # 归一化
        for col in range(features.shape[1]):
            col_mean = np.mean(features[:, col])
            col_std = np.std(features[:, col])
            if col_std > 1e-6:
                features[:, col] = (features[:, col] - col_mean) / col_std
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"[StructFeature] 结构特征提取完成。维度：{features.shape}")
        return torch.from_numpy(features)