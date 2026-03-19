"""
快速特征提取器 - V3 版本
核心优化：
1. 避免 groupby.apply，使用纯向量化操作
2. 避免 iterrows，使用矩阵操作
3. 修复数据泄露：归一化统计量仅从训练集计算
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class FastBotnetFeatureExtractor:
    """
    【速度优化版】统计特征提取器 (64-Dim)
    
    优化策略：
    1. 使用 groupby.agg 替代 apply
    2. 使用向量化操作计算 IAT 统计
    3. 使用矩阵操作填充特征，避免 iterrows
    """
    
    FEAT_DIM = 64
    
    @staticmethod
    def extract_features(df: pd.DataFrame, node_map: dict, fit_stats=None):
        """
        提取统计特征
        
        Args:
            df: 流数据 DataFrame
            node_map: IP 到索引的映射
            fit_stats: 用于拟合的统计量（避免数据泄露，仅用训练集）
            
        Returns:
            features: (num_nodes, 64) tensor
            norm_stats: 归一化统计量（用于验证/测试集）
        """
        num_nodes = len(node_map)
        eps = 1e-6
        
        # 数据预处理
        df = df.copy()
        df['dst_port'] = pd.to_numeric(df['dst_port'], errors='coerce').fillna(-1).astype(int)
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['ts'] = df['start_time'].astype(np.int64) // 10**9
        
        # 基础特征
        df['bpp'] = df['bytes'] / (df['packets'] + eps)
        df['pps'] = df['packets'] / (df['duration'] + 0.1)
        df['bps'] = df['bytes'] / (df['duration'] + 0.1)
        df['dst_port_priv'] = ((df['dst_port'] >= 0) & (df['dst_port'] < 1024)).astype(int)
        df['dst_port_high'] = (df['dst_port'] > 49151).astype(int)
        
        # 分组聚合
        grp = df.groupby('src_ip')
        
        agg_dict = {
            'dst_ip': 'nunique',
            'dst_port': 'nunique',
            'packets': ['mean', 'std', 'sum', 'max', 'min'],
            'bytes': ['mean', 'std', 'sum', 'max', 'min'],
            'duration': ['mean', 'std', 'sum', 'max'],
            'bpp': ['mean', 'max', 'std'],
            'pps': ['mean', 'max', 'std'],
            'bps': ['mean', 'max'],
            'dst_port_priv': ['sum', 'mean'],
            'dst_port_high': ['sum', 'mean']
        }
        
        stats = grp.agg(agg_dict)
        stats.columns = [f"{c[0]}_{c[1]}" for c in stats.columns]
        
        # IAT 统计 - 向量化优化版
        # 使用 sort + diff 替代 apply
        df_sorted = df.sort_values(['src_ip', 'ts'])
        df_sorted['ts_diff'] = df_sorted.groupby('src_ip')['ts'].diff()
        
        # 过滤有效间隔 (> 0.05 秒)
        df_valid = df_sorted[df_sorted['ts_diff'] > 0.05]
        
        iat_stats = df_valid.groupby('src_ip')['ts_diff'].agg(['mean', 'std']).fillna(10.0)
        iat_stats.columns = ['iat_mean', 'iat_std']
        
        # 协议分布
        proto_counts = pd.crosstab(df['src_ip'], df['protocol'])
        proto_sum = proto_counts.sum(axis=1) + eps
        for p in ['tcp', 'udp', 'icmp']:
            if p not in proto_counts.columns:
                proto_counts[p] = 0
            proto_counts[p] = proto_counts[p] / proto_sum
        
        # 构建特征 DataFrame
        fdf = pd.DataFrame(index=stats.index)
        fdf = fdf.join(stats)
        fdf['flow_count'] = grp.size()
        fdf = fdf.join(iat_stats, how='left').fillna(10.0)
        fdf = fdf.join(proto_counts[['tcp', 'udp', 'icmp']], how='left').fillna(0)
        
        # 高级特征
        fdf['pkts_cv'] = fdf['packets_std'] / (fdf['packets_mean'] + eps)
        fdf['bytes_cv'] = fdf['bytes_std'] / (fdf['bytes_mean'] + eps)
        fdf['dur_cv'] = fdf['duration_std'] / (fdf['duration_mean'] + eps)
        fdf['distinct_ip_ratio'] = fdf['dst_ip_nunique'] / fdf['flow_count']
        fdf['distinct_port_ratio'] = fdf['dst_port_nunique'] / fdf['flow_count']
        
        # 对数变换
        log_cols = [col for col in fdf.columns if any(x in col for x in ['mean', 'sum', 'max', 'std', 'count'])]
        for col in log_cols:
            fdf[col] = np.log1p(fdf[col].fillna(0))
        
        # 填充特征矩阵
        feat_dim = FastBotnetFeatureExtractor.FEAT_DIM
        x = np.zeros((num_nodes, feat_dim), dtype=np.float32)
        
        # 创建 IP 到索引的逆映射
        reverse_map = {v: k for k, v in node_map.items()}
        ips_array = np.array([reverse_map.get(i, '') for i in range(num_nodes)])
        
        # 向量化填充
        feat_cols = [
            'dst_ip_nunique', 'dst_port_nunique', 'flow_count',
            'distinct_ip_ratio', 'distinct_port_ratio',
            'packets_sum', 'packets_mean', 'packets_std', 'packets_max', 'packets_min',
            'pkts_cv',
            'bytes_sum', 'bytes_mean', 'bytes_std', 'bytes_max', 'bytes_min', 'bytes_cv',
            'duration_mean', 'duration_std', 'duration_sum', 'duration_max', 'dur_cv',
            'pps_mean', 'pps_max', 'pps_std', 'bps_mean', 'bps_max',
            'bpp_mean', 'bpp_max', 'bpp_std',
            'iat_mean', 'iat_std',
            'tcp', 'udp', 'icmp',
            'dst_port_priv_sum', 'dst_port_priv_mean',
            'dst_port_high_sum', 'dst_port_high_mean',
        ]
        
        # 确保所有列存在
        for col in feat_cols:
            if col not in fdf.columns:
                fdf[col] = 0
        
        # 填充矩阵 - 使用 merge 而非 iterrows
        fdf_with_idx = fdf.reset_index().merge(
            pd.DataFrame({'src_ip': ips_array, 'node_idx': range(num_nodes)}),
            on='src_ip',
            how='right'
        )
        
        # 填充特征
        for i, col in enumerate(feat_cols):
            if i < feat_dim:
                x[:, i] = fdf_with_idx[col].fillna(0).values
        
        # 额外特征填充
        x[:, 44] = fdf_with_idx['iat_std'].fillna(10.0).values - fdf_with_idx['iat_mean'].fillna(10.0).values
        
        # 归一化
        if fit_stats is None:
            # 训练集：计算统计量
            norm_stats = {
                'mean': np.nanmean(x, axis=0),
                'std': np.nanstd(x, axis=0) + eps
            }
        else:
            # 验证/测试集：使用训练集统计量
            norm_stats = fit_stats
        
        x = (x - norm_stats['mean']) / norm_stats['std']
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        return torch.from_numpy(x), norm_stats


class FastNodeSemanticFeatureExtractor:
    """
    【速度优化版】语义特征提取器 (32-Dim)
    
    优化策略：
    1. 使用 groupby.agg 替代 apply
    2. 避免 iterrows
    """
    
    SEMANTIC_DIM = 32
    
    @staticmethod
    def extract_semantic_features(df: pd.DataFrame, node_map: dict, fit_stats=None):
        num_nodes = len(node_map)
        eps = 1e-6
        
        df = df.copy()
        df['dst_port'] = pd.to_numeric(df['dst_port'], errors='coerce').fillna(-1).astype(int)
        df['protocol'] = df['protocol'].fillna('unknown').str.lower()
        
        # 分组聚合
        grp = df.groupby('src_ip')
        
        # 基本计数
        flow_counts = grp.size()
        
        # 端口统计
        port_stats = grp['dst_port'].agg(['nunique', 'mean', 'std', 'min', 'max']).fillna(0)
        
        # 字节统计
        if 'bytes' in df.columns:
            bytes_stats = grp['bytes'].agg(['sum', 'mean', 'std', 'max']).fillna(0)
        else:
            bytes_stats = pd.DataFrame(0, index=port_stats.index, columns=['sum', 'mean', 'std', 'max'])
        
        # 目标 IP 统计
        if 'dst_ip' in df.columns:
            dst_ip_stats = grp['dst_ip'].agg(['nunique']).fillna(0)
        else:
            dst_ip_stats = pd.DataFrame(0, index=port_stats.index, columns=['nunique'])
        
        # 协议分布 - 向量化
        proto_pivot = pd.crosstab(df['src_ip'], df['protocol'])
        for p in ['tcp', 'udp', 'icmp']:
            if p not in proto_pivot.columns:
                proto_pivot[p] = 0
        
        # 端口范围统计
        df['port_well_known'] = ((df['dst_port'] >= 0) & (df['dst_port'] <= 1023)).astype(int)
        df['port_registered'] = ((df['dst_port'] >= 1024) & (df['dst_port'] <= 49151)).astype(int)
        df['port_dynamic'] = (df['dst_port'] >= 49152).astype(int)
        
        port_range_stats = grp[['port_well_known', 'port_registered', 'port_dynamic']].sum()
        
        # 转换统计 - 简化版
        df_sorted = df.sort_values(['src_ip', 'dst_port'])
        df_sorted['port_changed'] = (df_sorted.groupby('src_ip')['dst_port'].shift() != df_sorted['dst_port']).astype(int)
        trans_stats = df_sorted.groupby('src_ip')['port_changed'].agg(['mean', 'sum'])
        trans_stats.columns = ['same_ratio', 'trans_count']
        trans_stats['same_ratio'] = 1 - trans_stats['same_ratio']
        
        # 构建特征矩阵
        features = np.zeros((num_nodes, FastNodeSemanticFeatureExtractor.SEMANTIC_DIM), dtype=np.float32)
        
        reverse_map = {v: k for k, v in node_map.items()}
        ips_array = np.array([reverse_map.get(i, '') for i in range(num_nodes)])
        
        fdf = pd.DataFrame({'src_ip': ips_array, 'node_idx': range(num_nodes)})
        fdf = fdf.merge(flow_counts.reset_index(name='flow_count'), on='src_ip', how='left').fillna(0)
        fdf = fdf.merge(port_stats.reset_index(), on='src_ip', how='left').fillna(0)
        fdf = fdf.merge(bytes_stats.reset_index(), on='src_ip', how='left').fillna(0)
        fdf = fdf.merge(dst_ip_stats.reset_index(), on='src_ip', how='left').fillna(0)
        fdf = fdf.merge(port_range_stats.reset_index(), on='src_ip', how='left').fillna(0)
        fdf = fdf.merge(trans_stats.reset_index(), on='src_ip', how='left').fillna(0)
        fdf = fdf.merge(proto_pivot.reset_index(), on='src_ip', how='left').fillna(0)
        
        # 填充特征
        idx = fdf['node_idx'].values
        n = fdf['flow_count'] + eps
        
        # [0-2] 端口范围分布
        features[:, 0] = fdf['port_well_known'] / n
        features[:, 1] = fdf['port_registered'] / n
        features[:, 2] = fdf['port_dynamic'] / n
        
        # [3-4] 端口统计 - 修复列名引用
        port_nunique_col = 'dst_port_nunique' if 'dst_port_nunique' in fdf.columns else 'nunique'
        if port_nunique_col in fdf.columns:
            features[:, 3] = fdf[port_nunique_col] / n
            features[:, 4] = np.minimum(1.0, fdf[port_nunique_col] / 100)
        else:
            features[:, 3] = 0
            features[:, 4] = 0
        
        # [5-6] 转换模式
        features[:, 5] = fdf.get('trans_count', 0) / n
        features[:, 6] = fdf.get('same_ratio', 1.0)
        
        # [7-11] 协议分布
        for i, p in enumerate(['tcp', 'udp', 'icmp']):
            features[:, 7+i] = fdf.get(p, 0)
        features[:, 10] = np.maximum(0, 1 - features[:, 7] - features[:, 8] - features[:, 9])
        
        # [11] 协议熵
        probs = np.stack([fdf.get(p, 0) for p in ['tcp', 'udp', 'icmp']], axis=1) + eps
        features[:, 11] = -np.sum(probs * np.log2(probs), axis=1)
        
        # [12-15] 字节统计
        for i, col in enumerate(['sum', 'mean', 'std', 'max']):
            features[:, 12+i] = np.log1p(np.abs(fdf.get(f'bytes_{col}', 0))) / (20 if i in [0, 3] else 10)
        
        # [16-17] 目标 IP 统计
        dst_ip_nunique_col = 'dst_ip_nunique' if 'dst_ip_nunique' in fdf.columns else 'nunique'
        if dst_ip_nunique_col in fdf.columns:
            features[:, 16] = fdf[dst_ip_nunique_col] / n
            features[:, 17] = np.minimum(1.0, fdf[dst_ip_nunique_col] / 100)
        else:
            features[:, 16] = 0
            features[:, 17] = 0
        
        # [18-19] 端口统计
        port_mean_col = 'port_mean' if 'port_mean' in fdf.columns else 'mean'
        port_std_col = 'port_std' if 'port_std' in fdf.columns else 'std'
        if port_mean_col in fdf.columns:
            features[:, 18] = np.log1p(np.abs(fdf[port_mean_col])) / 15
        else:
            features[:, 18] = 0
        if port_std_col in fdf.columns:
            features[:, 19] = np.log1p(np.abs(fdf[port_std_col])) / 15
        else:
            features[:, 19] = 0
        
        # [20-21] 流数量相关
        features[:, 20] = np.log1p(fdf['flow_count']) / 15
        features[:, 21] = np.minimum(1.0, fdf['flow_count'] / 1000)
        
        # 归一化
        if fit_stats is None:
            norm_stats = {
                'mean': np.nanmean(features, axis=0),
                'std': np.nanstd(features, axis=0) + eps
            }
        else:
            norm_stats = fit_stats
        
        features = (features - norm_stats['mean']) / norm_stats['std']
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return torch.from_numpy(features), norm_stats


class FastStructuralFeatureExtractor:
    """
    【速度优化版】结构特征提取器 (16-Dim)
    
    使用简化方法快速计算结构特征
    """
    
    STRUCT_DIM = 16
    
    @staticmethod
    def extract_structural_features(df: pd.DataFrame, node_map: dict, fit_stats=None):
        num_nodes = len(node_map)
        eps = 1e-6
        
        # 使用简化方法计算结构特征
        # 避免 apply，使用聚合操作
        
        # 出度和入度
        out_degrees = df.groupby('src_ip').size().to_dict()
        in_degrees = df.groupby('dst_ip').size().to_dict()
        
        # 确保所有节点都有度数
        for ip in node_map.keys():
            if ip not in out_degrees:
                out_degrees[ip] = 0
            if ip not in in_degrees:
                in_degrees[ip] = 0
        
        # 唯一目标 IP（使用 drop_duplicates + groupby 替代 nunique）
        unique_edges = df[['src_ip', 'dst_ip']].drop_duplicates()
        dst_ip_count = unique_edges.groupby('src_ip').size().to_dict()
        
        # 唯一目标端口
        if 'dst_port' in df.columns:
            unique_port_edges = df[['src_ip', 'dst_port']].drop_duplicates()
            port_count = unique_port_edges.groupby('src_ip').size().to_dict()
        else:
            port_count = {}
        
        # 协议多样性（简化）
        if 'protocol' in df.columns:
            unique_proto_edges = df[['src_ip', 'protocol']].drop_duplicates()
            proto_count = unique_proto_edges.groupby('src_ip').size().to_dict()
        else:
            proto_count = {}
        
        # 计算度统计
        all_degrees = list(out_degrees.values()) + list(in_degrees.values())
        max_degree = max(all_degrees) if all_degrees else 1
        mean_degree = np.mean(all_degrees) if all_degrees else 0
        std_degree = np.std(all_degrees) if all_degrees else 1
        
        # 构建特征矩阵 - 使用向量化方法
        reverse_map = {v: k for k, v in node_map.items()}
        ips_array = [reverse_map.get(i, -1) for i in range(num_nodes)]
        
        # 创建 DataFrame 用于向量化填充
        fdf = pd.DataFrame({
            'node_idx': range(num_nodes),
            'src_ip': [reverse_map.get(i, '') for i in range(num_nodes)]
        })
        
        # 反转映射用于 merge
        ip_to_idx = {v: k for k, v in node_map.items()}
        
        # 出度数据
        out_df = pd.DataFrame(list(out_degrees.items()), columns=['src_ip', 'out_degree'])
        in_df = pd.DataFrame(list(in_degrees.items()), columns=['src_ip', 'in_degree'])
        dst_df = pd.DataFrame(list(dst_ip_count.items()), columns=['src_ip', 'dst_ip_count'])
        
        # 合并数据
        fdf = fdf.merge(out_df, on='src_ip', how='left').fillna(0)
        fdf = fdf.merge(in_df, on='src_ip', how='left').fillna(0)
        fdf = fdf.merge(dst_df, on='src_ip', how='left').fillna(0)
        
        # 端口和协议多样性
        if port_count:
            port_df = pd.DataFrame(list(port_count.items()), columns=['src_ip', 'port_diversity'])
            fdf = fdf.merge(port_df, on='src_ip', how='left').fillna(0)
        else:
            fdf['port_diversity'] = 0
            
        if proto_count:
            proto_df = pd.DataFrame(list(proto_count.items()), columns=['src_ip', 'proto_diversity'])
            fdf = fdf.merge(proto_df, on='src_ip', how='left').fillna(0)
        else:
            fdf['proto_diversity'] = 0
        
        # 计算特征
        features = np.zeros((num_nodes, FastStructuralFeatureExtractor.STRUCT_DIM), dtype=np.float32)
        
        out_degree = fdf['out_degree'].values
        in_degree = fdf['in_degree'].values
        total_degree = out_degree + in_degree
        dst_ip_cnt = fdf['dst_ip_count'].values
        port_div = fdf['port_diversity'].values
        proto_div = fdf['proto_diversity'].values
        
        # 简化的邻居重叠估计（使用唯一性比率近似）
        # 真正的重叠计算需要集合操作，这里用启发式方法近似
        overlap_ratio = np.minimum(1.0, dst_ip_cnt / (np.maximum(out_degree, 1) + 1))
        
        # [0-3] 度特征
        features[:, 0] = out_degree / (max_degree + eps)
        features[:, 1] = in_degree / (max_degree + eps)
        features[:, 2] = total_degree / (max_degree * 2 + eps)
        features[:, 3] = out_degree / (in_degree + 1)
        
        # [4-7] 邻居多样性（简化估计）
        features[:, 4] = overlap_ratio  # 重叠比率估计
        features[:, 5] = 1 - overlap_ratio  # 出邻居特有比率
        features[:, 6] = 1 - overlap_ratio  # 入邻居特有比率
        features[:, 7] = overlap_ratio / (total_degree / max_degree + eps)
        
        # [8-11] 中心性相关
        features[:, 8] = total_degree / (num_nodes - 1 + eps)
        features[:, 9] = total_degree / (mean_degree + eps)
        features[:, 10] = (total_degree > mean_degree).astype(float)
        features[:, 11] = (total_degree - mean_degree) / (std_degree + eps)
        
        # [12-15] 流特征
        features[:, 12] = np.minimum(1.0, dst_ip_cnt / 100)
        features[:, 13] = np.minimum(1.0, port_div / 1000)
        features[:, 14] = np.minimum(1.0, proto_div / 3)
        features[:, 15] = np.minimum(1.0, total_degree / 1000)
        
        # 归一化
        if fit_stats is None:
            norm_stats = {
                'mean': np.nanmean(features, axis=0),
                'std': np.nanstd(features, axis=0) + eps
            }
        else:
            norm_stats = fit_stats
        
        features = (features - norm_stats['mean']) / norm_stats['std']
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return torch.from_numpy(features), norm_stats
