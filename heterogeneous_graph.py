import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
import gc
import warnings

warnings.filterwarnings('ignore')

class BotnetFeatureExtractor:
    """
    [修正版] 高维特征提取器 (64-Dim)
    Fix: 修复了端口数据类型错误 (TypeError)，并使用向量化加速了大规模数据的处理。
    """
    
    @staticmethod
    def extract_features(df: pd.DataFrame, node_map: dict) -> torch.Tensor:
        print("[Feature] 提取深度高维统计特征 (64-Dim)...")
        num_nodes = len(node_map)
        
        # === 1. 数据清洗与类型强制转换 (Fix TypeError) ===
        # 某些流数据中端口可能是字符串或空值，强制转为 int，错误转为 -1
        df['dst_port'] = pd.to_numeric(df['dst_port'], errors='coerce').fillna(-1).astype(int)
        
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['ts'] = df['start_time'].astype(np.int64) // 10**9
        
        # 基础辅助列
        eps = 1e-6
        df['bpp'] = df['bytes'] / (df['packets'] + eps)  # Bytes Per Packet
        df['pps'] = df['packets'] / (df['duration'] + 0.1) # Packets Per Second
        df['bps'] = df['bytes'] / (df['duration'] + 0.1)   # Bytes Per Second
        
        # === 端口语义特征 (向量化加速) ===
        # 替代原本的 apply(lambda x: ...)，解决 1700万行数据的性能瓶颈和类型报错
        # 是否为知名端口 (0-1023)
        df['dst_port_priv'] = ((df['dst_port'] >= 0) & (df['dst_port'] < 1024)).astype(int)
        # 是否为高位动态端口 (>49151)
        df['dst_port_high'] = (df['dst_port'] > 49151).astype(int)
        
        # === 2. 分组聚合 (Feature Engineering) ===
        grp = df.groupby('src_ip')
        
        agg_dict = {
            'dst_ip': 'nunique',          
            'dst_port': 'nunique',        
            'packets': ['mean', 'std', 'sum', 'max', 'min'],
            'bytes': ['mean', 'std', 'sum', 'max', 'min'],
            'duration': ['mean', 'max', 'std', 'sum'],
            'bpp': ['mean', 'max', 'std'],      
            'pps': ['mean', 'max', 'std'], 
            'bps': ['mean', 'max'],
            'dst_port_priv': ['sum', 'mean'], 
            'dst_port_high': ['sum', 'mean']
        }
        
        print(f"[Feature] 正在聚合 {len(df)} 条流记录...")
        stats = grp.agg(agg_dict)
        stats.columns = [f"{c[0]}_{c[1]}" for c in stats.columns]
        
        # === 3. 计算时序周期性 (IAT) ===
        print("[Feature] 计算时序周期性 (IAT)...")
        
        def calc_iat_stats(ts_array):
            if len(ts_array) < 3: return 10.0, 10.0
            ts_array = np.sort(ts_array)
            diffs = np.diff(ts_array)
            # 过滤极短间隔
            valid_diffs = diffs[diffs > 0.05]
            if len(valid_diffs) < 2: return 10.0, 10.0
            return np.mean(valid_diffs), np.std(valid_diffs)

        # 优化：仅对活跃IP计算 IAT (Top IP)
        heavy_hitters = grp.size()
        target_ips = heavy_hitters[heavy_hitters > 3].index
        
        iat_mean_map = {ip: 10.0 for ip in node_map.keys()}
        iat_std_map = {ip: 10.0 for ip in node_map.keys()}
        
        if len(target_ips) > 0:
            sub_df = df[df['src_ip'].isin(target_ips)][['src_ip', 'ts']]
            iat_series = sub_df.groupby('src_ip')['ts'].apply(lambda x: calc_iat_stats(x.values))
            
            for ip, (mean_val, std_val) in iat_series.items():
                iat_mean_map[ip] = mean_val
                iat_std_map[ip] = std_val

        # === 4. 协议分布 ===
        proto_counts = pd.crosstab(df['src_ip'], df['protocol'])
        proto_sum = proto_counts.sum(axis=1) + eps
        for p in ['tcp', 'udp', 'icmp']:
            if p not in proto_counts: proto_counts[p] = 0
            proto_counts[p] = proto_counts[p] / proto_sum
            
        # === 5. 构建特征矩阵 ===
        fdf = pd.DataFrame(index=stats.index)
        fdf = fdf.join(stats)
        fdf['flow_count'] = grp.size()
        
        fdf['iat_mean'] = fdf.index.map(iat_mean_map)
        fdf['iat_std'] = fdf.index.map(iat_std_map)
        fdf = fdf.join(proto_counts[['tcp', 'udp', 'icmp']], how='left').fillna(0)
        
        # === 6. 高级衍生特征 ===
        fdf['pkts_cv'] = fdf['packets_std'] / (fdf['packets_mean'] + eps)
        fdf['bytes_cv'] = fdf['bytes_std'] / (fdf['bytes_mean'] + eps)
        fdf['dur_cv'] = fdf['duration_std'] / (fdf['duration_mean'] + eps)
        
        fdf['distinct_ip_ratio'] = fdf['dst_ip_nunique'] / fdf['flow_count']
        fdf['distinct_port_ratio'] = fdf['dst_port_nunique'] / fdf['flow_count']
        
        # === 7. 归一化处理 ===
        print("[Feature] 归一化处理...")
        log_cols = [col for col in fdf.columns if 'mean' in col or 'sum' in col or 'max' in col or 'std' in col or 'count' in col]
        for col in log_cols:
            fdf[col] = np.log1p(fdf[col].fillna(0))
            
        for col in fdf.columns:
            if fdf[col].std() > eps:
                fdf[col] = (fdf[col] - fdf[col].mean()) / fdf[col].std()
            else:
                fdf[col] = 0.0

        # === 8. 填充 64维 Tensor ===
        feat_dim = 64
        x = np.zeros((num_nodes, feat_dim), dtype=np.float32)
        
        cnt = 0
        print("[Feature] 映射特征向量...")
        for ip, row in fdf.iterrows():
            if ip in node_map:
                idx = node_map[ip]
                feats = [
                    # [0-4] 广度与活跃度
                    row.get('dst_ip_nunique', 0), row.get('dst_port_nunique', 0),
                    row.get('flow_count', 0), 
                    row.get('distinct_ip_ratio', 0), row.get('distinct_port_ratio', 0),
                    
                    # [5-11] 包统计
                    row.get('packets_sum', 0), row.get('packets_mean', 0), row.get('packets_std', 0),
                    row.get('packets_max', 0), row.get('packets_min', 0), row.get('pkts_cv', 0),
                    
                    # [11-17] 字节统计
                    row.get('bytes_sum', 0), row.get('bytes_mean', 0), row.get('bytes_std', 0),
                    row.get('bytes_max', 0), row.get('bytes_min', 0), row.get('bytes_cv', 0),
                    
                    # [17-23] 持续时间
                    row.get('duration_mean', 0), row.get('duration_std', 0), row.get('duration_sum', 0),
                    row.get('duration_max', 0), row.get('dur_cv', 0),
                    
                    # [23-28] 速率特征
                    row.get('pps_mean', 0), row.get('pps_max', 0), row.get('pps_std', 0),
                    row.get('bps_mean', 0), row.get('bps_max', 0),
                    
                    # [28-33] 载荷特征
                    row.get('bpp_mean', 0), row.get('bpp_max', 0), row.get('bpp_std', 0),
                    row.get('bytes_sum', 0) - row.get('packets_sum', 0),
                    
                    # [33-36] 时序特征
                    row.get('iat_mean', 0), row.get('iat_std', 0),
                    row.get('iat_std', 0) - row.get('iat_mean', 0),
                    
                    # [36-39] 协议分布
                    row.get('tcp', 0), row.get('udp', 0), row.get('icmp', 0),
                    
                    # [39-43] 端口语义
                    row.get('dst_port_priv_sum', 0), row.get('dst_port_priv_mean', 0),
                    row.get('dst_port_high_sum', 0), row.get('dst_port_high_mean', 0),
                    
                    # Padding
                    0
                ]
                
                if len(feats) < feat_dim:
                    feats.extend([0] * (feat_dim - len(feats)))
                
                x[idx, :feat_dim] = feats[:feat_dim]
                cnt += 1
                
        print(f"[Feature] 特征提取完成。维度: {x.shape}")
        del fdf, stats, iat_mean_map
        gc.collect()
        return torch.from_numpy(x)

class HeterogeneousGraphBuilder:
    def __init__(self):
        self.data = HeteroData()
        self.ip_map = {}
        
    def build(self, df: pd.DataFrame):
        print("[Graph] 构建异构图...")
        
        # 1. 节点映射
        ips = pd.unique(np.concatenate([df['src_ip'].unique(), df['dst_ip'].unique()]))
        self.ip_map = {ip: i for i, ip in enumerate(ips)}
        
        # 2. 边构建 (聚合模式)
        edges = df.groupby(['src_ip', 'dst_ip']).size().reset_index(name='count')
        
        src_idx = [self.ip_map[ip] for ip in edges['src_ip']]
        dst_idx = [self.ip_map[ip] for ip in edges['dst_ip']]
        
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        edge_weight = torch.log1p(torch.tensor(edges['count'].values, dtype=torch.float)).unsqueeze(1)
        
        self.data['ip', 'flow', 'ip'].edge_index = edge_index
        self.data['ip', 'flow', 'ip'].edge_attr = edge_weight
        
        # 3. 特征提取
        self.data['ip'].x = BotnetFeatureExtractor.extract_features(df, self.ip_map)
        
        return self.data, self.ip_map