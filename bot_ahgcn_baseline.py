"""
Bot-AHGCN: Multi-attributed Heterogeneous Graph Convolutional Network for Bot Detection
Baseline implementation based on:
"Multi-attributed heterogeneous graph convolutional network for bot detection"
Information Sciences 537 (2020) 380–393
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import GCNConv
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class BotAHGCNGraphBuilder:
    """
    Builds Attributed Heterogeneous Information Network (AHIN) for bot detection.
    Models network flows as six-tuple: (IPsrc, IPdes, Port, Protocol, Request, Response)
    """
    
    def __init__(self):
        self.ip_map = {}
        self.port_map = {}
        self.protocol_map = {}
        self.request_map = {}
        self.response_map = {}
        
    def build_ahin(self, df: pd.DataFrame, max_ips: int = 10000) -> Tuple[HeteroData, Dict, Dict]:
        """
        Build AHIN from network flow data
        
        Args:
            df: DataFrame containing network flow records
            max_ips: Maximum number of IPs to include (for efficiency)
            
        Returns:
            HeteroData object, ip_map, node_type_sizes
        """
        print("[Bot-AHGCN] Building Attributed Heterogeneous Information Network...")
        import gc
        
        # Create node mappings for different node types
        print("[Bot-AHGCN] Step 1: Creating IP mappings...")
        
        # 使用采样方式减少内存使用
        if len(df) > 1000000:
            print(f"[Bot-AHGCN] Large dataset ({len(df)} rows), using sampled unique IPs...")
            sample_df = df.sample(n=min(500000, len(df)), random_state=42)
            unique_src = set(sample_df['src_ip'].unique())
            unique_dst = set(sample_df['dst_ip'].unique())
            unique_ips = list(unique_src | unique_dst)
            del sample_df
            gc.collect()
        else:
            unique_ips = pd.unique(np.concatenate([df['src_ip'].unique(), df['dst_ip'].unique()]))
        
        # 如果IP数量过多，采样最活跃的IP
        if len(unique_ips) > max_ips:
            print(f"[Bot-AHGCN] Too many IPs ({len(unique_ips)}), sampling top {max_ips} active IPs...")
            # 使用采样减少内存
            sample_size = min(1000000, len(df))
            sample_df = df.sample(n=sample_size, random_state=42)
            ip_counts = sample_df['src_ip'].value_counts().head(max_ips)
            top_ips = set(ip_counts.index)
            dst_counts = sample_df['dst_ip'].value_counts().head(max_ips // 2)
            top_ips.update(dst_counts.index)
            unique_ips = list(top_ips)[:max_ips]
            del sample_df, ip_counts, dst_counts
            gc.collect()
        
        self.ip_map = {ip: i for i, ip in enumerate(unique_ips)}
        print(f"[Bot-AHGCN] Step 1 done: {len(self.ip_map)} unique IPs")
        
        # Map ports (limit to top 1000 most common ports for efficiency)
        print("[Bot-AHGCN] Step 2: Creating port mappings...")
        try:
            # 使用采样方式处理大端口数据
            if len(df) > 1000000:
                print("[Bot-AHGCN] Sampling for port analysis...")
                sample_df = df.sample(n=min(500000, len(df)), random_state=42)
                # 安全转换端口为数值
                ports_numeric = pd.to_numeric(sample_df['dst_port'], errors='coerce')
                ports_numeric = ports_numeric.dropna().astype(int)
                top_ports_series = ports_numeric.value_counts().head(1000)
                del sample_df, ports_numeric
                gc.collect()
            else:
                ports_numeric = pd.to_numeric(df['dst_port'], errors='coerce')
                ports_numeric = ports_numeric.dropna().astype(int)
                top_ports_series = ports_numeric.value_counts().head(1000)
                del ports_numeric
                gc.collect()
            
            top_ports = top_ports_series.index.tolist()
            self.port_map = {port: i for i, port in enumerate(top_ports)}
            print(f"[Bot-AHGCN] Step 2 done: {len(self.port_map)} unique ports")
        except Exception as e:
            print(f"[Bot-AHGCN] Warning: Port mapping failed ({e}), using default ports")
            # 使用常见端口作为后备
            common_ports = [80, 443, 22, 21, 25, 53, 110, 143, 3306, 3389, 5900, 8080]
            self.port_map = {port: i for i, port in enumerate(common_ports)}
            print(f"[Bot-AHGCN] Step 2 done: {len(self.port_map)} default ports")
        
        # Map protocols
        print("[Bot-AHGCN] Step 3: Creating protocol mappings...")
        unique_protocols = df['protocol'].unique()
        self.protocol_map = {proto: i for i, proto in enumerate(unique_protocols)}
        print(f"[Bot-AHGCN] Step 3 done: {len(self.protocol_map)} unique protocols")
        
        # Map requests (top 500 most common) - 使用采样优化
        print("[Bot-AHGCN] Step 4: Creating request mappings...")
        try:
            if 'request' not in df.columns:
                # 为整个df创建request列（用于后续边构建）
                print("[Bot-AHGCN] Creating request column...")
                df['request'] = df['src_ip'].astype(str) + '-' + df['dst_ip'].astype(str) + '-' + df['dst_port'].astype(str)
            
            # 使用采样获取top requests
            if len(df) > 1000000:
                sample_df = df.sample(n=min(500000, len(df)), random_state=42)
                top_requests = sample_df['request'].value_counts().head(500).index.tolist()
                del sample_df
                gc.collect()
            else:
                top_requests = df['request'].value_counts().head(500).index.tolist()
            self.request_map = {req: i for i, req in enumerate(top_requests)}
            print(f"[Bot-AHGCN] Step 4 done: {len(self.request_map)} unique requests")
        except Exception as e:
            print(f"[Bot-AHGCN] Warning: Request mapping failed ({e}), using empty map")
            self.request_map = {}
        
        # Map responses (top 500 most common) - 使用采样优化
        print("[Bot-AHGCN] Step 5: Creating response mappings...")
        try:
            if 'response' not in df.columns:
                # 为整个df创建response列（用于后续边构建）
                print("[Bot-AHGCN] Creating response column...")
                df['response'] = df['bytes'].apply(lambda x: f"size_{int(x/1000)}k" if pd.notna(x) and x > 0 else "empty")
            
            # 使用采样获取top responses
            if len(df) > 1000000:
                sample_df = df.sample(n=min(500000, len(df)), random_state=42)
                top_responses = sample_df['response'].value_counts().head(500).index.tolist()
                del sample_df
                gc.collect()
            else:
                top_responses = df['response'].value_counts().head(500).index.tolist()
            self.response_map = {resp: i for i, resp in enumerate(top_responses)}
            print(f"[Bot-AHGCN] Step 5 done: {len(self.response_map)} unique responses")
        except Exception as e:
            print(f"[Bot-AHGCN] Warning: Response mapping failed ({e}), using empty map")
            self.response_map = {}
        
        data = HeteroData()
        
        # Build edges for different relationships (R1-R10 from the paper)
        # R1: Source IP -> Destination IP
        print("[Bot-AHGCN] Step 6: Building IP-to-IP edges...")
        self._build_ip_to_ip_edges(df, data)
        
        # R2, R7: Source/Dest IP -> Protocol
        print("[Bot-AHGCN] Step 7: Building IP-Protocol edges...")
        self._build_ip_protocol_edges(df, data)
        
        # R3, R8: Source/Dest IP -> Port
        print("[Bot-AHGCN] Step 8: Building IP-Port edges...")
        self._build_ip_port_edges(df, data)
        
        # R4, R9: Source/Dest IP -> Request
        print("[Bot-AHGCN] Step 9: Building IP-Request edges...")
        self._build_ip_request_edges(df, data)
        
        # R5, R10: Source/Dest IP -> Response
        print("[Bot-AHGCN] Step 10: Building IP-Response edges...")
        self._build_ip_response_edges(df, data)
        
        # R6: Protocol -> Port
        print("[Bot-AHGCN] Step 11: Building Protocol-Port edges...")
        self._build_protocol_port_edges(df, data)
        
        # Build node attributes
        print("[Bot-AHGCN] Step 12: Building node attributes...")
        self._build_node_attributes(df, data)
        
        node_type_sizes = {
            'ip': len(self.ip_map),
            'port': len(self.port_map),
            'protocol': len(self.protocol_map),
            'request': len(self.request_map),
            'response': len(self.response_map)
        }
        
        print(f"[Bot-AHGCN] AHIN built with {len(self.ip_map)} IP nodes, "
              f"{len(self.port_map)} port nodes, {len(self.protocol_map)} protocol nodes")
        
        return data, self.ip_map, node_type_sizes
    
    def _build_ip_to_ip_edges(self, df: pd.DataFrame, data: HeteroData):
        """R1: Source IP -> Destination IP connection"""
        import gc
        
        # 对于大数据集，使用采样
        if len(df) > 1000000:
            print("[Bot-AHGCN] Sampling for IP-to-IP edges...")
            sample_df = df.sample(n=min(1000000, len(df)), random_state=42)
            edges = sample_df.groupby(['src_ip', 'dst_ip']).size().reset_index(name='count')
            del sample_df
            gc.collect()
        else:
            edges = df.groupby(['src_ip', 'dst_ip']).size().reset_index(name='count')
        
        # 只处理存在于ip_map中的IP - 使用向量化操作
        edges = edges[edges['src_ip'].isin(self.ip_map.keys()) & edges['dst_ip'].isin(self.ip_map.keys())]
        
        if len(edges) > 0:
            src_idx = [self.ip_map[ip] for ip in edges['src_ip']]
            dst_idx = [self.ip_map[ip] for ip in edges['dst_ip']]
            weights = edges['count'].values
            
            edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            edge_weight = torch.log1p(torch.tensor(weights, dtype=torch.float)).unsqueeze(1)
            data['ip', 'connects', 'ip'].edge_index = edge_index
            data['ip', 'connects', 'ip'].edge_attr = edge_weight
            print(f"[Bot-AHGCN] Built {len(src_idx)} IP-to-IP edges")
        del edges
        gc.collect()
        
    def _build_ip_protocol_edges(self, df: pd.DataFrame, data: HeteroData):
        """R2: Source IP -> Protocol, R7: Destination IP -> Protocol"""
        import gc
        
        for src_col, edge_type in [('src_ip', 'uses'), ('dst_ip', 'uses')]:
            # 对于大数据集使用采样
            if len(df) > 1000000:
                sample_df = df[[src_col, 'protocol']].sample(n=min(500000, len(df)), random_state=42)
                edges = sample_df.drop_duplicates()
                del sample_df
                gc.collect()
            else:
                edges = df[[src_col, 'protocol']].drop_duplicates()
            
            # 只保留存在于映射中的边
            edges = edges[edges[src_col].isin(self.ip_map.keys()) & edges['protocol'].isin(self.protocol_map.keys())]
            
            if len(edges) > 0:
                src_idx = [self.ip_map[ip] for ip in edges[src_col]]
                dst_idx = [self.protocol_map[p] for p in edges['protocol']]
                edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
                data['ip', edge_type, 'protocol'].edge_index = edge_index
                print(f"[Bot-AHGCN] Built {len(src_idx)} IP-{edge_type}-Protocol edges")
            del edges
            gc.collect()
        
    def _build_ip_port_edges(self, df: pd.DataFrame, data: HeteroData):
        """R3: Source IP -> Port, R8: Destination IP -> Port"""
        import gc
        
        for src_col, edge_type in [('src_ip', 'uses'), ('dst_ip', 'uses')]:
            # 对于大数据集使用采样
            if len(df) > 1000000:
                sample_df = df[[src_col, 'dst_port']].sample(n=min(500000, len(df)), random_state=42)
                edges = sample_df.drop_duplicates()
                del sample_df
                gc.collect()
            else:
                edges = df[[src_col, 'dst_port']].drop_duplicates()
            
            # 安全转换端口
            edges['dst_port_int'] = pd.to_numeric(edges['dst_port'], errors='coerce').fillna(-1).astype(int)
            # 只保留存在于映射中的边
            edges = edges[edges[src_col].isin(self.ip_map.keys()) & edges['dst_port_int'].isin(self.port_map.keys())]
            
            if len(edges) > 0:
                src_idx = [self.ip_map[ip] for ip in edges[src_col]]
                dst_idx = [self.port_map[p] for p in edges['dst_port_int']]
                edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
                data['ip', edge_type, 'port'].edge_index = edge_index
                print(f"[Bot-AHGCN] Built {len(src_idx)} IP-{edge_type}-Port edges")
            del edges
            gc.collect()
    
    def _build_ip_request_edges(self, df: pd.DataFrame, data: HeteroData):
        """R4: Source IP -> Request, R9: Destination IP -> Request"""
        import gc
        
        if not self.request_map:
            print("[Bot-AHGCN] Skipping IP-Request edges (no request map)")
            return
            
        for src_col, edge_type in [('src_ip', 'sends'), ('dst_ip', 'receives')]:
            # 对于大数据集使用采样
            if len(df) > 1000000:
                sample_df = df[[src_col, 'request']].sample(n=min(500000, len(df)), random_state=42)
                edges = sample_df.drop_duplicates()
                del sample_df
                gc.collect()
            else:
                edges = df[[src_col, 'request']].drop_duplicates()
            
            # 只保留存在于映射中的边
            edges = edges[edges[src_col].isin(self.ip_map.keys()) & edges['request'].isin(self.request_map.keys())]
            
            if len(edges) > 0:
                src_idx = [self.ip_map[ip] for ip in edges[src_col]]
                dst_idx = [self.request_map[r] for r in edges['request']]
                edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
                data['ip', edge_type, 'request'].edge_index = edge_index
                print(f"[Bot-AHGCN] Built {len(src_idx)} IP-{edge_type}-Request edges")
            del edges
            gc.collect()
    
    def _build_ip_response_edges(self, df: pd.DataFrame, data: HeteroData):
        """R5: Source IP -> Response, R10: Destination IP -> Response"""
        import gc
        
        if not self.response_map:
            print("[Bot-AHGCN] Skipping IP-Response edges (no response map)")
            return
            
        for src_col, edge_type in [('src_ip', 'receives'), ('dst_ip', 'sends')]:
            # 对于大数据集使用采样
            if len(df) > 1000000:
                sample_df = df[[src_col, 'response']].sample(n=min(500000, len(df)), random_state=42)
                edges = sample_df.drop_duplicates()
                del sample_df
                gc.collect()
            else:
                edges = df[[src_col, 'response']].drop_duplicates()
            
            # 只保留存在于映射中的边
            edges = edges[edges[src_col].isin(self.ip_map.keys()) & edges['response'].isin(self.response_map.keys())]
            
            if len(edges) > 0:
                src_idx = [self.ip_map[ip] for ip in edges[src_col]]
                dst_idx = [self.response_map[r] for r in edges['response']]
                edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
                data['ip', edge_type, 'response'].edge_index = edge_index
                print(f"[Bot-AHGCN] Built {len(src_idx)} IP-{edge_type}-Response edges")
            del edges
            gc.collect()
    
    def _build_protocol_port_edges(self, df: pd.DataFrame, data: HeteroData):
        """R6: Protocol -> Port"""
        import gc
        
        # 对于大数据集使用采样
        if len(df) > 1000000:
            sample_df = df[['protocol', 'dst_port']].sample(n=min(500000, len(df)), random_state=42)
            edges = sample_df.drop_duplicates()
            del sample_df
            gc.collect()
        else:
            edges = df[['protocol', 'dst_port']].drop_duplicates()
        
        # 安全转换端口
        edges['dst_port_int'] = pd.to_numeric(edges['dst_port'], errors='coerce').fillna(-1).astype(int)
        # 只保留存在于映射中的边
        edges = edges[edges['protocol'].isin(self.protocol_map.keys()) & edges['dst_port_int'].isin(self.port_map.keys())]
        
        if len(edges) > 0:
            src_idx = [self.protocol_map[p] for p in edges['protocol']]
            dst_idx = [self.port_map[p] for p in edges['dst_port_int']]
            edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            data['protocol', 'utilizes', 'port'].edge_index = edge_index
            print(f"[Bot-AHGCN] Built {len(src_idx)} Protocol-Port edges")
        del edges
        gc.collect()
    
    def _build_node_attributes(self, df: pd.DataFrame, data: HeteroData):
        """Build attribute features for IP nodes (using statistical features)"""
        import gc
        print("[Bot-AHGCN] Extracting node attributes...")
        
        num_ips = len(self.ip_map)
        
        # 始终使用统计特征，但采用采样方式处理大数据
        sample_size = min(500000, len(df)) if len(df) > 500000 else len(df)
        if sample_size < len(df):
            print(f"[Bot-AHGCN] Using sampled data ({sample_size} rows) for feature extraction...")
            sample_df = df.sample(n=sample_size, random_state=42)
        else:
            sample_df = df
        
        # 过滤只包含在ip_map中的IP
        sample_df = sample_df[sample_df['src_ip'].isin(self.ip_map.keys())]
        
        # 使用简化的聚合 - 计算统计特征
        print("[Bot-AHGCN] Computing IP statistics...")
        try:
            agg_stats = sample_df.groupby('src_ip').agg({
                'dst_ip': 'nunique',
                'dst_port': 'nunique',
                'bytes': ['count', 'mean', 'std'],
                'packets': ['mean', 'sum'],
                'duration': ['mean', 'std']
            }).reset_index()
            
            # 展平列名
            agg_stats.columns = ['src_ip', 'dst_ip_nunique', 'dst_port_nunique', 
                                'bytes_count', 'bytes_mean', 'bytes_std',
                                'packets_mean', 'packets_sum', 'duration_mean', 'duration_std']
        except Exception as e:
            print(f"[Bot-AHGCN] Warning: Full aggregation failed ({e}), using simplified features...")
            agg_stats = sample_df.groupby('src_ip').agg({
                'dst_ip': 'nunique',
                'dst_port': 'nunique',
                'bytes': 'count'
            }).reset_index()
            agg_stats.columns = ['src_ip', 'dst_ip_nunique', 'dst_port_nunique', 'bytes_count']
            # 添加缺失列
            for col in ['bytes_mean', 'bytes_std', 'packets_mean', 'packets_sum', 'duration_mean', 'duration_std']:
                agg_stats[col] = 0
        
        del sample_df
        gc.collect()
        
        # 创建特征矩阵 (100维)
        ip_features = np.zeros((num_ips, 100), dtype=np.float32)
        
        # 使用字典映射加速
        stats_dict = {}
        for _, row in agg_stats.iterrows():
            ip = row['src_ip']
            stats_dict[ip] = [
                row.get('bytes_count', 0) / 1000.0,
                row.get('dst_ip_nunique', 0) / 100.0,
                row.get('dst_port_nunique', 0) / 50.0,
                row.get('bytes_mean', 0) / 10000.0 if pd.notna(row.get('bytes_mean', 0)) else 0,
                row.get('bytes_std', 0) / 10000.0 if pd.notna(row.get('bytes_std', 0)) else 0,
                row.get('packets_mean', 0) / 100.0 if pd.notna(row.get('packets_mean', 0)) else 0,
                row.get('packets_sum', 0) / 1000.0 if pd.notna(row.get('packets_sum', 0)) else 0,
                row.get('duration_mean', 0) / 100.0 if pd.notna(row.get('duration_mean', 0)) else 0,
                row.get('duration_std', 0) / 100.0 if pd.notna(row.get('duration_std', 0)) else 0,
            ]
        
        for ip, idx in self.ip_map.items():
            if ip in stats_dict:
                features = stats_dict[ip]
                ip_features[idx, :9] = features
                # 添加一些统计衍生特征
                ip_features[idx, 9] = features[0] * features[1]  # bytes_count * unique_dst_ips
                ip_features[idx, 10] = features[2] / (features[1] + 1e-6)  # ports_per_dst
                ip_features[idx, 11] = features[3] / (features[5] + 1e-6)  # bytes_per_packet
        
        del agg_stats, stats_dict
        gc.collect()
        
        # 标准化特征
        mean_vals = ip_features.mean(axis=0)
        std_vals = ip_features.std(axis=0) + 1e-6
        ip_features = (ip_features - mean_vals) / std_vals
        
        data['ip'].x = torch.from_numpy(ip_features.astype(np.float32))
        
        # Initialize attributes for other node types
        data['port'].x = torch.randn(len(self.port_map), 32) * 0.1
        data['protocol'].x = torch.randn(len(self.protocol_map), 16) * 0.1
        data['request'].x = torch.randn(len(self.request_map), 64) * 0.1
        data['response'].x = torch.randn(len(self.response_map), 64) * 0.1
        
        print(f"[Bot-AHGCN] Node attributes built: IP features shape {data['ip'].x.shape}")


class MetaPathSimilarity:
    """
    Computes similarity matrices based on meta-paths and meta-graphs
    Implements Equations (1) and (2) from the paper
    
    Paper定义的10种元路径 (P1-P10):
    P1: IPsrc -uses-> Protocol -uses-> IPsrc (使用相同协议)
    P2: IPsrc -sends-> Request -sends-> IPsrc (发送相同请求)
    P3: IPsrc -uses-> Port -uses-> IPsrc (使用相同端口)
    P4: IPsrc -connects-> IPdst -connects-> IPsrc (访问相同目的IP)
    P5: IPsrc -uses-> Protocol -connects-> IPdst (使用相同协议访问同一目的)
    P6: IPsrc -uses-> Port -connects-> IPdst (使用相同端口访问同一目的)
    P7: IPsrc -sends-> Request -connects-> IPdst (发送相同请求到同一目的)
    P8: IPsrc -receives-> Response -receives-> IPsrc (接收相同响应)
    P9: IPsrc -uses-> Protocol -uses-> Port -uses-> IPsrc (使用相同协议和端口)
    P10: IPsrc -sends-> Request -receives-> Response -receives-> IPsrc (发送请求并接收响应)
    
    7种元图 (M1-M7):
    M1: P1 * P2 (使用相同协议并发送相同请求)
    M2: P2 * P8 (发送相同请求并接收相同响应)
    M3: P1 * P3 (使用相同协议和端口)
    M4: P5 * P7 (使用相同协议访问同一目的并发送相同请求)
    M5: P1 * P3 * P2 (使用相同协议、端口并发送相同请求)
    M6: P1 * P3 * P7 (使用相同协议、端口并发送相同请求到同一目的)
    M7: P3 * P2 (使用相同端口并发送相同请求)
    """
    
    def __init__(self, ahin: HeteroData, ip_map: Dict):
        self.ahin = ahin
        self.ip_map = ip_map
        self.num_ips = len(ip_map)
        self.idx_to_ip = {v: k for k, v in ip_map.items()}
        
        # Cache for computed matrices
        self._cache = {}
        
    def _get_adjacency_matrix(self, edge_key: tuple) -> torch.Tensor:
        """Get adjacency matrix for a given edge type"""
        if edge_key in self._cache:
            return self._cache[edge_key]
            
        try:
            edge_store = self.ahin[edge_key]
            if not hasattr(edge_store, 'edge_index') or edge_store.edge_index is None:
                return None
            edge_index = edge_store.edge_index
        except (KeyError, AttributeError):
            return None
        
        # Create sparse adjacency matrix
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()
        
        # Determine matrix dimensions based on edge type
        if edge_key[0] == 'ip' and edge_key[2] == 'ip':
            n_src, n_dst = self.num_ips, self.num_ips
        else:
            # For heterogeneous edges, we need node type sizes
            n_src = self.num_ips
            n_dst = len(edge_index[1].unique())
        
        # Build sparse matrix using COO format
        import scipy.sparse as sp
        adj = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n_src, n_dst))
        adj = torch.from_numpy(adj.todense()).float()
        
        self._cache[edge_key] = adj
        return adj
    
    def compute_meta_path_similarity(self, meta_path_type: str = 'combined', 
                                      weights: torch.Tensor = None) -> torch.Tensor:
        """
        Compute similarity matrix AM based on meta-paths
        Implements Equation (1): SM(hi, hj) = Σ_m wm * 2*|{hi→j ∈ Pm}| / (|{hi→i ∈ Pm}| + |{hj→j ∈ Pm}|)
        
        Args:
            meta_path_type: Type of meta-path ('protocol', 'port', 'request', 'combined')
            weights: Optional weight vector for meta-paths
            
        Returns:
            Similarity matrix AM (N x N)
        """
        print(f"[Bot-AHGCN] Computing meta-path based similarity ({meta_path_type})...")
        
        N = self.num_ips
        
        # Compute individual meta-path similarity matrices
        meta_path_matrices = {}
        
        # P1: Hosts using the same protocol (IPsrc -uses-> Protocol -uses-> IPsrc)
        if meta_path_type in ['protocol', 'combined']:
            P1 = self._compute_symmetric_meta_path('protocol', 'uses')
            if P1 is not None:
                meta_path_matrices['P1'] = P1
                print(f"[Bot-AHGCN] P1 (protocol) computed, non-zero: {(P1 > 0).sum().item()}")
        
        # P2: Hosts sending the same request
        if meta_path_type in ['request', 'combined']:
            P2 = self._compute_symmetric_meta_path('request', 'sends')
            if P2 is not None:
                meta_path_matrices['P2'] = P2
                print(f"[Bot-AHGCN] P2 (request) computed, non-zero: {(P2 > 0).sum().item()}")
        
        # P3: Hosts accessing the same port
        if meta_path_type in ['port', 'combined']:
            P3 = self._compute_symmetric_meta_path('port', 'uses')
            if P3 is not None:
                meta_path_matrices['P3'] = P3
                print(f"[Bot-AHGCN] P3 (port) computed, non-zero: {(P3 > 0).sum().item()}")
        
        # P4: Hosts accessing the same destination IP
        if meta_path_type in ['combined']:
            P4 = self._compute_destination_similarity()
            if P4 is not None:
                meta_path_matrices['P4'] = P4
                print(f"[Bot-AHGCN] P4 (destination) computed, non-zero: {(P4 > 0).sum().item()}")
        
        # Store for later use in meta-graph computation
        self.meta_path_matrices = meta_path_matrices
        
        # Combine meta-paths with weights
        if weights is None:
            # Default weights based on paper's findings (Table 1)
            default_weights = {
                'P1': 0.15,  # Protocol
                'P2': 0.30,  # Request (high importance per paper)
                'P3': 0.10,  # Port
                'P4': 0.25,  # Destination
            }
        else:
            default_weights = weights
        
        AM = torch.zeros(N, N)
        for name, matrix in meta_path_matrices.items():
            w = default_weights.get(name, 0.1)
            AM += w * matrix
        
        # Normalize
        AM = AM / (AM.sum(dim=1, keepdim=True) + 1e-6)
        
        return AM
    
    def _compute_symmetric_meta_path(self, node_type: str, edge_type: str) -> torch.Tensor:
        """
        Compute symmetric meta-path similarity: IP -edge_type-> node_type -edge_type-> IP
        Using Equation (1) from the paper
        """
        edge_key = ('ip', edge_type, node_type)
        adj = self._get_adjacency_matrix(edge_key)
        
        if adj is None:
            return torch.zeros(self.num_ips, self.num_ips)
        
        N = self.num_ips
        
        # Compute co-occurrence: C = A @ A^T
        # This counts how many intermediate nodes two IPs share
        cooccurrence = torch.mm(adj, adj.t())
        
        # Apply PathSim normalization (Equation 1)
        # SM(hi, hj) = 2 * C(i,j) / (C(i,i) + C(j,j))
        diag = torch.diag(cooccurrence)
        
        # Avoid division by zero
        diag_safe = diag.clamp(min=1e-6)
        
        # Compute denominator matrix
        denom = diag_safe.unsqueeze(1) + diag_safe.unsqueeze(0)
        
        # Compute similarity
        similarity = 2 * cooccurrence / denom
        
        # Set diagonal to 0 (no self-similarity in off-diagonal)
        similarity.fill_diagonal_(0)
        
        return similarity
    
    def _compute_destination_similarity(self) -> torch.Tensor:
        """
        P4: Compute similarity based on common destination IPs
        IPsrc -connects-> IPdst -connects-> IPsrc
        """
        edge_key = ('ip', 'connects', 'ip')
        adj = self._get_adjacency_matrix(edge_key)
        
        if adj is None:
            return torch.zeros(self.num_ips, self.num_ips)
        
        # For destination similarity, we want: IP -> same destination
        # C = A @ A^T where A is source-to-destination adjacency
        cooccurrence = torch.mm(adj, adj.t())
        
        # Apply PathSim normalization
        diag = torch.diag(cooccurrence)
        diag_safe = diag.clamp(min=1e-6)
        denom = diag_safe.unsqueeze(1) + diag_safe.unsqueeze(0)
        
        similarity = 2 * cooccurrence / denom
        similarity.fill_diagonal_(0)
        
        return similarity
    
    def compute_meta_graph_similarity(self, AM: torch.Tensor = None, 
                                       weights: torch.Tensor = None) -> torch.Tensor:
        """
        Compute similarity matrix AG based on meta-graphs
        Implements Equation (2): SG(hi, hj) = Σ_m wg * 2*CouMG(hi, hj) / (CouMG(hi, hi) + CouMG(hj, hj))
        
        Meta-graphs use Hadamard product to combine multiple meta-paths
        
        Args:
            AM: Meta-path based similarity matrix (optional)
            weights: Optional weight vector for meta-graphs
            
        Returns:
            Similarity matrix AG (N x N)
        """
        print("[Bot-AHGCN] Computing meta-graph based similarity...")
        
        N = self.num_ips
        
        # Get meta-path matrices
        if not hasattr(self, 'meta_path_matrices'):
            # Compute them first
            self.compute_meta_path_similarity()
        
        P = self.meta_path_matrices
        
        # Compute meta-graph matrices using Hadamard product
        meta_graph_matrices = {}
        
        # M1: P1 * P2 (use same protocol AND send same request)
        if 'P1' in P and 'P2' in P:
            M1 = P['P1'] * P['P2']
            meta_graph_matrices['M1'] = M1
            print(f"[Bot-AHGCN] M1 computed, non-zero: {(M1 > 0).sum().item()}")
        
        # M2: P2 * (response similarity) if available
        # M3: P1 * P3 (use same protocol AND port)
        if 'P1' in P and 'P3' in P:
            M3 = P['P1'] * P['P3']
            meta_graph_matrices['M3'] = M3
            print(f"[Bot-AHGCN] M3 computed, non-zero: {(M3 > 0).sum().item()}")
        
        # M4: P1 * P4 (use same protocol to same destination)
        if 'P1' in P and 'P4' in P:
            M4 = P['P1'] * P['P4']
            meta_graph_matrices['M4'] = M4
            print(f"[Bot-AHGCN] M4 computed, non-zero: {(M4 > 0).sum().item()}")
        
        # M5: P1 * P3 * P2 (use same protocol, port AND send same request)
        if 'P1' in P and 'P2' in P and 'P3' in P:
            M5 = P['P1'] * P['P3'] * P['P2']
            meta_graph_matrices['M5'] = M5
            print(f"[Bot-AHGCN] M5 computed, non-zero: {(M5 > 0).sum().item()}")
        
        # M7: P3 * P2 (use same port AND send same request)
        if 'P2' in P and 'P3' in P:
            M7 = P['P3'] * P['P2']
            meta_graph_matrices['M7'] = M7
            print(f"[Bot-AHGCN] M7 computed, non-zero: {(M7 > 0).sum().item()}")
        
        # Store meta-graph matrices
        self.meta_graph_matrices = meta_graph_matrices
        
        # Combine meta-graphs with weights
        if weights is None:
            # Default weights based on paper's findings (Table 1 shows M6 is best)
            default_weights = {
                'M1': 0.20,
                'M3': 0.15,
                'M4': 0.25,  # High importance (similar to M4 in paper)
                'M5': 0.25,  # High importance (similar to M5 in paper)
                'M7': 0.15,
            }
        else:
            default_weights = weights
        
        AG = torch.zeros(N, N)
        for name, matrix in meta_graph_matrices.items():
            w = default_weights.get(name, 0.1)
            AG += w * matrix
        
        # Normalize
        AG = AG / (AG.sum(dim=1, keepdim=True) + 1e-6)
        
        return AG


class GraphConvLayer(nn.Module):
    """
    Graph Convolution Layer for dense adjacency matrices
    Implements: H' = σ(D^(-1/2) A D^(-1/2) H W)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x: Node features (N x in_features)
            adj: Normalized adjacency matrix (N x N)
        Returns:
            Output features (N x out_features)
        """
        # Linear transformation
        support = torch.mm(x, self.weight)
        # Graph convolution
        output = torch.spmm(adj, support) if adj.is_sparse else torch.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class BotAHGCNModel(nn.Module):
    """
    Bot-AHGCN: Multi-attributed Heterogeneous Graph Convolutional Network
    Implements the model from Section 3.4 of the paper
    
    Combines:
    - GCN on meta-path similarity graph AM
    - GCN on meta-graph similarity graph AG
    - Weight α to combine both representations
    """
    
    def __init__(self, in_dim: int = 100, hidden_dim: int = 64, out_dim: int = 32):
        super(BotAHGCNModel, self).__init__()
        
        # GCN for meta-path similarity graph AM
        self.gcn_m1 = GraphConvLayer(in_dim, hidden_dim)
        self.gcn_m2 = GraphConvLayer(hidden_dim, out_dim)
        
        # GCN for meta-graph similarity graph AG
        self.gcn_g1 = GraphConvLayer(in_dim, hidden_dim)
        self.gcn_g2 = GraphConvLayer(hidden_dim, out_dim)
        
        # Learnable coefficient α to combine AM and AG
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, 64),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor, AM: torch.Tensor, AG: torch.Tensor):
        """
        Forward pass
        
        Args:
            x: Node feature matrix (N x in_dim)
            AM: Meta-path similarity adjacency matrix (N x N) - should be normalized
            AG: Meta-graph similarity adjacency matrix (N x N) - should be normalized
            
        Returns:
            Logits (N x 1)
        """
        # Process AM (meta-path similarity)
        ZM = self.gcn_m1(x, AM)
        ZM = F.relu(ZM)
        ZM = F.dropout(ZM, p=0.3, training=self.training)
        ZM = self.gcn_m2(ZM, AM)
        
        # Process AG (meta-graph similarity)
        ZG = self.gcn_g1(x, AG)
        ZG = F.relu(ZG)
        ZG = F.dropout(ZG, p=0.3, training=self.training)
        ZG = self.gcn_g2(ZG, AG)
        
        # Combine using learnable α (Equation 7)
        alpha_sigmoid = torch.sigmoid(self.alpha)
        Z = alpha_sigmoid * ZM + (1 - alpha_sigmoid) * ZG
        
        # Concatenate both representations for classifier
        Z_combined = torch.cat([ZM, ZG], dim=1)
        
        # Classification
        logits = self.classifier(Z_combined)
        
        return logits
    
    def get_embeddings(self, x: torch.Tensor, AM: torch.Tensor, AG: torch.Tensor):
        """Get node embeddings for visualization"""
        ZM = self.gcn_m1(x, AM)
        ZM = F.relu(ZM)
        ZM = self.gcn_m2(ZM, AM)
        
        ZG = self.gcn_g1(x, AG)
        ZG = F.relu(ZG)
        ZG = self.gcn_g2(ZG, AG)
        
        return ZM, ZG


class BotAHGCNTrainer:
    """Trainer for Bot-AHGCN model"""
    
    def __init__(self, model: nn.Module, lr: float = 0.01, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def train_step(self, x: torch.Tensor, AM: torch.Tensor, AG: torch.Tensor, 
                   labels: torch.Tensor, train_mask: torch.Tensor):
        """Single training step"""
        self.model.train()
        
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(x, AM, AG)
        
        # Compute loss only on training nodes
        loss = self.criterion(logits[train_mask].squeeze(), labels[train_mask])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, x: torch.Tensor, AM: torch.Tensor, AG: torch.Tensor,
                 labels: torch.Tensor, test_mask: torch.Tensor):
        """Evaluate model"""
        self.model.eval()
        
        logits = self.model(x, AM, AG)
        probs = torch.sigmoid(logits).squeeze()
        
        pred = (probs[test_mask] > 0.5).float()
        true = labels[test_mask]
        
        correct = (pred == true).sum().item()
        total = test_mask.sum().item()
        accuracy = correct / total if total > 0 else 0
        
        return accuracy, probs, pred
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'state_dict': self.model.state_dict(),
            'alpha': self.model.alpha.item()
        }, path)
        print(f"[Bot-AHGCN] Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.alpha.data = torch.tensor(checkpoint['alpha'])
        print(f"[Bot-AHGCN] Model loaded from {path}")


def prepare_ahgcn_data(ahin: HeteroData, similarity_calculator: MetaPathSimilarity,
                      labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare data for Bot-AHGCN model
    
    Args:
        ahin: Heterogeneous information network
        similarity_calculator: MetaPathSimilarity instance
        labels: Node labels
        
    Returns:
        x: Feature matrix, AM: Meta-path similarity matrix, AG: Meta-graph similarity matrix
    """
    import gc
    print("[Bot-AHGCN] Preparing data for training...")
    
    # Get IP node features
    x = ahin['ip'].x
    n_nodes = x.shape[0]
    print(f"[Bot-AHGCN] Number of nodes: {n_nodes}")
    
    # 使用完整的元路径/元图相似性计算
    print("[Bot-AHGCN] Computing meta-path similarity matrix (AM)...")
    try:
        AM = similarity_calculator.compute_meta_path_similarity(meta_path_type='combined')
        print(f"[Bot-AHGCN] AM computed successfully, shape: {AM.shape}")
    except Exception as e:
        print(f"[Bot-AHGCN] Warning: Meta-path computation failed ({e}), using identity matrix")
        AM = torch.eye(n_nodes)
    
    gc.collect()
    
    print("[Bot-AHGCN] Computing meta-graph similarity matrix (AG)...")
    try:
        AG = similarity_calculator.compute_meta_graph_similarity(AM)
        print(f"[Bot-AHGCN] AG computed successfully, shape: {AG.shape}")
    except Exception as e:
        print(f"[Bot-AHGCN] Warning: Meta-graph computation failed ({e}), using identity matrix")
        AG = torch.eye(n_nodes)
    
    gc.collect()
    
    # Add self-loops (important for GCN)
    print("[Bot-AHGCN] Adding self-loops and normalizing...")
    I = torch.eye(n_nodes)
    AM = AM + I
    AG = AG + I
    
    # Normalize adjacency matrices (symmetric normalization)
    def normalize_adj(A):
        D = torch.sum(A, dim=1)
        D_sqrt_inv = torch.pow(D, -0.5).where(torch.isfinite(torch.pow(D, -0.5)), torch.zeros_like(D))
        D_sqrt_inv = torch.diag(D_sqrt_inv)
        return torch.matmul(torch.matmul(D_sqrt_inv, A), D_sqrt_inv)
    
    AM_norm = normalize_adj(AM)
    AG_norm = normalize_adj(AG)
    
    print(f"[Bot-AHGCN] Data prepared: x={x.shape}, AM={AM_norm.shape}, AG={AG_norm.shape}")
    
    return x, AM_norm, AG_norm, labels
