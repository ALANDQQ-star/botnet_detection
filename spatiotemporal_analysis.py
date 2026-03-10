import pandas as pd
import networkx as nx
import numpy as np
import json
import time
import warnings
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

class SpatioTemporalAnalyzer:
    """
    时空关联分析器 (Enhanced for Visualization)
    """
    
    def __init__(self, bot_ips: list):
        self.bot_ips = set(bot_ips)
        self.graph = nx.DiGraph()
        self.common_ports = {80, 443, 53, 8080, 25, 110, 995, 143, 993} 
        
    def analyze(self, df: pd.DataFrame, ground_truth_cc: set = None):
        print("[Analysis] 启动时空关联分析与拓扑构建...")
        
        # 1. 基础数据切片
        src_mask = df['src_ip'].isin(self.bot_ips).to_numpy()
        dst_mask = df['dst_ip'].isin(self.bot_ips).to_numpy()
        rel_df = df[src_mask | dst_mask].copy()
        
        if rel_df.empty: return None, None

        # 2. 全局流行度：仅在关联子集上计算，避免对全量 df 做 groupby
        global_pop = rel_df.groupby('dst_ip')['src_ip'].nunique().to_dict()
        
        # 3. 提取特征并识别 C2
        feature_df = self._extract_advanced_features(rel_df, global_pop)
        c2_candidates = []
        if not feature_df.empty:
            c2_candidates = self._train_predict_c2(feature_df)
        
        # 4. 构建用于可视化的拓扑数据 (新增功能)
        self._export_visualization_data(rel_df, c2_candidates, "viz_data.json")

        # 5. 传统路径追踪
        paths = self._trace_propagation_paths(rel_df)
        timeline = self._build_timeline(rel_df)
        
        # 6. 评估 (如果存在真值)
        eval_metrics = {}
        if ground_truth_cc and c2_candidates:
            preds = set([x['ip'] for x in c2_candidates])
            tp = len(preds & ground_truth_cc)
            fp = len(preds - ground_truth_cc)
            fn = len(ground_truth_cc - preds)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            eval_metrics = {'precision': round(p,4), 'recall': round(r,4), 'f1': round(f1,4)}

        return {
            'c2_candidates': c2_candidates,
            'evaluation': eval_metrics,
            'propagation_paths': paths
        }, timeline

    def _export_visualization_data(self, df, c2_candidates, filename):
        """
        [新增] 导出前端可视化专用的图数据结构
        包含：Nodes (Bot, C2, Victim) 和 Edges (通信链路)
        """
        print(f"[Viz] 正在生成全量拓扑数据: {filename} ...")
        
        # 识别角色
        c2_ips = set([x['ip'] for x in c2_candidates]) if c2_candidates else set()
        
        # 聚合边：减少数据量，合并同一对IP的多次通信
        # 取每个链接的第一次通信时间、总包数、总字节
        edges_grp = df.groupby(['src_ip', 'dst_ip']).agg({
            'start_time': 'min',
            'packets': 'sum',
            'bytes': 'sum',
            'dst_port': lambda x: list(x.unique())[:3] # 只记录前3个端口
        }).reset_index()
        
        nodes = {}
        links = []
        # 先过滤出仅与 Bot/C2 相关的边，避免对海量边做 iterrows
        keep = edges_grp['src_ip'].isin(self.bot_ips) | edges_grp['dst_ip'].isin(self.bot_ips) | edges_grp['src_ip'].isin(c2_ips) | edges_grp['dst_ip'].isin(c2_ips)
        edges_grp = edges_grp.loc[keep]

        for row in edges_grp.itertuples(index=False):
            src, dst = row.src_ip, row.dst_ip
            # 添加节点
            for ip in [src, dst]:
                if ip not in nodes:
                    role = 'NORMAL'
                    if ip in c2_ips: role = 'C2'
                    elif ip in self.bot_ips: role = 'BOT'
                    nodes[ip] = {'id': ip, 'role': role, 'val': 1}
                else:
                    nodes[ip]['val'] += 1 # 活跃度
            
            # 添加边
            port_val = getattr(row, 'dst_port', None)
            if isinstance(port_val, (list, tuple)):
                port_str = ", ".join(str(int(p)) for p in port_val if p is not None and int(p) >= 0)
            else:
                port_str = str(int(port_val)) if port_val is not None else "N/A"
            links.append({
                'source': src,
                'target': dst,
                'pkts': int(getattr(row, 'packets', 0)),
                'bytes': int(getattr(row, 'bytes', 0)),
                'port': port_str,
                'time': str(getattr(row, 'start_time', ''))
            })
            
        data = {
            'nodes': list(nodes.values()),
            'links': links
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def _extract_advanced_features(self, df, global_pop_map):
        candidates = set(df['src_ip'].unique()) | set(df['dst_ip'].unique())
        candidates = list(candidates - self.bot_ips)
        if not candidates: return pd.DataFrame()

        out_df = df[df['src_ip'].isin(self.bot_ips) & df['dst_ip'].isin(candidates)]
        in_df = df[df['src_ip'].isin(candidates) & df['dst_ip'].isin(self.bot_ips)]
        has_duration = 'duration' in df.columns

        def get_stats(sub_df, group_col):
            if sub_df.empty: return pd.DataFrame()
            agg_dict = {
                'packets': ['sum', 'count'],
                'bytes': 'sum',
                'start_time': list
            }
            if has_duration:
                agg_dict['duration'] = 'mean'
            base = sub_df.groupby(group_col).agg(agg_dict)
            base.columns = ['pkts', 'flows', 'bytes', 'ts_list'] + (['duration_avg'] if has_duration else [])
            if not has_duration:
                base['duration_avg'] = 0.0

            def advanced_metrics(row):
                avg_bpp = row['bytes'] / (row['pkts'] + 1)
                try:
                    ts = np.sort(np.array(row['ts_list'], dtype='datetime64[ns]'))
                    ts = ts[~np.isnat(ts)]
                except: ts = np.array([])
                if len(ts) > 2:
                    ts_unix = ts.astype(np.int64) // 10**9
                    diffs = np.diff(ts_unix)
                    iat_mean, iat_std = np.mean(diffs), np.std(diffs)
                else:
                    iat_mean, iat_std = 300.0, 300.0
                return pd.Series([avg_bpp, iat_mean, iat_std], index=['bpp', 'iat_mean', 'iat_std'])

            metrics = base.apply(advanced_metrics, axis=1)
            out = base.drop(columns=['ts_list']).join(metrics)
            if 'duration_avg' in base.columns:
                out['duration_avg'] = base['duration_avg']
            return out

        feat_out = get_stats(out_df, 'dst_ip')
        feat_in = get_stats(in_df, 'src_ip')
        full_idx = list(set(feat_out.index) | set(feat_in.index))
        # 限制候选数量，避免后续 O(candidates) 循环导致卡死
        if len(full_idx) > 3000:
            out_flows = feat_out.get('flows', pd.Series(0, index=feat_out.index)).reindex(full_idx).fillna(0)
            in_flows = feat_in.get('flows', pd.Series(0, index=feat_in.index)).reindex(full_idx).fillna(0)
            tot = out_flows + in_flows
            full_idx = tot.nlargest(3000).index.tolist()
        features = pd.DataFrame(index=full_idx)
        features = features.join(feat_out, rsuffix='_in').join(feat_in, rsuffix='_out').fillna(0)
        features['global_pop'] = features.index.map(lambda x: global_pop_map.get(x, 0))
        features['purity'] = features['flows'] / (features['global_pop'] + features['flows'] + 1)
        features['min_iat_std'] = features[['iat_std', 'iat_std_out']].min(axis=1).replace(0, 300)
        features['total_flows'] = features['flows'] + features['flows_out']
        features['bpp'] = features['bpp'].fillna(0)

        # bot_ratio: 与该候选 IP 通信的 Bot 数 / 总通信对端数（向量化，避免逐候选扫描）
        rel_sub = df[df['dst_ip'].isin(full_idx) | df['src_ip'].isin(full_idx)]
        out_pairs = rel_sub[rel_sub['dst_ip'].isin(full_idx)][['dst_ip', 'src_ip']].drop_duplicates().rename(columns={'dst_ip': 'c', 'src_ip': 'peer'})
        in_pairs = rel_sub[rel_sub['src_ip'].isin(full_idx)][['src_ip', 'dst_ip']].drop_duplicates().rename(columns={'src_ip': 'c', 'dst_ip': 'peer'})
        all_pairs = pd.concat([out_pairs, in_pairs], ignore_index=True).drop_duplicates()
        all_pairs['is_bot'] = all_pairs['peer'].isin(self.bot_ips)
        bot_per_c = all_pairs.groupby('c')['is_bot'].sum()
        tot_per_c = all_pairs.groupby('c')['peer'].nunique()
        ratio_ser = (bot_per_c / tot_per_c.replace(0, np.nan)).fillna(0)
        features['bot_ratio'] = ratio_ser.reindex(features.index).fillna(0).values

        # port_concentration: 目的端口集中度 (top-1 占比)，向量化
        port_src = rel_sub[rel_sub['dst_ip'].isin(full_idx)][['dst_ip', 'dst_port']].rename(columns={'dst_ip': 'c'})
        port_dst = rel_sub[rel_sub['src_ip'].isin(full_idx)][['src_ip', 'dst_port']].rename(columns={'src_ip': 'c'})
        port_df = pd.concat([port_src, port_dst], ignore_index=True)
        port_df = port_df[port_df['dst_port'].notna()]
        if not port_df.empty:
            port_cnt = port_df.groupby(['c', 'dst_port']).size()
            tot = port_cnt.groupby(level=0).sum()
            top = port_cnt.groupby(level=0).max()
            port_conc_ser = (top / tot).fillna(0)
            features['port_concentration'] = port_conc_ser.reindex(features.index).fillna(0).values
        else:
            features['port_concentration'] = 0.0

        # conn_duration: 与 Bot 相关流的平均时长
        if 'duration_avg' in features.columns and features['duration_avg'].abs().sum() > 0:
            features['conn_duration'] = features[['duration_avg', 'duration_avg_out']].fillna(0).max(axis=1)
        else:
            features['conn_duration'] = 0.0
        features['conn_duration'] = features['conn_duration'].fillna(0)
        return features

    def _train_predict_c2(self, df):
        for col in ['bot_ratio', 'port_concentration', 'conn_duration']:
            if col not in df.columns:
                df[col] = 0.0
        df['bot_ratio'] = df['bot_ratio'].fillna(0)
        df['port_concentration'] = df['port_concentration'].fillna(0)
        df['conn_duration'] = df['conn_duration'].fillna(0)

        # 弱标签：负样本 = 流行度高或明显非 C2；正样本 = 高纯度/低 IAT 方差 + 一定 Bot 比例
        cond_neg = (df['global_pop'] > 20) | ((df['global_pop'] > 10) & (df['purity'] < 0.2))
        cond_pos = (df['total_flows'] >= 2) & (
            (df['purity'] > 0.5) | (df['min_iat_std'] < 120)
        ) & (df['bot_ratio'] > 0.3)
        df['weak_label'] = -1
        df.loc[cond_neg, 'weak_label'] = 0
        df.loc[cond_pos, 'weak_label'] = 1

        train_df = df[df['weak_label'] != -1]
        if train_df.empty:
            return self._fallback_rule_based(df)

        feat_cols = ['flows', 'purity', 'min_iat_std', 'global_pop', 'bot_ratio', 'port_concentration', 'conn_duration']
        X_train = train_df[feat_cols].fillna(0).values
        y_train = train_df['weak_label'].values
        X_all = df[feat_cols].fillna(0).values

        try:
            clf = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_all)[:, 1]
        except Exception:
            return self._fallback_rule_based(df)

        df['score'] = probs
        # 阈值略放宽并配合 top-K：取 score > 0.45 的前 20，或至少前 5 个
        thresh = 0.45
        scored = df[df['score'] > thresh].sort_values('score', ascending=False)
        if len(scored) < 5:
            scored = df.nlargest(10, 'score')
        candidates = []
        for ip, row in scored.head(20).iterrows():
            if str(ip).endswith('.255'):
                continue
            if row.get('global_pop', 0) > 50 and row.get('purity', 0) < 0.2:
                continue
            candidates.append({
                'ip': ip, 'role': 'C2', 'confidence': float(row['score']),
                'flows': int(row.get('total_flows', 0))
            })
        return candidates if candidates else self._fallback_rule_based(df)

    def _fallback_rule_based(self, df):
        if df.empty or len(df) == 0:
            return []
        for col in ['purity', 'min_iat_std', 'bot_ratio']:
            if col not in df.columns:
                df[col] = 0.0
        df['purity'] = df['purity'].fillna(0)
        df['min_iat_std'] = df['min_iat_std'].fillna(300)
        df['bot_ratio'] = df['bot_ratio'].fillna(0)
        # 按纯度降序、IAT 方差升序、bot_ratio 降序排序
        fallback_df = df.sort_values(
            by=['purity', 'bot_ratio'],
            ascending=[False, False]
        )
        fallback_df = fallback_df[fallback_df['min_iat_std'] < 200].head(15)
        if fallback_df.empty:
            fallback_df = df.nlargest(10, 'purity')
        return [
            {'ip': ip, 'role': 'Fallback', 'confidence': 0.5, 'flows': int(row.get('total_flows', 0))}
            for ip, row in fallback_df.head(15).iterrows() if not str(ip).endswith('.255')
        ]
        
    def _trace_propagation_paths(self, df):
        # 简化版路径追踪
        return []

    def _build_timeline(self, df):
        return []