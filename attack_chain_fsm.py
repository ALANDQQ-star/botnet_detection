import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from enum import Enum
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 定义状态与观察空间 (保持不变)
# ==========================================
class BotState(Enum):
    BOOT = 0
    DNS_QUERY = 1
    C2_SETUP = 2
    C2_MAINTAIN = 3
    SCANNING = 4
    ATTACKING = 5
    DATA_EXFIL = 6
    DEAD = 7

class TrafficObs(Enum):
    SILENCE = 0
    DNS_PKT = 1
    TCP_SYN_LOW = 2
    TCP_SYN_HIGH = 3
    UDP_HIGH = 4
    HIGH_BPP = 5
    LONG_CONN = 6
    UNKNOWN = 7

class ObservationQuantizer:
    @staticmethod
    def quantize(row):
        try:
            dst_port = int(float(row.get('dst_port', -1))) if row.get('dst_port') is not None else -1
            pkts = float(row.get('packets', 0))
            bytes_t = float(row.get('bytes', 0))
            dur = float(row.get('duration', 0))
            proto = str(row.get('protocol', '')).lower()
        except:
            return TrafficObs.UNKNOWN

        pps = pkts / (dur + 0.001)
        bpp = bytes_t / (pkts + 1e-6)

        if dst_port == 53 or proto == 'dns':
            return TrafficObs.DNS_PKT
        if (proto == 'udp' and pkts > 10) or pps > 20:
            return TrafficObs.UDP_HIGH
        if bytes_t > 50000 and bpp > 500:
            return TrafficObs.HIGH_BPP
        if proto in ['tcp', 'sctp']:
            if pkts <= 5:
                if dst_port in [22, 23, 80, 443, 445, 3389]:
                    return TrafficObs.TCP_SYN_LOW
                return TrafficObs.TCP_SYN_HIGH
        if dur > 30 and pps < 2:
            return TrafficObs.LONG_CONN
        if pkts < 3:
            return TrafficObs.SILENCE
        # 减少 UNKNOWN：中等包数、低字节视为静默；中等时长低 pps 视为长连接
        if proto in ['tcp', 'udp', 'sctp'] and 3 <= pkts <= 25 and bytes_t < 3000:
            return TrafficObs.SILENCE
        if dur > 5 and pps < 4 and pkts >= 3:
            return TrafficObs.LONG_CONN
        if proto in ['tcp', 'sctp'] and 5 < pkts <= 20 and dst_port not in [22, 23, 80, 443, 445, 3389]:
            return TrafficObs.TCP_SYN_HIGH
            
        return TrafficObs.UNKNOWN

# ==========================================
# 2. HMM 核心引擎 (升级为二阶上下文感知)
# ==========================================
class HMMModel:
    def __init__(self, num_states, num_obs):
        self.N = num_states
        self.M = num_obs
        
        # HMM 基础参数
        self.A = np.zeros((self.N, self.N))
        self.B = np.zeros((self.N, self.M))
        self.Pi = np.zeros(self.N)
        
        self.trans_counts = np.zeros((self.N, self.N))
        self.emit_counts = np.zeros((self.N, self.M))
        self.start_counts = np.zeros(self.N)
        
        # [Level 1] 全局窗口统计 (fallback)
        # Shape: [Current, Future]
        self.global_horizon_counts = np.zeros((self.N, self.N))
        self.global_horizon_probs = np.zeros((self.N, self.N))
        
        # [Level 2] 上下文窗口统计 (Context-Aware)
        # Shape: [Previous, Current, Future]
        # 记录从 Previous 跳转到 Current 后，未来的状态分布
        self.context_horizon_counts = np.zeros((self.N, self.N, self.N))
        self.context_horizon_probs = np.zeros((self.N, self.N, self.N))
        
        self.is_trained = False

    def partial_fit(self, state_sequences, obs_sequences):
        """HMM 参数训练"""
        for states, obs in zip(state_sequences, obs_sequences):
            if len(states) > 0: self.start_counts[states[0]] += 1
            for t in range(len(states)):
                s_curr = states[t]
                o_curr = obs[t]
                if t < len(obs): self.emit_counts[s_curr][o_curr] += 1
                if t < len(states) - 1:
                    s_next = states[t+1]
                    self.trans_counts[s_curr][s_next] += 1

    def update_horizon_stats(self, prev_state, curr_state, future_states):
        """
        [新增] 同时更新全局和上下文统计
        prev_state: int or None (None 表示序列起始)
        curr_state: int
        future_states: array of int
        """
        if len(future_states) == 0: return

        # 1. 更新全局分布 (作为兜底)
        np.add.at(self.global_horizon_counts[curr_state], future_states, 1)
        
        # 2. 更新上下文分布 (如果存在前驱状态)
        if prev_state is not None:
            np.add.at(self.context_horizon_counts[prev_state, curr_state], future_states, 1)

    def finalize(self):
        # 狄利克雷先验，减少极少样本导致的极端概率
        alpha_a = 0.2
        alpha_b = 0.1
        epsilon = 1e-6

        total_starts = np.sum(self.start_counts) + epsilon * self.N
        self.Pi = (self.start_counts + epsilon) / total_starts

        BOOT, DNS_QUERY, C2_SETUP, C2_MAINTAIN, SCANNING, ATTACKING, DATA_EXFIL, DEAD = range(8)
        for i in range(self.N):
            raw_trans = self.trans_counts[i].astype(float) + alpha_a
            # 状态转移约束：DEAD 只进不出；BOOT 仅能到 DNS_QUERY、C2_SETUP、SCANNING、BOOT
            if i == DEAD:
                raw_trans[:] = epsilon
                raw_trans[DEAD] = 1.0
            elif i == BOOT:
                mask = np.ones(self.N, dtype=float)
                for j in [C2_MAINTAIN, ATTACKING, DATA_EXFIL, DEAD]:
                    mask[j] = 0.0
                raw_trans = raw_trans * mask + epsilon * (1 - mask)
            self.A[i] = raw_trans / (raw_trans.sum() + 1e-20)

            raw_emit = self.emit_counts[i] + alpha_b
            # 确保每个状态至少能发射若干观测，避免某状态发射概率全近 0
            raw_emit = np.maximum(raw_emit, 1e-4)
            self.B[i] = raw_emit / raw_emit.sum()

            total_g = np.sum(self.global_horizon_counts[i]) + epsilon * self.N
            self.global_horizon_probs[i] = (self.global_horizon_counts[i] + epsilon) / total_g

            for prev in range(self.N):
                total_c = np.sum(self.context_horizon_counts[prev, i])
                if total_c > 0:
                    self.context_horizon_probs[prev, i] = self.context_horizon_counts[prev, i] / total_c
                else:
                    self.context_horizon_probs[prev, i] = 0.0

        self.is_trained = True

    def viterbi(self, obs_seq):
        if not self.is_trained or len(obs_seq) == 0: return []
        T = len(obs_seq)
        log_A = np.log(self.A + 1e-20)
        log_B = np.log(self.B + 1e-20)
        log_Pi = np.log(self.Pi + 1e-20)
        
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        
        delta[0] = log_Pi + log_B[:, obs_seq[0]]
        
        for t in range(1, T):
            for j in range(self.N):
                probs = delta[t-1] + log_A[:, j]
                best_prev = np.argmax(probs)
                delta[t][j] = probs[best_prev] + log_B[j, obs_seq[t]]
                psi[t][j] = best_prev
                
        best_last_state = np.argmax(delta[T-1])
        path = [0] * T
        path[T-1] = best_last_state
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1][path[t+1]]
        return path

    def predict_context_aware(self, curr_state, prev_state=None):
        """
        [改进] 二阶上下文预测
        策略：
        1. 优先查表 Context[Prev][Curr]
        2. 回退到 Global[Curr]
        3. 最终回退到转移矩阵 A[Curr]
        返回: (next_state, probability_distribution)
        """
        if not self.is_trained: return curr_state, np.zeros(self.N)

        # 1. 尝试上下文匹配
        if prev_state is not None:
            ctx_probs = self.context_horizon_probs[prev_state, curr_state]
            if np.sum(ctx_probs) > 0.01:
                return np.argmax(ctx_probs), ctx_probs.copy()

        # 2. 回退到全局分布
        g_probs = self.global_horizon_probs[curr_state]
        if np.sum(g_probs) > 0.01:
            return np.argmax(g_probs), g_probs.copy()

        # 3. 回退到转移矩阵
        a_probs = self.A[curr_state]
        return np.argmax(a_probs), a_probs.copy()

    def save(self, path='hmm_model.json'):
        data = {
            'A': self.A.tolist(), 
            'B': self.B.tolist(), 
            'Pi': self.Pi.tolist(), 
            'global_horizon_probs': self.global_horizon_probs.tolist(),
            'context_horizon_probs': self.context_horizon_probs.tolist(),
            'trained': self.is_trained
        }
        with open(path, 'w') as f: json.dump(data, f)
        
    def load(self, path='hmm_model.json'):
        if not os.path.exists(path): return False
        try:
            with open(path, 'r') as f: data = json.load(f)
            self.A = np.array(data['A'])
            self.B = np.array(data['B'])
            self.Pi = np.array(data['Pi'])
            self.is_trained = data.get('trained', True)
            
            if 'global_horizon_probs' in data:
                self.global_horizon_probs = np.array(data['global_horizon_probs'])
                
            if 'context_horizon_probs' in data:
                self.context_horizon_probs = np.array(data['context_horizon_probs'])
            else:
                # 兼容旧版本，初始化为空
                self.context_horizon_probs = np.zeros((self.N, self.N, self.N))
                
            return True
        except: return False

# ==========================================
# 3. 业务逻辑封装
# ==========================================
class AttackChainInference:
    def __init__(self, target_ips: list = None, model_path='hmm_model.json'):
        self.bot_ips = set(target_ips) if target_ips else None
        self.model = HMMModel(num_states=len(BotState), num_obs=len(TrafficObs))
        self.model_path = model_path
        self.model.load(model_path)
        
        self.min_horizon = 1
        self.max_horizon = 500

    def _pseudo_label(self, row):
        obs = ObservationQuantizer.quantize(row)
        if obs == TrafficObs.DNS_PKT: return BotState.DNS_QUERY
        if obs == TrafficObs.UDP_HIGH: return BotState.ATTACKING
        if obs == TrafficObs.HIGH_BPP: return BotState.DATA_EXFIL
        if obs == TrafficObs.TCP_SYN_LOW: return BotState.C2_SETUP
        if obs == TrafficObs.LONG_CONN: return BotState.C2_MAINTAIN
        if obs == TrafficObs.TCP_SYN_HIGH: return BotState.SCANNING
        return BotState.BOOT

    def train_on_dataset(self, df):
        """
        训练：更新 Global 和 Context 统计
        """
        if df.empty: return
        if 'is_bot' in df.columns:
            relevant_df = df[df['is_bot'] == True].sort_values('ts')
        else:
            relevant_df = df.sort_values('ts')
        if relevant_df.empty: return

        state_seqs, obs_seqs = [], []
        grouped = relevant_df.groupby('src_ip')
        min_seq_len = 2  # 过滤过短序列，减少噪声

        for ip, group in grouped:
            if len(group) < min_seq_len:
                continue
            obs_seq = []
            group_states = []
            for row in group.itertuples(index=False):
                row_dict = row._asdict()
                obs = ObservationQuantizer.quantize(row_dict)
                obs_seq.append(obs.value)
                if obs == TrafficObs.DNS_PKT: state = BotState.DNS_QUERY
                elif obs == TrafficObs.UDP_HIGH: state = BotState.ATTACKING
                elif obs == TrafficObs.HIGH_BPP: state = BotState.DATA_EXFIL
                elif obs == TrafficObs.TCP_SYN_LOW: state = BotState.C2_SETUP
                elif obs == TrafficObs.LONG_CONN: state = BotState.C2_MAINTAIN
                elif obs == TrafficObs.TCP_SYN_HIGH: state = BotState.SCANNING
                else: state = BotState.BOOT
                group_states.append(state.value)
                
            group_obs = np.array(obs_seq)
            group_states = np.array(group_states)
            group_ts = group['ts'].values
            
            state_seqs.append(group_states)
            obs_seqs.append(group_obs)
            
            # === 二阶窗口统计更新 ===
            t_start = group_ts + self.min_horizon
            t_end = group_ts + self.max_horizon
            
            idx_start = np.searchsorted(group_ts, t_start, side='left')
            idx_end = np.searchsorted(group_ts, t_end, side='right')
            
            n_samples = len(group_ts)
            for i in range(n_samples):
                s = idx_start[i]
                e = idx_end[i]
                
                if s < n_samples and s < e:
                    valid_e = min(e, n_samples)
                    if s < valid_e:
                        future_states = group_states[s:valid_e]
                        
                        # 获取前一状态 (Previous State)
                        curr_s = group_states[i]
                        prev_s = group_states[i-1] if i > 0 else None
                        
                        # 同时更新 Global 和 Context
                        self.model.update_horizon_stats(prev_s, curr_s, future_states)
        
        self.model.partial_fit(state_seqs, obs_seqs)

    def finalize_training(self):
        self.model.finalize()
        self.model.save(self.model_path)

    def run_inference_with_evaluation(self, df, prediction_horizon=500):
        if not self.model.is_trained:
            return {}

        df = df.copy()
        if 'ts' not in df.columns and 'start_time' in df.columns:
            df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
            df['ts'] = df['start_time'].astype(np.int64) // 10**9

        # Only filter by src_ip (the bot as initiator of traffic)
        relevant_df = df[df['src_ip'].isin(self.bot_ips)].sort_values('ts')
        
        if relevant_df.empty:
            return {}

        print(f"[HMM] 正在对 {len(self.bot_ips)} 个 Bot 进行 Context-Aware Trend Prediction...")
        print(f"[HMM] 相关流量记录: {len(relevant_df)} 条")

        # Vectorized feature extraction for the entire dataframe at once
        dst_port = pd.to_numeric(relevant_df.get('dst_port', pd.Series(dtype=float)), errors='coerce').fillna(-1).values
        pkts = pd.to_numeric(relevant_df.get('packets', pd.Series(dtype=float)), errors='coerce').fillna(0).values
        bytes_t = pd.to_numeric(relevant_df.get('bytes', pd.Series(dtype=float)), errors='coerce').fillna(0).values
        dur = pd.to_numeric(relevant_df.get('duration', pd.Series(dtype=float)), errors='coerce').fillna(0).values
        proto = relevant_df.get('protocol', pd.Series(dtype=str)).fillna('').str.lower().values

        pps = pkts / (dur + 0.001)
        bpp = bytes_t / (pkts + 1e-6)

        obs_arr = np.full(len(relevant_df), TrafficObs.UNKNOWN.value, dtype=int)
        obs_arr[(dst_port == 53) | (proto == 'dns')] = TrafficObs.DNS_PKT.value
        
        udp_high = ((proto == 'udp') & (pkts > 10)) | (pps > 20)
        obs_arr[udp_high & (obs_arr == TrafficObs.UNKNOWN.value)] = TrafficObs.UDP_HIGH.value
        
        high_bpp = (bytes_t > 50000) & (bpp > 500)
        obs_arr[high_bpp & (obs_arr == TrafficObs.UNKNOWN.value)] = TrafficObs.HIGH_BPP.value
        
        tcp_mask = np.isin(proto, ['tcp', 'sctp'])
        low_port = np.isin(dst_port.astype(int), [22, 23, 80, 443, 445, 3389])
        tcp_syn_low = tcp_mask & (pkts <= 5) & low_port
        tcp_syn_high = tcp_mask & (pkts <= 5) & ~low_port
        obs_arr[tcp_syn_low & (obs_arr == TrafficObs.UNKNOWN.value)] = TrafficObs.TCP_SYN_LOW.value
        obs_arr[tcp_syn_high & (obs_arr == TrafficObs.UNKNOWN.value)] = TrafficObs.TCP_SYN_HIGH.value
        
        long_conn = (dur > 30) & (pps < 2)
        obs_arr[long_conn & (obs_arr == TrafficObs.UNKNOWN.value)] = TrafficObs.LONG_CONN.value
        
        silence = pkts < 3
        obs_arr[silence & (obs_arr == TrafficObs.UNKNOWN.value)] = TrafficObs.SILENCE.value

        # 减少 UNKNOWN：与 ObservationQuantizer 一致的补充规则
        mid_pkts = (pkts >= 3) & (pkts <= 25) & (bytes_t < 3000)
        tcp_udp = np.isin(proto, ['tcp', 'udp', 'sctp'])
        obs_arr[tcp_udp & mid_pkts & (obs_arr == TrafficObs.UNKNOWN.value)] = TrafficObs.SILENCE.value
        long_conn_alt = (dur > 5) & (pps < 4) & (pkts >= 3)
        obs_arr[long_conn_alt & (obs_arr == TrafficObs.UNKNOWN.value)] = TrafficObs.LONG_CONN.value
        tcp_mid = tcp_mask & (pkts > 5) & (pkts <= 20) & ~low_port
        obs_arr[tcp_mid & (obs_arr == TrafficObs.UNKNOWN.value)] = TrafficObs.TCP_SYN_HIGH.value

        # Vectorized pseudo-label mapping
        OBS_TO_STATE = {
            TrafficObs.DNS_PKT.value: BotState.DNS_QUERY.value,
            TrafficObs.UDP_HIGH.value: BotState.ATTACKING.value,
            TrafficObs.HIGH_BPP.value: BotState.DATA_EXFIL.value,
            TrafficObs.TCP_SYN_LOW.value: BotState.C2_SETUP.value,
            TrafficObs.LONG_CONN.value: BotState.C2_MAINTAIN.value,
            TrafficObs.TCP_SYN_HIGH.value: BotState.SCANNING.value,
        }
        state_arr = np.full(len(relevant_df), BotState.BOOT.value, dtype=int)
        for obs_val, state_val in OBS_TO_STATE.items():
            state_arr[obs_arr == obs_val] = state_val

        relevant_df = relevant_df.copy()
        relevant_df['_obs'] = obs_arr
        relevant_df['_state'] = state_arr

        report = {}
        grouped = relevant_df.groupby('src_ip')
        
        for ip, group in grouped:
            if ip not in self.bot_ips:
                continue
                
            obs_seq = group['_obs'].values.tolist()
            pseudo_labels = group['_state'].values.tolist()
            times = group['ts'].values
            
            hidden_path = self.model.viterbi(obs_seq)
            
            if len(hidden_path) > 0:
                path_names = [BotState(s).name for s in hidden_path]
                obs_names = [TrafficObs(o).name for o in obs_seq]
                
                chain_log = []
                sample_step = max(1, len(hidden_path) // 30)
                for i in range(0, len(hidden_path), sample_step):
                    # 将 Unix 时间戳转换为年月日时分秒格式
                    try:
                        ts_str = datetime.fromtimestamp(times[i]).strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        ts_str = str(times[i])
                    chain_log.append({'ts': ts_str, 'state': path_names[i], 'obs': obs_names[i]})
                # 最后一条记录
                try:
                    ts_str = datetime.fromtimestamp(times[-1]).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    ts_str = str(times[-1])
                chain_log.append({'ts': ts_str, 'state': path_names[-1], 'obs': obs_names[-1]})

                report[ip] = {
                    'final_state': path_names[-1],
                    'chain_len': len(hidden_path),
                    'recent_logs': chain_log[-10:],
                }

                # 预测下一步状态
                curr_s = hidden_path[-1]
                prev_s = hidden_path[-2] if len(hidden_path) > 1 else None
                next_state, next_probs = self.model.predict_context_aware(curr_s, prev_s)
                next_state_name = BotState(next_state).name

                # 构建概率分布（top-3）
                top_indices = np.argsort(next_probs)[::-1][:3]
                next_dist = []
                for idx in top_indices:
                    if next_probs[idx] > 0.01:
                        next_dist.append({
                            'state': BotState(idx).name,
                            'prob': round(float(next_probs[idx]), 4)
                        })

                report[ip]['predicted_next_state'] = next_state_name
                report[ip]['next_state_distribution'] = next_dist
        
        print(f"[HMM] 完成。生成了 {len(report)} 个 Bot 的攻击链报告。")
        return report

    def identify_c2_nodes(self, df):
        """
        基于攻击链状态识别 C2 服务器节点。
        C2 特征：
        1. 作为 dst_ip 被多个 bot 连接（高入度）
        2. 关联流量以 C2_SETUP / C2_MAINTAIN 状态为主
        3. 长连接、低包速率、固定端口模式
        返回: list of dict，每个包含 ip, score, evidence
        """
        if not self.bot_ips or df.empty:
            return []

        # 筛选 bot 发出的流量
        bot_traffic = df[df['src_ip'].isin(self.bot_ips)].copy()
        if bot_traffic.empty:
            return []

        # 统计每个 dst_ip 被多少个不同 bot 连接
        dst_bot_counts = bot_traffic.groupby('dst_ip')['src_ip'].nunique()
        # 排除 bot 自身作为 dst
        dst_bot_counts = dst_bot_counts[~dst_bot_counts.index.isin(self.bot_ips)]

        if dst_bot_counts.empty:
            return []

        # 对每个候选 dst_ip 分析流量特征
        c2_candidates = []
        for dst_ip, n_bots in dst_bot_counts.items():
            dst_flows = bot_traffic[bot_traffic['dst_ip'] == dst_ip]
            n_flows = len(dst_flows)
            if n_flows < 3:
                continue

            # 量化观测
            c2_obs_count = 0
            for _, row in dst_flows.iterrows():
                obs = ObservationQuantizer.quantize(row.to_dict())
                if obs in (TrafficObs.TCP_SYN_LOW, TrafficObs.LONG_CONN):
                    c2_obs_count += 1

            c2_ratio = c2_obs_count / (n_flows + 1e-10)

            # 端口集中度
            if 'dst_port' in dst_flows.columns:
                port_counts = dst_flows['dst_port'].value_counts()
                port_concentration = port_counts.iloc[0] / n_flows if len(port_counts) > 0 else 0
            else:
                port_concentration = 0

            # 综合评分
            score = (
                0.35 * min(1.0, n_bots / 3) +        # 被多个 bot 连接
                0.30 * c2_ratio +                      # C2 相关流量占比
                0.20 * port_concentration +             # 端口集中度
                0.15 * min(1.0, n_flows / 20)          # 流量量
            )

            if score > 0.25:
                c2_candidates.append({
                    'ip': dst_ip,
                    'score': round(float(score), 4),
                    'n_bots_connected': int(n_bots),
                    'n_flows': int(n_flows),
                    'c2_traffic_ratio': round(float(c2_ratio), 4),
                    'port_concentration': round(float(port_concentration), 4),
                    'evidence': f"被{n_bots}个bot连接, C2流量占比{c2_ratio:.1%}, 端口集中度{port_concentration:.1%}"
                })

        c2_candidates.sort(key=lambda x: x['score'], reverse=True)
        print(f"[C2] 识别到 {len(c2_candidates)} 个 C2 候选节点")
        return c2_candidates

    def evaluate_state_prediction(self, df):
        """
        评估状态预测准确率。
        方法：对每个 bot 的流量序列，先用 Viterbi 解码得到隐状态路径，
        然后用前 t 步预测第 t+1 步状态，与 Viterbi 路径对比。
        同时计算 top-1 和 top-3 准确率。
        返回: dict 包含各项准确率指标
        """
        if not self.model.is_trained or df.empty:
            return {'top1_accuracy': 0, 'top3_accuracy': 0, 'total': 0}

        if 'ts' not in df.columns and 'start_time' in df.columns:
            df = df.copy()
            df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
            df['ts'] = df['start_time'].astype(np.int64) // 10**9

        relevant_df = df[df['src_ip'].isin(self.bot_ips)].sort_values('ts') if self.bot_ips else df.sort_values('ts')
        if relevant_df.empty:
            return {'top1_accuracy': 0, 'top3_accuracy': 0, 'total': 0}

        top1_correct = 0
        top3_correct = 0
        total = 0
        state_correct = defaultdict(int)
        state_total = defaultdict(int)

        grouped = relevant_df.groupby('src_ip')
        for ip, group in grouped:
            if len(group) < 5:
                continue

            # 提取观测序列
            obs_seq = []
            for _, row in group.iterrows():
                obs = ObservationQuantizer.quantize(row.to_dict())
                obs_seq.append(obs.value)

            # Viterbi 解码得到隐状态路径
            hidden_path = self.model.viterbi(obs_seq)
            if len(hidden_path) < 3:
                continue

            # 逐步预测
            for t in range(1, len(hidden_path) - 1):
                prev_s = hidden_path[t - 1]
                curr_s = hidden_path[t]
                actual_next = hidden_path[t + 1]

                predicted_next, probs = self.model.predict_context_aware(curr_s, prev_s)

                state_name = BotState(curr_s).name
                state_total[state_name] += 1
                total += 1

                # Top-1 准确率
                if predicted_next == actual_next:
                    top1_correct += 1
                    state_correct[state_name] += 1

                # Top-3 准确率
                top3_indices = np.argsort(probs)[::-1][:3]
                if actual_next in top3_indices:
                    top3_correct += 1

        top1_acc = top1_correct / total if total > 0 else 0
        top3_acc = top3_correct / total if total > 0 else 0

        # 每个状态的预测准确率
        per_state_acc = {}
        for state_name in state_total:
            per_state_acc[state_name] = {
                'accuracy': round(state_correct[state_name] / state_total[state_name], 4) if state_total[state_name] > 0 else 0,
                'total': state_total[state_name],
                'correct': state_correct[state_name]
            }

        result = {
            'top1_accuracy': round(top1_acc, 4),
            'top3_accuracy': round(top3_acc, 4),
            'total_predictions': total,
            'top1_correct': top1_correct,
            'top3_correct': top3_correct,
            'per_state_accuracy': per_state_acc
        }

        print(f"[HMM评估] 状态预测 Top-1 准确率: {top1_acc:.4f} ({top1_correct}/{total})")
        print(f"[HMM评估] 状态预测 Top-3 准确率: {top3_acc:.4f} ({top3_correct}/{total})")
        for s, info in sorted(per_state_acc.items(), key=lambda x: x[1]['total'], reverse=True):
            print(f"  {s:15s}: {info['accuracy']:.4f} ({info['correct']}/{info['total']})")

        return result