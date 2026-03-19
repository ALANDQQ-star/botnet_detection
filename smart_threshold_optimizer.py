"""
智能阈值优化器 - 高效版

基于分数分布的统计特性，快速找到最优阈值
核心改进：
1. 使用快速统计方法而非迭代聚类
2. 利用分布交叉点特性
3. 自适应估计异常比例
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
import warnings
warnings.filterwarnings('ignore')


class SmartThresholdOptimizer:
    """
    智能阈值优化器 - 高效版

    基于分数分布统计特性，不依赖真实标签
    支持训练集先验阈值引导
    """

    def __init__(self, verbose=False, train_prior=None):
        self.threshold = None
        self.verbose = verbose
        self.debug_info = {}
        # train_prior: dict with keys like 'threshold', 'bot_ratio', 'bot_score_stats'
        self.train_prior = train_prior
    
    def _log_transform(self, x, eps=1e-10):
        """安全的对数变换"""
        return np.log10(x + eps)
    
    def _estimate_anomaly_ratio(self, probs):
        """
        估计异常比例
        
        使用多种统计方法的融合
        """
        log_probs = self._log_transform(probs)
        
        # 方法1：基于偏度的估计
        # 高度右偏的分布表示存在小比例异常值
        skewness = stats.skew(log_probs)
        
        # 偏度越大，异常比例越小
        if skewness > 5:
            skew_ratio = 0.005
        elif skewness > 3:
            skew_ratio = 0.01
        elif skewness > 1:
            skew_ratio = 0.02
        else:
            skew_ratio = 0.03
        
        # 方法2：基于分位数的估计
        # 正常节点和高分异常节点之间应该有明显的差距
        p99 = np.percentile(probs, 99)
        p99_5 = np.percentile(probs, 99.5)
        p99_9 = np.percentile(probs, 99.9)
        
        # 如果99.5%-99%的差距远大于99%-98%，说明异常集中在尾部
        p98 = np.percentile(probs, 98)
        gap_low = p99 - p98
        gap_high = p99_5 - p99
        
        if gap_high > gap_low * 2:
            # 尾部有明显的异常聚集
            quantile_ratio = 0.005
        elif gap_high > gap_low * 1.5:
            quantile_ratio = 0.01
        else:
            quantile_ratio = 0.02
        
        # 融合估计
        estimated_ratio = (skew_ratio + quantile_ratio) / 2
        
        # 限制在合理范围
        estimated_ratio = max(0.001, min(0.05, estimated_ratio))
        
        return estimated_ratio
    
    def _find_threshold_distribution_crossing(self, probs):
        """
        方法1：分布交叉点检测
        
        核心思想：找到正常分布尾部和异常分布头部的交叉点
        """
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        
        # 使用对数变换增强低值区域分辨率
        log_probs = self._log_transform(sorted_probs)
        
        # 计算概率密度（使用差分近似）
        # 在高尾区域计算局部密度变化
        tail_start = int(n * 0.95)
        tail_probs = sorted_probs[tail_start:]
        tail_log = log_probs[tail_start:]
        
        # 计算对数空间的一阶差分（近似密度）
        diff = np.diff(tail_log)
        
        # 找到差分变化最大的点（密度拐点）
        # 这通常对应于正常节点和异常节点的边界
        diff_diff = np.diff(diff)
        
        # 找到最大的正向变化（从平稳到陡峭的转变）
        # 使用平滑避免噪声
        if len(diff_diff) > 10:
            window = min(11, len(diff_diff) // 2)
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                diff_diff_smooth = savgol_filter(diff_diff, window, 2)
            else:
                diff_diff_smooth = diff_diff
            
            # 找到拐点
            knee_idx = np.argmax(np.abs(diff_diff_smooth))
            threshold = tail_probs[knee_idx]
        else:
            threshold = np.percentile(probs, 98)
        
        return threshold
    
    def _find_threshold_percentile_gap(self, probs):
        """
        方法2：分位数间距分析
        
        找到分位数间距突变点
        """
        # 在高分区域分析
        percentiles = np.linspace(95, 99.9, 100)
        p_values = np.percentile(probs, percentiles)
        
        # 计算相邻分位数的相对间距
        gaps = np.diff(p_values)
        percentile_steps = np.diff(percentiles)
        
        # 归一化间距
        normalized_gaps = gaps / (percentile_steps + 1e-10)
        
        # 找到间距突增的点
        gap_changes = np.diff(normalized_gaps)
        
        # 找到最大变化点
        max_change_idx = np.argmax(gap_changes)
        
        threshold = (p_values[max_change_idx + 1] + p_values[max_change_idx + 2]) / 2
        
        return threshold
    
    def _find_threshold_double_mad(self, probs):
        """
        方法3：双MAD（Median Absolute Deviation）方法
        
        使用MAD检测异常值
        """
        log_probs = self._log_transform(probs)
        
        # 计算MAD
        median = np.median(log_probs)
        mad = np.median(np.abs(log_probs - median))
        
        # 修正的Z-score
        modified_z = 0.6745 * (log_probs - median) / (mad + 1e-10)
        
        # 使用3.5作为阈值（对应于约0.5%的异常比例）
        threshold_idx = np.where(modified_z > 3.5)[0]
        
        if len(threshold_idx) > 0:
            threshold = probs[threshold_idx[0]]
        else:
            threshold = np.percentile(probs, 99)
        
        return threshold
    
    def _find_threshold_iqr(self, probs):
        """
        方法4：IQR方法（改进版）
        
        使用四分位距检测异常
        """
        log_probs = self._log_transform(probs)
        
        q1, q3 = np.percentile(log_probs, [25, 75])
        iqr = q3 - q1
        
        # 使用更严格的阈值（3倍IQR而非1.5倍）
        upper_bound = q3 + 3 * iqr
        
        threshold = 10**upper_bound
        
        return threshold
    
    def _find_threshold_density_change(self, probs):
        """
        方法5：密度变化检测
        
        分析概率密度的变化率
        """
        # 使用有限差分估计密度
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        
        # 使用核密度估计的思想
        # 在对数空间均匀采样
        log_probs = self._log_transform(sorted_probs)
        
        # 计算局部密度（使用k近邻）
        k = max(10, int(n * 0.001))  # 0.1%的邻居
        
        # 在高尾区域计算密度
        tail_start = int(n * 0.90)
        tail_indices = np.arange(tail_start, n - k)
        
        if len(tail_indices) < 10:
            return np.percentile(probs, 98)
        
        # 计算局部密度
        local_densities = []
        for i in tail_indices:
            # k近邻的平均距离
            local_density = k / (sorted_probs[i + k] - sorted_probs[i] + 1e-10)
            local_densities.append(local_density)
        
        local_densities = np.array(local_densities)
        
        # 找到密度变化最大的点
        density_diff = np.diff(np.log(local_densities + 1e-10))
        
        if len(density_diff) > 5:
            # 平滑
            window = min(11, len(density_diff) // 2)
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                density_diff_smooth = savgol_filter(density_diff, window, 2)
            else:
                density_diff_smooth = density_diff
            
            # 找最大变化点
            max_change_idx = np.argmax(np.abs(density_diff_smooth))
            threshold = sorted_probs[tail_start + max_change_idx]
        else:
            threshold = np.percentile(probs, 98)
        
        return threshold
    
    def _find_threshold_two_phase(self, probs):
        """
        方法6：两阶段阈值
        
        基于分数分布的自然分割
        """
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        
        # 估计异常比例
        estimated_ratio = self._estimate_anomaly_ratio(probs)
        
        # 基于估计比例的初步阈值
        initial_threshold = np.percentile(probs, (1 - estimated_ratio) * 100)
        
        # 在初步阈值附近精细搜索
        # 找到分数间距最大的点
        candidates_idx = np.where(probs >= initial_threshold * 0.5)[0]
        
        if len(candidates_idx) < 10:
            return initial_threshold
        
        candidate_probs = np.sort(probs[candidates_idx])
        gaps = np.diff(candidate_probs)
        
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            threshold = (candidate_probs[max_gap_idx] + candidate_probs[max_gap_idx + 1]) / 2
        else:
            threshold = initial_threshold
        
        return threshold

    def _find_threshold_train_prior(self, probs):
        """
        方法7：训练集先验引导

        利用训练集上已知的最优阈值和bot分数统计，
        在测试集分布上做自适应校准。
        优先使用分位数映射（更鲁棒），回退到分布缩放。
        """
        if self.train_prior is None:
            return None

        train_thresh = self.train_prior.get('threshold', None)
        if train_thresh is None:
            return None

        # 策略1（优先）：分位数映射
        # 训练集阈值在训练分布中的分位数位置 → 映射到测试分布
        train_percentile = self.train_prior.get('threshold_percentile', None)
        if train_percentile is not None and train_percentile > 50:
            # 在分位数附近搜索最大gap点，做微调
            base_thresh = float(np.percentile(probs, train_percentile))

            # 在 base_thresh 附近 ±2个百分点范围内找最大gap
            low_pct = max(90, train_percentile - 2)
            high_pct = min(99.9, train_percentile + 2)
            search_range = np.percentile(probs, np.linspace(low_pct, high_pct, 50))
            sorted_search = np.sort(search_range)
            gaps = np.diff(sorted_search)
            if len(gaps) > 0:
                max_gap_idx = np.argmax(gaps)
                gap_thresh = (sorted_search[max_gap_idx] + sorted_search[max_gap_idx + 1]) / 2
                # 如果gap点和base_thresh差距不大，用gap点（更精确）
                if abs(gap_thresh - base_thresh) / (base_thresh + 1e-10) < 0.5:
                    return float(gap_thresh)

            return base_thresh

        # 策略2（回退）：直接使用训练阈值
        return float(train_thresh)

    def _find_threshold_adaptive_f1(self, probs):
        """
        方法8：自适应F1优化

        在无标签情况下，通过分析分数分布的双峰性来
        寻找最大化预期F1的阈值
        """
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)

        # 估计异常比例
        estimated_ratio = self._estimate_anomaly_ratio(probs)
        estimated_n_pos = max(1, int(n * estimated_ratio))

        # 在候选阈值范围内搜索
        # 候选范围：从 top estimated_ratio*3 到 top estimated_ratio*0.3
        low_pct = max(90, (1 - estimated_ratio * 3) * 100)
        high_pct = min(99.9, (1 - estimated_ratio * 0.3) * 100)

        candidates = np.percentile(probs, np.linspace(low_pct, high_pct, 50))

        best_score = -1
        best_thresh = candidates[len(candidates) // 2]

        for thresh in candidates:
            n_pred = np.sum(probs >= thresh)
            if n_pred == 0:
                continue

            # 假设模型排序正确（AUC高），预测的top-k中大部分是真正的异常
            # 估计precision: 真正异常数 / 预测数
            # 如果预测数 <= 真实异常数，precision接近1
            # 如果预测数 > 真实异常数，precision下降
            est_tp = min(n_pred, estimated_n_pos)
            est_precision = est_tp / n_pred
            est_recall = est_tp / estimated_n_pos

            est_f1 = 2 * est_precision * est_recall / (est_precision + est_recall + 1e-10)

            if est_f1 > best_score:
                best_score = est_f1
                best_thresh = thresh

        return float(best_thresh)

    def find_threshold(self, probs, return_all=False):
        """
        主方法：综合多种方法找最优阈值
        """
        probs = np.asarray(probs).flatten()
        n = len(probs)
        
        # 基本统计
        self.debug_info['n_samples'] = n
        self.debug_info['prob_stats'] = {
            'min': float(probs.min()),
            'max': float(probs.max()),
            'mean': float(probs.mean()),
            'median': float(np.median(probs)),
            'std': float(probs.std()),
            'skewness': float(stats.skew(probs)),
            'kurtosis': float(stats.kurtosis(probs))
        }
        
        # 估计异常比例
        estimated_ratio = self._estimate_anomaly_ratio(probs)
        self.debug_info['estimated_ratio'] = float(estimated_ratio)
        
        # ==============================
        # 应用所有方法
        # ==============================
        methods = {
            'distribution_crossing': self._find_threshold_distribution_crossing,
            'percentile_gap': self._find_threshold_percentile_gap,
            'double_mad': self._find_threshold_double_mad,
            'iqr': self._find_threshold_iqr,
            'density_change': self._find_threshold_density_change,
            'two_phase': self._find_threshold_two_phase,
            'adaptive_f1': self._find_threshold_adaptive_f1,
        }

        thresholds = {}
        for name, method in methods.items():
            try:
                thresh = method(probs)
                if thresh is not None and 0 < thresh < 1:
                    thresholds[name] = thresh
            except Exception as e:
                if self.verbose:
                    print(f"[Warning] {name} failed: {e}")

        # 训练集先验方法（优先级高）
        try:
            prior_thresh = self._find_threshold_train_prior(probs)
            if prior_thresh is not None and 0 < prior_thresh < 1:
                thresholds['train_prior'] = prior_thresh
        except Exception as e:
            if self.verbose:
                print(f"[Warning] train_prior failed: {e}")
        
        self.debug_info['all_thresholds'] = {k: float(v) for k, v in thresholds.items()}
        
        if not thresholds:
            self.threshold = np.percentile(probs, 99)
            self.debug_info['fallback'] = True
            return self.threshold
        
        # ==============================
        # 计算每个阈值的预测比例
        # ==============================
        pred_ratios = {}
        for name, thresh in thresholds.items():
            pred_ratios[name] = np.mean(probs >= thresh)
        
        self.debug_info['pred_ratios'] = {k: float(v) for k, v in pred_ratios.items()}
        
        # ==============================
        # 最终决策逻辑（固定倍数分位数法）
        # ==============================
        # 经验发现：Youden最优解预测比例 ≈ bot_ratio * 5
        # 在无标签情况下，用 estimated_ratio * 5 作为预测比例
        # 这样在极端不平衡下能保证高recall

        pred_target_ratio = min(0.05, estimated_ratio * 5)
        self.threshold = float(np.percentile(probs, (1 - pred_target_ratio) * 100))
        self.debug_info['selection_method'] = f'fixed_5x(ratio={pred_target_ratio:.4f})'
        
        self.debug_info['final_threshold'] = float(self.threshold)
        self.debug_info['final_pred_ratio'] = float(np.mean(probs >= self.threshold))
        
        if return_all:
            return self.threshold, thresholds
        return self.threshold
    
    def predict(self, probs):
        """生成预测"""
        if self.threshold is None:
            self.find_threshold(probs)
        return (np.asarray(probs) >= self.threshold).astype(int)
    
    def get_debug_report(self):
        """获取详细的调试报告"""
        report = []
        report.append("=" * 60)
        report.append("智能阈值优化器 - 详细报告")
        report.append("=" * 60)
        
        if 'prob_stats' in self.debug_info:
            stats = self.debug_info['prob_stats']
            report.append("\n【分数分布统计】")
            report.append(f"  样本数: {self.debug_info.get('n_samples', 'N/A')}")
            report.append(f"  均值: {stats['mean']:.6f}")
            report.append(f"  中位数: {stats['median']:.6f}")
            report.append(f"  标准差: {stats['std']:.6f}")
            report.append(f"  偏度: {stats['skewness']:.4f}")
        
        if 'estimated_ratio' in self.debug_info:
            report.append(f"\n【估计异常比例】: {self.debug_info['estimated_ratio']:.4f}")
        
        if 'all_thresholds' in self.debug_info:
            report.append("\n【各方法阈值】")
            for name, thresh in self.debug_info['all_thresholds'].items():
                ratio = self.debug_info.get('pred_ratios', {}).get(name, 'N/A')
                report.append(f"  {name:25s}: {thresh:.6f} (预测比例: {ratio:.4f})")
        
        if 'final_threshold' in self.debug_info:
            report.append("\n【最终决策】")
            report.append(f"  选择方法: {self.debug_info.get('selection_method', 'N/A')}")
            report.append(f"  最终阈值: {self.debug_info['final_threshold']:.6f}")
            report.append(f"  预测比例: {self.debug_info.get('final_pred_ratio', 'N/A'):.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def compute_botnet_metrics_smart(y_true: np.ndarray, probs: np.ndarray, verbose: bool = False) -> dict:
    """
    使用智能阈值优化器计算指标
    """
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    
    optimizer = SmartThresholdOptimizer(verbose=verbose)
    threshold = optimizer.find_threshold(probs)
    
    preds = (probs >= threshold).astype(int)
    
    # 计算指标（仅用于评估）
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average='binary', zero_division=0
    )
    auc = roc_auc_score(y_true, probs)
    
    result = {
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'threshold': float(threshold),
        'num_predicted': int(preds.sum()),
        'num_true': int(y_true.sum()),
        'method': 'Smart_Optimizer',
        'predicted_ratio': float(preds.sum() / len(probs)),
        'debug_info': optimizer.debug_info
    }
    
    if verbose:
        print(optimizer.get_debug_report())
    
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("智能阈值优化器测试")
    print("=" * 70)
    
    try:
        data = np.load('v3_final_score_distribution.npz', allow_pickle=True)
        probs = data['probs']
        y_true = data['y_true']
        
        print(f"\n[数据概览]")
        print(f"  总节点数：{len(probs)}")
        print(f"  僵尸节点数：{int(y_true.sum())} ({y_true.sum()/len(probs)*100:.2f}%)")
        
        # 测试不同阈值的效果
        print(f"\n[阈值效果分析（使用真实标签评估）]")
        for thresh in [0.001, 0.002, 0.003, 0.005, 0.008, 0.01]:
            preds = (probs >= thresh).astype(int)
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  阈值 {thresh:.4f}: 预测 {preds.sum():5d}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        # 测试智能优化器
        print("\n[测试智能阈值优化器]...")
        result = compute_botnet_metrics_smart(y_true, probs, verbose=True)
        
        print(f"\n评估结果（智能优化器）")
        print("=" * 60)
        print(f"  AUC:       {result['auc']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1-Score:  {result['f1']:.4f}")
        print(f"  Threshold: {result['threshold']:.6f}")
        print("=" * 60)
        print(f"  预测为僵尸网络的节点：{result['num_predicted']}")
        print(f"  实际僵尸网络节点：{result['num_true']}")
        print(f"  预测比例：{result['predicted_ratio']:.4f}")
        
    except FileNotFoundError:
        print("[Error] 请先运行 main_improved_v3_final.py 生成分数数据")
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()