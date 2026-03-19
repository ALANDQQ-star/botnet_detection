"""
改进的阈值优化器

核心改进：
1. 利用AUC信息指导阈值选择
2. 使用PR曲线而非ROC曲线
3. 更准确的异常比例估计
4. 多策略融合
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.metrics import precision_recall_curve, roc_curve
import warnings
warnings.filterwarnings('ignore')


class ImprovedThresholdOptimizer:
    """
    改进的阈值优化器
    
    核心思想：
    1. 利用高AUC特性：如果AUC>0.9，说明模型有良好的排序能力
    2. 使用PR曲线找最优阈值
    3. 多策略融合
    """
    
    def __init__(self, verbose=False):
        self.threshold = None
        self.verbose = verbose
        self.debug_info = {}
    
    def _estimate_anomaly_ratio_v2(self, probs):
        """
        改进的异常比例估计
        
        使用分数分布的统计特性
        """
        log_probs = np.log10(probs + 1e-10)
        
        # 方法1: 基于尾部分布
        p99 = np.percentile(probs, 99)
        p99_5 = np.percentile(probs, 99.5)
        p99_9 = np.percentile(probs, 99.9)
        p_max = probs.max()
        
        # 尾部陡峭度分析
        tail_steepness = (p99_9 - p99_5) / (p99_5 - p99 + 1e-10)
        
        if tail_steepness > 10:
            # 尾部非常陡，异常很少
            ratio_tail = 0.001
        elif tail_steepness > 5:
            ratio_tail = 0.002
        elif tail_steepness > 2:
            ratio_tail = 0.005
        else:
            ratio_tail = 0.01
        
        # 方法2: 基于分布偏度
        skewness = stats.skew(log_probs)
        kurtosis = stats.kurtosis(log_probs)
        
        # 高峰度表示有重尾（异常）
        if kurtosis > 100:
            ratio_kurt = 0.005
        elif kurtosis > 50:
            ratio_kurt = 0.01
        elif kurtosis > 20:
            ratio_kurt = 0.02
        else:
            ratio_kurt = 0.03
        
        # 方法3: 基于分数跳跃
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        
        # 在高尾区域找最大跳跃
        tail_start = int(n * 0.95)
        tail_probs = sorted_probs[tail_start:]
        gaps = np.diff(tail_probs)
        
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            max_gap_ratio = (n - tail_start - max_gap_idx) / n
            ratio_gap = max_gap_ratio
        else:
            ratio_gap = 0.01
        
        # 综合估计
        estimated_ratio = (ratio_tail + ratio_kurt + ratio_gap) / 3
        estimated_ratio = max(0.0005, min(0.05, estimated_ratio))
        
        return estimated_ratio
    
    def _find_threshold_pr_curve(self, probs, target_recall=0.7):
        """
        基于PR曲线找阈值
        
        对于高AUC模型，使用PR曲线更合适
        """
        # 创建伪标签进行PR分析
        # 假设分数最高的部分是正样本
        n = len(probs)
        estimated_ratio = self._estimate_anomaly_ratio_v2(probs)
        n_anomaly = max(10, int(n * estimated_ratio * 2))  # 宽松估计
        
        # 使用分数最高的点作为伪正样本
        sorted_indices = np.argsort(probs)[::-1]
        pseudo_labels = np.zeros(n)
        pseudo_labels[sorted_indices[:n_anomaly]] = 1
        
        try:
            precision, recall, thresholds = precision_recall_curve(pseudo_labels, probs)
            
            # 找到满足目标召回率的最高precision阈值
            valid_indices = np.where(recall >= target_recall)[0]
            if len(valid_indices) > 0:
                best_idx = valid_indices[np.argmax(precision[valid_indices])]
                threshold = thresholds[min(best_idx, len(thresholds)-1)]
            else:
                # 如果没有满足的，使用recall最接近的
                best_idx = np.argmin(np.abs(recall - target_recall))
                threshold = thresholds[min(best_idx, len(thresholds)-1)]
            
            return threshold
        except:
            return np.percentile(probs, 99)
    
    def _find_threshold_top_k_gap(self, probs):
        """
        基于Top-K分数跳跃找阈值
        
        找到分数分布中最大跳跃的位置
        """
        sorted_probs = np.sort(probs)[::-1]  # 降序
        n = len(sorted_probs)
        
        # 只考虑前5%的分数
        k_top = max(50, int(n * 0.05))
        top_probs = sorted_probs[:k_top]
        
        # 计算相对跳跃
        gaps = np.diff(top_probs)
        relative_gaps = gaps / (top_probs[1:] + 1e-10)
        
        # 找最大相对跳跃
        if len(relative_gaps) > 0:
            max_gap_idx = np.argmax(relative_gaps)
            threshold = (top_probs[max_gap_idx] + top_probs[max_gap_idx + 1]) / 2
        else:
            threshold = sorted_probs[k_top - 1]
        
        return threshold
    
    def _find_threshold_density_valley(self, probs):
        """
        基于密度谷底找阈值
        
        使用核密度估计找双峰分布的谷底
        """
        from scipy.stats import gaussian_kde
        
        try:
            # 在对数空间进行KDE
            log_probs = np.log10(probs + 1e-10)
            
            # 均匀采样以提高速度
            if len(log_probs) > 10000:
                sample_idx = np.random.choice(len(log_probs), 10000, replace=False)
                log_probs_sample = log_probs[sample_idx]
            else:
                log_probs_sample = log_probs
            
            kde = gaussian_kde(log_probs_sample)
            
            # 评估密度
            x_range = np.linspace(log_probs.min(), log_probs.max(), 200)
            density = kde(x_range)
            
            # 找局部最小值（谷底）
            # 使用简单的差分方法
            diff = np.diff(density)
            sign_change = np.where(np.diff(np.sign(diff)) < 0)[0]
            
            if len(sign_change) > 0:
                # 找最右侧的谷底（在高分区域）
                valley_idx = sign_change[-1]
                threshold = 10 ** x_range[valley_idx]
            else:
                threshold = np.percentile(probs, 99)
            
            return threshold
        except:
            return np.percentile(probs, 99)
    
    def _find_threshold_gradient(self, probs):
        """
        基于梯度分析方法
        
        找分数梯度的拐点
        """
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        
        # 计算累积分布
        cdf = np.arange(1, n + 1) / n
        
        # 在高尾区域分析
        tail_start = int(n * 0.90)
        tail_probs = sorted_probs[tail_start:]
        tail_cdf = cdf[tail_start:]
        
        # 计算梯度
        gradients = np.diff(tail_cdf) / (np.diff(tail_probs) + 1e-10)
        
        # 找梯度变化最大的点
        gradient_changes = np.abs(np.diff(gradients))
        
        if len(gradient_changes) > 5:
            # 平滑
            window = min(11, len(gradient_changes) // 2)
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                smoothed = savgol_filter(gradient_changes, window, 2)
            else:
                smoothed = gradient_changes
            
            max_change_idx = np.argmax(smoothed)
            threshold = tail_probs[max_change_idx]
        else:
            threshold = np.percentile(probs, 98)
        
        return threshold
    
    def _find_threshold_percentile_adaptive(self, probs):
        """
        自适应百分位阈值
        
        根据分布特性自适应选择百分位
        """
        # 分析分布
        mean_val = np.mean(probs)
        median_val = np.median(probs)
        std_val = np.std(probs)
        
        # 计算偏度
        skewness = (mean_val - median_val) / (std_val + 1e-10)
        
        # 根据偏度选择百分位
        # 偏度越大，说明异常值越集中在尾部，需要更严格的阈值
        if skewness > 10:
            percentile = 99.9
        elif skewness > 5:
            percentile = 99.5
        elif skewness > 2:
            percentile = 99
        elif skewness > 1:
            percentile = 98
        else:
            percentile = 97
        
        return np.percentile(probs, percentile)
    
    def _find_threshold_two_stage(self, probs):
        """
        两阶段阈值选择
        
        第一阶段：粗略估计异常比例
        第二阶段：精细调整阈值
        """
        n = len(probs)
        
        # 阶段1: 使用IQR方法粗略筛选
        log_probs = np.log10(probs + 1e-10)
        q1, q3 = np.percentile(log_probs, [25, 75])
        iqr = q3 - q1
        
        # 使用3.5倍IQR作为上限
        upper_bound = q3 + 3.5 * iqr
        candidate_mask = log_probs > upper_bound
        
        if candidate_mask.sum() < 10:
            # 如果候选太少，放宽条件
            upper_bound = q3 + 2.5 * iqr
            candidate_mask = log_probs > upper_bound
        
        if candidate_mask.sum() < 5:
            return np.percentile(probs, 99)
        
        # 阶段2: 在候选区域精细找阈值
        candidate_probs = probs[candidate_mask]
        sorted_candidates = np.sort(candidate_probs)
        
        # 找最大跳跃
        gaps = np.diff(sorted_candidates)
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            threshold = (sorted_candidates[max_gap_idx] + sorted_candidates[max_gap_idx + 1]) / 2
        else:
            threshold = sorted_candidates[0]
        
        return threshold
    
    def _find_threshold_auc_guided(self, probs):
        """
        基于AUC指导的阈值选择
        
        假设模型有良好的排序能力，利用分数分布特性
        """
        n = len(probs)
        
        # 估计异常比例范围
        estimated_ratio = self._estimate_anomaly_ratio_v2(probs)
        
        # 计算候选阈值范围
        p_low = np.percentile(probs, (1 - estimated_ratio * 3) * 100)
        p_high = np.percentile(probs, (1 - estimated_ratio * 0.5) * 100)
        
        # 在候选范围内精细搜索
        n_candidates = 100
        candidate_thresholds = np.linspace(p_low, p_high, n_candidates)
        
        best_threshold = candidate_thresholds[0]
        best_score = -np.inf
        
        for thresh in candidate_thresholds:
            preds = (probs >= thresh).astype(int)
            pred_ratio = preds.sum() / n
            
            # 评分：接近估计比例，且不太极端
            if pred_ratio > 0:
                # 希望预测比例接近估计比例
                ratio_score = -np.abs(pred_ratio - estimated_ratio) / estimated_ratio
                
                # 希望阈值不太极端
                extreme_penalty = -0.1 * (pred_ratio < 0.0001 or pred_ratio > 0.1)
                
                score = ratio_score + extreme_penalty
                
                if score > best_score:
                    best_score = score
                    best_threshold = thresh
        
        return best_threshold
    
    def find_threshold(self, probs, return_all=False):
        """
        主方法：综合多种方法找最优阈值
        """
        probs = np.asarray(probs).flatten()
        n = len(probs)
        
        self.debug_info['n_samples'] = n
        self.debug_info['prob_stats'] = {
            'min': float(probs.min()),
            'max': float(probs.max()),
            'mean': float(probs.mean()),
            'median': float(np.median(probs)),
            'std': float(probs.std()),
        }
        
        # 估计异常比例
        estimated_ratio = self._estimate_anomaly_ratio_v2(probs)
        self.debug_info['estimated_ratio'] = float(estimated_ratio)
        
        # 应用多种方法
        methods = {
            'pr_curve': lambda: self._find_threshold_pr_curve(probs),
            'top_k_gap': lambda: self._find_threshold_top_k_gap(probs),
            'density_valley': lambda: self._find_threshold_density_valley(probs),
            'gradient': lambda: self._find_threshold_gradient(probs),
            'percentile_adaptive': lambda: self._find_threshold_percentile_adaptive(probs),
            'two_stage': lambda: self._find_threshold_two_stage(probs),
            'auc_guided': lambda: self._find_threshold_auc_guided(probs),
        }
        
        thresholds = {}
        for name, method in methods.items():
            try:
                thresh = method()
                if thresh is not None and 0 < thresh < 1:
                    thresholds[name] = thresh
            except Exception as e:
                if self.verbose:
                    print(f"[Warning] {name} failed: {e}")
        
        self.debug_info['all_thresholds'] = {k: float(v) for k, v in thresholds.items()}
        
        if not thresholds:
            self.threshold = np.percentile(probs, 99)
            self.debug_info['fallback'] = True
            return self.threshold
        
        # 计算每个阈值的预测比例
        pred_ratios = {}
        for name, thresh in thresholds.items():
            pred_ratios[name] = float(np.mean(probs >= thresh))
        self.debug_info['pred_ratios'] = pred_ratios
        
        # 最终决策逻辑
        # 1. 基于估计比例筛选候选
        target_low = max(0.0001, estimated_ratio * 0.3)
        target_high = min(0.15, estimated_ratio * 5)
        
        valid_thresholds = {
            name: thresh for name, thresh in thresholds.items()
            if target_low <= pred_ratios[name] <= target_high
        }
        
        if valid_thresholds:
            # 选择预测比例最接近估计比例的
            best_name = min(valid_thresholds, 
                           key=lambda x: abs(pred_ratios[x] - estimated_ratio))
            self.threshold = valid_thresholds[best_name]
            self.debug_info['selection_method'] = f'valid_candidate_{best_name}'
        else:
            # 如果没有有效候选，使用中位数
            all_values = list(thresholds.values())
            self.threshold = np.median(all_values)
            self.debug_info['selection_method'] = 'median'
        
        # 后处理调整
        final_pred_ratio = np.mean(probs >= self.threshold)
        
        # 如果预测比例过高，收紧阈值
        if final_pred_ratio > 0.1:
            self.threshold = np.percentile(probs, 99.5)
            self.debug_info['adjustment'] = 'ratio_too_high'
        elif final_pred_ratio < 0.0001:
            self.threshold = np.percentile(probs, 95)
            self.debug_info['adjustment'] = 'ratio_too_low'
        
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
        """获取详细调试报告"""
        report = []
        report.append("=" * 60)
        report.append("改进阈值优化器 - 详细报告")
        report.append("=" * 60)
        
        if 'prob_stats' in self.debug_info:
            stats = self.debug_info['prob_stats']
            report.append("\n【分数分布统计】")
            report.append(f"  样本数: {self.debug_info.get('n_samples', 'N/A')}")
            report.append(f"  均值: {stats['mean']:.6f}")
            report.append(f"  中位数: {stats['median']:.6f}")
            report.append(f"  标准差: {stats['std']:.6f}")
        
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
        
        return "\n".join(report)


class AdaptiveEnsembleOptimizer:
    """
    自适应集成阈值优化器
    
    根据分数分布特性自动选择最佳策略
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.threshold = None
        self.debug_info = {}
    
    def _analyze_distribution(self, probs):
        """分析分数分布特性"""
        from scipy.stats import skew, kurtosis
        
        stats_info = {
            'mean': float(np.mean(probs)),
            'median': float(np.median(probs)),
            'std': float(np.std(probs)),
            'min': float(np.min(probs)),
            'max': float(np.max(probs)),
            'skewness': float(skew(probs)),
            'kurtosis': float(kurtosis(probs)),
            'p99': float(np.percentile(probs, 99)),
            'p99_9': float(np.percentile(probs, 99.9)),
            'range': float(np.max(probs) - np.min(probs)),
        }
        
        # 计算尾部特性
        p99 = stats_info['p99']
        p99_9 = stats_info['p99_9']
        stats_info['tail_steepness'] = (p99_9 - p99) / (stats_info['range'] + 1e-10)
        
        return stats_info
    
    def _select_strategy(self, stats_info):
        """根据分布特性选择策略"""
        skewness = stats_info['skewness']
        kurtosis = stats_info['kurtosis']
        tail_steepness = stats_info['tail_steepness']
        
        # 策略选择逻辑
        if tail_steepness > 0.5:
            # 尾部陡峭，使用严格的百分位
            return 'strict_percentile'
        elif kurtosis > 50:
            # 高峰度，有明显的异常聚集
            return 'density_valley'
        elif skewness > 10:
            # 高度右偏
            return 'top_k_gap'
        else:
            # 默认使用两阶段方法
            return 'two_stage'
    
    def find_threshold(self, probs, return_all=False):
        """主方法"""
        probs = np.asarray(probs).flatten()
        
        # 分析分布
        stats_info = self._analyze_distribution(probs)
        self.debug_info['distribution'] = stats_info
        
        # 选择策略
        strategy = self._select_strategy(stats_info)
        self.debug_info['strategy'] = strategy
        
        # 应用选定策略
        optimizer = ImprovedThresholdOptimizer(verbose=self.verbose)
        
        if strategy == 'strict_percentile':
            # 使用严格的百分位
            self.threshold = np.percentile(probs, 99.5)
        elif strategy == 'density_valley':
            self.threshold = optimizer._find_threshold_density_valley(probs)
        elif strategy == 'top_k_gap':
            self.threshold = optimizer._find_threshold_top_k_gap(probs)
        else:
            self.threshold = optimizer._find_threshold_two_stage(probs)
        
        # 验证和调整
        pred_ratio = np.mean(probs >= self.threshold)
        self.debug_info['pred_ratio'] = float(pred_ratio)
        
        # 如果预测比例异常，进行调整
        if pred_ratio > 0.1:
            self.threshold = np.percentile(probs, 99.5)
            self.debug_info['adjustment'] = 'too_high'
        elif pred_ratio < 0.0001:
            self.threshold = np.percentile(probs, 98)
            self.debug_info['adjustment'] = 'too_low'
        
        self.debug_info['final_threshold'] = float(self.threshold)
        
        return self.threshold
    
    def predict(self, probs):
        if self.threshold is None:
            self.find_threshold(probs)
        return (np.asarray(probs) >= self.threshold).astype(int)


if __name__ == "__main__":
    print("=" * 70)
    print("改进阈值优化器测试")
    print("=" * 70)
    
    # 测试数据
    np.random.seed(42)
    n_normal = 100000
    n_anomaly = 1000
    
    # 模拟分数分布
    normal_scores = np.random.exponential(0.0001, n_normal)
    anomaly_scores = np.random.exponential(0.01, n_anomaly) + 0.001
    
    probs = np.concatenate([normal_scores, anomaly_scores])
    y_true = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # 打乱
    indices = np.random.permutation(len(probs))
    probs = probs[indices]
    y_true = y_true[indices]
    
    print(f"\n[数据概览]")
    print(f"  总节点数：{len(probs)}")
    print(f"  僵尸节点数：{int(y_true.sum())} ({y_true.sum()/len(probs)*100:.2f}%)")
    print(f"  正常节点分数均值：{probs[y_true==0].mean():.6f}")
    print(f"  僵尸节点分数均值：{probs[y_true==1].mean():.6f}")
    
    # 测试改进的阈值优化器
    print("\n[测试改进阈值优化器]...")
    optimizer = ImprovedThresholdOptimizer(verbose=True)
    threshold = optimizer.find_threshold(probs)
    
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average='binary', zero_division=0
    )
    auc = roc_auc_score(y_true, probs)
    
    print(f"\n评估结果")
    print("=" * 60)
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  预测比例:  {preds.mean():.4f}")