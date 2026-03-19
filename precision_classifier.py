"""
精确阈值优化分类器

核心洞察：
- 分数极度集中在低值区域（正常节点中位数≈0.000002）
- 僵尸节点分数略高（中位数≈0.001），但不是"异常值"
- 需要在分数分布的"稠密区域"找阈值，而不是找异常值

策略：
1. 分析分数分布的累积分布函数(CDF)
2. 使用分数间隔（gap）分析找"自然分界点"
3. 结合分数分布的导数分析
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')


class PrecisionThresholdFinder:
    """
    精确阈值查找器
    
    专门针对分数极度集中在低值区域的情况
    """
    
    def __init__(self):
        self.threshold = None
        self.debug_info = {}
    
    def _log_transform(self, x, eps=1e-10):
        """安全的对数变换"""
        return np.log(x + eps)
    
    def find_threshold(self, probs):
        """
        主方法：找到最优阈值
        
        综合多种策略，选择最合适的阈值
        """
        probs = np.asarray(probs).flatten()
        n = len(probs)
        
        # 排序
        sorted_probs = np.sort(probs)
        
        # 1. 分析分数分布的基本统计量
        q25 = np.percentile(probs, 25)
        q50 = np.percentile(probs, 50)
        q75 = np.percentile(probs, 75)
        q90 = np.percentile(probs, 90)
        q95 = np.percentile(probs, 95)
        q99 = np.percentile(probs, 99)
        
        self.debug_info['quantiles'] = {
            'q25': float(q25),
            'q50': float(q50),
            'q75': float(q75),
            'q90': float(q90),
            'q95': float(q95),
            'q99': float(q99)
        }
        
        # 2. 使用多种方法估计阈值
        
        # 方法A：分数间隔分析（Gap Analysis）
        thresh_gap = self._method_gap_analysis(probs, sorted_probs)
        
        # 方法B：密度变化分析（Density Change）
        thresh_density = self._method_density_change(probs, sorted_probs)
        
        # 方法C：双峰分布分析（Bimodal Detection）
        thresh_bimodal = self._method_bimodal_detection(probs)
        
        # 方法D：累积分布拐点（CDF Knee）
        thresh_cdf = self._method_cdf_knee(probs, sorted_probs)
        
        # 方法E：对比度最大化（Contrast Maximization）
        thresh_contrast = self._method_contrast_max(probs, sorted_probs)
        
        # 方法F：Top-K 密度跳跃
        thresh_topk = self._method_topk_density_jump(probs, sorted_probs)
        
        # 方法G：基于熵的阈值
        thresh_entropy = self._method_entropy_based(probs, sorted_probs)
        
        # 方法H：自适应百分位
        thresh_adaptive = self._method_adaptive_percentile(probs)
        
        # 方法I：基于AUC的优化阈值
        # 高AUC意味着分数有良好的排序能力
        thresh_auc_optimized = self._method_auc_optimized(probs, sorted_probs)
        
        # 方法J：分数比值的稳定点
        thresh_ratio_stable = self._method_ratio_stability(probs, sorted_probs)
        
        # 收集所有阈值
        all_thresholds = {
            'gap': thresh_gap,
            'density': thresh_density,
            'bimodal': thresh_bimodal,
            'cdf': thresh_cdf,
            'contrast': thresh_contrast,
            'topk': thresh_topk,
            'entropy': thresh_entropy,
            'adaptive': thresh_adaptive,
            'auc_opt': thresh_auc_optimized,
            'ratio_stable': thresh_ratio_stable
        }
        
        self.debug_info['all_thresholds'] = {k: float(v) for k, v in all_thresholds.items() if v is not None}
        
        # 3. 综合选择 - 新策略：基于分数分布特征
        
        # 计算基本统计量
        log_probs = self._log_transform(probs)
        log_mean = np.mean(log_probs)
        log_std = np.std(log_probs)
        
        # 分析分数分布的特征
        # 如果大部分分数集中在极低值，说明需要更低的阈值
        
        # 找到分数开始"稀疏"的位置
        # 即：从这个位置开始，分数间隔变大
        threshold_sparse = self._find_sparse_point(probs, sorted_probs)
        all_thresholds['sparse'] = threshold_sparse
        
        # 策略：选择能预测合理数量节点的最低阈值
        # 合理范围：1% - 6%（僵尸节点约占1.5%）
        
        valid_thresholds = []
        for name, thresh in all_thresholds.items():
            if thresh is not None and 0 < thresh < 1:
                pred_ratio = np.mean(probs >= thresh)
                # 扩大合理范围：0.5% - 15%
                if 0.005 <= pred_ratio <= 0.15:
                    valid_thresholds.append((name, thresh, pred_ratio))
        
        self.debug_info['valid_thresholds'] = [(n, float(t), float(r)) for n, t, r in valid_thresholds]
        
        if not valid_thresholds:
            # 回退策略：使用 q95
            self.threshold = q95
            self.debug_info['fallback'] = True
        else:
            # 新策略：优先选择预测比例在 2%-6% 的阈值
            # 这个范围更接近实际僵尸节点比例（1.5%）
            
            # 按预测比例排序
            sorted_by_ratio = sorted(valid_thresholds, key=lambda x: x[2])
            
            # 优先选择预测比例在 2%-6% 的阈值
            primary_candidates = [(n, t, r) for n, t, r in sorted_by_ratio if 0.02 <= r <= 0.06]
            
            if primary_candidates:
                # 取这些候选的中位数
                self.threshold = np.median([t for _, t, _ in primary_candidates])
            else:
                # 扩大范围：1%-8%
                secondary_candidates = [(n, t, r) for n, t, r in sorted_by_ratio if 0.01 <= r <= 0.08]
                
                if secondary_candidates:
                    # 取预测比例最小的那个（更保守）
                    self.threshold = secondary_candidates[0][1]
                else:
                    # 取所有有效阈值的中位数
                    self.threshold = np.median([t for _, t, _ in valid_thresholds])
        
        # 4. 验证和微调
        pred_ratio = np.mean(probs >= self.threshold)
        
        # 如果预测比例太高（>8%），尝试提高阈值
        if pred_ratio > 0.08:
            for pct in [96, 97, 98]:
                test_thresh = np.percentile(probs, pct)
                if np.mean(probs >= test_thresh) <= 0.06:
                    self.threshold = test_thresh
                    break
        
        # 如果预测比例太低（<1%），尝试降低阈值
        if pred_ratio < 0.01:
            for pct in [94, 93, 92, 91, 90]:
                test_thresh = np.percentile(probs, pct)
                if np.mean(probs >= test_thresh) >= 0.01:
                    self.threshold = test_thresh
                    break
        
        self.debug_info['final_threshold'] = float(self.threshold)
        self.debug_info['final_pred_ratio'] = float(np.mean(probs >= self.threshold))
        
        return self.threshold
    
    def _method_gap_analysis(self, probs, sorted_probs):
        """
        方法A：分数间隔分析
        
        找到排序分数中"间隔"最大的位置
        但要在合适的范围内搜索
        """
        n = len(sorted_probs)
        
        # 计算相邻分数的间隔
        gaps = np.diff(sorted_probs)
        
        # 在前 95% 的数据中搜索最大间隔
        # 因为僵尸节点分数通常在低值区域
        search_end = int(n * 0.98)
        
        # 找前search_end个数据中最大的间隔
        max_gap_idx = np.argmax(gaps[:search_end])
        max_gap = gaps[max_gap_idx]
        
        # 如果最大间隔太小，说明没有明显分界
        if max_gap < 1e-8:
            return np.percentile(probs, 95)
        
        # 阈值设置在间隔处
        threshold = sorted_probs[max_gap_idx]
        
        return threshold
    
    def _method_density_change(self, probs, sorted_probs):
        """
        方法B：密度变化分析
        
        分析分数密度的变化，找到密度骤降的位置
        """
        n = len(sorted_probs)
        
        # 使用核密度估计
        from scipy.stats import gaussian_kde
        
        # 采样以提高效率
        if n > 10000:
            sample_idx = np.random.choice(n, 10000, replace=False)
            sample_probs = sorted_probs[sample_idx]
            sample_probs = np.sort(sample_probs)
        else:
            sample_probs = sorted_probs
        
        # KDE
        kde = gaussian_kde(sample_probs)
        
        # 在分数范围内评估密度
        x_range = np.linspace(sample_probs.min(), sample_probs.max(), 1000)
        density = kde(x_range)
        
        # 计算密度的导数
        density_deriv = np.gradient(density, x_range)
        
        # 找密度下降最快的点（导数最小）
        # 但要在有实际数据的区域
        min_deriv_idx = np.argmin(density_deriv)
        threshold = x_range[min_deriv_idx]
        
        return threshold
    
    def _method_bimodal_detection(self, probs):
        """
        方法C：双峰分布检测
        
        检测对数分数是否呈现双峰分布
        """
        log_probs = self._log_transform(probs)
        
        # 使用KDE检测峰值
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(log_probs)
        
        x_range = np.linspace(log_probs.min(), log_probs.max(), 1000)
        density = kde(x_range)
        
        # 找峰值
        peaks, _ = find_peaks(density, height=np.max(density) * 0.05)
        
        if len(peaks) >= 2:
            # 有多个峰，找两个最高峰之间的谷
            sorted_peaks = peaks[np.argsort(density[peaks])[-2:]]
            sorted_peaks = np.sort(sorted_peaks)
            
            # 在两个峰之间找最小值
            valley_region = density[sorted_peaks[0]:sorted_peaks[1]+1]
            valley_idx = sorted_peaks[0] + np.argmin(valley_region)
            
            threshold_log = x_range[valley_idx]
            threshold = np.exp(threshold_log) - 1e-10
        else:
            # 只有一个峰，使用统计方法
            log_mean = np.mean(log_probs)
            log_std = np.std(log_probs)
            threshold = np.exp(log_mean + 2 * log_std) - 1e-10
        
        return threshold
    
    def _method_cdf_knee(self, probs, sorted_probs):
        """
        方法D：CDF 拐点检测
        
        找到 CDF 曲线的"膝点"
        """
        n = len(sorted_probs)
        
        # 计算 CDF
        cdf = np.arange(1, n + 1) / n
        
        # 归一化
        x_norm = (sorted_probs - sorted_probs[0]) / (sorted_probs[-1] - sorted_probs[0] + 1e-10)
        y_norm = cdf
        
        # 计算 CDF 曲线到对角线的距离
        # 膝点是距离最大的点
        distances = np.abs(y_norm - x_norm)
        
        # 在前90%的数据中搜索（因为僵尸节点分数较低）
        search_end = int(n * 0.9)
        knee_idx = np.argmax(distances[:search_end])
        
        threshold = sorted_probs[knee_idx]
        
        return threshold
    
    def _method_contrast_max(self, probs, sorted_probs):
        """
        方法E：对比度最大化
        
        找到使两组分数差异最大的阈值
        """
        n = len(probs)
        
        # 尝试多个候选阈值
        candidates = np.percentile(probs, np.arange(70, 99, 1))
        
        best_threshold = candidates[0]
        best_contrast = 0
        
        for thresh in candidates:
            below = probs[probs < thresh]
            above = probs[probs >= thresh]
            
            if len(below) == 0 or len(above) == 0:
                continue
            
            # 对比度 = (above.mean - below.mean) / (above.std + below.std)
            contrast = (np.mean(above) - np.mean(below)) / (np.std(above) + np.std(below) + 1e-10)
            
            # 同时考虑两组的大小平衡
            size_penalty = abs(len(above) / n - 0.02)  # 希望above组约2%
            score = contrast / (1 + size_penalty * 10)
            
            if score > best_contrast:
                best_contrast = score
                best_threshold = thresh
        
        return best_threshold
    
    def _method_topk_density_jump(self, probs, sorted_probs):
        """
        方法F：Top-K 密度跳跃
        
        分析高分数区域的密度跳跃
        """
        n = len(sorted_probs)
        
        # 分析 Top 5% 的分数
        top_5_pct = sorted_probs[int(n * 0.95):]
        
        # 计算这个区域的密度跳跃
        if len(top_5_pct) < 10:
            return np.percentile(probs, 95)
        
        # 计算相邻分数的比值
        ratios = top_5_pct[1:] / (top_5_pct[:-1] + 1e-10)
        
        # 找比值突然增大的位置
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        
        # 找超过均值+3倍标准差的跳跃
        jump_indices = np.where(ratios > mean_ratio + 3 * std_ratio)[0]
        
        if len(jump_indices) > 0:
            # 使用第一个跳跃点
            idx = jump_indices[0]
            threshold = top_5_pct[idx]
        else:
            # 没有明显跳跃，使用95%分位
            threshold = np.percentile(probs, 95)
        
        return threshold
    
    def _method_entropy_based(self, probs, sorted_probs):
        """
        方法G：基于熵的阈值选择
        
        选择使分数分布熵最大的阈值
        """
        n = len(probs)
        
        # 尝试多个候选阈值
        candidates = np.percentile(probs, np.arange(80, 99.5, 0.5))
        
        best_threshold = candidates[0]
        best_entropy = -np.inf
        
        for thresh in candidates:
            below = probs[probs < thresh]
            above = probs[probs >= thresh]
            
            if len(below) == 0 or len(above) == 0:
                continue
            
            # 计算两组的熵
            p_below = len(below) / n
            p_above = len(above) / n
            
            # 信息增益
            entropy = -p_below * np.log(p_below + 1e-10) - p_above * np.log(p_above + 1e-10)
            
            # 加权：考虑分数差异
            score_diff = np.mean(above) - np.mean(below)
            score = entropy * (1 + score_diff * 10)
            
            if score > best_entropy:
                best_entropy = score
                best_threshold = thresh
        
        return best_threshold
    
    def _method_adaptive_percentile(self, probs):
        """
        方法H：自适应百分位
        
        根据分布特征自适应选择百分位
        """
        # 分析分布的偏度
        log_probs = self._log_transform(probs)
        
        skewness = stats.skew(log_probs)
        kurtosis = stats.kurtosis(log_probs)
        
        # 如果分布极度偏斜，使用较高的百分位
        if skewness > 10:
            percentile = 99
        elif skewness > 5:
            percentile = 97
        elif skewness > 2:
            percentile = 95
        else:
            percentile = 90
        
        # 根据峰度调整
        if kurtosis > 100:
            percentile = min(percentile + 2, 99.5)
        
        threshold = np.percentile(probs, percentile)
        
        return threshold
    
    def _method_auc_optimized(self, probs, sorted_probs):
        """
        方法I：基于AUC的优化阈值
        
        高AUC意味着分数有良好的排序能力
        利用这个特性，找到能最好地区分两组的阈值
        """
        n = len(probs)
        
        # 计算对数分数
        log_probs = self._log_transform(probs)
        
        # 使用 KMeans 找到两个群体
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(log_probs.reshape(-1, 1))
        
        # 确定高低组
        centers = kmeans.cluster_centers_.flatten()
        high_label = 0 if centers[0] > centers[1] else 1
        
        # 获取高分数组的统计量
        high_probs = probs[labels == high_label]
        low_probs = probs[labels != high_label]
        
        # 使用高分数组的最小值作为阈值
        # 这确保我们捕获所有高分数的节点
        threshold = high_probs.min()
        
        return threshold
    
    def _method_ratio_stability(self, probs, sorted_probs):
        """
        方法J：分数比值稳定性分析
        
        找到分数比值开始变得不稳定的位置
        这通常意味着从"正常"区域进入"异常"区域
        """
        n = len(sorted_probs)
        
        # 计算相邻分数的比值
        ratios = sorted_probs[1:] / (sorted_probs[:-1] + 1e-10)
        
        # 计算比值的滑动平均和标准差
        window = max(100, n // 1000)
        
        # 使用 cumsum 计算滑动平均
        cumsum = np.cumsum(ratios)
        moving_avg = (cumsum[window:] - cumsum[:-window]) / window
        
        # 找到比值突然增大的位置
        # 这表示分数间隔变大，可能是分界点
        threshold_idx = None
        
        for i in range(len(moving_avg)):
            idx = i + window
            # 如果当前比值超过滑动平均的3倍
            if ratios[idx] > moving_avg[i] * 3 and ratios[idx] > 1.1:
                threshold_idx = idx
                break
        
        if threshold_idx is not None:
            threshold = sorted_probs[threshold_idx]
        else:
            # 没有找到不稳定点，使用统计方法
            threshold = np.percentile(probs, 95)
        
        return threshold
    
    def _find_sparse_point(self, probs, sorted_probs):
        """
        找到分数分布开始变得稀疏的位置
        
        稀疏点定义：从这个点开始，相邻分数的间隔开始增大
        """
        n = len(sorted_probs)
        
        # 计算相邻分数的间隔
        gaps = np.diff(sorted_probs)
        
        # 计算间隔的累积分布
        cum_gaps = np.cumsum(gaps)
        total_gap = cum_gaps[-1]
        
        # 找到累积间隔达到总间隔50%的位置
        # 这表示分数分布的"中位数"位置
        sparse_idx = np.searchsorted(cum_gaps, total_gap * 0.5)
        
        if sparse_idx < n - 1:
            threshold = sorted_probs[sparse_idx]
        else:
            threshold = np.percentile(probs, 50)
        
        return threshold


class PrecisionBotnetClassifier:
    """
    精确僵尸网络分类器
    
    结合多种策略进行精确分类
    """
    
    def __init__(self):
        self.threshold_finder = PrecisionThresholdFinder()
        self.threshold = None
    
    def fit_predict(self, probs, y_true=None):
        """
        执行分类
        
        Args:
            probs: 模型输出概率
            y_true: 真实标签（仅用于评估）
        
        Returns:
            result: 包含预测结果的字典
        """
        probs = np.asarray(probs).flatten()
        n = len(probs)
        
        # 找最优阈值
        self.threshold = self.threshold_finder.find_threshold(probs)
        debug_info = self.threshold_finder.debug_info
        
        # 生成预测
        preds = (probs >= self.threshold).astype(int)
        
        # 计算指标
        if y_true is not None:
            y_true = np.asarray(y_true).flatten()[:len(probs)]
            probs = probs[:len(y_true)]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, preds, average='binary', zero_division=0
            )
            auc = roc_auc_score(y_true, probs)
        else:
            precision = recall = f1 = auc = None
        
        return {
            'preds': preds,
            'probs': probs,
            'threshold': float(self.threshold),
            'num_predicted': int(preds.sum()),
            'predicted_ratio': float(preds.sum() / n),
            'precision': float(precision) if precision is not None else None,
            'recall': float(recall) if recall is not None else None,
            'f1': float(f1) if f1 is not None else None,
            'auc': float(auc) if auc is not None else None,
            'debug_info': debug_info
        }


def compute_botnet_metrics_precision(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """
    使用精确分类器计算指标
    """
    classifier = PrecisionBotnetClassifier()
    result = classifier.fit_predict(probs, y_true)
    
    return {
        'auc': result['auc'],
        'precision': result['precision'],
        'recall': result['recall'],
        'f1': result['f1'],
        'threshold': result['threshold'],
        'num_predicted': result['num_predicted'],
        'num_true': int(y_true.sum()),
        'method': 'Precision',
        'ratio': result['predicted_ratio'],
        'debug_info': result['debug_info']
    }


if __name__ == "__main__":
    # 测试
    print("="*70)
    print("精确分类器测试")
    print("="*70)
    
    # 加载保存的分数数据
    import numpy as np
    
    try:
        data = np.load('s12_scores.npz', allow_pickle=True)
        probs = data['probs']
        y_true = data['y_true']
        
        print(f"\n[数据概览]")
        print(f"  总节点数：{len(probs)}")
        print(f"  僵尸节点数：{y_true.sum()} ({y_true.sum()/len(probs)*100:.2f}%)")
        
        # 测试精确分类器
        print("\n[测试精确分类器]...")
        result = compute_botnet_metrics_precision(y_true, probs)
        
        print(f"\n评估结果（精确分类器）")
        print("="*60)
        print(f"  AUC:       {result['auc']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1-Score:  {result['f1']:.4f}")
        print(f"  Threshold: {result['threshold']:.6f}")
        print("="*60)
        print(f"  预测为僵尸网络的节点：{result['num_predicted']}")
        print(f"  实际僵尸网络节点：{result['num_true']}")
        
        # 打印调试信息
        print(f"\n[调试信息]")
        print(f"  最终阈值：{result['debug_info'].get('final_threshold', 'N/A')}")
        print(f"  最终预测比例：{result['debug_info'].get('final_pred_ratio', 'N/A'):.4f}")
        print(f"  有效阈值方法：{result['debug_info'].get('valid_thresholds', [])[:5]}")
        
    except FileNotFoundError:
        print("[Error] 请先运行 analyze_s12.py 生成分数数据")