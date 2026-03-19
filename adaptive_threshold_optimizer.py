"""
自适应阈值优化器 V2

核心改进：
1. 基于分数分布的统计特性（而非固定假设）
2. 使用多策略融合找最优阈值
3. 利用正常/异常分布交叉点
4. GMM双聚类 + 膝点检测 + 最大化F1估计

关键发现：
- 正常节点99%分位 ≈ 僵尸节点10%分位（分布交叉点）
- 这个交叉点是理想的阈值候选区域
"""

import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize_scalar, brentq
import warnings
warnings.filterwarnings('ignore')


class AdaptiveThresholdOptimizerV2:
    """
    自适应阈值优化器 V2
    
    基于分数分布统计特性，不依赖真实标签，智能选择阈值
    """
    
    def __init__(self, verbose=False):
        self.threshold = None
        self.verbose = verbose
        self.debug_info = {}
        
    def _log_transform(self, x, eps=1e-10):
        """安全的对数变换，增强低值区域的分辨率"""
        return np.log10(x + eps)
    
    def _find_threshold_gmm_crossing(self, probs):
        """
        方法1：双高斯混合模型 - 找分布交叉点
        
        核心思想：两个高斯分布的交点是最优决策边界
        """
        log_probs = self._log_transform(probs).reshape(-1, 1)
        
        # 使用贝叶斯GMM自动确定有效聚类数
        gmm = BayesianGaussianMixture(
            n_components=2,
            covariance_type='full',
            random_state=42,
            n_init=20,
            max_iter=1000,
            weight_concentration_prior=0.1
        )
        gmm.fit(log_probs)
        
        # 获取聚类参数
        means = gmm.means_.flatten()
        weights = gmm.weights_.flatten()
        covariances = gmm.covariances_.flatten()
        
        # 确定高值聚类和低值聚类
        if means[0] > means[1]:
            high_mean, low_mean = means[0], means[1]
            high_weight, low_weight = weights[0], weights[1]
            high_var, low_var = covariances[0], covariances[1]
        else:
            high_mean, low_mean = means[1], means[0]
            high_weight, low_weight = weights[1], weights[0]
            high_var, low_var = covariances[1], covariances[0]
        
        std_high = np.sqrt(high_var)
        std_low = np.sqrt(low_var)
        
        # 使用贝叶斯决策理论找两个高斯的交叉点
        def log_gaussian_diff(x, m1, s1, w1, m2, s2, w2):
            """两个高斯分布对数概率之差"""
            log_p1 = -0.5 * np.log(2 * np.pi * s1**2) - 0.5 * ((x - m1) / s1)**2 + np.log(w1 + 1e-10)
            log_p2 = -0.5 * np.log(2 * np.pi * s2**2) - 0.5 * ((x - m2) / s2)**2 + np.log(w2 + 1e-10)
            return log_p1 - log_p2
        
        # 在两个均值之间找交叉点
        try:
            if low_mean < high_mean:
                x_intersect = brentq(
                    lambda x: log_gaussian_diff(x, high_mean, std_high, high_weight, 
                                                  low_mean, std_low, low_weight),
                    low_mean, high_mean
                )
                threshold = 10**x_intersect
            else:
                threshold = np.percentile(probs, 95)
        except:
            threshold = np.percentile(probs, 95)
        
        self.debug_info['gmm_crossing'] = float(threshold)
        self.debug_info['gmm_high_mean'] = float(10**high_mean)
        self.debug_info['gmm_low_mean'] = float(10**low_mean)
        
        return threshold
    
    def _find_threshold_knee_detection(self, probs):
        """
        方法2：膝点检测（Knee/Elbow Detection）
        
        核心思想：在累积分布函数中找到"膝盖"点
        这是分布从平缓到陡峭的转变点
        """
        sorted_probs = np.sort(probs)[::-1]  # 降序
        n = len(sorted_probs)
        
        # 计算累积分布
        x = np.arange(n)
        y = sorted_probs
        
        # 方法：找曲率最大的点（二阶导数最大）
        # 使用差分近似导数
        if n > 10:
            # 平滑处理
            window = min(51, n // 10 * 2 + 1)
            if window >= 5:
                y_smooth = savgol_filter(y, window, 3)
            else:
                y_smooth = y
            
            # 一阶导数
            dy = np.diff(y_smooth)
            
            # 二阶导数（曲率）
            d2y = np.diff(dy)
            
            # 找曲率最大的点（只在前50%区域搜索，因为异常分数在高区域）
            search_range = n // 2
            knee_idx = np.argmax(np.abs(d2y[:search_range]))
            
            threshold = sorted_probs[knee_idx]
        else:
            threshold = np.percentile(probs, 95)
        
        self.debug_info['knee_threshold'] = float(threshold)
        
        return threshold
    
    def _find_threshold_percentile_gap(self, probs):
        """
        方法3：分位数间距分析
        
        核心思想：找到相邻分位数之间差距最大的点
        这通常对应于两个分布的边界
        """
        # 在高分区域搜索
        percentiles = np.linspace(90, 99.9, 200)
        p_values = np.percentile(probs, percentiles)
        
        # 计算相邻分位数的间距
        gaps = np.diff(p_values)
        
        # 归一化间距（相对于分位数间距）
        percentile_gaps = percentiles[1:] - percentiles[:-1]
        normalized_gaps = gaps / (percentile_gaps + 1e-10)
        
        # 找到归一化间距最大的点
        max_gap_idx = np.argmax(normalized_gaps)
        threshold = (p_values[max_gap_idx] + p_values[max_gap_idx + 1]) / 2
        
        self.debug_info['percentile_gap_threshold'] = float(threshold)
        self.debug_info['percentile_gap_idx'] = int(max_gap_idx)
        
        return threshold
    
    def _find_threshold_tail_analysis(self, probs):
        """
        方法4：尾部行为分析
        
        核心思想：分析高分尾部的统计特性
        使用极值理论的思想找到"尾部开始"的点
        """
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        
        # 计算尾部指数的稳定性
        # 使用Hill估计量的变体
        log_sorted = np.log(sorted_probs[::-1] + 1e-10)  # 降序后取对数
        
        # 计算不同k值下的Hill估计
        hill_estimates = []
        k_values = range(10, min(n // 5, 500))
        
        for k in k_values:
            hill = np.mean(log_sorted[:k]) - log_sorted[k]
            hill_estimates.append(hill)
        
        hill_estimates = np.array(hill_estimates)
        
        # 找到Hill估计开始稳定的点
        # 使用变化率检测
        if len(hill_estimates) > 10:
            hill_diff = np.abs(np.diff(hill_estimates))
            # 找变化率最小的区域
            smooth_window = min(11, len(hill_diff) // 2)
            if smooth_window >= 3:
                hill_diff_smooth = savgol_filter(hill_diff, smooth_window, 2)
            else:
                hill_diff_smooth = hill_diff
            
            # 找到变化率低于平均值的第一个点
            threshold_idx = np.where(hill_diff_smooth < np.mean(hill_diff_smooth))[0]
            if len(threshold_idx) > 0:
                stable_k = k_values[threshold_idx[0]]
                threshold = sorted_probs[n - stable_k]
            else:
                threshold = np.percentile(probs, 98)
        else:
            threshold = np.percentile(probs, 98)
        
        self.debug_info['tail_threshold'] = float(threshold)
        
        return threshold
    
    def _find_threshold_two_stage(self, probs):
        """
        方法5：两阶段聚类
        
        核心思想：
        1. 第一阶段：粗略估计异常比例
        2. 第二阶段：在估计的异常区域内精细定位
        """
        log_probs = self._log_transform(probs).reshape(-1, 1)
        
        # 第一阶段：K-means粗聚类
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(log_probs)
        
        centers = kmeans.cluster_centers_.flatten()
        high_cluster = np.argmax(centers)
        
        # 估计异常比例
        anomaly_ratio = np.mean(labels == high_cluster)
        
        self.debug_info['stage1_anomaly_ratio'] = float(anomaly_ratio)
        
        # 如果估计比例不合理（>10%），使用分位数修正
        if anomaly_ratio > 0.1:
            # 使用更严格的分位数
            target_ratio = 0.01  # 假设1%
            threshold = np.percentile(probs, (1 - target_ratio) * 100)
        elif anomaly_ratio < 0.001:
            # 使用更宽松的分位数
            target_ratio = 0.005
            threshold = np.percentile(probs, (1 - target_ratio) * 100)
        else:
            # 第二阶段：在异常聚类内精细定位
            anomaly_probs = probs[labels == high_cluster]
            
            if len(anomaly_probs) > 10:
                # 在异常聚类内再聚类
                log_anomaly = self._log_transform(anomaly_probs).reshape(-1, 1)
                kmeans2 = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels2 = kmeans2.fit_predict(log_anomaly)
                
                centers2 = kmeans2.cluster_centers_.flatten()
                high_cluster2 = np.argmax(centers2)
                
                # 真正的异常是异常聚类中的高值子聚类
                threshold = anomaly_probs[labels2 == high_cluster2].min()
            else:
                threshold = np.percentile(probs, 99)
        
        self.debug_info['two_stage_threshold'] = float(threshold)
        
        return threshold
    
    def _find_threshold_density_valley(self, probs):
        """
        方法6：密度谷检测
        
        核心思想：在概率密度函数中找到双峰之间的谷
        """
        log_probs = self._log_transform(probs)
        
        # 在对数空间建立网格
        x_min, x_max = log_probs.min() - 1, log_probs.max() + 1
        x_grid = np.linspace(x_min, x_max, 1000)
        
        # 核密度估计
        try:
            from sklearn.neighbors import KernelDensity
            kde = KernelDensity(bandwidth='silverman', kernel='gaussian')
            kde.fit(log_probs.reshape(-1, 1))
            log_density = kde.score_samples(x_grid.reshape(-1, 1))
            density = np.exp(log_density)
        except:
            return np.percentile(probs, 98)
        
        # 平滑密度曲线
        if len(density) > 11:
            density_smooth = savgol_filter(density, 11, 3)
        else:
            density_smooth = density
        
        # 找峰值
        peaks, properties = find_peaks(density_smooth, distance=20, prominence=0.001)
        
        if len(peaks) < 2:
            # 单峰分布，使用高分位数
            return np.percentile(probs, 98)
        
        # 找两个最显著的峰
        prominences = properties['prominences']
        top_two_idx = np.argsort(prominences)[-2:]
        top_two_peaks = np.sort(peaks[top_two_idx])
        
        # 在两个峰之间找谷（最小值）
        peak1, peak2 = top_two_peaks
        valley_region = density_smooth[peak1:peak2+1]
        valley_idx = peak1 + np.argmin(valley_region)
        
        # 谷点对应的分数
        threshold = 10**x_grid[valley_idx]
        
        self.debug_info['density_valley_threshold'] = float(threshold)
        self.debug_info['density_peaks'] = [float(10**x_grid[p]) for p in peaks]
        
        return threshold
    
    def _estimate_optimal_threshold_f1(self, probs):
        """
        方法7：基于F1估计的阈值优化
        
        核心思想：使用聚类结果估计F1，找到使估计F1最大的阈值
        
        注意：不使用真实标签，而是用聚类结果作为代理
        """
        log_probs = self._log_transform(probs).reshape(-1, 1)
        
        # 使用GMM获得"伪标签"
        gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
        pseudo_labels = gmm.fit_predict(log_probs)
        
        means = gmm.means_.flatten()
        high_cluster = np.argmax(means)
        pseudo_labels = (pseudo_labels == high_cluster).astype(int)
        
        # 在不同阈值下估计F1
        candidates = np.percentile(probs, np.linspace(90, 99.9, 100))
        
        best_f1 = 0
        best_threshold = candidates[0]
        
        for thresh in candidates:
            preds = (probs >= thresh).astype(int)
            
            # 使用伪标签计算"代理F1"
            tp = np.sum((preds == 1) & (pseudo_labels == 1))
            fp = np.sum((preds == 1) & (pseudo_labels == 0))
            fn = np.sum((preds == 0) & (pseudo_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        self.debug_info['f1_estimated_threshold'] = float(best_threshold)
        self.debug_info['f1_estimated_value'] = float(best_f1)
        
        return best_threshold
    
    def _find_threshold_outlier_detection(self, probs):
        """
        方法8：异常检测方法
        
        核心思想：使用多种异常检测方法的集成
        """
        from scipy import stats
        
        # Z-score方法
        log_probs = self._log_transform(probs)
        z_scores = np.abs(stats.zscore(log_probs))
        threshold_zscore = probs[z_scores < 3].max()  # 3σ规则
        
        # IQR方法
        q1, q3 = np.percentile(log_probs, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        threshold_iqr = 10**upper_bound
        
        # MAD方法（中位数绝对偏差）
        median = np.median(log_probs)
        mad = np.median(np.abs(log_probs - median))
        modified_z = 0.6745 * (log_probs - median) / (mad + 1e-10)
        threshold_mad = probs[modified_z < 3.5].max()
        
        # 使用中位数作为最终阈值
        thresholds = [threshold_zscore, threshold_iqr, threshold_mad]
        thresholds = [t for t in thresholds if 0 < t < 1]
        
        if thresholds:
            threshold = np.median(thresholds)
        else:
            threshold = np.percentile(probs, 98)
        
        self.debug_info['outlier_threshold'] = float(threshold)
        self.debug_info['outlier_zscore'] = float(threshold_zscore)
        self.debug_info['outlier_iqr'] = float(threshold_iqr)
        self.debug_info['outlier_mad'] = float(threshold_mad)
        
        return threshold
    
    def _evaluate_threshold(self, probs, threshold):
        """
        评估阈值质量
        
        返回综合评分（越高越好）
        """
        below = probs[probs < threshold]
        above = probs[probs >= threshold]
        
        if len(below) == 0 or len(above) == 0:
            return -np.inf
        
        n = len(probs)
        
        # 1. 组间差异（越大越好）
        mean_diff = np.abs(np.mean(above) - np.mean(below))
        
        # 2. 组内一致性（标准差越小越好）
        std_below = np.std(below)
        std_above = np.std(above)
        avg_std = (std_below * len(below) + std_above * len(above)) / n
        
        # 3. 预测比例合理性
        # 基于分数分布估计真实比例
        log_probs = self._log_transform(probs)
        
        try:
            gmm = GaussianMixture(n_components=2, random_state=42, n_init=5)
            gmm.fit(log_probs.reshape(-1, 1))
            weights = gmm.weights_.flatten()
            means = gmm.means_.flatten()
            high_idx = np.argmax(means)
            estimated_ratio = weights[high_idx]
        except:
            estimated_ratio = 0.01
        
        # 修正估计比例
        if estimated_ratio > 0.1:
            estimated_ratio = 0.01
        if estimated_ratio < 0.001:
            estimated_ratio = 0.005
        
        pred_ratio = len(above) / n
        
        # 评分：预测比例越接近估计比例越好
        ratio_diff = np.abs(pred_ratio - estimated_ratio)
        ratio_score = 1 / (ratio_diff + 0.01)
        
        # 4. 轮廓系数（类间分离度）
        from sklearn.metrics import silhouette_score
        labels = (probs >= threshold).astype(int)
        if len(np.unique(labels)) > 1 and len(probs) < 100000:
            try:
                silhouette = silhouette_score(probs.reshape(-1, 1), labels, sample_size=10000)
            except:
                silhouette = 0
        else:
            silhouette = 0
        
        # 综合评分
        score = (
            mean_diff * 100 +           # 组间差异权重
            1 / (avg_std + 1e-10) +     # 组内一致性权重
            ratio_score * 10 +          # 预测比例合理性权重
            silhouette * 5              # 轮廓系数权重
        )
        
        return score
    
    def find_threshold(self, probs, return_all=False):
        """
        主方法：使用多种方法找阈值，综合投票
        
        Args:
            probs: 概率分数数组
            return_all: 是否返回所有方法的阈值
            
        Returns:
            threshold: 最优阈值
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
        
        # ==============================
        # 应用所有方法
        # ==============================
        methods = {
            'gmm_crossing': self._find_threshold_gmm_crossing,
            'knee_detection': self._find_threshold_knee_detection,
            'percentile_gap': self._find_threshold_percentile_gap,
            'tail_analysis': self._find_threshold_tail_analysis,
            'two_stage': self._find_threshold_two_stage,
            'density_valley': self._find_threshold_density_valley,
            'f1_estimation': self._estimate_optimal_threshold_f1,
            'outlier_detection': self._find_threshold_outlier_detection
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
        
        self.debug_info['all_thresholds'] = {k: float(v) for k, v in thresholds.items()}
        
        if not thresholds:
            # 回退到98%分位数
            self.threshold = np.percentile(probs, 98)
            self.debug_info['fallback'] = True
            return self.threshold
        
        # ==============================
        # 评估每个阈值的质量
        # ==============================
        quality_scores = {}
        for name, thresh in thresholds.items():
            quality_scores[name] = self._evaluate_threshold(probs, thresh)
        
        self.debug_info['quality_scores'] = {k: float(v) for k, v in quality_scores.items()}
        
        # ==============================
        # 估计真实的异常比例
        # ==============================
        log_probs = self._log_transform(probs)
        
        try:
            gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
            gmm.fit(log_probs.reshape(-1, 1))
            weights = gmm.weights_.flatten()
            means = gmm.means_.flatten()
            high_idx = np.argmax(means)
            estimated_ratio = weights[high_idx]
        except:
            estimated_ratio = 0.01
        
        # 修正估计比例
        if estimated_ratio > 0.1:
            estimated_ratio = 0.01
        if estimated_ratio < 0.001:
            estimated_ratio = 0.005
        
        self.debug_info['estimated_ratio'] = float(estimated_ratio)
        
        # ==============================
        # 计算每个阈值的预测比例
        # ==============================
        pred_ratios = {}
        for name, thresh in thresholds.items():
            pred_ratios[name] = np.mean(probs >= thresh)
        
        self.debug_info['pred_ratios'] = {k: float(v) for k, v in pred_ratios.items()}
        
        # ==============================
        # 最终决策逻辑
        # ==============================
        
        # 策略1：选择预测比例最接近估计比例的阈值
        target_ratio = estimated_ratio
        best_diff = np.inf
        best_name = None
        
        for name, ratio in pred_ratios.items():
            diff = np.abs(ratio - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_name = name
        
        threshold_ratio_based = thresholds[best_name]
        
        # 策略2：选择质量评分最高的阈值
        best_quality_name = max(quality_scores, key=quality_scores.get)
        threshold_quality_based = thresholds[best_quality_name]
        
        # 策略3：使用加权中位数
        # 按质量分数加权
        total_quality = sum(max(0, q) for q in quality_scores.values())
        if total_quality > 0:
            weights_list = [max(0, quality_scores[name]) / total_quality for name in thresholds]
            # 使用加权中位数
            sorted_indices = np.argsort(list(thresholds.values()))
            cumsum = np.cumsum([weights_list[i] for i in sorted_indices])
            median_idx = sorted_indices[np.searchsorted(cumsum, 0.5)]
            threshold_weighted = list(thresholds.values())[median_idx]
        else:
            threshold_weighted = np.median(list(thresholds.values()))
        
        self.debug_info['threshold_ratio_based'] = float(threshold_ratio_based)
        self.debug_info['threshold_quality_based'] = float(threshold_quality_based)
        self.debug_info['threshold_weighted'] = float(threshold_weighted)
        
        # ==============================
        # 最终选择：基于多种策略的综合
        # ==============================
        
        # 计算三个候选阈值
        candidates = [threshold_ratio_based, threshold_quality_based, threshold_weighted]
        
        # 计算每个候选的预测比例
        candidate_ratios = [np.mean(probs >= t) for t in candidates]
        
        # 选择预测比例在合理范围内的最佳阈值
        # 合理范围：估计比例的0.5-2倍
        low_ratio = max(0.001, estimated_ratio * 0.5)
        high_ratio = min(0.1, estimated_ratio * 2)
        
        valid_candidates = [
            (t, r) for t, r in zip(candidates, candidate_ratios)
            if low_ratio <= r <= high_ratio
        ]
        
        if valid_candidates:
            # 选择预测比例最接近估计比例的
            valid_candidates.sort(key=lambda x: abs(x[1] - estimated_ratio))
            self.threshold = valid_candidates[0][0]
            self.debug_info['selection_method'] = 'valid_candidate'
        else:
            # 如果没有有效候选，使用几何平均
            self.threshold = np.exp(np.mean(np.log(candidates)))
            self.debug_info['selection_method'] = 'geometric_mean'
        
        # 最终验证
        final_pred_ratio = np.mean(probs >= self.threshold)
        
        # 如果预测比例过高（>10%），调整为更严格的阈值
        if final_pred_ratio > 0.1:
            self.threshold = np.percentile(probs, 99)
            self.debug_info['adjustment'] = 'too_high_ratio'
        # 如果预测比例过低（<0.1%），调整为更宽松的阈值
        elif final_pred_ratio < 0.001:
            self.threshold = np.percentile(probs, 95)
            self.debug_info['adjustment'] = 'too_low_ratio'
        
        self.debug_info['final_threshold'] = float(self.threshold)
        self.debug_info['final_pred_ratio'] = float(np.mean(probs >= self.threshold))
        self.debug_info['final_quality'] = float(self._evaluate_threshold(probs, self.threshold))
        
        if return_all:
            return self.threshold, thresholds, quality_scores
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
        report.append("自适应阈值优化器 V2 - 详细报告")
        report.append("=" * 60)
        
        if 'prob_stats' in self.debug_info:
            stats = self.debug_info['prob_stats']
            report.append("\n【分数分布统计】")
            report.append(f"  样本数: {self.debug_info.get('n_samples', 'N/A')}")
            report.append(f"  最小值: {stats['min']:.6f}")
            report.append(f"  最大值: {stats['max']:.6f}")
            report.append(f"  均值: {stats['mean']:.6f}")
            report.append(f"  中位数: {stats['median']:.6f}")
            report.append(f"  标准差: {stats['std']:.6f}")
            report.append(f"  偏度: {stats['skewness']:.4f}")
            report.append(f"  峰度: {stats['kurtosis']:.4f}")
        
        if 'all_thresholds' in self.debug_info:
            report.append("\n【各方法阈值】")
            for name, thresh in self.debug_info['all_thresholds'].items():
                quality = self.debug_info.get('quality_scores', {}).get(name, 'N/A')
                ratio = self.debug_info.get('pred_ratios', {}).get(name, 'N/A')
                report.append(f"  {name:25s}: {thresh:.6f} (质量: {quality:.4f}, 预测比例: {ratio:.4f})")
        
        if 'estimated_ratio' in self.debug_info:
            report.append(f"\n【估计异常比例】: {self.debug_info['estimated_ratio']:.4f}")
        
        if 'final_threshold' in self.debug_info:
            report.append("\n【最终决策】")
            report.append(f"  选择方法: {self.debug_info.get('selection_method', 'N/A')}")
            report.append(f"  最终阈值: {self.debug_info['final_threshold']:.6f}")
            report.append(f"  预测比例: {self.debug_info.get('final_pred_ratio', 'N/A'):.4f}")
            report.append(f"  最终质量: {self.debug_info.get('final_quality', 'N/A'):.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def compute_botnet_metrics_adaptive(y_true: np.ndarray, probs: np.ndarray, verbose: bool = False) -> dict:
    """
    使用自适应阈值优化器V2计算指标
    
    Args:
        y_true: 真实标签（仅用于评估，不参与阈值选择）
        probs: 概率分数
        verbose: 是否打印详细信息
        
    Returns:
        评估结果字典
    """
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    
    optimizer = AdaptiveThresholdOptimizerV2(verbose=verbose)
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
        'method': 'Adaptive_V2',
        'predicted_ratio': float(preds.sum() / len(probs)),
        'debug_info': optimizer.debug_info
    }
    
    if verbose:
        print(optimizer.get_debug_report())
    
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("自适应阈值优化器 V2 测试")
    print("=" * 70)
    
    try:
        data = np.load('v3_final_score_distribution.npz', allow_pickle=True)
        probs = data['probs']
        y_true = data['y_true']
        
        print(f"\n[数据概览]")
        print(f"  总节点数：{len(probs)}")
        print(f"  僵尸节点数：{y_true.sum()} ({y_true.sum()/len(probs)*100:.2f}%)")
        
        # 测试不同阈值的效果
        print(f"\n[阈值效果分析（使用真实标签评估）]")
        from sklearn.metrics import precision_recall_fscore_support
        for thresh in [0.001, 0.002, 0.003, 0.005, 0.008, 0.01]:
            preds = (probs >= thresh).astype(int)
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  阈值 {thresh:.4f}: 预测 {preds.sum():5d}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        # 测试自适应优化器
        print("\n[测试自适应阈值优化器 V2]...")
        result = compute_botnet_metrics_adaptive(y_true, probs, verbose=True)
        
        print(f"\n评估结果（自适应优化器 V2）")
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