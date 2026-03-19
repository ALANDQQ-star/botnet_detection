"""
智能阈值优化器

采用聚类算法和统计学习方法智能确定动态阈值：
1. 双聚类GMM（高斯混合模型）- 自动识别两个聚类群
2. 核密度估计(KDE) + 谷点检测 - 找到分布双峰之间的谷
3. Otsu自适应阈值 - 最大化类间方差
4. 分位数差距分析 - 找到分布突变点
5. 综合投票机制 - 多方法融合决策

核心思想：
- 分数分布呈现"一大一小"两个聚类群（正常节点群大，僵尸节点群小）
- 智能预测聚类群分界线
"""

import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize_scalar, brentq
import warnings
warnings.filterwarnings('ignore')


class IntelligentThresholdOptimizer:
    """
    智能阈值优化器
    
    使用聚类算法和统计学习方法，不依赖真实标签
    自动找到分数分布中两个聚类群的最优分界线
    """
    
    def __init__(self, verbose=False):
        self.threshold = None
        self.verbose = verbose
        self.debug_info = {}
        
    def _log_transform(self, x, eps=1e-10):
        """安全的对数变换，增强低值区域的分辨率"""
        return np.log10(x + eps)
    
    def _find_threshold_gmm_bic(self, probs):
        """
        方法1：GMM + BIC模型选择
        
        自动确定最佳聚类数并找阈值
        """
        # 使用对数变换增强分离度
        log_probs = self._log_transform(probs).reshape(-1, 1)
        
        best_bic = np.inf
        best_gmm = None
        best_n = 1
        
        # 尝试1-5个聚类
        for n_components in range(1, 6):
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=42,
                    n_init=10,
                    max_iter=500
                )
                gmm.fit(log_probs)
                bic = gmm.bic(log_probs)
                
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                    best_n = n_components
            except:
                continue
        
        self.debug_info['gmm_bic_n_components'] = best_n
        
        if best_n == 1:
            # 单峰分布，使用高分位数
            return np.percentile(probs, 98)
        
        # 多个聚类：找两个最大聚类中心之间的分界点
        means = best_gmm.means_.flatten()
        sorted_idx = np.argsort(means)
        
        # 最高的两个聚类中心
        high_cluster_idx = sorted_idx[-1]
        second_high_idx = sorted_idx[-2]
        
        # 计算两聚类之间的分界点
        # 使用后验概率的等分点
        posteriors = best_gmm.predict_proba(log_probs)
        
        # 找到两个最大聚类
        high_posterior = posteriors[:, high_cluster_idx]
        second_posterior = posteriors[:, second_high_idx]
        
        # 找到两个后验概率相等的点
        diff = high_posterior - second_posterior
        
        # 从高分向低分扫描，找到符号变化的点
        sorted_probs_idx = np.argsort(probs)[::-1]
        
        threshold = None
        for i in range(len(sorted_probs_idx) - 1):
            idx1 = sorted_probs_idx[i]
            idx2 = sorted_probs_idx[i + 1]
            if diff[idx1] > 0 and diff[idx2] <= 0:
                # 符号变化点
                threshold = (probs[idx1] + probs[idx2]) / 2
                break
        
        if threshold is None:
            # 回退到聚类中心的中间点
            threshold = (10**means[high_cluster_idx] + 10**means[second_high_idx]) / 2
        
        return threshold
    
    def _find_threshold_gmm_two_cluster(self, probs):
        """
        方法2：双高斯混合模型
        
        强制双聚类假设，找聚类分界线
        """
        log_probs = self._log_transform(probs).reshape(-1, 1)
        
        # 使用贝叶斯GMM自动确定有效聚类数
        bgmm = BayesianGaussianMixture(
            n_components=2,
            covariance_type='full',
            random_state=42,
            n_init=20,
            max_iter=1000,
            weight_concentration_prior=0.1
        )
        bgmm.fit(log_probs)
        
        # 获取聚类参数
        means = bgmm.means_.flatten()
        weights = bgmm.weights_.flatten()
        covariances = bgmm.covariances_.flatten()
        
        # 确定高值聚类和低值聚类
        if means[0] > means[1]:
            high_mean, low_mean = means[0], means[1]
            high_weight, low_weight = weights[0], weights[1]
            high_var, low_var = covariances[0], covariances[1]
        else:
            high_mean, low_mean = means[1], means[0]
            high_weight, low_weight = weights[1], weights[0]
            high_var, low_var = covariances[1], covariances[0]
        
        self.debug_info['gmm_high_mean'] = float(10**high_mean)
        self.debug_info['gmm_low_mean'] = float(10**low_mean)
        self.debug_info['gmm_high_weight'] = float(high_weight)
        self.debug_info['gmm_low_weight'] = float(low_weight)
        
        # 方法2a: 使用后验概率决策边界
        posteriors = bgmm.predict_proba(log_probs)
        high_cluster = 0 if means[0] > means[1] else 1
        high_posterior = posteriors[:, high_cluster]
        
        # 找到后验概率 > 0.5 的边界
        sorted_idx = np.argsort(probs)[::-1]
        threshold_posterior = None
        
        for i in range(len(sorted_idx) - 1):
            idx1 = sorted_idx[i]
            idx2 = sorted_idx[i + 1]
            if high_posterior[idx1] >= 0.5 and high_posterior[idx2] < 0.5:
                threshold_posterior = probs[sorted_idx[i + 1]]
                break
        
        # 方法2b: 使用贝叶斯决策理论
        # 找到两个高斯分布交叉点（对数空间）
        # log(p1(x)) = log(p2(x))
        # 展开: -0.5*log(2πσ1²) - 0.5(x-μ1)²/σ1² + log(w1) = -0.5*log(2πσ2²) - 0.5(x-μ2)²/σ2² + log(w2)
        
        std_high = np.sqrt(high_var)
        std_low = np.sqrt(low_var)
        
        # 数值求解两个高斯的交叉点
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
                threshold_bayes = 10**x_intersect
            else:
                threshold_bayes = np.percentile(probs, 95)
        except:
            threshold_bayes = np.percentile(probs, 95)
        
        self.debug_info['threshold_posterior'] = float(threshold_posterior) if threshold_posterior else None
        self.debug_info['threshold_bayes'] = float(threshold_bayes)
        
        # 使用后验概率方法
        if threshold_posterior is not None and 0 < threshold_posterior < 1:
            return threshold_posterior
        elif 0 < threshold_bayes < 1:
            return threshold_bayes
        else:
            return np.percentile(probs, 97)
    
    def _find_threshold_kde_valley(self, probs):
        """
        方法3：核密度估计 + 谷点检测
        
        找到概率密度分布的双峰之间的谷
        """
        # 使用对数变换
        log_probs = self._log_transform(probs)
        
        # 在对数空间建立网格
        x_min, x_max = log_probs.min() - 1, log_probs.max() + 1
        x_grid = np.linspace(x_min, x_max, 1000)
        
        # 核密度估计
        try:
            kde = KernelDensity(bandwidth='silverman', kernel='gaussian')
            kde.fit(log_probs.reshape(-1, 1))
            log_density = kde.score_samples(x_grid.reshape(-1, 1))
            density = np.exp(log_density)
        except:
            return np.percentile(probs, 97)
        
        # 平滑密度曲线
        if len(density) > 11:
            density_smooth = savgol_filter(density, 11, 3)
        else:
            density_smooth = density
        
        # 找峰值
        peaks, peak_properties = find_peaks(density_smooth, distance=20, prominence=0.001)
        
        if len(peaks) < 2:
            # 单峰，使用高分位数
            return np.percentile(probs, 97)
        
        # 找两个最显著的峰
        prominences = peak_properties['prominences']
        top_two_idx = np.argsort(prominences)[-2:]
        top_two_peaks = np.sort(peaks[top_two_idx])
        
        # 在两个峰之间找谷（最小值）
        peak1, peak2 = top_two_peaks
        valley_region = density_smooth[peak1:peak2+1]
        valley_idx = peak1 + np.argmin(valley_region)
        
        # 谷点对应的分数
        threshold = 10**x_grid[valley_idx]
        
        self.debug_info['kde_peaks'] = [float(10**x_grid[p]) for p in peaks]
        self.debug_info['kde_valley'] = float(threshold)
        
        return threshold
    
    def _find_threshold_otsu(self, probs):
        """
        方法4：Otsu自适应阈值
        
        最大化类间方差
        """
        # 将概率离散化为256个bin
        bins = 256
        hist, bin_edges = np.histogram(probs, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 计算累积统计量
        total = len(probs)
        weight1 = np.cumsum(hist) / total
        weight2 = 1 - weight1
        
        mean1 = np.cumsum(hist * bin_centers) / (np.cumsum(hist) + 1e-10)
        mean2 = (np.sum(hist * bin_centers) - np.cumsum(hist * bin_centers)) / (np.cumsum(hist[::-1])[::-1] + 1e-10)
        
        # 类间方差
        between_class_variance = weight1 * weight2 * (mean1 - mean2)**2
        
        # 找最大方差点
        best_idx = np.argmax(between_class_variance)
        threshold = bin_centers[best_idx]
        
        self.debug_info['otsu_threshold'] = float(threshold)
        
        return threshold
    
    def _find_threshold_percentile_gap(self, probs):
        """
        方法5：分位数差距分析
        
        找到分布突变点（相邻分位数差距最大的点）
        """
        # 分析高分区域（90%以上）的分位数
        percentiles = np.linspace(90, 99.9, 100)
        p_values = np.percentile(probs, percentiles)
        
        # 计算相邻分位数的差距
        gaps = np.diff(p_values)
        
        # 找到差距最大的点（突变点）
        max_gap_idx = np.argmax(gaps)
        threshold = (p_values[max_gap_idx] + p_values[max_gap_idx + 1]) / 2
        
        self.debug_info['percentile_gap_threshold'] = float(threshold)
        self.debug_info['percentile_gap_idx'] = int(max_gap_idx)
        self.debug_info['percentile_gap_value'] = float(gaps[max_gap_idx])
        
        return threshold
    
    def _find_threshold_kmeans(self, probs):
        """
        方法6：K-means聚类
        
        强制分成两类
        """
        # 使用对数变换
        log_probs = self._log_transform(probs).reshape(-1, 1)
        
        # K-means
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
        labels = kmeans.fit_predict(log_probs)
        
        centers = kmeans.cluster_centers_.flatten()
        high_center_idx = np.argmax(centers)
        
        # 找到高值聚类中的最小分数
        high_cluster_mask = labels == high_center_idx
        high_probs = probs[high_cluster_mask]
        
        if len(high_probs) > 0:
            threshold = high_probs.min()
        else:
            threshold = np.percentile(probs, 95)
        
        self.debug_info['kmeans_threshold'] = float(threshold)
        self.debug_info['kmeans_centers'] = [float(10**c) for c in centers]
        
        return threshold
    
    def _find_threshold_jenks_natural_breaks(self, probs):
        """
        方法7：Jenks自然断点
        
        最小化类内方差，最大化类间方差
        """
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        
        # 只搜索合理的分割点范围（高分区域）
        start_idx = int(n * 0.90)
        end_idx = int(n * 0.99)
        
        best_threshold = sorted_probs[start_idx]
        best_gvf = 0  # Goodness of Variance Fit
        
        # 计算总方差
        total_var = np.var(sorted_probs)
        
        for i in range(start_idx, end_idx):
            below = sorted_probs[:i]
            above = sorted_probs[i:]
            
            if len(below) == 0 or len(above) == 0:
                continue
            
            # 类内方差
            var_within = (np.var(below) * len(below) + np.var(above) * len(above)) / n
            
            # 类间方差
            mean_below = np.mean(below)
            mean_above = np.mean(above)
            grand_mean = np.mean(sorted_probs)
            var_between = (len(below) * (mean_below - grand_mean)**2 + 
                          len(above) * (mean_above - grand_mean)**2) / n
            
            # GVF (Goodness of Variance Fit)
            gvf = (total_var - var_within) / total_var if total_var > 0 else 0
            
            if gvf > best_gvf:
                best_gvf = gvf
                best_threshold = sorted_probs[i]
        
        self.debug_info['jenks_threshold'] = float(best_threshold)
        self.debug_info['jenks_gvf'] = float(best_gvf)
        
        return best_threshold
    
    def _evaluate_threshold_quality(self, probs, threshold):
        """
        评估阈值质量
        
        返回一个综合评分（越高越好）
        """
        below = probs[probs < threshold]
        above = probs[probs >= threshold]
        
        if len(below) == 0 or len(above) == 0:
            return 0
        
        n = len(probs)
        
        # 评分维度：
        # 1. 组间差异（越高越好）
        mean_diff = np.abs(np.mean(above) - np.mean(below))
        
        # 2. 组内一致性（标准差越小越好）
        std_below = np.std(below)
        std_above = np.std(above)
        avg_std = (std_below + std_above) / 2
        
        # 3. 轮廓系数（类间分离度）
        labels = (probs >= threshold).astype(int)
        if len(np.unique(labels)) > 1:
            try:
                silhouette = silhouette_score(probs.reshape(-1, 1), labels)
            except:
                silhouette = 0
        else:
            silhouette = 0
        
        # 4. 预测比例合理性（假设僵尸网络占0.3%-2%）
        # 调整为更合理的范围
        pred_ratio = len(above) / n
        ratio_score = 0
        if 0.003 <= pred_ratio <= 0.02:
            ratio_score = 1.0
        elif pred_ratio < 0.003:
            ratio_score = pred_ratio / 0.003  # 过低
        else:
            ratio_score = max(0, 1 - (pred_ratio - 0.02) / 0.05)  # 过高
        
        # 综合评分
        quality_score = (
            mean_diff * 10 +           # 组间差异权重
            1 / (avg_std + 1e-10) +    # 组内一致性权重
            silhouette +                # 轮廓系数权重
            ratio_score * 5             # 预测比例合理性权重（提高权重）
        )
        
        return quality_score
    
    def _find_threshold_optimized(self, probs):
        """
        方法8：优化的阈值选择
        
        基于分数分布的统计特性，找到最优分割点
        核心思想：使用分数分布的"拐点"作为阈值
        """
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        
        # 计算分数的累积分布
        # 找到分数分布的"肘部"（elbow point）
        
        # 方法1: 使用二阶导数找拐点
        # 只关注高分区域 (90%以上)
        start_idx = int(n * 0.90)
        
        # 计算一阶差分（斜率）
        high_probs = sorted_probs[start_idx:]
        first_diff = np.diff(high_probs)
        
        # 计算二阶差分（曲率）
        second_diff = np.diff(first_diff)
        
        # 找到曲率最大的点（最显著的拐点）
        # 这通常对应于正常节点和异常节点的分界
        elbow_idx = np.argmax(second_diff)
        threshold_elbow = high_probs[elbow_idx + 1]
        
        # 方法2: 使用 K-均值聚类的边界
        log_probs = self._log_transform(sorted_probs[start_idx:])
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(log_probs.reshape(-1, 1))
        
        # 找到聚类边界
        boundary_idx = np.where(np.diff(labels) != 0)[0]
        if len(boundary_idx) > 0:
            boundary_idx = boundary_idx[0]
            threshold_boundary = sorted_probs[start_idx + boundary_idx]
        else:
            threshold_boundary = sorted_probs[int(n * 0.97)]
        
        # 方法3: 基于分数间距
        # 找到相邻分数间距最大的点
        gaps = first_diff
        max_gap_idx = np.argmax(gaps)
        threshold_gap = high_probs[max_gap_idx + 1]
        
        self.debug_info['threshold_elbow'] = float(threshold_elbow)
        self.debug_info['threshold_boundary'] = float(threshold_boundary)
        self.debug_info['threshold_gap'] = float(threshold_gap)
        
        # 综合三个阈值，选择中间值
        thresholds = [threshold_elbow, threshold_boundary, threshold_gap]
        return np.median(thresholds)
    
    def _find_threshold_adaptive(self, probs):
        """
        方法9：自适应阈值选择
        
        基于分数分布的形状特征自适应选择阈值
        """
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        
        # 分析分数分布的形状
        # 使用对数变换后的分布
        log_probs = self._log_transform(probs)
        
        # 计算分布的偏度
        skewness = stats.skew(log_probs)
        
        # 如果分布高度偏斜（右偏），说明有明显的异常值
        # 阈值应该在高分区域
        
        if skewness > 10:  # 高度偏斜
            # 使用更高的分位数
            percentile = 98.5
        elif skewness > 5:
            percentile = 97
        else:
            percentile = 95
        
        # 计算分位数
        threshold_percentile = np.percentile(probs, percentile)
        
        # 使用 GMM 精细调整
        log_probs_2d = log_probs.reshape(-1, 1)
        
        try:
            gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
            gmm.fit(log_probs_2d)
            
            means = gmm.means_.flatten()
            high_mean_idx = np.argmax(means)
            
            # 获取高值聚类的后验概率
            posteriors = gmm.predict_proba(log_probs_2d)
            high_posterior = posteriors[:, high_mean_idx]
            
            # 找到后验概率 > 0.5 的边界
            sorted_idx = np.argsort(probs)[::-1]
            threshold_gmm = None
            
            for i in range(len(sorted_idx) - 1):
                if high_posterior[sorted_idx[i]] >= 0.5 and high_posterior[sorted_idx[i+1]] < 0.5:
                    threshold_gmm = probs[sorted_idx[i+1]]
                    break
            
            if threshold_gmm is None:
                threshold_gmm = threshold_percentile
                
        except:
            threshold_gmm = threshold_percentile
        
        # 综合决策
        # 选择使预测比例在合理范围内的阈值
        pred_ratio_percentile = np.mean(probs >= threshold_percentile)
        pred_ratio_gmm = np.mean(probs >= threshold_gmm)
        
        # 目标比例：0.5%-2%
        target_ratio = 0.01  # 1%
        
        # 选择更接近目标比例的阈值
        if abs(pred_ratio_percentile - target_ratio) < abs(pred_ratio_gmm - target_ratio):
            return threshold_percentile
        else:
            return threshold_gmm
    
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
            'gmm_bic': self._find_threshold_gmm_bic,
            'gmm_two_cluster': self._find_threshold_gmm_two_cluster,
            'kde_valley': self._find_threshold_kde_valley,
            'otsu': self._find_threshold_otsu,
            'percentile_gap': self._find_threshold_percentile_gap,
            'kmeans': self._find_threshold_kmeans,
            'jenks': self._find_threshold_jenks_natural_breaks,
            'optimized': self._find_threshold_optimized,
            'adaptive': self._find_threshold_adaptive
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
            # 回退到96%分位数
            self.threshold = np.percentile(probs, 96)
            self.debug_info['fallback'] = True
            return self.threshold
        
        # ==============================
        # 评估每个阈值的质量
        # ==============================
        quality_scores = {}
        for name, thresh in thresholds.items():
            quality_scores[name] = self._evaluate_threshold_quality(probs, thresh)
        
        self.debug_info['quality_scores'] = {k: float(v) for k, v in quality_scores.items()}
        
        # ==============================
        # 综合决策
        # ==============================
        
        # 方法1：加权平均（按质量分数加权）
        total_quality = sum(quality_scores.values())
        if total_quality > 0:
            weighted_threshold = sum(
                thresholds[name] * quality_scores[name] / total_quality 
                for name in thresholds
            )
        else:
            weighted_threshold = np.mean(list(thresholds.values()))
        
        # 方法2：中位数（鲁棒）
        median_threshold = np.median(list(thresholds.values()))
        
        # 方法3：选择质量最高的
        best_method = max(quality_scores, key=quality_scores.get)
        best_threshold = thresholds[best_method]
        
        # ==============================
        # 最终决策逻辑
        # ==============================
        
        # 分析预测比例
        pred_ratios = {name: np.mean(probs >= thresh) for name, thresh in thresholds.items()}
        self.debug_info['pred_ratios'] = {k: float(v) for k, v in pred_ratios.items()}
        
        # 估计真实的僵尸网络比例（基于分数分布的统计特性）
        # 使用对数变换后的分布来估计
        log_probs = self._log_transform(probs)
        
        # 使用双高斯混合模型估计真实比例
        try:
            gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
            gmm.fit(log_probs.reshape(-1, 1))
            weights = gmm.weights_.flatten()
            means = gmm.means_.flatten()
            
            # 高值聚类作为异常
            high_idx = np.argmax(means)
            estimated_ratio = weights[high_idx]
        except:
            estimated_ratio = 0.01  # 默认1%
        
        # 修正估计比例（如果看起来不合理）
        if estimated_ratio > 0.1:
            estimated_ratio = 0.01
        if estimated_ratio < 0.001:
            estimated_ratio = 0.005
            
        self.debug_info['estimated_ratio'] = float(estimated_ratio)
        
        # 设置目标比例范围：估计值的0.8-1.5倍
        target_ratio_low = estimated_ratio * 0.8
        target_ratio_high = estimated_ratio * 1.5
        
        self.debug_info['target_ratio_range'] = [float(target_ratio_low), float(target_ratio_high)]
        
        # 选择使预测比例在合理范围内的阈值
        valid_thresholds = {
            name: thresh for name, thresh in thresholds.items()
            if target_ratio_low <= pred_ratios[name] <= target_ratio_high
        }
        
        if valid_thresholds:
            # 在有效阈值中选择质量最高的
            valid_quality = {name: quality_scores[name] for name in valid_thresholds}
            best_valid = max(valid_quality, key=valid_quality.get)
            self.threshold = valid_thresholds[best_valid]
            self.debug_info['selection_method'] = 'best_valid'
        else:
            # 如果没有有效阈值，选择最接近目标比例的
            target_ratio = estimated_ratio
            best_diff = np.inf
            best_name = None
            
            for name, ratio in pred_ratios.items():
                diff = abs(ratio - target_ratio)
                if diff < best_diff:
                    best_diff = diff
                    best_name = name
            
            self.threshold = thresholds[best_name]
            self.debug_info['selection_method'] = 'closest_to_estimate'
        
        # 验证最终阈值
        final_pred_ratio = np.mean(probs >= self.threshold)
        self.debug_info['final_threshold'] = float(self.threshold)
        self.debug_info['final_pred_ratio'] = float(final_pred_ratio)
        self.debug_info['final_quality'] = float(self._evaluate_threshold_quality(probs, self.threshold))
        
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
        report.append("智能阈值优化器 - 详细报告")
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
                report.append(f"  {name:20s}: {thresh:.6f} (质量: {quality:.4f}, 预测比例: {ratio:.4f})")
        
        if 'final_threshold' in self.debug_info:
            report.append("\n【最终决策】")
            report.append(f"  选择方法: {self.debug_info.get('selection_method', 'N/A')}")
            report.append(f"  最终阈值: {self.debug_info['final_threshold']:.6f}")
            report.append(f"  预测比例: {self.debug_info.get('final_pred_ratio', 'N/A'):.4f}")
            report.append(f"  最终质量: {self.debug_info.get('final_quality', 'N/A'):.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def compute_botnet_metrics_intelligent(y_true: np.ndarray, probs: np.ndarray, verbose: bool = False) -> dict:
    """
    使用智能阈值优化器计算指标
    
    Args:
        y_true: 真实标签
        probs: 概率分数
        verbose: 是否打印详细信息
        
    Returns:
        评估结果字典
    """
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    
    optimizer = IntelligentThresholdOptimizer(verbose=verbose)
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
        'method': 'Intelligent_Clustering',
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
        print(f"  僵尸节点数：{y_true.sum()} ({y_true.sum()/len(probs)*100:.2f}%)")
        
        # 测试不同阈值的效果
        print(f"\n[阈值效果分析]")
        for thresh in [0.0005, 0.001, 0.002, 0.005, 0.01]:
            preds = (probs >= thresh).astype(int)
            tp = ((preds == 1) & (y_true == 1)).sum()
            precision = tp / preds.sum() if preds.sum() > 0 else 0
            recall = tp / y_true.sum() if y_true.sum() > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  阈值 {thresh:.4f}: 预测 {preds.sum():5d}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        # 测试智能优化器
        print("\n[测试智能阈值优化器]...")
        result = compute_botnet_metrics_intelligent(y_true, probs, verbose=True)
        
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
