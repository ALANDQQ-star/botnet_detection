"""
优化的僵尸网络分类器

核心洞察（基于分数分布分析）：
- 僵尸节点分数均值：0.002，中位数：0.001
- 正常节点分数均值：0.0004，中位数：0.000002
- 僵尸节点分数显著高于正常节点（AUC=0.958）
- 当前阈值选择算法失效，选出太高阈值

无监督统计学策略：
1. 对数变换扩展动态范围
2. 使用 KMeans 聚类找到两个群体
3. 使用 Gap Statistic 或轮廓系数确定分界点
4. 结合极值理论(EVT)优化阈值

目标：不使用标签，仅基于分数分布特征找到最优阈值
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')


class OptimizedBotnetClassifier:
    """
    优化的僵尸网络分类器（纯无监督）
    
    核心思想：
    1. 分析分数分布的统计特征
    2. 使用多种无监督方法估计阈值
    3. 综合多种方法的结论
    """
    
    def __init__(self):
        self.threshold = None
        self.method_used = None
        
    def _log_transform(self, probs, eps=1e-10):
        """对数变换：扩展低值区域的动态范围"""
        return np.log(probs + eps)
    
    def _method_kmeans(self, probs):
        """
        方法1：KMeans 聚类
        
        使用对数变换后的分数进行 KMeans 聚类
        假设：分数分布呈现两个聚类
        """
        log_probs = self._log_transform(probs).reshape(-1, 1)
        
        # KMeans 聚类
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(log_probs)
        
        # 确定哪个聚类是高分数组
        centers = kmeans.cluster_centers_.flatten()
        high_cluster = 0 if centers[0] > centers[1] else 1
        
        # 找到两个聚类之间的分界点
        high_probs = probs[labels == high_cluster]
        low_probs = probs[labels != high_cluster]
        
        # 阈值设置为低分数组的最大值和高分数组的最小值的中点
        threshold = (low_probs.max() + high_probs.min()) / 2
        
        return threshold, {
            'method': 'KMeans',
            'high_center': float(centers[high_cluster]),
            'low_center': float(centers[1 - high_cluster]),
            'high_cluster_size': int(np.sum(labels == high_cluster)),
            'low_cluster_size': int(np.sum(labels != high_cluster))
        }
    
    def _method_gmm_elbow(self, probs):
        """
        方法2：GMM + 肘点法
        
        使用高斯混合模型拟合对数分数分布
        找到两个高斯分布之间的肘点
        """
        log_probs = self._log_transform(probs).reshape(-1, 1)
        
        # 拟合双高斯混合模型
        gmm = GaussianMixture(n_components=2, covariance_type='full', 
                               random_state=42, n_init=5, max_iter=200)
        gmm.fit(log_probs)
        
        # 获取参数
        means = gmm.means_.flatten()
        covariances = gmm.covariances_.flatten()
        weights = gmm.weights_
        
        # 确定高低分数组
        high_idx = 0 if means[0] > means[1] else 1
        low_idx = 1 - high_idx
        
        # 方法2a：使用两个高斯分布的交点
        # 解方程：w1 * N(x|μ1,σ1) = w2 * N(x|μ2,σ2)
        mu1, sigma1, w1 = means[low_idx], np.sqrt(covariances[low_idx]), weights[low_idx]
        mu2, sigma2, w2 = means[high_idx], np.sqrt(covariances[high_idx]), weights[high_idx]
        
        # 简化：使用对数似然比 = 0 的点
        # log(w1) - (x-μ1)^2/(2σ1^2) - log(σ1) = log(w2) - (x-μ2)^2/(2σ2^2) - log(σ2)
        # 这是一个二次方程，我们取两个分布均值的中点作为近似
        threshold_intersection_log = (mu1 + mu2) / 2
        threshold_intersection = np.exp(threshold_intersection_log) - 1e-10
        
        # 方法2b：基于 FPR 控制
        # 假设低分数组是正常节点，设置 FPR = 1%
        from scipy.stats import norm
        z_score = norm.ppf(0.99)  # 99% 分位点
        threshold_fpr_log = mu1 + z_score * sigma1
        threshold_fpr = np.exp(threshold_fpr_log) - 1e-10
        
        # 方法2c：使用后验概率
        posteriors = gmm.predict_proba(log_probs)
        high_posterior = posteriors[:, high_idx]
        
        # 找到后验概率首次超过 0.5 的分数
        sorted_indices = np.argsort(probs)
        sorted_posteriors = high_posterior[sorted_indices]
        above_half = np.where(sorted_posteriors > 0.5)[0]
        
        if len(above_half) > 0:
            idx = above_half[0]
            threshold_posterior = probs[sorted_indices[idx]]
        else:
            threshold_posterior = threshold_fpr
        
        # 综合选择：取中等值
        thresholds = [threshold_intersection, threshold_fpr, threshold_posterior]
        thresholds = [t for t in thresholds if t > 0 and t < 1]
        
        if thresholds:
            threshold = np.median(thresholds)
        else:
            threshold = threshold_fpr
        
        return threshold, {
            'method': 'GMM_Elbow',
            'threshold_intersection': float(threshold_intersection),
            'threshold_fpr': float(threshold_fpr),
            'threshold_posterior': float(threshold_posterior),
            'means': means.tolist(),
            'weights': weights.tolist()
        }
    
    def _method_percentile_gap(self, probs):
        """
        方法3：分位数间隔分析
        
        找到分数分布中"间隔"最大的位置
        假设：正常节点和僵尸节点分数之间存在间隔
        """
        # 排序分数
        sorted_probs = np.sort(probs)
        
        # 计算相邻分数的间隔
        gaps = np.diff(sorted_probs)
        
        # 在低分数区域（95%分位之前）找最大间隔
        search_range = int(len(sorted_probs) * 0.99)
        max_gap_idx = np.argmax(gaps[:search_range])
        
        # 阈值设置在间隔中间
        threshold = (sorted_probs[max_gap_idx] + sorted_probs[max_gap_idx + 1]) / 2
        
        # 如果间隔太小，使用分位数方法
        if gaps[max_gap_idx] < 1e-6:
            # 使用99%分位数
            threshold = np.percentile(probs, 99)
        
        return threshold, {
            'method': 'Percentile_Gap',
            'max_gap': float(gaps[max_gap_idx]),
            'max_gap_idx': int(max_gap_idx),
            'gap_percentile': float(max_gap_idx / len(sorted_probs) * 100)
        }
    
    def _method_derivative(self, probs):
        """
        方法4：导数分析（基于 CDF 的拐点）
        
        找到 CDF 曲线斜率变化最大的点
        """
        # 计算 ECDF
        sorted_probs = np.sort(probs)
        ecdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
        
        # 计算 CDF 的导数（即 PDF 近似）
        # 使用滑动窗口平滑
        window_size = max(100, len(sorted_probs) // 100)
        
        # 计算 PDF 近似
        pdf_approx = np.gradient(ecdf, sorted_probs)
        
        # 平滑 PDF
        if len(pdf_approx) > window_size:
            pdf_smooth = savgol_filter(pdf_approx, window_size, 3)
        else:
            pdf_smooth = pdf_approx
        
        # 找 PDF 的峰值（即分布的众数）
        peaks, properties = find_peaks(pdf_smooth, height=np.max(pdf_smooth) * 0.1)
        
        if len(peaks) >= 2:
            # 如果有两个以上的峰，取两个最高峰之间的谷
            sorted_peaks = peaks[np.argsort(pdf_smooth[peaks])[-2:]]
            sorted_peaks = np.sort(sorted_peaks)
            
            # 找谷点
            valley_idx = sorted_peaks[0] + np.argmin(pdf_smooth[sorted_peaks[0]:sorted_peaks[1]+1])
            threshold = sorted_probs[valley_idx]
        elif len(peaks) == 1:
            # 只有一个峰，使用尾部
            peak = peaks[0]
            # 在峰值右侧找下降最快的位置
            right_side = pdf_smooth[peak:]
            if len(right_side) > 10:
                decline_rate = np.diff(right_side)
                steepest_idx = np.argmin(decline_rate)
                threshold = sorted_probs[peak + steepest_idx + 1]
            else:
                threshold = np.percentile(probs, 95)
        else:
            # 没有峰值，使用分位数
            threshold = np.percentile(probs, 95)
        
        return threshold, {
            'method': 'Derivative',
            'num_peaks': len(peaks),
            'peaks': peaks.tolist() if len(peaks) > 0 else []
        }
    
    def _method_knee_point(self, probs):
        """
        方法5：膝点检测 (Knee Point Detection)
        
        找到排序分数曲线的"膝点"
        这是分数开始快速上升的位置
        使用 Kneedle 算法的简化版本
        """
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        
        # 归一化
        x = np.arange(n) / (n - 1)  # [0, 1]
        y = (sorted_probs - sorted_probs[0]) / (sorted_probs[-1] - sorted_probs[0] + 1e-10)
        
        # 计算到对角线的距离
        # 膝点是曲线偏离对角线最远的点
        distances = np.abs(y - x)
        
        # 找到最大距离点
        knee_idx = np.argmax(distances)
        threshold = sorted_probs[knee_idx]
        
        return threshold, {
            'method': 'Knee_Point',
            'knee_idx': int(knee_idx),
            'knee_percentile': float(knee_idx / n * 100)
        }
    
    def _method_outlier_detection(self, probs):
        """
        方法6：异常检测方法
        
        使用 IQR 和 MAD 方法检测异常高分
        """
        # 方法6a：IQR 方法
        q1 = np.percentile(probs, 25)
        q3 = np.percentile(probs, 75)
        iqr = q3 - q1
        threshold_iqr = q3 + 1.5 * iqr
        
        # 方法6b：MAD 方法 (Median Absolute Deviation)
        median = np.median(probs)
        mad = np.median(np.abs(probs - median))
        # 使用修正因子使其与正态分布一致
        mad_scaled = mad * 1.4826
        threshold_mad = median + 3 * mad_scaled
        
        # 方法6c：Z-score 方法（对数尺度）
        log_probs = self._log_transform(probs)
        log_mean = np.mean(log_probs)
        log_std = np.std(log_probs)
        threshold_zscore = np.exp(log_mean + 3 * log_std) - 1e-10
        
        # 方法6d：百分位数方法
        threshold_percentile = np.percentile(probs, 99)
        
        # 综合选择：使用中位数或最小值
        thresholds = [threshold_iqr, threshold_mad, threshold_zscore, threshold_percentile]
        thresholds = [t for t in thresholds if t > 0 and t < 1]
        
        # 取最小值（更敏感）
        threshold = min(thresholds)
        
        return threshold, {
            'method': 'Outlier_Detection',
            'threshold_iqr': float(threshold_iqr),
            'threshold_mad': float(threshold_mad),
            'threshold_zscore': float(threshold_zscore),
            'threshold_percentile': float(threshold_percentile),
            'q1': float(q1),
            'q3': float(q3),
            'median': float(median),
            'mad': float(mad)
        }
    
    def _method_log_distribution(self, probs):
        """
        方法7：对数分布分析
        
        分析对数分数的分布特征
        假设：对数分数呈现双峰分布
        """
        log_probs = self._log_transform(probs)
        
        # 分析对数分数的分布
        log_mean = np.mean(log_probs)
        log_std = np.std(log_probs)
        log_median = np.median(log_probs)
        
        # 使用核密度估计找峰值
        from scipy.stats import gaussian_kde
        
        # 在对数域进行 KDE
        kde = gaussian_kde(log_probs)
        x_range = np.linspace(log_probs.min(), log_probs.max(), 1000)
        kde_values = kde(x_range)
        
        # 找峰值
        peaks, _ = find_peaks(kde_values, height=np.max(kde_values) * 0.1)
        
        if len(peaks) >= 2:
            # 有两个峰，取两个峰之间的谷
            sorted_peaks = peaks[np.argsort(kde_values[peaks])[-2:]]
            sorted_peaks = np.sort(sorted_peaks)
            
            # 找谷点
            valley_idx = sorted_peaks[0] + np.argmin(kde_values[sorted_peaks[0]:sorted_peaks[1]+1])
            threshold_log = x_range[valley_idx]
            threshold = np.exp(threshold_log) - 1e-10
        elif len(peaks) == 1:
            # 只有一个峰，使用统计阈值
            # 取均值 + 2 倍标准差
            threshold_log = log_mean + 2 * log_std
            threshold = np.exp(threshold_log) - 1e-10
        else:
            # 没有峰值，使用简单阈值
            threshold = np.percentile(probs, 95)
        
        return threshold, {
            'method': 'Log_Distribution',
            'log_mean': float(log_mean),
            'log_std': float(log_std),
            'log_median': float(log_median),
            'num_peaks': len(peaks)
        }
    
    def _method_quantile_ratio(self, probs):
        """
        方法8：分位数比值分析
        
        基于分布的分位数比值来识别异常
        """
        q50 = np.percentile(probs, 50)
        q75 = np.percentile(probs, 75)
        q90 = np.percentile(probs, 90)
        q95 = np.percentile(probs, 95)
        q99 = np.percentile(probs, 99)
        
        # 计算分位数比值
        # 对于高度偏斜的分布（正常节点集中在低分），这些比值会很大
        ratio_75_50 = q75 / (q50 + 1e-10)
        ratio_90_75 = q90 / (q75 + 1e-10)
        ratio_95_90 = q95 / (q90 + 1e-10)
        ratio_99_95 = q99 / (q95 + 1e-10)
        
        # 使用分位数间隔来确定阈值
        # 假设：正常节点分数 < q75，异常节点在 q95 以上
        # 阈值设置在 q75 和 q95 之间，偏向 q95
        
        # 分析分位数之间的分布
        # 计算每个分位数区间的密度
        total = len(probs)
        n_75_90 = np.sum((probs >= q75) & (probs < q90))
        n_90_95 = np.sum((probs >= q90) & (probs < q95))
        n_95_99 = np.sum((probs >= q95) & (probs < q99))
        n_above_99 = np.sum(probs >= q99)
        
        # 计算密度比
        density_75_90 = n_75_90 / (total * 0.15)
        density_90_95 = n_90_95 / (total * 0.05)
        density_95_99 = n_95_99 / (total * 0.04)
        density_above_99 = n_above_99 / (total * 0.01)
        
        # 阈值选择策略：
        # 如果高分数区的密度异常高，说明有僵尸网络聚集
        if density_above_99 > density_95_99 * 2:
            threshold = q99
        elif density_95_99 > density_90_95 * 2:
            threshold = q95
        elif density_90_95 > density_75_90 * 2:
            threshold = q90
        else:
            # 使用更保守的阈值
            threshold = q99
        
        # 验证：预测比例应该在合理范围内
        pred_ratio = np.mean(probs >= threshold)
        if pred_ratio < 0.005:
            # 太少，降低阈值
            threshold = q95
        elif pred_ratio > 0.1:
            # 太多，提高阈值
            threshold = q99
        
        return threshold, {
            'method': 'Quantile_Ratio',
            'q50': float(q50),
            'q75': float(q75),
            'q90': float(q90),
            'q95': float(q95),
            'q99': float(q99),
            'ratio_75_50': float(ratio_75_50),
            'ratio_90_75': float(ratio_90_75),
            'density_95_99': float(density_95_99),
            'density_above_99': float(density_above_99)
        }
    
    def _ensemble_threshold(self, thresholds, infos, probs):
        """
        综合多种方法的阈值选择
        
        策略：
        1. 过滤掉不合理的阈值
        2. 使用加权投票或中位数
        3. 考虑预测比例的合理性
        """
        # 过滤掉无效阈值
        valid_thresholds = []
        valid_methods = []
        
        for thresh, info in zip(thresholds, infos):
            if thresh > 0 and thresh < 1:
                pred_ratio = np.mean(probs >= thresh)
                # 合理的预测比例：0.1% - 20%
                if 0.001 <= pred_ratio <= 0.2:
                    valid_thresholds.append(thresh)
                    valid_methods.append(info['method'])
        
        if not valid_thresholds:
            # 如果没有合理阈值，使用默认
            return np.percentile(probs, 99), {'method': 'Default'}
        
        # 策略：使用中位数阈值
        threshold = np.median(valid_thresholds)
        
        # 验证
        pred_ratio = np.mean(probs >= threshold)
        
        # 如果预测比例太低或太高，调整
        if pred_ratio < 0.005:
            # 太保守，使用最小阈值
            threshold = min(valid_thresholds)
        elif pred_ratio > 0.1:
            # 太激进，使用最大阈值
            threshold = max(valid_thresholds)
        
        return threshold, {
            'method': 'Ensemble',
            'valid_methods': valid_methods,
            'valid_thresholds': [float(t) for t in valid_thresholds],
            'median_threshold': float(np.median(valid_thresholds)),
            'final_threshold': float(threshold),
            'predicted_ratio': float(pred_ratio)
        }
    
    def fit_predict(self, probs, y_true=None):
        """
        执行分类（纯无监督，不使用标签）
        
        Args:
            probs: 模型输出概率
            y_true: 真实标签（仅用于评估）
        
        Returns:
            result: 包含预测结果和中间信息的字典
        """
        probs = np.asarray(probs).flatten()
        n = len(probs)
        
        # 1. 分析分数分布的基本统计量
        mean_prob = np.mean(probs)
        std_prob = np.std(probs)
        median_prob = np.median(probs)
        q25 = np.percentile(probs, 25)
        q75 = np.percentile(probs, 75)
        q95 = np.percentile(probs, 95)
        q99 = np.percentile(probs, 99)
        
        # 对数域统计
        log_probs = self._log_transform(probs)
        log_mean = np.mean(log_probs)
        log_std = np.std(log_probs)
        
        # 2. 使用多种方法估计阈值
        threshold_kmeans, info_kmeans = self._method_kmeans(probs)
        threshold_gmm, info_gmm = self._method_gmm_elbow(probs)
        threshold_gap, info_gap = self._method_percentile_gap(probs)
        threshold_knee, info_knee = self._method_knee_point(probs)
        threshold_outlier, info_outlier = self._method_outlier_detection(probs)
        threshold_log, info_log = self._method_log_distribution(probs)
        threshold_quantile, info_quantile = self._method_quantile_ratio(probs)
        
        # 3. 综合选择阈值
        all_thresholds = [
            threshold_kmeans,
            threshold_gmm,
            threshold_gap,
            threshold_knee,
            threshold_outlier,
            threshold_log,
            threshold_quantile
        ]
        all_infos = [
            info_kmeans,
            info_gmm,
            info_gap,
            info_knee,
            info_outlier,
            info_log,
            info_quantile
        ]
        
        self.threshold, ensemble_info = self._ensemble_threshold(all_thresholds, all_infos, probs)
        self.method_used = ensemble_info
        
        # 4. 生成预测
        preds = (probs >= self.threshold).astype(int)
        
        # 5. 计算指标（仅用于评估）
        if y_true is not None:
            y_true = np.asarray(y_true).flatten()[:len(probs)]
            probs = probs[:len(y_true)]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, preds, average='binary', zero_division=0
            )
            auc = roc_auc_score(y_true, probs)
        else:
            precision = recall = f1 = auc = None
        
        result = {
            'preds': preds,
            'probs': probs,
            'threshold': float(self.threshold),
            'method': self.method_used.get('method', 'Ensemble'),
            'method_details': self.method_used,
            'num_predicted': int(preds.sum()),
            'predicted_ratio': float(preds.sum() / n),
            'statistics': {
                'mean': float(mean_prob),
                'std': float(std_prob),
                'median': float(median_prob),
                'q25': float(q25),
                'q75': float(q75),
                'q95': float(q95),
                'q99': float(q99),
                'log_mean': float(log_mean),
                'log_std': float(log_std)
            },
            'all_methods': {
                'kmeans': {'threshold': float(threshold_kmeans), 'info': info_kmeans},
                'gmm': {'threshold': float(threshold_gmm), 'info': info_gmm},
                'gap': {'threshold': float(threshold_gap), 'info': info_gap},
                'knee': {'threshold': float(threshold_knee), 'info': info_knee},
                'outlier': {'threshold': float(threshold_outlier), 'info': info_outlier},
                'log': {'threshold': float(threshold_log), 'info': info_log},
                'quantile': {'threshold': float(threshold_quantile), 'info': info_quantile}
            },
            'precision': float(precision) if precision is not None else None,
            'recall': float(recall) if recall is not None else None,
            'f1': float(f1) if f1 is not None else None,
            'auc': float(auc) if auc is not None else None,
        }
        
        return result


def compute_botnet_metrics_optimized(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """
    使用优化分类器计算指标
    """
    classifier = OptimizedBotnetClassifier()
    result = classifier.fit_predict(probs, y_true)
    
    return {
        'auc': result['auc'],
        'precision': result['precision'],
        'recall': result['recall'],
        'f1': result['f1'],
        'threshold': result['threshold'],
        'num_predicted': result['num_predicted'],
        'num_true': int(y_true.sum()),
        'method': 'Optimized_' + result['method'],
        'ratio': result['predicted_ratio'],
    }


if __name__ == "__main__":
    # 测试
    np.random.seed(42)
    
    print("="*70)
    print("优化分类器测试")
    print("="*70)
    
    # 加载场景12数据
    import torch
    from data_loader import CTU13Loader
    from improved_heterogeneous_graph import ImprovedHeterogeneousGraphBuilder
    from improved_model import ImprovedBotnetDetector
    from torch_geometric.loader import NeighborLoader
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Config] 使用设备: {device}")
    
    # 加载数据
    print("\n[Phase 1] 加载测试数据...")
    loader = CTU13Loader('/root/autodl-fs/CTU-13/CTU-13-Dataset')
    df = loader.load_data([12])
    
    # 构建图
    print("\n[Phase 2] 构建测试图...")
    builder = ImprovedHeterogeneousGraphBuilder()
    data, ip_map = builder.build(df, include_semantic=True, include_struct=True)
    
    # 获取标签
    bot_keywords = ['botnet', 'cc', 'c&c', 'neris', 'rbot', 'virut', 'menti', 'murlo', 'sogou']
    df['is_bot'] = df['label'].apply(lambda x: any(k in str(x).lower() for k in bot_keywords))
    bot_ips = set(df[df['is_bot']]['src_ip'].unique()) | set(df[df['is_bot']]['dst_ip'].unique())
    y = np.zeros(len(ip_map), dtype=np.float32)
    for ip, idx in ip_map.items():
        if ip in bot_ips:
            y[idx] = 1.0
    labels = torch.tensor(y)
    
    # 加载模型
    print("\n[Phase 3] 加载模型...")
    model = ImprovedBotnetDetector.load('improved_botnet_model.pth', device=device)
    model.eval()
    
    # 推理
    print("\n[Phase 4] 执行推理...")
    infer_loader = NeighborLoader(
        data, 
        num_neighbors=[15, 10], 
        batch_size=4096, 
        input_nodes='ip', 
        shuffle=False
    )
    
    all_logits = []
    with torch.no_grad():
        for batch in infer_loader:
            batch = batch.to(device)
            out = model(batch)[:batch['ip'].batch_size]
            all_logits.append(out.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    probs = torch.sigmoid(logits).squeeze().numpy()
    
    # 确保对齐
    y_true = labels.numpy()[:len(probs)]
    probs = probs[:len(y_true)]
    
    print(f"\n[数据概览]")
    print(f"  总节点数：{len(probs)}")
    print(f"  僵尸节点数：{y_true.sum()} ({y_true.sum()/len(probs)*100:.2f}%)")
    
    # 测试优化分类器
    print("\n[Phase 5] 测试优化分类器...")
    result = compute_botnet_metrics_optimized(y_true, probs)
    
    print(f"\n评估结果（优化分类器）")
    print("="*60)
    print(f"  AUC:       {result['auc']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1-Score:  {result['f1']:.4f}")
    print(f"  Threshold: {result['threshold']:.6f}")
    print("="*60)
    print(f"  预测为僵尸网络的节点：{result['num_predicted']}")
    print(f"  实际僵尸网络节点：{result['num_true']}")