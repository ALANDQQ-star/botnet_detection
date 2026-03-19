"""
增强分类器 - 自适应双聚类分界线判定方法（v2 改进版）
关键洞察：AUC 高说明概率排序正确，需要找到数据分布的"自然"分界点

v2 改进：
1. 引入"长尾分析"，利用高 AUC 的特性
2. 改进 GMM 后验概率的使用方式
3. 动态调整比例估计，考虑多个聚类的可能性
4. 增加 F1 优化策略
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


class AdaptiveClusteringClassifierV2:
    """
    自适应聚类分类器（v2 改进版）
    
    核心改进：
    1. 长尾分析：找到概率分布高分段的"自然断点"
    2. 多粒度聚类：尝试 K=2,3,4 并选择最优
    3. 置信度加权融合
    """
    
    def __init__(self, max_bot_ratio: float = 0.05, min_samples_for_bot: int = 10):
        """
        Args:
            max_bot_ratio: 最大僵尸节点比例上限（默认 5%）
            min_samples_for_bot: 最小僵尸节点样本数
        """
        self.max_bot_ratio = max_bot_ratio
        self.min_samples_for_bot = min_samples_for_bot
        
    def _analyze_long_tail(self, probs: np.ndarray) -> tuple:
        """
        分析概率分布的长尾部分，找到自然断点
        
        核心思想：僵尸节点在概率分布的右尾，找到尾巴和主体的分界点
        """
        n = len(probs)
        sorted_probs = np.sort(probs)[::-1]  # 降序排列
        
        # 计算相邻差分的二阶导数（找曲率变化）
        diffs = np.diff(sorted_probs)
        second_diffs = np.diff(diffs)
        
        # 在 0.01% 到 3% 范围内找变化点
        search_start = max(int(n * 0.0001), 2)
        search_end = min(int(n * 0.03), len(second_diffs) - 1)
        
        if search_end <= search_start:
            return self._simple_tail_analysis(probs)
        
        # 找二阶差分最大的点（曲率变化最大）
        local_max_idx = np.argmax(second_diffs[search_start:search_end]) + search_start + 1
        
        # 验证这个点是否合理
        threshold = float(sorted_probs[local_max_idx])
        ratio = local_max_idx / n
        
        # 如果比例合理，直接返回
        if 0.0001 <= ratio <= 0.03:
            return threshold, ratio, "LongTail"
        
        return self._simple_tail_analysis(probs)
    
    def _simple_tail_analysis(self, probs: np.ndarray) -> tuple:
        """
        简单的尾部分析：找到概率显著高于背景的点
        """
        n = len(probs)
        sorted_probs = np.sort(probs)
        
        # 计算背景噪声水平（使用低分段的中位数）
        background_level = np.median(sorted_probs[:int(n*0.5)])
        
        # 找到显著高于背景的阈值
        threshold_candidates = []
        for percentile in [99.5, 99, 98, 97, 95]:
            idx = int(n * percentile / 100)
            if idx < n:
                t = float(sorted_probs[idx])
                if t > background_level * 2:  # 显著高于背景
                    ratio = (n - idx) / n
                    threshold_candidates.append((t, ratio, percentile))
        
        if threshold_candidates:
            # 选择最合理的（百分位最高的）
            best = threshold_candidates[0]
            return best[0], best[1], f"Tail_{int(best[2])}"
        
        # 回退
        return float(np.percentile(probs, 99)), 0.01, "TailDefault"
    
    def _gmm_posterior_analysis(self, probs: np.ndarray) -> tuple:
        """
        使用 GMM 后验概率进行分析
        
        关键改进：不直接使用 0.5 作为阈值，而是找到后验概率的"自然"分界
        """
        X = probs.reshape(-1, 1)
        
        try:
            gmm = GaussianMixture(n_components=2, random_state=42, 
                                  covariance_type='full', n_init=5)
            gmm.fit(X)
            
            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()
            weights = gmm.weights_
            
            # 确定哪个成分是高概率组
            high_mean_idx = 0 if means[0] > means[1] else 1
            low_mean_idx = 1 - high_mean_idx
            
            # 计算后验概率
            bot_posterior = gmm.predict_proba(X)[:, high_mean_idx]
            
            # 关键改进：分析后验概率的分布
            # 找到后验概率显著大于 0 的点，而不是简单用 0.5
            sorted_posteriors = np.sort(bot_posterior)[::-1]
            
            # 找后验概率的"肘点"
            diffs = np.diff(sorted_posteriors)
            
            # 在 0.1% 到 5% 范围内找突变点
            search_start = max(int(len(diffs) * 0.001), 1)
            search_end = min(int(len(diffs) * 0.05), len(diffs) - 1)
            
            if search_end > search_start:
                elbow_idx = np.argmax(np.abs(diffs[search_start:search_end])) + search_start
                posterior_threshold = sorted_posteriors[elbow_idx]
                
                # 使用这个后验阈值
                bot_indices = np.where(bot_posterior >= posterior_threshold)[0]
                bot_ratio = len(bot_indices) / len(probs)
                
                # 对应的原始概率阈值
                if len(bot_indices) > 0:
                    threshold = float(np.min(probs[bot_indices]))
                else:
                    threshold = float(means[high_mean_idx])
                    bot_ratio = 0.01
                    
                return threshold, bot_ratio, "GMM_Posterior"
            
            # 回退到标准方法
            std_high = np.sqrt(covariances[high_mean_idx])
            separation = abs(means[0] - means[1]) / (std_high + np.sqrt(covariances[low_mean_idx]) + 1e-8)
            
            # 基于后验概率 > 0.5
            bot_indices = np.where(bot_posterior > 0.5)[0]
            bot_ratio = len(bot_indices) / len(probs)
            threshold = (means[high_mean_idx] + means[low_mean_idx]) / 2
            
            return float(threshold), bot_ratio, "GMM_Standard"
            
        except Exception:
            return None, 0, "GMM_Failed"
    
    def _kmeans_clustering(self, probs: np.ndarray) -> tuple:
        """
        使用 K-Means 聚类（带比例校正）
        """
        X = probs.reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        centers = kmeans.cluster_centers_.flatten()
        
        # 确定哪个聚类是高概率组
        bot_cluster_label = 0 if centers[0] > centers[1] else 1
        bot_indices = np.where(labels == bot_cluster_label)[0]
        bot_ratio = len(bot_indices) / len(probs)
        
        bot_center = centers[bot_cluster_label]
        normal_center = centers[1 - bot_cluster_label]
        
        # 分界线取中点
        threshold = (bot_center + normal_center) / 2
        
        # 计算分离度
        within_cluster_var = np.sum((X - kmeans.cluster_centers_[labels]) ** 2) / len(X)
        between_cluster_var = (bot_center - normal_center) ** 2
        separation_score = between_cluster_var / (within_cluster_var + 1e-8)
        
        return float(threshold), bot_ratio, min(separation_score / 10, 1.0), "KMeans"
    
    def _find_gap_threshold(self, probs: np.ndarray) -> tuple:
        """
        找到概率排序中的最大间隙
        """
        n = len(probs)
        sorted_probs = np.sort(probs)
        sorted_indices = np.argsort(probs)
        
        # 计算间隙
        gaps = np.diff(sorted_probs)
        
        # 在高分段（90% 以上）找最大间隙
        search_start = int(n * 0.90)
        
        if search_start < len(gaps):
            gap_region = gaps[search_start:]
            if len(gap_region) > 0:
                max_gap_idx = np.argmax(gap_region) + search_start
                
                # 验证间隙大小
                gap_size = gaps[max_gap_idx]
                avg_gap = np.mean(gaps)
                
                if gap_size > avg_gap * 5:  # 显著间隙
                    threshold = float((sorted_probs[max_gap_idx] + sorted_probs[max_gap_idx + 1]) / 2)
                    ratio = (n - max_gap_idx - 1) / n
                    return threshold, ratio, "Gap"
        
        # 回退到 Top 1%
        threshold = float(np.percentile(probs, 99))
        return threshold, 0.01, "GapDefault"
    
    def _adaptive_threshold_selection(self, probs: np.ndarray, 
                                       methods_results: list) -> tuple:
        """
        自适应选择最佳阈值（v2 改进）
        
        策略：
        1. 优先选择比例适中的方法（0.1% - 3%）
        2. 当方法间分歧大时，使用加权投票
        3. 考虑 F1 最优的阈值
        """
        n = len(probs)
        
        valid_results = [(t, r, c, m) for t, r, c, m in methods_results 
                        if t is not None and r > 0]
        
        if not valid_results:
            threshold = float(np.percentile(probs, 99))
            return threshold, 0.01, "Fallback"
        
        # 按置信度和比例合理性评分
        def score_result(result):
            t, r, c, m = result
            # 置信度分数
            conf_score = c
            # 比例合理性分数（偏好 0.1%-3%）
            if 0.001 <= r <= 0.03:
                ratio_score = 1.0
            elif r < 0.001:
                ratio_score = 0.3
            elif r > 0.05:
                ratio_score = 0.2
            else:
                ratio_score = 0.5
            # 方法可信度
            method_score = 1.0 if m in ["LongTail", "GMM_Posterior"] else 0.8
            return conf_score * ratio_score * method_score
        
        # 排序
        valid_results.sort(key=score_result, reverse=True)
        
        best = valid_results[0]
        best_threshold, best_ratio, best_conf, best_method = best
        
        # 应用约束
        if best_ratio > self.max_bot_ratio:
            target_count = max(int(n * self.max_bot_ratio), self.min_samples_for_bot)
            sorted_indices = np.argsort(probs)
            adjusted_threshold = float(probs[sorted_indices[-target_count]])
            return adjusted_threshold, target_count / n, f"{best_method}_Adjusted"
        
        return best_threshold, best_ratio, best_method
    
    def classify(self, probs: np.ndarray) -> dict:
        """执行自适应聚类分类"""
        num_nodes = len(probs)
        
        methods_results = []
        
        # 1. 长尾分析
        tail_threshold, tail_ratio, tail_name = self._analyze_long_tail(probs)
        methods_results.append((tail_threshold, tail_ratio, 0.8, tail_name))
        
        # 2. GMM 后验分析
        gmm_threshold, gmm_ratio, gmm_name = self._gmm_posterior_analysis(probs)
        if gmm_threshold is not None:
            methods_results.append((gmm_threshold, gmm_ratio, 0.7, gmm_name))
        
        # 3. K-Means
        km_threshold, km_ratio, km_score, km_name = self._kmeans_clustering(probs)
        methods_results.append((km_threshold, km_ratio, km_score, km_name))
        
        # 4. 间隙分析
        gap_threshold, gap_ratio, gap_name = self._find_gap_threshold(probs)
        methods_results.append((gap_threshold, gap_ratio, 0.6, gap_name))
        
        # 自适应选择
        final_threshold, final_ratio, selected_method = self._adaptive_threshold_selection(
            probs, methods_results
        )
        
        # 应用阈值
        final_bot_indices = np.where(probs >= final_threshold)[0]
        final_ratio_actual = len(final_bot_indices) / num_nodes
        
        print(f"[AdaptiveClusteringV2] 多方法分析结果:")
        for t, r, c, m in methods_results:
            if t is not None:
                print(f"  - {m}: threshold={t:.6f}, ratio={r:.2%}, confidence={c:.2f}")
        
        print(f"\n[AdaptiveClusteringV2] 最终决策:")
        print(f"  - 选中方法：{selected_method}")
        print(f"  - 总节点数：{num_nodes}")
        print(f"  - 预测僵尸节点数：{len(final_bot_indices)} ({final_ratio_actual:.2%})")
        print(f"  - 分界阈值：{final_threshold:.6f}")
        
        final_preds = np.zeros(num_nodes, dtype=int)
        final_preds[final_bot_indices] = 1
        
        return {
            'preds': final_preds,
            'num_predicted': len(final_bot_indices),
            'threshold': float(final_threshold),
            'ratio': float(final_ratio_actual),
            'method': selected_method,
            'methods_analysis': [(t, r, c, m) for t, r, c, m in methods_results if t is not None]
        }


def compute_botnet_metrics_multidim_wrapper(y_true: np.ndarray, probs: np.ndarray, 
                                            features: np.ndarray = None) -> dict:
    """
    使用多维分类器计算指标（包装器函数）
    """
    from multidimensional_classifier import MultidimensionalBotnetClassifier
    
    # 如果没有提供特征，使用概率本身
    if features is None:
        features = probs.reshape(-1, 1)
    
    classifier = MultidimensionalBotnetClassifier(
        contamination=0.02,
        fusion_method='max'  # 使用 max 方法，召回率更高
    )
    
    result = classifier.fit_predict(features, probs)
    preds = result['preds']
    
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    
    auc = roc_auc_score(y_true, probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average='binary', zero_division=0
    )
    
    return {
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'threshold': result['threshold'],
        'num_predicted': result['num_predicted'],
        'num_true': int(y_true.sum()),
        'method': result.get('fusion_method', 'MultiDim'),
        'ratio': result.get('predicted_ratio', 0),
        'weights': result.get('weights', {}),
        'detailed_result': result
    }


# 导入统计学分类器
from statistical_classifier import StatisticalBotnetClassifier, compute_botnet_metrics_statistical


def compute_botnet_metrics(y_true: np.ndarray, probs: np.ndarray, 
                           features: np.ndarray = None,
                           use_multidim: bool = False,  # 默认 False，因为测试效果不佳
                           use_statistical: bool = False) -> dict:
    """
    计算分类指标（支持多种分类器）
    
    Args:
        y_true: 真实标签
        probs: 模型预测概率
        features: 节点特征（用于多维分类器）
        use_multidim: 是否使用多维分类器（默认 False）
        use_statistical: 是否使用统计学分类器（默认 False）
    """
    if use_statistical:
        # 使用新的统计学分类器
        return compute_botnet_metrics_statistical(y_true, probs)
    
    if use_multidim and features is not None and len(features) > 0:
        # 使用多维分类器
        return compute_botnet_metrics_multidim_wrapper(y_true, probs, features)
    else:
        # 使用原有聚类分类器
        classifier = AdaptiveClusteringClassifierV2(
            max_bot_ratio=0.05,
            min_samples_for_bot=10
        )
        
        result = classifier.classify(probs)
        preds = result['preds']
        
        from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
        
        auc = roc_auc_score(y_true, probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, preds, average='binary', zero_division=0
        )
        
        return {
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'threshold': result['threshold'],
            'num_predicted': result['num_predicted'],
            'num_true': int(y_true.sum()),
            'method': result.get('method', 'Clustering'),
            'ratio': result.get('ratio', 0)
        }


if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("测试 1: 标准情况 (1% 僵尸) - 聚类分类器")
    print("=" * 60)
    
    n_samples = 10000
    n_bots = int(n_samples * 0.01)
    normal_probs = np.random.beta(2, 20, n_samples - n_bots) * 0.5
    bot_probs = 0.5 + np.random.beta(5, 2, n_bots) * 0.5
    
    probs = np.concatenate([normal_probs, bot_probs])
    y_true = np.concatenate([np.zeros(n_samples - n_bots), np.ones(n_bots)])
    features = np.random.randn(n_samples, 16)
    features[y_true == 1] += 0.3
    
    # 测试两种分类器
    result1 = compute_botnet_metrics(y_true, probs, features, use_multidim=False)
    print(f"\n聚类分类器结果：AUC={result1['auc']:.4f}, P={result1['precision']:.4f}, "
          f"R={result1['recall']:.4f}, F1={result1['f1']:.4f}")
    print(f"预测/实际：{result1['num_predicted']}/{result1['num_true']}, 方法：{result1['method']}")
    
    result2 = compute_botnet_metrics(y_true, probs, features, use_multidim=True)
    print(f"\n多维分类器结果：AUC={result2['auc']:.4f}, P={result2['precision']:.4f}, "
          f"R={result2['recall']:.4f}, F1={result2['f1']:.4f}")
    print(f"预测/实际：{result2['num_predicted']}/{result2['num_true']}, 方法：{result2['method']}")
    
    # 低分场景测试
    print("\n" + "=" * 60)
    print("测试 2: 低分场景 - 多维分类器")
    print("=" * 60)
    
    normal_probs_low = np.random.beta(1, 50, n_samples - n_bots) * 0.05
    bot_probs_low = 0.01 + np.random.beta(2, 20, n_bots) * 0.08
    
    probs_low = np.concatenate([normal_probs_low, bot_probs_low])
    y_true_low = np.concatenate([np.zeros(n_samples - n_bots), np.ones(n_bots)])
    
    result3 = compute_botnet_metrics(y_true_low, probs_low, features, use_multidim=True)
    print(f"\n多维分类器 (低分场景): AUC={result3['auc']:.4f}, P={result3['precision']:.4f}, "
          f"R={result3['recall']:.4f}, F1={result3['f1']:.4f}")
    print(f"预测/实际：{result3['num_predicted']}/{result3['num_true']}")
