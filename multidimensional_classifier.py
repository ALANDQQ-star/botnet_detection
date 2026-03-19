"""
多维打分分类器 - 针对低分场景优化的异常检测系统

核心思想：
1. 利用 AUC 高的特性（排序正确），在保持排序不变的前提下重新校准分数
2. 多视图异常检测：从统计、语义、结构三个维度独立打分
3. 基于局部密度的异常检测：僵尸节点在特征空间中形成紧密簇
4. 自适应分数校准：使用温度缩放和百分位映射

关键创新：
- 不偷看标签，纯无监督/自监督方式
- 结合多种异常检测算法的集成决策
- 针对"分数普遍偏低"场景的专门优化
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.spatial.distance import cdist, mahalanobis
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


class MultiDimensionalScorer:
    """
    多维打分器：从多个视角评估节点异常程度
    """
    
    def __init__(self, contamination=0.02):
        """
        Args:
            contamination: 预期的异常比例（默认 2%）
        """
        self.contamination = contamination
        self.scalers = {}
        self.pca = None
        self.gmm = None
        self.lof = None
        self.is_fitted = False
        
    def fit(self, features, probs):
        """
        拟合多维打分器
        
        Args:
            features: 原始特征 [n_samples, n_features]
            probs: 模型输出的一维概率 [n_samples]
        """
        n_samples = len(probs)
        
        # 1. 特征预处理
        self.feature_scaler = RobustScaler()
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # 2. PCA 降维用于某些检测器
        n_components = min(32, features_scaled.shape[1], n_samples // 10)
        self.pca = PCA(n_components=n_components, svd_solver='full' if n_samples > 100 else 'auto')
        features_pca = self.pca.fit_transform(features_scaled)
        
        # 3. 拟合 GMM（用于后验概率）
        # 使用更稳健的配置
        n_gmm = min(4, max(2, n_samples // 100))  # 动态调整组件数
        self.gmm = GaussianMixture(
            n_components=n_gmm, 
            covariance_type='diag',  # 使用对角协方差，更稳定
            reg_covar=1e-4,  # 增加正则化
            n_init=3,
            max_iter=200,
            tol=1e-3,
            random_state=42
        )
        try:
            self.gmm.fit(features_scaled.astype(np.float64))  # 使用 float64 提高数值稳定性
        except Exception as e:
            print(f"[MultiDim] GMM 拟合失败：{e}，使用简化模式")
            self.gmm = None
        
        # 4. 拟合 LOF（局部异常因子）
        lof_neighbors = min(20, max(5, n_samples // 500))
        self.lof = LocalOutlierFactor(
            n_neighbors=lof_neighbors,
            novelty=False,
            contamination=self.contamination,
            n_jobs=-1
        )
        self.lof.fit_predict(features_scaled)  # 只需拟合
        self.lof_scores_ = -self.lof.negative_outlier_factor_  # 转为正向分数
        
        # 5. 计算马氏距离
        self.mean_vec = np.mean(features_scaled, axis=0)
        self.cov_matrix = np.cov(features_scaled, rowvar=False)
        try:
            self.cov_inv = np.linalg.inv(self.cov_matrix + 1e-6 * np.eye(self.cov_matrix.shape[0]))
        except:
            self.cov_inv = np.eye(self.cov_matrix.shape[0])
        
        # 6. 基于原始概率的排序信息
        self.prob_ranks = np.argsort(np.argsort(probs))  # 秩次
        self.prob_percentiles = stats.percentileofscore(probs, probs, kind='mean') / 100
        
        self.is_fitted = True
        self.features_scaled = features_scaled
        self.features_pca = features_pca
        
        return self
    
    def compute_multidimensional_scores(self, features, probs):
        """
        计算多维异常分数
        
        返回多个独立维度的分数，用于后续融合
        """
        if not self.is_fitted:
            self.fit(features, probs)
        
        n_samples = len(probs)
        
        # 确保输入是 numpy 数组
        if isinstance(features, np.matrix):
            features = np.asarray(features)
        
        # 特征缩放
        features_scaled = self.feature_scaler.transform(features)
        
        scores = {}
        
        # 1. GMM 后验概率分数（如果 GMM 可用）
        if self.gmm is not None:
            gmm_posterior = self.gmm.predict_proba(features_scaled.astype(np.float64))
            # 找到最可能的组件，计算其"异常程度"
            component_probs = gmm_posterior.max(axis=1)
            # 低概率组件成员 = 高异常
            scores['gmm_posterior'] = 1 - component_probs
            
            # 2. GMM 密度分数（基于对数似然）
            log_likelihood = self.gmm.score_samples(features_scaled.astype(np.float64))
            # 归一化到 [0, 1]
            ll_min, ll_max = log_likelihood.min(), log_likelihood.max()
            if ll_max > ll_min:
                scores['gmm_density'] = 1 - (log_likelihood - ll_min) / (ll_max - ll_min)
            else:
                scores['gmm_density'] = np.zeros(n_samples)
            
            # 7. 组件间距离分数（检测处于簇边界的点）
            if gmm_posterior.shape[1] >= 2:
                # 计算每个点到各组件中心的"均衡度"
                component_entropy = -np.sum(gmm_posterior * np.log(gmm_posterior + 1e-10), axis=1)
                max_entropy = np.log(gmm_posterior.shape[1])
                scores['component_entropy'] = component_entropy / max_entropy
            else:
                scores['component_entropy'] = np.zeros(n_samples)
        else:
            # GMM 不可用时，使用其他分数替代
            scores['gmm_posterior'] = np.zeros(n_samples)
            scores['gmm_density'] = np.zeros(n_samples)
            scores['component_entropy'] = np.zeros(n_samples)
        
        # 3. LOF 分数（已拟合）
        scores['lof'] = self.lof_scores_
        # 归一化
        if scores['lof'].max() > scores['lof'].min():
            scores['lof'] = (scores['lof'] - scores['lof'].min()) / (scores['lof'].max() - scores['lof'].min())
        
        # 4. 马氏距离分数 - 使用向量化计算加速
        # 原始实现使用循环，对于大数据集非常慢
        # 使用向量化公式：mahal^2 = (x-mu)^T @ Sigma^-1 @ (x-mu)
        diff = features_scaled - self.mean_vec
        # 使用 Cholesky 分解加速计算
        try:
            L = np.linalg.cholesky(self.cov_inv + 1e-8 * np.eye(self.cov_inv.shape[0]))
            mahal_scores = np.sqrt(np.sum((diff @ L)**2, axis=1))
        except:
            # Cholesky 失败时回退到简单欧氏距离
            mahal_scores = np.sqrt(np.sum(diff**2, axis=1))
        
        # 使用卡方分布转换
        scores['mahalanobis'] = 1 - stats.chi2.cdf(mahal_scores**2, df=min(features.shape[1], 32))
        
        # 5. 概率秩次分数（保持 AUC 排序）
        scores['prob_rank'] = self.prob_ranks / n_samples
        
        # 6. 概率百分位分数
        scores['prob_percentile'] = self.prob_percentiles
        
        return scores
    
    def fuse_scores(self, scores_dict, method='adaptive'):
        """
        融合多维分数
        
        Args:
            scores_dict: 各维度分数字典
            method: 融合方法 ('weighted', 'adaptive', 'max', 'mean')
        
        Returns:
            final_scores: 融合后的异常分数
            weights: 各维度权重
        """
        n_samples = len(list(scores_dict.values())[0])
        
        # 归一化所有分数到 [0, 1]
        normalized = {}
        for name, s in scores_dict.items():
            s = np.asarray(s).flatten()
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                normalized[name] = (s - s_min) / (s_max - s_min)
            else:
                normalized[name] = np.zeros(n_samples)
        
        if method == 'max':
            # 取最大值（最激进）
            final = np.maximum.reduce(list(normalized.values()))
            weights = {k: 1.0 for k in normalized.keys()}
            
        elif method == 'mean':
            # 简单平均
            final = np.mean(list(normalized.values()), axis=0)
            weights = {k: 1.0 / len(normalized) for k in normalized.keys()}
            
        elif method == 'weighted':
            # 基于信息熵的加权
            weights = {}
            for name, s in normalized.items():
                # 高分散度 = 高权重
                std = np.std(s)
                weights[name] = std + 0.1  # 平滑
            total_w = sum(weights.values())
            weights = {k: v / total_w for k, v in weights.items()}
            final = np.sum([normalized[k] * weights[k] for k in normalized.keys()], axis=0)
            
        elif method == 'adaptive':
            # 自适应融合：根据分数分布动态调整
            # 核心思想：如果概率分数和其他维度一致，则提高其权重
            
            prob_rank = normalized.get('prob_rank', np.zeros(n_samples))
            prob_pct = normalized.get('prob_percentile', np.zeros(n_samples))
            
            # 计算其他维度的平均分数
            other_scores = [v for k, v in normalized.items() if k not in ['prob_rank', 'prob_percentile']]
            if other_scores:
                other_mean = np.mean(other_scores, axis=0)
                
                # 计算一致性
                correlation = np.corrcoef(prob_rank.flatten(), other_mean.flatten())[0, 1]
                
                if np.isnan(correlation):
                    correlation = 0.5
                
                # 根据一致性调整权重
                if correlation > 0.3:
                    # 高一致性：信任概率排序
                    w_prob = 0.4
                    w_other = 0.6 / len(other_scores)
                    weights = {'prob_rank': w_prob, 'prob_percentile': w_prob}
                    weights.update({k: w_other for k in normalized.keys() if k not in ['prob_rank', 'prob_percentile']})
                else:
                    # 低一致性：更信任结构异常检测
                    w_prob = 0.1
                    w_other = 0.9 / len(other_scores)
                    weights = {'prob_rank': w_prob, 'prob_percentile': w_prob}
                    weights.update({k: w_other for k in normalized.keys() if k not in ['prob_rank', 'prob_percentile']})
                
                final = np.sum([normalized[k] * weights[k] for k in normalized.keys()], axis=0)
            else:
                final = np.mean([prob_rank, prob_pct], axis=0)
                weights = {'prob_rank': 0.5, 'prob_percentile': 0.5}
        else:
            final = np.mean(list(normalized.values()), axis=0)
            weights = {k: 1.0 / len(normalized) for k in normalized.keys()}
        
        return final, weights


class AdaptiveThresholdSelector:
    """
    自适应阈值选择器
    
    根据分数分布自动选择最优阈值
    """
    
    def __init__(self, expected_bot_ratio_range=(0.001, 0.05)):
        self.expected_min, self.expected_max = expected_bot_ratio_range
        
    def select_threshold(self, scores, method='multi'):
        """
        选择最优阈值
        
        Args:
            scores: 融合后的异常分数
            method: 选择方法
        
        Returns:
            threshold: 选择的阈值
            predicted_ratio: 预测的异常比例
        """
        n = len(scores)
        sorted_scores = np.sort(scores)[::-1]  # 降序
        
        if method == 'gap':
            # 在高分段找最大间隙
            top_n = int(n * self.expected_max)
            if top_n < 2:
                top_n = 2
            
            top_scores = sorted_scores[:top_n]
            gaps = np.diff(top_scores)
            
            # 找最大间隙
            max_gap_idx = np.argmax(np.abs(gaps))
            threshold = (top_scores[max_gap_idx] + top_scores[max_gap_idx + 1]) / 2
            
        elif method == 'percentile':
            # 使用百分位数
            target_ratio = np.sqrt(self.expected_min * self.expected_max)  # 几何平均
            threshold = float(np.percentile(scores, 100 * (1 - target_ratio)))
            
        elif method == 'knee':
            # 寻找"肘点"
            # 计算累积和的曲率
            cumsum = np.cumsum(sorted_scores)
            cumsum_norm = cumsum / cumsum[-1]
            x = np.arange(len(cumsum_norm))
            
            # 找曲率最大点
            if len(x) > 2:
                # 二阶导数
                dy = np.diff(cumsum_norm)
                d2y = np.diff(dy)
                
                # 在期望范围内找最大曲率
                search_start = max(0, int(len(d2y) * 0.01))
                search_end = min(len(d2y), int(len(d2y) * 0.1))
                
                if search_end > search_start:
                    knee_idx = np.argmax(np.abs(d2y[search_start:search_end])) + search_start
                    threshold = float(sorted_scores[knee_idx])
                else:
                    threshold = float(np.percentile(scores, 100 * (1 - self.expected_max)))
            else:
                threshold = float(np.percentile(scores, 100 * (1 - self.expected_max)))
                
        elif method == 'multi':
            # 多方法集成
            thresholds = []
            
            # Gap 方法
            top_n = max(2, int(n * self.expected_max))
            top_scores = sorted_scores[:top_n]
            gaps = np.diff(top_scores)
            max_gap_idx = np.argmax(np.abs(gaps))
            thresholds.append((top_scores[max_gap_idx] + top_scores[max_gap_idx + 1]) / 2)
            
            # 百分位方法
            target_ratio = np.sqrt(self.expected_min * self.expected_max)
            thresholds.append(float(np.percentile(scores, 100 * (1 - target_ratio))))
            
            # 取中位数
            threshold = float(np.median(thresholds))
            
        else:  # default
            threshold = float(np.percentile(scores, 100 * (1 - self.expected_max)))
        
        # 计算预测比例
        predicted_bots = (scores >= threshold).sum()
        predicted_ratio = predicted_bots / n
        
        return threshold, predicted_ratio


class MultidimensionalBotnetClassifier:
    """
    多维僵尸网络分类器
    
    整合：
    1. 原始模型概率输出
    2. 多维异常检测分数
    3. 自适应阈值选择
    
    核心改进：
    - 使用 max 融合方法：捕获所有可能的异常信号
    - 降低污染率假设：适应低比例场景
    """
    
    def __init__(self, contamination=0.01, fusion_method='max', target_precision=0.0):
        self.contamination = contamination
        self.fusion_method = fusion_method
        self.target_precision = target_precision  # 0 表示不使用有监督阈值
        self.scorer = MultiDimensionalScorer(contamination=contamination)
        self.threshold_selector = AdaptiveThresholdSelector(
            expected_bot_ratio_range=(0.0005, 0.02)  # 更宽的比率范围
        )
        
    def fit_predict(self, features, probs, y_true=None):
        """
        执行多维打分和预测
        
        Args:
            features: 节点特征 [n_nodes, n_features]
            probs: 模型输出的概率 [n_nodes]
            y_true: 真实标签（可选，用于优化阈值）
        
        Returns:
            result: 包含预测结果和中间信息的字典
        """
        n_nodes = len(probs)
        
        # 1. 计算多维分数
        multidim_scores = self.scorer.compute_multidimensional_scores(features, probs)
        
        # 2. 融合分数
        fused_scores, weights = self.scorer.fuse_scores(multidim_scores, method=self.fusion_method)
        
        # 3. 分数校准（针对低分场景优化）
        calibrated_scores = self._calibrate_scores(fused_scores, probs)
        
        # 4. 选择阈值 - 改进版本
        if y_true is not None and self.target_precision > 0:
            # 如果有真实标签，使用精确率导向的阈值选择
            threshold, predicted_ratio = self._select_threshold_precision_oriented(
                calibrated_scores, y_true, target_precision=self.target_precision
            )
        else:
            # 无监督方式
            threshold, predicted_ratio = self.threshold_selector.select_threshold(
                calibrated_scores, method='multi'
            )
        
        # 5. 生成预测
        preds = (calibrated_scores >= threshold).astype(int)
        
        # 6. 诊断信息
        bot_indices = np.where(preds == 1)[0]
        
        result = {
            'preds': preds,
            'scores': calibrated_scores,
            'fused_scores': fused_scores,
            'multidim_scores': multidim_scores,
            'weights': weights,
            'threshold': float(threshold),
            'num_predicted': int(preds.sum()),
            'predicted_ratio': float(predicted_ratio),
            'bot_indices': bot_indices.tolist(),
            'fusion_method': self.fusion_method,
            'calibration_applied': True
        }
        
        return result
    
    def _select_threshold_precision_oriented(self, scores, y_true, target_precision=0.5):
        """
        以提高精确率为目标的阈值选择
        
        策略：找到能达到目标精确率的最低阈值
        """
        n = len(scores)
        n_bots = y_true.sum()
        
        if n_bots == 0:
            return float(scores.max()), 0.0
        
        # 按分数降序排序
        sorted_indices = np.argsort(scores)[::-1]
        sorted_y_true = y_true[sorted_indices]
        sorted_scores = scores[sorted_indices]
        
        # 计算累积精确率
        cumsum_tp = np.cumsum(sorted_y_true)
        cumsum_pred = np.arange(1, n + 1)
        precision_curve = cumsum_tp / cumsum_pred
        
        # 找到能达到目标精确率的位置
        valid_mask = precision_curve >= target_precision
        if not valid_mask.any():
            # 如果达不到目标，使用最高阈值
            return float(sorted_scores[0]), 1.0 / n
        
        # 找到最远的有效位置（预测最多但仍满足精确率）
        best_idx = np.where(valid_mask)[0][-1]
        
        # 设置阈值为该位置的分数
        threshold = float(sorted_scores[best_idx])
        predicted_ratio = (best_idx + 1) / n
        
        return threshold, predicted_ratio
    
    def _calibrate_scores(self, scores, probs):
        """
        分数校准 - 优化版本
        
        策略：
        1. 保留原始排序（保持 AUC）
        2. 使用快速百分位映射（向量化）
        3. 结合原始概率和多维异常分数
        """
        n = len(scores)
        
        # 使用 argsort 的快速百分位计算（向量化，无循环）
        # prob_percentiles = stats.percentileofscore(probs, probs, kind='mean') / 100
        # 快速实现：使用排序索引直接计算百分位
        prob_ranks = np.argsort(np.argsort(probs))
        prob_percentiles = prob_ranks / n
        
        # 多维分数的百分位 - 同样使用快速方法
        score_ranks = np.argsort(np.argsort(scores))
        score_percentiles = score_ranks / n
        
        # 融合：原始概率百分位 + 多维异常百分位
        # 使用加权平均
        calibrated = 0.5 * prob_percentiles + 0.5 * score_percentiles
        
        # 增强高分区域：对高分进行指数放大
        # 这使得真正的异常点分数更高
        calibrated = np.power(calibrated, 0.8)  # 轻微压缩低分，保持高分
        
        return calibrated


def compute_botnet_metrics_multidim(y_true, probs, features, 
                                     contamination=0.02,
                                     fusion_method='max'):  # 改为 max 方法，召回率更高
    """
    使用多维分类器计算僵尸网络检测指标
    
    Args:
        y_true: 真实标签
        probs: 模型输出概率
        features: 节点特征
        contamination: 预期异常比例
        fusion_method: 融合方法
    
    Returns:
        metrics: 包含各种评估指标的字典
    """
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    
    classifier = MultidimensionalBotnetClassifier(
        contamination=contamination,
        fusion_method=fusion_method
    )
    
    result = classifier.fit_predict(features, probs)
    preds = result['preds']
    
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
        'method': f'MultiDim_{fusion_method}',
        'ratio': result['predicted_ratio'],
        'weights': result['weights'],
        'detailed_result': result
    }


if __name__ == "__main__":
    # 测试
    np.random.seed(42)
    
    print("="*60)
    print("多维分类器测试")
    print("="*60)
    
    # 模拟数据：低分场景
    n_samples = 10000
    n_bots = int(n_samples * 0.02)
    
    # 正常节点：分数集中在 0 附近
    normal_probs = np.random.beta(1, 50, n_samples - n_bots) * 0.1
    # 僵尸节点：分数略高但仍然很低
    bot_probs = 0.02 + np.random.beta(2, 20, n_bots) * 0.15
    
    probs = np.concatenate([normal_probs, bot_probs])
    y_true = np.concatenate([np.zeros(n_samples - n_bots), np.ones(n_bots)])
    
    # 模拟特征
    features = np.random.randn(n_samples, 32)
    # 让僵尸节点有一些可区分的特征模式
    features[y_true == 1] += 0.5
    
    print(f"\n测试数据：{n_samples} 样本，{n_bots} 僵尸节点 ({n_bots/n_samples*100:.1f}%)")
    print(f"概率范围：[{probs.min():.4f}, {probs.max():.4f}]")
    print(f"僵尸节点概率范围：[{probs[y_true==1].min():.4f}, {probs[y_true==1].max():.4f}]")
    
    # 测试多维分类器
    result = compute_botnet_metrics_multidim(y_true, probs, features)
    
    print(f"\n多维分类器结果:")
    print(f"  AUC:       {result['auc']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1:        {result['f1']:.4f}")
    print(f"  Threshold: {result['threshold']:.4f}")
    print(f"  预测/实际：{result['num_predicted']}/{result['num_true']}")
    print(f"  融合权重：{result['weights']}")