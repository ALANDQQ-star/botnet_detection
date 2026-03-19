"""
基于统计学分析的改进分类器

核心思想：
1. 使用对数变换扩展分数动态范围
2. 使用分位数法找到最优阈值
3. 针对极度不平衡数据设计分类策略
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_recall_curve, f1_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class StatisticalBotnetClassifier:
    """
    基于统计学的僵尸网络分类器（增强版 - 高精度）
    
    关键洞察：
    - 正常节点分数高度集中在 0 附近（偏度>25，峰度>800）
    - 僵尸节点分数虽然也低，但显著高于正常节点
    - 需要使用对数变换和分位数方法
    
    分类策略（纯无监督）：
    1. 使用对数变换将分数映射到更大范围
    2. 对变换后的分数使用高斯混合模型（GMM）
    3. 基于后验概率和控制 FPR 进行分类
    
    增强：
    - 使用更严格的阈值选择，提升精确率
    - 基于卡方分布控制假阳性率
    """
    
    def __init__(self, eps=1e-10, target_fpr=0.01, min_precision=0.3, prior_bot_ratio=0.01):
        """
        Args:
            eps: 对数变换的小常数，避免 log(0)
            target_fpr: 目标假阳性率
            min_precision: 最小精确率目标
            prior_bot_ratio: 先验僵尸节点比例
        """
        self.eps = eps
        self.target_fpr = target_fpr  # 目标假阳性率
        self.min_precision = min_precision  # 最小精确率目标
        self.prior_bot_ratio = prior_bot_ratio  # 先验僵尸节点比例
        self.log_transformed = False
        self.threshold = None
        
    def _log_transform(self, probs):
        """对数变换：扩展 [0, 1] 到 [-∞, 0]"""
        return np.log(probs + self.eps)
    
    def _find_threshold_gmm(self, probs):
        """使用对数变换后的高斯混合模型找阈值"""
        log_probs = self._log_transform(probs)
        
        from sklearn.mixture import GaussianMixture
        
        X = log_probs.reshape(-1, 1)
        
        # 使用两个高斯成分
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        gmm.fit(X)
        
        # 获取后验概率
        posteriors = gmm.predict_proba(X)
        
        # 确定哪个成分是高概率组
        means = gmm.means_.flatten()
        high_mean_idx = 0 if means[0] > means[1] else 1
        
        # 使用后验概率 > 0.5 作为分类标准
        bot_posterior = posteriors[:, high_mean_idx]
        
        # 找最优阈值
        # 策略：找到后验概率显著上升的点
        sorted_idx = np.argsort(probs)[::-1]  # 降序
        sorted_posteriors = bot_posterior[sorted_idx]
        
        # 找后验概率 > 0.5 的第一个点
        above_05 = sorted_posteriors > 0.5
        if above_05.any():
            target_idx = np.where(above_05)[0][-1]
            threshold = probs[sorted_idx[target_idx]]
        else:
            # 如果没有后验概率 > 0.5 的点，取 Top 1%
            threshold = np.percentile(probs, 99)
        
        return threshold
    
    def _find_threshold_quantile(self, probs, y_true=None):
        """
        使用分位数方法找阈值
        
        关键洞察：正常节点的 Q3（75% 分位数）是很好的参考点
        僵尸节点分数通常显著高于正常节点的 Q3
        """
        # 正常节点的分布特征
        # 由于数据极度不平衡，总体分布的 Q3 接近正常节点的 Q3
        q75 = np.percentile(probs, 75)
        q90 = np.percentile(probs, 90)
        q95 = np.percentile(probs, 95)
        q99 = np.percentile(probs, 99)
        
        # 使用对数尺度找"自然"分界点
        log_probs = self._log_transform(probs)
        log_mean = np.mean(log_probs)
        log_std = np.std(log_probs)
        
        # 阈值设置在均值 + 2-3 倍标准差处
        # 这对应于正态分布的 97.5%-99.9% 分位点
        threshold_log = log_mean + 2.5 * log_std
        threshold = np.exp(threshold_log) - self.eps
        
        # 验证阈值合理性
        ratio = (probs >= threshold).mean()
        
        # 如果比例太小，调整阈值
        if ratio < 0.001:
            threshold = np.percentile(probs, 99.5)
        elif ratio > 0.05:
            threshold = np.percentile(probs, 95)
        
        return threshold
    
    def _find_threshold_f1_optimal(self, probs, y_true):
        """使用 F1 优化找阈值（需要真实标签）"""
        prec_curve, rec_curve, thresholds = precision_recall_curve(y_true, probs)
        
        # 计算 F1 曲线
        f1_curve = 2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-10)
        
        # 找 F1 最大点
        best_idx = np.argmax(f1_curve[:-1])  # 最后一个是 0
        best_threshold = thresholds[best_idx]
        
        return best_threshold
    
    def _find_threshold_youden(self, probs, y_true):
        """使用 Youden's J 找阈值（需要真实标签）"""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        return thresholds[optimal_idx]
    
    def _find_threshold_gmm_enhanced(self, probs):
        """
        使用对数 GMM + FPR 控制找阈值（增强版）
        
        策略：
        1. 对对数分数拟合双高斯混合模型
        2. 计算每个点属于"异常"成分的后验概率
        3. 基于后验概率和控制 FPR 选择阈值
        """
        from sklearn.mixture import GaussianMixture
        
        log_probs = self._log_transform(probs).reshape(-1, 1)
        
        # 拟合双高斯混合模型
        gmm = GaussianMixture(
            n_components=2, 
            covariance_type='full',
            n_init=5,
            random_state=42,
            max_iter=200
        )
        gmm.fit(log_probs)
        
        # 获取参数
        means = gmm.means_.flatten()
        covariances = gmm.covariances_.flatten()
        weights = gmm.weights_
        
        # 确定哪个成分是高概率组（均值更大的）
        high_mean_idx = 0 if means[0] > means[1] else 1
        low_mean_idx = 1 - high_mean_idx
        
        high_mean = means[high_mean_idx]
        low_mean = means[low_mean_idx]
        high_std = np.sqrt(covariances[high_mean_idx])
        low_std = np.sqrt(covariances[low_mean_idx])
        
        # 计算后验概率
        posteriors = gmm.predict_proba(log_probs)
        bot_posterior = posteriors[:, high_mean_idx]
        
        # 策略 1：基于后验概率 > 0.5
        threshold_posterior_05_idx = np.where(bot_posterior > 0.5)[0]
        if len(threshold_posterior_05_idx) > 0:
            threshold_posterior_05 = probs[threshold_posterior_05_idx].min()
        else:
            threshold_posterior_05 = np.percentile(probs, 99)
        
        # 策略 2：基于低成分高斯分布的 FPR 控制
        # 找到使低成分（正常节点）只有 target_fpr 比例超过的阈值
        from scipy.stats import norm
        z_score = norm.ppf(1 - self.target_fpr)
        threshold_fpr_gaussian = low_mean + z_score * low_std
        threshold_fpr = np.exp(threshold_fpr_gaussian) - self.eps
        
        # 策略 3：使用两高斯分布的交点
        # 解方程：w1 * N(x|u1,s1) = w2 * N(x|u2,s2)
        # 简化：取两均值的加权平均
        threshold_intersection = (weights[0] * means[0] + weights[1] * means[1])
        threshold_intersection = np.exp(threshold_intersection) - self.eps
        
        # 综合策略：取最大值（最严格）
        # 这确保同时满足：
        # 1. 后验概率 > 0.5
        # 2. FPR 控制
        candidates = [threshold_fpr]
        if threshold_posterior_05 > 0:
            candidates.append(threshold_posterior_05)
        
        # 选择较大的阈值以提升精确率
        self.threshold = max(candidates)
        
        # 验证阈值合理性
        ratio = (probs >= self.threshold).mean()
        
        # 如果预测比例太小，适当降低阈值
        if ratio < 0.005:
            self.threshold = np.percentile(probs, 99)
        # 如果预测比例太大，提高阈值
        elif ratio > 0.05:
            self.threshold = np.percentile(probs, 99.5)
        
        return self.threshold
    
    def fit_predict(self, probs, y_true=None):
        """
        执行分类（纯无监督，不偷看标注）
        
        Args:
            probs: 模型输出概率
            y_true: 真实标签（仅用于评估，不用于阈值选择）
        
        Returns:
            result: 包含预测结果和中间信息的字典
        """
        n = len(probs)
        
        # 1. 分析分布特征
        log_probs = self._log_transform(probs)
        log_mean = np.mean(log_probs)
        log_std = np.std(log_probs)
        
        # 2. 计算关键分位数
        q50 = np.percentile(probs, 50)
        q75 = np.percentile(probs, 75)
        q90 = np.percentile(probs, 90)
        q95 = np.percentile(probs, 95)
        q99 = np.percentile(probs, 99)
        
        # 3. 选择阈值（使用增强 GMM 方法）
        self.threshold = self._find_threshold_gmm_enhanced(probs)
        
        # 4. 生成预测
        preds = (probs >= self.threshold).astype(int)
        
        # 5. 计算指标（仅用于评估，不影响阈值选择）
        if y_true is not None:
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
            'num_predicted': int(preds.sum()),
            'predicted_ratio': float(preds.sum() / n),
            'log_mean': float(log_mean),
            'log_std': float(log_std),
            'q50': float(q50),
            'q75': float(q75),
            'q90': float(q90),
            'q95': float(q95),
            'q99': float(q99),
            'precision': float(precision) if precision is not None else None,
            'recall': float(recall) if recall is not None else None,
            'f1': float(f1) if f1 is not None else None,
            'auc': float(auc) if auc is not None else None,
        }
        
        return result
    
    def fit_predict_with_calibration(self, probs, y_true=None):
        """
        使用校准分数的分类
        
        策略：
        1. 将分数映射到对数尺度
        2. 使用对数尺度的分位数作为分类标准
        """
        n = len(probs)
        log_probs = self._log_transform(probs)
        
        # 对数尺度的分位数
        log_q75 = np.percentile(log_probs, 75)
        log_q90 = np.percentile(log_probs, 90)
        log_q99 = np.percentile(log_probs, 99)
        
        # 校准分数：基于对数尺度的位置
        calibrated = (log_probs - log_q75) / (log_q99 - log_q75 + 1e-10)
        
        # 校正值应该在 [0, 1] 范围
        calibrated = np.clip(calibrated, 0, 1)
        
        # 使用校准后的分数
        if y_true is not None:
            threshold_cal = self._find_threshold_f1_optimal(calibrated, y_true)
        else:
            threshold_cal = 0.5
        
        preds = (calibrated >= threshold_cal).astype(int)
        
        if y_true is not None:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, preds, average='binary', zero_division=0
            )
            auc = roc_auc_score(y_true, probs)
        else:
            precision = recall = f1 = auc = None
        
        return {
            'preds': preds,
            'probs': probs,
            'calibrated': calibrated,
            'threshold_calibrated': float(threshold_cal),
            'threshold_original': float(probs[np.argmin(np.abs(calibrated - threshold_cal))]) if threshold_cal else None,
            'num_predicted': int(preds.sum()),
            'precision': float(precision) if precision is not None else None,
            'recall': float(recall) if recall is not None else None,
            'f1': float(f1) if f1 is not None else None,
            'auc': float(auc) if auc is not None else None,
        }


def compute_botnet_metrics_statistical(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """
    使用统计学分类器计算指标
    """
    classifier = StatisticalBotnetClassifier()
    result = classifier.fit_predict(probs, y_true)
    
    return {
        'auc': result['auc'],
        'precision': result['precision'],
        'recall': result['recall'],
        'f1': result['f1'],
        'threshold': result['threshold'],
        'num_predicted': result['num_predicted'],
        'num_true': int(y_true.sum()),
        'method': 'Statistical',
        'ratio': result['predicted_ratio'],
    }


if __name__ == "__main__":
    # 测试
    np.random.seed(42)
    
    print("="*70)
    print("统计学分类器测试")
    print("="*70)
    
    # 模拟场景 12 的分布
    n_neg = 92966
    n_pos = 1468
    
    # 正常节点：极度集中在 0 附近
    normal_probs = np.random.beta(0.17, 2e12, n_neg) * 0.0001
    normal_probs = np.clip(normal_probs, 0, 1)
    
    # 僵尸节点：集中在 0.001 附近
    bot_probs = 0.0005 + np.random.beta(0.6, 3e8, n_pos) * 0.001
    bot_probs = np.clip(bot_probs, 0, 1)
    
    probs = np.concatenate([normal_probs, bot_probs])
    y_true = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    
    print(f"\n测试数据：{len(probs)} 样本，{n_pos} 僵尸节点 ({n_pos/len(probs)*100:.2f}%)")
    print(f"概率范围：[{probs.min():.6f}, {probs.max():.6f}]")
    print(f"正常节点 Q3: {np.percentile(normal_probs, 75):.6f}")
    print(f"僵尸节点 Q1: {np.percentile(bot_probs, 25):.6f}, 中位数：{np.median(bot_probs):.6f}, Q3: {np.percentile(bot_probs, 75):.6f}")
    
    # 测试统计学分类器
    result = compute_botnet_metrics_statistical(y_true, probs)
    
    print(f"\n统计学分类器结果:")
    print(f"  AUC:       {result['auc']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1:        {result['f1']:.4f}")
    print(f"  Threshold: {result['threshold']:.6f}")
    print(f"  预测/实际：{result['num_predicted']}/{result['num_true']}")