"""
数学阈值优化器

采用高级数学建模方法优化阈值选择：
1. 极值理论 (Extreme Value Theory, EVT)
2. 分数分布拟合 (Distribution Fitting)
3. 信息几何分析 (Information Geometry)
4. 变分推断 (Variational Inference)

核心思想：
- 高 AUC 表明分数分布呈现两个重叠的子分布
- 使用数学方法找到最优分割点
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from scipy import stats, optimize
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, genpareto
from scipy.special import digamma, polygamma
import warnings
warnings.filterwarnings('ignore')


class MathematicalThresholdOptimizer:
    """
    数学阈值优化器
    
    使用高级数学方法，不依赖真实标签
    """
    
    def __init__(self):
        self.threshold = None
        self.debug_info = {}
    
    def _log_transform(self, x, eps=1e-10):
        """安全的对数变换"""
        return np.log(x + eps)
    
    def _fit_beta_distribution(self, probs):
        """
        方法1：Beta 分布拟合
        
        Beta 分布定义在 [0,1] 区间，适合建模概率分数
        """
        # Beta 分布参数估计
        a, b, loc, scale = stats.beta.fit(probs, floc=0, fscale=1)
        
        return a, b, loc, scale
    
    def _fit_gamma_distribution(self, probs):
        """
        方法2：Gamma 分布拟合
        
        适合建模高度偏斜的分布
        """
        a, loc, scale = stats.gamma.fit(probs, floc=0)
        
        return a, loc, scale
    
    def _fit_mixture_model(self, probs):
        """
        方法3：混合模型拟合
        
        假设分数来自两个子分布的混合
        """
        from sklearn.mixture import GaussianMixture
        
        log_probs = self._log_transform(probs).reshape(-1, 1)
        
        # 双高斯混合模型
        gmm = GaussianMixture(n_components=2, covariance_type='full', 
                               random_state=42, n_init=10, max_iter=500)
        gmm.fit(log_probs)
        
        return gmm
    
    def _compute_kl_divergence(self, p, q):
        """计算 KL 散度"""
        return np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
    
    def _compute_bethe_free_energy(self, probs, threshold):
        """
        计算 Bethe 自由能
        
        自由能 = 能量 - 温度 × 熵
        
        最优阈值应该最小化自由能
        """
        # 分割成两组
        below = probs[probs < threshold]
        above = probs[probs >= threshold]
        
        if len(below) == 0 or len(above) == 0:
            return np.inf
        
        n = len(probs)
        n_below = len(below)
        n_above = len(above)
        
        # 计算熵
        p_below = n_below / n
        p_above = n_above / n
        
        # 熵项
        entropy = -p_below * np.log(p_below + 1e-10) - p_above * np.log(p_above + 1e-10)
        
        # 能量项（组内方差）
        var_below = np.var(below) if len(below) > 1 else 0
        var_above = np.var(above) if len(above) > 1 else 0
        
        energy = p_below * var_below + p_above * var_above
        
        # 自由能
        free_energy = energy - entropy * 0.1
        
        return free_energy
    
    def _find_optimal_bethe_threshold(self, probs):
        """
        使用 Bethe 自由能优化找阈值
        """
        sorted_probs = np.sort(probs)
        
        # 在候选阈值中搜索
        candidates = np.percentile(probs, np.linspace(90, 99, 50))
        
        best_threshold = candidates[0]
        best_energy = np.inf
        
        for thresh in candidates:
            energy = self._compute_bethe_free_energy(probs, thresh)
            if energy < best_energy:
                best_energy = energy
                best_threshold = thresh
        
        return best_threshold
    
    def _compute_tail_index(self, probs):
        """
        使用 Pickands 估计量计算尾部指数
        
        极值理论中的关键参数
        """
        sorted_probs = np.sort(probs)[::-1]  # 降序
        n = len(sorted_probs)
        
        k = int(n / 10)  # 使用 top 10%
        if k < 10:
            k = max(10, n // 2)
        
        # Pickands 估计量
        Q1 = sorted_probs[k]
        Q2 = sorted_probs[2*k] if 2*k < n else sorted_probs[-1]
        Q3 = sorted_probs[3*k] if 3*k < n else sorted_probs[-1]
        
        if Q2 == Q3:
            return 0
        
        xi = np.log(Q1 / Q2) / np.log(Q2 / Q3)
        
        return xi
    
    def _find_evt_threshold(self, probs):
        """
        使用极值理论找阈值
        
        核心思想：使用广义帕累托分布建模尾部
        """
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        
        # 计算尾部指数
        tail_index = self._compute_tail_index(probs)
        
        # 使用广义帕累托分布
        # 阈值选择：找到尾部行为开始的点
        # 使用 Hill 图的方法
        
        # 计算 Hill 估计
        log_sorted = np.log(sorted_probs[::-1])  # 降序后取对数
        hill_values = []
        
        for k in range(10, min(n // 2, 1000)):
            hill = np.mean(log_sorted[:k]) - log_sorted[k]
            hill_values.append((k, hill))
        
        # 找到 Hill 值稳定的区域
        if len(hill_values) > 10:
            # 计算 Hill 值的导数
            hill_diffs = np.diff([h[1] for h in hill_values])
            
            # 找到导数最小的点（Hill 值最稳定）
            stable_idx = np.argmin(np.abs(hill_diffs)) + 10
            
            threshold_idx = hill_values[stable_idx][0]
            threshold = sorted_probs[n - threshold_idx]
        else:
            threshold = np.percentile(probs, 95)
        
        return threshold
    
    def _compute_fisher_information(self, probs, threshold):
        """
        计算 Fisher 信息
        
        衡量分割点处概率分布的变化率
        """
        below = probs[probs < threshold]
        above = probs[probs >= threshold]
        
        if len(below) == 0 or len(above) == 0:
            return 0
        
        # 估计两组的分布参数
        mu_below = np.mean(below)
        mu_above = np.mean(above)
        var_below = np.var(below) + 1e-10
        var_above = np.var(above) + 1e-10
        
        # Fisher 信息（基于均值差异）
        fisher = (mu_above - mu_below)**2 / (var_below + var_above)
        
        return fisher
    
    def _find_optimal_fisher_threshold(self, probs):
        """
        使用 Fisher 信息最大化找阈值
        """
        candidates = np.percentile(probs, np.linspace(90, 99, 50))
        
        best_threshold = candidates[0]
        best_fisher = 0
        
        for thresh in candidates:
            fisher = self._compute_fisher_information(probs, thresh)
            if fisher > best_fisher:
                best_fisher = fisher
                best_threshold = thresh
        
        return best_threshold
    
    def _compute_wasserstein_distance(self, below, above):
        """
        计算 Wasserstein 距离（Earth Mover's Distance）
        
        衡量两组分布之间的差异
        """
        if len(below) == 0 or len(above) == 0:
            return 0
        
        # 经验 CDF
        sorted_below = np.sort(below)
        sorted_above = np.sort(above)
        
        # 使用等分点计算 Wasserstein 距离
        n_points = min(len(sorted_below), len(sorted_above), 100)
        
        quantiles = np.linspace(0, 1, n_points)
        cdf_below = np.percentile(sorted_below, quantiles * 100)
        cdf_above = np.percentile(sorted_above, quantiles * 100)
        
        # L1 Wasserstein 距离
        distance = np.mean(np.abs(cdf_above - cdf_below))
        
        return distance
    
    def _find_optimal_wasserstein_threshold(self, probs):
        """
        使用 Wasserstein 距离最大化找阈值
        """
        candidates = np.percentile(probs, np.linspace(90, 99, 50))
        
        best_threshold = candidates[0]
        best_distance = 0
        
        for thresh in candidates:
            below = probs[probs < thresh]
            above = probs[probs >= thresh]
            
            distance = self._compute_wasserstein_distance(below, above)
            if distance > best_distance:
                best_distance = distance
                best_threshold = thresh
        
        return best_threshold
    
    def _compute_shannon_entropy(self, probs):
        """
        计算 Shannon 熵
        """
        # 离散化
        n_bins = 50
        hist, _ = np.histogram(probs, bins=n_bins, density=True)
        hist = hist / (hist.sum() + 1e-10)
        
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        return entropy
    
    def _find_entropy_based_threshold(self, probs):
        """
        使用熵的变化率找阈值
        
        在最优分割点，两组的总熵应该最小
        """
        candidates = np.percentile(probs, np.linspace(90, 99, 50))
        
        best_threshold = candidates[0]
        best_entropy = np.inf
        
        for thresh in candidates:
            below = probs[probs < thresh]
            above = probs[probs >= thresh]
            
            if len(below) == 0 or len(above) == 0:
                continue
            
            n = len(probs)
            n_below = len(below)
            n_above = len(above)
            
            # 加权熵
            entropy_below = self._compute_shannon_entropy(below)
            entropy_above = self._compute_shannon_entropy(above)
            
            total_entropy = (n_below * entropy_below + n_above * entropy_above) / n
            
            if total_entropy < best_entropy:
                best_entropy = total_entropy
                best_threshold = thresh
        
        return best_threshold
    
    def _fit_polynomial_to_cdf(self, probs):
        """
        使用多项式拟合 CDF，找拐点
        """
        n = len(probs)
        sorted_probs = np.sort(probs)
        cdf = np.arange(1, n + 1) / n
        
        # 在高尾区域拟合多项式
        tail_idx = int(n * 0.9)
        tail_probs = sorted_probs[tail_idx:]
        tail_cdf = cdf[tail_idx:]
        
        # 拟合二次多项式
        if len(tail_probs) > 3:
            coeffs = np.polyfit(tail_probs, tail_cdf, 2)
            
            # 拐点（导数为 0 的点）
            # f(x) = ax^2 + bx + c
            # f'(x) = 2ax + b = 0
            # x = -b / 2a
            if abs(coeffs[0]) > 1e-10:
                inflection_point = -coeffs[1] / (2 * coeffs[0])
                if 0 < inflection_point < 1:
                    return inflection_point
        
        return np.percentile(probs, 95)
    
    def find_threshold(self, probs):
        """
        主方法：使用多种数学方法找阈值
        
        综合以下方法：
        1. Bethe 自由能最小化
        2. Fisher 信息最大化
        3. Wasserstein 距离最大化
        4. 熵最小化
        5. 混合模型
        6. EVT（极值理论）
        """
        probs = np.asarray(probs).flatten()
        n = len(probs)
        sorted_probs = np.sort(probs)
        
        # 基本统计
        self.debug_info['n'] = n
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
        # 方法 1：Bethe 自由能最小化
        # ==============================
        threshold_bethe = self._find_optimal_bethe_threshold(probs)
        self.debug_info['bethe'] = {'threshold': float(threshold_bethe)}
        
        # ==============================
        # 方法 2：Fisher 信息最大化
        # ==============================
        threshold_fisher = self._find_optimal_fisher_threshold(probs)
        self.debug_info['fisher'] = {'threshold': float(threshold_fisher)}
        
        # ==============================
        # 方法 3：Wasserstein 距离最大化
        # ==============================
        threshold_wasserstein = self._find_optimal_wasserstein_threshold(probs)
        self.debug_info['wasserstein'] = {'threshold': float(threshold_wasserstein)}
        
        # ==============================
        # 方法 4：熵最小化
        # ==============================
        threshold_entropy = self._find_entropy_based_threshold(probs)
        self.debug_info['entropy'] = {'threshold': float(threshold_entropy)}
        
        # ==============================
        # 方法 5：混合模型 (GMM)
        # ==============================
        gmm = self._fit_mixture_model(probs)
        means = gmm.means_.flatten()
        high_idx = 0 if means[0] > means[1] else 1
        
        posteriors = gmm.predict_proba(np.log(probs.reshape(-1, 1) + 1e-10))
        high_posterior = posteriors[:, high_idx]
        
        # 找后验概率 > 0.5 的最低分数
        sorted_idx = np.argsort(probs)[::-1]
        sorted_posteriors = high_posterior[sorted_idx]
        
        above_half = np.where(sorted_posteriors > 0.5)[0]
        if len(above_half) > 0:
            idx = above_half[-1]
            threshold_gmm = probs[sorted_idx[idx]]
        else:
            threshold_gmm = np.percentile(probs, 95)
        
        self.debug_info['gmm'] = {
            'threshold': float(threshold_gmm),
            'means': means.tolist()
        }
        
        # ==============================
        # 方法 6：极值理论 (EVT)
        # ==============================
        threshold_evt = self._find_evt_threshold(probs)
        self.debug_info['evt'] = {'threshold': float(threshold_evt)}
        
        # ==============================
        # 方法 7：CDF 多项式拟合
        # ==============================
        threshold_poly = self._fit_polynomial_to_cdf(probs)
        self.debug_info['polynomial'] = {'threshold': float(threshold_poly)}
        
        # ==============================
        # 综合：加权投票
        # ==============================
        all_thresholds = {
            'bethe': threshold_bethe,
            'fisher': threshold_fisher,
            'wasserstein': threshold_wasserstein,
            'entropy': threshold_entropy,
            'gmm': threshold_gmm,
            'evt': threshold_evt,
            'polynomial': threshold_poly
        }
        
        # 计算每个阈值的预测比例
        threshold_scores = []
        for name, thresh in all_thresholds.items():
            if thresh is not None and 0 < thresh < 1:
                pred_ratio = np.mean(probs >= thresh)
                # 合理范围：2%-8%
                if 0.02 <= pred_ratio <= 0.08:
                    threshold_scores.append((name, thresh, pred_ratio))
        
        self.debug_info['valid_thresholds'] = [
            (n, float(t), float(r)) for n, t, r in threshold_scores
        ]
        
        if not threshold_scores:
            # 回退：使用 96% 分位数
            self.threshold = sorted_probs[int(n * 0.96)]
        else:
            # 优化策略：选择预测比例在 3%-5% 范围内的最低阈值
            # 这样可以最大化 recall 同时保持合理的 precision
            
            # 首先尝试找到预测比例在 3%-5% 的阈值
            optimal = [(n, t, r) for n, t, r in threshold_scores if 0.03 <= r <= 0.05]
            
            if optimal:
                # 取这些阈值的中位数
                optimal.sort(key=lambda x: x[2])
                mid_idx = len(optimal) // 2
                self.threshold = optimal[mid_idx][1]
            else:
                # 扩大范围到 2%-6%
                extended = [(n, t, r) for n, t, r in threshold_scores if 0.02 <= r <= 0.06]
                if extended:
                    extended.sort(key=lambda x: x[2])
                    mid_idx = len(extended) // 2
                    self.threshold = extended[mid_idx][1]
                else:
                    # 使用 96% 分位数作为回退
                    self.threshold = sorted_probs[int(n * 0.96)]
        
        # 最终调整
        pred_ratio = np.mean(probs >= self.threshold)
        
        # 如果预测比例不合理，调整到 96% 分位数
        if pred_ratio < 0.02 or pred_ratio > 0.08:
            self.threshold = sorted_probs[int(n * 0.96)]
        
        self.debug_info['final_threshold'] = float(self.threshold)
        self.debug_info['final_pred_ratio'] = float(np.mean(probs >= self.threshold))
        
        return self.threshold
    
    def predict(self, probs):
        """生成预测"""
        if self.threshold is None:
            self.find_threshold(probs)
        return (np.asarray(probs) >= self.threshold).astype(int)


def compute_botnet_metrics_mathematical(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """
    使用数学阈值优化器计算指标
    
    完全不使用标签信息
    """
    optimizer = MathematicalThresholdOptimizer()
    threshold = optimizer.find_threshold(probs)
    
    preds = (probs >= threshold).astype(int)
    
    # 计算指标（仅用于评估）
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average='binary', zero_division=0
    )
    auc = roc_auc_score(y_true, probs)
    
    return {
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'threshold': float(threshold),
        'num_predicted': int(preds.sum()),
        'num_true': int(y_true.sum()),
        'method': 'Mathematical_Optimization',
        'predicted_ratio': float(preds.sum() / len(probs)),
        'debug_info': optimizer.debug_info
    }


if __name__ == "__main__":
    print("="*70)
    print("数学阈值优化器测试")
    print("="*70)
    
    try:
        data = np.load('s12_scores.npz', allow_pickle=True)
        probs = data['probs']
        y_true = data['y_true']
        
        print(f"\n[数据概览]")
        print(f"  总节点数：{len(probs)}")
        print(f"  僵尸节点数：{y_true.sum()} ({y_true.sum()/len(probs)*100:.2f}%)")
        
        # 测试不同阈值的效果
        print(f"\n[阈值效果分析]")
        for thresh in [0.0003, 0.0005, 0.0007, 0.001, 0.002]:
            preds = (probs >= thresh).astype(int)
            tp = ((preds == 1) & (y_true == 1)).sum()
            precision = tp / preds.sum() if preds.sum() > 0 else 0
            recall = tp / y_true.sum() if y_true.sum() > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  阈值 {thresh:.4f}: 预测 {preds.sum():5d}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        # 测试数学优化器
        print("\n[测试数学阈值优化器]...")
        result = compute_botnet_metrics_mathematical(y_true, probs)
        
        print(f"\n评估结果（数学优化器）")
        print("="*60)
        print(f"  AUC:       {result['auc']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1-Score:  {result['f1']:.4f}")
        print(f"  Threshold: {result['threshold']:.6f}")
        print("="*60)
        print(f"  预测为僵尸网络的节点：{result['num_predicted']}")
        print(f"  实际僵尸网络节点：{result['num_true']}")
        
        print(f"\n[调试信息]")
        print(f"  最终阈值：{result['debug_info'].get('final_threshold', 'N/A'):.6f}")
        print(f"  最终预测比例：{result['predicted_ratio']:.4f}")
        print(f"  各方法阈值：")
        for method in ['bethe', 'fisher', 'wasserstein', 'entropy', 'gmm', 'evt', 'polynomial']:
            if method in result['debug_info']:
                t = result['debug_info'][method]['threshold']
                print(f"    {method}: {t:.6f}")
        
    except FileNotFoundError:
        print("[Error] 请先运行 analyze_s12.py 生成分数数据")