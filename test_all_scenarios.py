"""
在场景 1-13 上测试 8 种阈值选择方法，选择综合效果最好的方法。
"""

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader
from data_loader import CTU13Loader
from improved_heterogeneous_graph import ImprovedHeterogeneousGraphBuilder
from improved_model import ImprovedBotnetDetector
from sklearn.metrics import precision_recall_fscore_support
import json
import os


def get_labels(df, ip_map):
    """提取节点标签（仅用于评估，不用于阈值选择）"""
    bot_keywords = ['botnet', 'cc', 'c&c', 'neris', 'rbot', 'virut', 'menti', 'murlo', 'sogou']
    df['is_bot'] = df['label'].apply(lambda x: any(k in str(x).lower() for k in bot_keywords))
    
    bot_ips = set(df[df['is_bot']]['src_ip'].unique()) | set(df[df['is_bot']]['dst_ip'].unique())
    y = np.zeros(len(ip_map), dtype=np.float32)
    for ip, idx in ip_map.items():
        if ip in bot_ips:
            y[idx] = 1.0
    return torch.tensor(y)


class SimpleThresholdMethods:
    """8 种简单的阈值计算方法（无标签）"""
    
    @staticmethod
    def bethe(probs):
        """Bethe 自由能最小化"""
        sorted_probs = np.sort(probs)
        n = len(probs)
        candidates = np.percentile(probs, np.linspace(90, 99, 50))
        
        best_threshold = candidates[0]
        best_energy = np.inf
        
        for thresh in candidates:
            below = probs[probs < thresh]
            above = probs[probs >= thresh]
            
            if len(below) == 0 or len(above) == 0:
                continue
            
            p_below = len(below) / n
            p_above = len(above) / n
            
            entropy = -p_below * np.log(p_below + 1e-10) - p_above * np.log(p_above + 1e-10)
            var_below = np.var(below)
            var_above = np.var(above)
            energy = p_below * var_below + p_above * var_above
            
            free_energy = energy - entropy * 0.1
            
            if free_energy < best_energy:
                best_energy = free_energy
                best_threshold = thresh
        
        return best_threshold
    
    @staticmethod
    def fisher(probs):
        """Fisher 信息最大化"""
        candidates = np.percentile(probs, np.linspace(90, 99, 50))
        
        best_threshold = candidates[0]
        best_fisher = 0
        
        for thresh in candidates:
            below = probs[probs < thresh]
            above = probs[probs >= thresh]
            
            if len(below) == 0 or len(above) == 0:
                continue
            
            mu_below = np.mean(below)
            mu_above = np.mean(above)
            var_below = np.var(below) + 1e-10
            var_above = np.var(above) + 1e-10
            
            fisher = (mu_above - mu_below)**2 / (var_below + var_above)
            
            if fisher > best_fisher:
                best_fisher = fisher
                best_threshold = thresh
        
        return best_threshold
    
    @staticmethod
    def wasserstein(probs):
        """Wasserstein 距离最大化"""
        candidates = np.percentile(probs, np.linspace(90, 99, 50))
        
        best_threshold = candidates[0]
        best_distance = 0
        
        for thresh in candidates:
            below = probs[probs < thresh]
            above = probs[probs >= thresh]
            
            if len(below) == 0 or len(above) == 0:
                continue
            
            sorted_below = np.sort(below)
            sorted_above = np.sort(above)
            
            n_points = min(len(sorted_below), len(sorted_above), 100)
            quantiles = np.linspace(0, 1, n_points)
            cdf_below = np.percentile(sorted_below, quantiles * 100)
            cdf_above = np.percentile(sorted_above, quantiles * 100)
            
            distance = np.mean(np.abs(cdf_above - cdf_below))
            
            if distance > best_distance:
                best_distance = distance
                best_threshold = thresh
        
        return best_threshold
    
    @staticmethod
    def entropy(probs):
        """Shannon 熵最小化"""
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
            
            hist_below, _ = np.histogram(below, bins=20, density=True)
            hist_below = hist_below / (hist_below.sum() + 1e-10)
            entropy_below = -np.sum(hist_below * np.log(hist_below + 1e-10))
            
            hist_above, _ = np.histogram(above, bins=20, density=True)
            hist_above = hist_above / (hist_above.sum() + 1e-10)
            entropy_above = -np.sum(hist_above * np.log(hist_above + 1e-10))
            
            total_entropy = (n_below * entropy_below + n_above * entropy_above) / n
            
            if total_entropy < best_entropy:
                best_entropy = total_entropy
                best_threshold = thresh
        
        return best_threshold
    
    @staticmethod
    def gmm(probs):
        """高斯混合模型"""
        from sklearn.mixture import GaussianMixture
        
        log_probs = np.log(probs + 1e-10).reshape(-1, 1)
        
        gmm = GaussianMixture(n_components=2, covariance_type='full', 
                               random_state=42, n_init=3, max_iter=200)
        gmm.fit(log_probs)
        
        means = gmm.means_.flatten()
        high_idx = 0 if means[0] > means[1] else 1
        
        posteriors = gmm.predict_proba(log_probs)
        high_posterior = posteriors[:, high_idx]
        
        sorted_idx = np.argsort(probs)[::-1]
        sorted_posteriors = high_posterior[sorted_idx]
        
        above_half = np.where(sorted_posteriors > 0.5)[0]
        if len(above_half) > 0:
            idx = above_half[-1]
            threshold = probs[sorted_idx[idx]]
        else:
            threshold = np.percentile(probs, 95)
        
        return threshold
    
    @staticmethod
    def evt(probs):
        """极值理论"""
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        
        log_sorted = np.log(sorted_probs[::-1])
        hill_values = []
        
        for k in range(10, min(n // 2, 1000)):
            hill = np.mean(log_sorted[:k]) - log_sorted[k]
            hill_values.append((k, hill))
        
        if len(hill_values) > 10:
            hill_diffs = np.diff([h[1] for h in hill_values])
            stable_idx = np.argmin(np.abs(hill_diffs)) + 10
            threshold_idx = hill_values[stable_idx][0]
            threshold = sorted_probs[n - threshold_idx]
        else:
            threshold = np.percentile(probs, 95)
        
        return threshold
    
    @staticmethod
    def polynomial(probs):
        """CDF 多项式拟合"""
        n = len(probs)
        sorted_probs = np.sort(probs)
        cdf = np.arange(1, n + 1) / n
        
        tail_idx = int(n * 0.9)
        tail_probs = sorted_probs[tail_idx:]
        tail_cdf = cdf[tail_idx:]
        
        if len(tail_probs) > 3:
            coeffs = np.polyfit(tail_probs, tail_cdf, 2)
            if abs(coeffs[0]) > 1e-10:
                inflection_point = -coeffs[1] / (2 * coeffs[0])
                if 0 < inflection_point < 1:
                    return inflection_point
        
        return np.percentile(probs, 95)
    
    @staticmethod
    def quantile_96(probs):
        """固定 96% 分位数"""
        return np.percentile(probs, 96)


METHODS = {
    'bethe': SimpleThresholdMethods.bethe,
    'fisher': SimpleThresholdMethods.fisher,
    'wasserstein': SimpleThresholdMethods.wasserstein,
    'entropy': SimpleThresholdMethods.entropy,
    'gmm': SimpleThresholdMethods.gmm,
    'evt': SimpleThresholdMethods.evt,
    'polynomial': SimpleThresholdMethods.polynomial,
    'quantile_96': SimpleThresholdMethods.quantile_96
}


def test_scenario(scenario, data_dir, model_path, device):
    """测试单个场景"""
    print(f"\n{'='*60}")
    print(f"测试场景 {scenario}")
    print(f"{'='*60}")
    
    try:
        # 加载数据
        loader = CTU13Loader(data_dir)
        df = loader.load_data([scenario])
        
        if df.empty:
            print(f"[Error] 场景 {scenario} 数据加载失败!")
            return None
        
        # 构建图
        builder = ImprovedHeterogeneousGraphBuilder()
        data, ip_map = builder.build(df, include_semantic=True, include_struct=True)
        
        # 获取标签（仅用于评估）
        labels = get_labels(df, ip_map)
        
        # 加载模型
        model = ImprovedBotnetDetector.load(model_path, device=device)
        model.eval()
        
        # 推理
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
        
        # 测试所有方法
        results = {}
        
        print(f"\n  {'方法':<15} {'阈值':<12} {'预测数':<10} {'P':<8} {'R':<8} {'F1':<8}")
        print(f"  {'-'*70}")
        
        for name, method in METHODS.items():
            threshold = method(probs)
            preds = (probs >= threshold).astype(int)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, preds, average='binary', zero_division=0
            )
            
            results[name] = {
                'threshold': float(threshold),
                'num_predicted': int(preds.sum()),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
            
            print(f"  {name:<15} {threshold:<12.6f} {preds.sum():<10} {precision:<8.3f} {recall:<8.3f} {f1:<8.3f}")
        
        return results
    
    except Exception as e:
        print(f"[Error] 场景 {scenario} 测试失败：{e}")
        return None


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Config] 使用设备：{device}")
    
    # 检查模型
    model_path = 'improved_botnet_model.pth'
    if not os.path.exists(model_path):
        print(f"[Error] 模型不存在：{model_path}")
        return
    
    # 测试场景 1-13
    test_scenarios = list(range(1, 14))  # 1-13
    all_results = {}
    
    for scenario in test_scenarios:
        results = test_scenario(scenario, '/root/autodl-fs/CTU-13/CTU-13-Dataset', model_path, device)
        if results:
            all_results[scenario] = results
        
        # 清理内存
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 综合分析
    print("\n" + "="*80)
    print("综合分析")
    print("="*80)
    
    # 计算每个方法的平均 F1
    method_scores = {name: [] for name in METHODS.keys()}
    
    for scenario, results in all_results.items():
        for name, metrics in results.items():
            method_scores[name].append(metrics['f1'])
    
    print(f"\n  {'方法':<15}", end='')
    for s in all_results.keys():
        print(f" {s:<8}", end='')
    print(f" {'平均 F1':<10}")
    print(f"  {'-'*15 + '-'*8*len(all_results) + '-'*10}")
    
    avg_f1_scores = []
    for name, f1_scores in method_scores.items():
        if f1_scores:
            avg_f1 = np.mean(f1_scores)
            avg_f1_scores.append((name, avg_f1))
            
            print(f"  {name:<15}", end='')
            for s in f1_scores:
                print(f" {s:<8.3f}", end='')
            print(f" {avg_f1:<10.4f}")
    
    # 排序
    avg_f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*80)
    print("方法排名（按平均 F1）")
    print("="*80)
    
    for i, (name, avg_f1) in enumerate(avg_f1_scores):
        print(f"  {i+1}. {name}: F1 = {avg_f1:.4f}")
    
    # 推荐最佳方法
    best_method, best_f1 = avg_f1_scores[0]
    
    print(f"\n[推荐] 最佳方法：{best_method} (平均 F1 = {best_f1:.4f})")
    
    # 保存结果
    with open('threshold_method_comparison_all.json', 'w') as f:
        json.dump({
            'scenarios_tested': list(all_results.keys()),
            'scenario_results': all_results,
            'method_ranking': [{'method': name, 'avg_f1': score, 'f1_scores': method_scores[name]} 
                              for name, score in avg_f1_scores],
            'best_method': best_method
        }, f, indent=2)
    
    print(f"\n[Info] 结果已保存到 threshold_method_comparison_all.json")
    
    # 生成最终分类器代码
    print(f"\n[生成] 创建最终分类器文件...")
    
    final_code = f'''"""
最终僵尸网络分类器

基于场景 1-13 的测试结果，选择 {best_method} 方法作为最终判定方法。
该方法在 13 个场景上平均 F1 = {best_f1:.4f}
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


class FinalBotnetClassifier:
    """
    最终僵尸网络分类器
    
    使用 {best_method} 方法
    """
    
    def __init__(self):
        self.threshold = None
        self.method_name = "{best_method}"
    
    def find_threshold(self, probs):
        """
        找到最优阈值
        
        方法：{best_method}
        """
        probs = np.asarray(probs).flatten()
        
        # 使用方法：{best_method}
        self.threshold = SimpleThresholdMethods.{best_method}(probs)
        return self.threshold
    
    def predict(self, probs):
        """生成预测"""
        if self.threshold is None:
            self.find_threshold(probs)
        return (np.asarray(probs) >= self.threshold).astype(int)


class SimpleThresholdMethods:
    """阈值计算方法集合"""
    
    @staticmethod
    def bethe(probs):
        sorted_probs = np.sort(probs)
        n = len(probs)
        candidates = np.percentile(probs, np.linspace(90, 99, 50))
        best_threshold = candidates[0]
        best_energy = np.inf
        for thresh in candidates:
            below = probs[probs < thresh]
            above = probs[probs >= thresh]
            if len(below) == 0 or len(above) == 0:
                continue
            p_below = len(below) / n
            p_above = len(above) / n
            entropy = -p_below * np.log(p_below + 1e-10) - p_above * np.log(p_above + 1e-10)
            var_below = np.var(below)
            var_above = np.var(above)
            energy = p_below * var_below + p_above * var_above
            free_energy = energy - entropy * 0.1
            if free_energy < best_energy:
                best_energy = free_energy
                best_threshold = thresh
        return best_threshold
    
    @staticmethod
    def fisher(probs):
        candidates = np.percentile(probs, np.linspace(90, 99, 50))
        best_threshold = candidates[0]
        best_fisher = 0
        for thresh in candidates:
            below = probs[probs < thresh]
            above = probs[probs >= thresh]
            if len(below) == 0 or len(above) == 0:
                continue
            mu_below = np.mean(below)
            mu_above = np.mean(above)
            var_below = np.var(below) + 1e-10
            var_above = np.var(above) + 1e-10
            fisher = (mu_above - mu_below)**2 / (var_below + var_above)
            if fisher > best_fisher:
                best_fisher = fisher
                best_threshold = thresh
        return best_threshold
    
    @staticmethod
    def wasserstein(probs):
        candidates = np.percentile(probs, np.linspace(90, 99, 50))
        best_threshold = candidates[0]
        best_distance = 0
        for thresh in candidates:
            below = probs[probs < thresh]
            above = probs[probs >= thresh]
            if len(below) == 0 or len(above) == 0:
                continue
            sorted_below = np.sort(below)
            sorted_above = np.sort(above)
            n_points = min(len(sorted_below), len(sorted_above), 100)
            quantiles = np.linspace(0, 1, n_points)
            cdf_below = np.percentile(sorted_below, quantiles * 100)
            cdf_above = np.percentile(sorted_above, quantiles * 100)
            distance = np.mean(np.abs(cdf_above - cdf_below))
            if distance > best_distance:
                best_distance = distance
                best_threshold = thresh
        return best_threshold
    
    @staticmethod
    def entropy(probs):
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
            hist_below, _ = np.histogram(below, bins=20, density=True)
            hist_below = hist_below / (hist_below.sum() + 1e-10)
            entropy_below = -np.sum(hist_below * np.log(hist_below + 1e-10))
            hist_above, _ = np.histogram(above, bins=20, density=True)
            hist_above = hist_above / (hist_above.sum() + 1e-10)
            entropy_above = -np.sum(hist_above * np.log(hist_above + 1e-10))
            total_entropy = (n_below * entropy_below + n_above * entropy_above) / n
            if total_entropy < best_entropy:
                best_entropy = total_entropy
                best_threshold = thresh
        return best_threshold
    
    @staticmethod
    def gmm(probs):
        from sklearn.mixture import GaussianMixture
        log_probs = np.log(probs + 1e-10).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type='full', 
                               random_state=42, n_init=3, max_iter=200)
        gmm.fit(log_probs)
        means = gmm.means_.flatten()
        high_idx = 0 if means[0] > means[1] else 1
        posteriors = gmm.predict_proba(log_probs)
        high_posterior = posteriors[:, high_idx]
        sorted_idx = np.argsort(probs)[::-1]
        sorted_posteriors = high_posterior[sorted_idx]
        above_half = np.where(sorted_posteriors > 0.5)[0]
        if len(above_half) > 0:
            idx = above_half[-1]
            threshold = probs[sorted_idx[idx]]
        else:
            threshold = np.percentile(probs, 95)
        return threshold
    
    @staticmethod
    def evt(probs):
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        log_sorted = np.log(sorted_probs[::-1])
        hill_values = []
        for k in range(10, min(n // 2, 1000)):
            hill = np.mean(log_sorted[:k]) - log_sorted[k]
            hill_values.append((k, hill))
        if len(hill_values) > 10:
            hill_diffs = np.diff([h[1] for h in hill_values])
            stable_idx = np.argmin(np.abs(hill_diffs)) + 10
            threshold_idx = hill_values[stable_idx][0]
            threshold = sorted_probs[n - threshold_idx]
        else:
            threshold = np.percentile(probs, 95)
        return threshold
    
    @staticmethod
    def polynomial(probs):
        n = len(probs)
        sorted_probs = np.sort(probs)
        cdf = np.arange(1, n + 1) / n
        tail_idx = int(n * 0.9)
        tail_probs = sorted_probs[tail_idx:]
        tail_cdf = cdf[tail_idx:]
        if len(tail_probs) > 3:
            coeffs = np.polyfit(tail_probs, tail_cdf, 2)
            if abs(coeffs[0]) > 1e-10:
                inflection_point = -coeffs[1] / (2 * coeffs[0])
                if 0 < inflection_point < 1:
                    return inflection_point
        return np.percentile(probs, 95)
    
    @staticmethod
    def quantile_96(probs):
        return np.percentile(probs, 96)


def compute_botnet_metrics_final(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """
    使用最终分类器计算指标
    """
    classifier = FinalBotnetClassifier()
    threshold = classifier.find_threshold(probs)
    
    preds = (probs >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average='binary', zero_division=0
    )
    auc = roc_auc_score(y_true, probs)
    
    return {{
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'threshold': float(threshold),
        'num_predicted': int(preds.sum()),
        'num_true': int(y_true.sum()),
        'method': '{best_method}',
        'predicted_ratio': float(preds.sum() / len(probs))
    }}
'''
    
    with open('final_botnet_classifier.py', 'w') as f:
        f.write(final_code)
    
    print(f"[Info] 最终分类器已更新为：{best_method}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()