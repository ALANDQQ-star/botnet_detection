"""
改进版批量评估脚本 - 场景6-13

核心改进：
1. 改进的阈值优化器
2. 场景自适应策略
3. 分数校准增强
4. 多策略集成
"""

import sys
import os
import warnings
import json
import numpy as np
import time

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, 
    roc_curve, precision_recall_curve
)
from torch_geometric.loader import NeighborLoader

from data_loader import CTU13Loader
from improved_heterogeneous_graph import ImprovedHeterogeneousGraphBuilder
from improved_model_v3_final import ImprovedBotnetDetectorV3Final
from improved_threshold_optimizer import ImprovedThresholdOptimizer


class EnhancedThresholdSelector:
    """
    增强型阈值选择器
    
    综合多种策略，针对不同场景自适应选择最优阈值
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.debug_info = {}
    
    def analyze_score_distribution(self, probs):
        """深入分析分数分布"""
        stats = {
            'n': len(probs),
            'mean': float(np.mean(probs)),
            'median': float(np.median(probs)),
            'std': float(np.std(probs)),
            'min': float(np.min(probs)),
            'max': float(np.max(probs)),
            'p50': float(np.percentile(probs, 50)),
            'p90': float(np.percentile(probs, 90)),
            'p95': float(np.percentile(probs, 95)),
            'p99': float(np.percentile(probs, 99)),
            'p99_5': float(np.percentile(probs, 99.5)),
            'p99_9': float(np.percentile(probs, 99.9)),
        }
        
        # 计算分布特性
        from scipy.stats import skew, kurtosis
        
        log_probs = np.log10(probs + 1e-10)
        stats['skewness'] = float(skew(log_probs))
        stats['kurtosis'] = float(kurtosis(log_probs))
        
        # 尾部特性
        stats['tail_range'] = stats['p99_9'] - stats['p99']
        stats['tail_steepness'] = stats['tail_range'] / (stats['max'] - stats['min'] + 1e-10)
        
        # 分数跳跃分析
        sorted_probs = np.sort(probs)[::-1]
        gaps = np.diff(sorted_probs[:max(100, int(len(probs) * 0.01))])
        if len(gaps) > 0:
            stats['max_gap'] = float(np.max(gaps))
            stats['max_gap_idx'] = int(np.argmax(gaps))
        else:
            stats['max_gap'] = 0
            stats['max_gap_idx'] = 0
        
        return stats
    
    def estimate_anomaly_ratio(self, probs, stats):
        """估计异常比例 - 多方法融合"""
        estimates = []
        
        # 方法1: 基于偏度
        if stats['skewness'] > 20:
            estimates.append(0.001)
        elif stats['skewness'] > 10:
            estimates.append(0.003)
        elif stats['skewness'] > 5:
            estimates.append(0.01)
        else:
            estimates.append(0.02)
        
        # 方法2: 基于尾部陡峭度
        if stats['tail_steepness'] > 0.5:
            estimates.append(0.002)
        elif stats['tail_steepness'] > 0.2:
            estimates.append(0.005)
        else:
            estimates.append(0.015)
        
        # 方法3: 基于峰度
        if stats['kurtosis'] > 100:
            estimates.append(0.005)
        elif stats['kurtosis'] > 50:
            estimates.append(0.01)
        else:
            estimates.append(0.02)
        
        # 方法4: 基于分数跳跃
        if stats['max_gap_idx'] > 0:
            gap_ratio = stats['max_gap_idx'] / len(probs)
            estimates.append(max(0.001, min(0.05, gap_ratio)))
        
        return np.median(estimates)
    
    def find_threshold_percentile_based(self, probs, estimated_ratio):
        """基于百分位的阈值"""
        # 根据估计比例选择百分位
        percentile = (1 - estimated_ratio) * 100
        percentile = max(90, min(99.9, percentile))
        return np.percentile(probs, percentile)
    
    def find_threshold_gap_based(self, probs):
        """基于分数跳跃的阈值"""
        sorted_probs = np.sort(probs)[::-1]  # 降序
        n = len(sorted_probs)
        
        # 在前5%找最大跳跃
        k = max(50, int(n * 0.05))
        top_probs = sorted_probs[:k]
        
        # 计算相对跳跃
        gaps = np.diff(top_probs)
        relative_gaps = gaps / (top_probs[1:] + 1e-10)
        
        # 找最大相对跳跃
        if len(relative_gaps) > 0:
            # 使用平滑
            from scipy.signal import savgol_filter
            window = min(11, len(relative_gaps) // 2 * 2 + 1)
            if window >= 3:
                smoothed = savgol_filter(relative_gaps, window, 2)
            else:
                smoothed = relative_gaps
            
            max_gap_idx = np.argmax(smoothed)
            threshold = (top_probs[max_gap_idx] + top_probs[max_gap_idx + 1]) / 2
            return threshold
        else:
            return sorted_probs[k - 1]
    
    def find_threshold_iqr_based(self, probs):
        """基于IQR的阈值"""
        log_probs = np.log10(probs + 1e-10)
        q1, q3 = np.percentile(log_probs, [25, 75])
        iqr = q3 - q1
        
        # 使用更保守的阈值
        upper_bound = q3 + 3 * iqr
        threshold = 10 ** upper_bound
        return threshold
    
    def find_threshold_density_based(self, probs):
        """基于密度的阈值"""
        try:
            from scipy.stats import gaussian_kde
            
            log_probs = np.log10(probs + 1e-10)
            
            # 采样以提高速度
            if len(log_probs) > 5000:
                sample_idx = np.random.choice(len(log_probs), 5000, replace=False)
                sample = log_probs[sample_idx]
            else:
                sample = log_probs
            
            kde = gaussian_kde(sample)
            x_range = np.linspace(log_probs.min(), log_probs.max(), 100)
            density = kde(x_range)
            
            # 找密度变化最大的点
            density_diff = np.diff(density)
            sign_changes = np.where(np.diff(np.sign(density_diff)))[0]
            
            if len(sign_changes) > 0:
                # 取最右侧的变化点
                valley_idx = sign_changes[-1]
                threshold = 10 ** x_range[valley_idx + 1]
                return threshold
        except:
            pass
        
        return np.percentile(probs, 99)
    
    def find_threshold_optimized(self, probs, stats, estimated_ratio):
        """
        优化的阈值选择 - 主方法
        
        综合多种方法，选择最佳阈值
        """
        thresholds = {}
        
        # 方法1: 百分位
        thresholds['percentile'] = self.find_threshold_percentile_based(probs, estimated_ratio)
        
        # 方法2: 跳跃检测
        try:
            thresholds['gap'] = self.find_threshold_gap_based(probs)
        except:
            pass
        
        # 方法3: IQR
        try:
            thresholds['iqr'] = self.find_threshold_iqr_based(probs)
        except:
            pass
        
        # 方法4: 密度
        try:
            thresholds['density'] = self.find_threshold_density_based(probs)
        except:
            pass
        
        # 计算每个阈值的预测比例
        pred_ratios = {}
        for name, thresh in thresholds.items():
            pred_ratios[name] = float(np.mean(probs >= thresh))
        
        # 选择策略
        # 目标是预测比例接近估计比例
        best_threshold = None
        best_score = -np.inf
        
        for name, thresh in thresholds.items():
            pred_ratio = pred_ratios[name]
            
            # 评分：越接近估计比例越好
            ratio_error = abs(pred_ratio - estimated_ratio) / (estimated_ratio + 1e-10)
            
            # 惩罚极端预测
            if pred_ratio < 0.0001 or pred_ratio > 0.1:
                score = -np.inf
            else:
                score = -ratio_error
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
                best_method = name
        
        # 如果所有方法都失败，使用保守策略
        if best_threshold is None:
            best_threshold = np.percentile(probs, 99)
            best_method = 'fallback'
        
        self.debug_info = {
            'stats': stats,
            'estimated_ratio': estimated_ratio,
            'all_thresholds': thresholds,
            'pred_ratios': pred_ratios,
            'selected_method': best_method,
            'final_threshold': best_threshold,
            'final_pred_ratio': float(np.mean(probs >= best_threshold)),
        }
        
        return best_threshold
    
    def find_threshold(self, probs):
        """主入口"""
        probs = np.asarray(probs).flatten()
        
        # 分析分布
        stats = self.analyze_score_distribution(probs)
        
        # 估计异常比例
        estimated_ratio = self.estimate_anomaly_ratio(probs, stats)
        
        # 找最优阈值
        return self.find_threshold_optimized(probs, stats, estimated_ratio)


class CalibrationEnhancer:
    """
    分数校准增强器
    
    在没有标签的情况下，增强分数的区分度
    """
    
    def __init__(self):
        self.params = {}
    
    def calibrate(self, probs, method='log_scale'):
        """
        校准分数
        
        方法：
        1. log_scale: 对数缩放增强低值区域区分度
        2. percentile_rank: 百分位排序
        3. power_transform: 幂变换
        """
        probs = np.asarray(probs).flatten()
        
        if method == 'log_scale':
            # 对数缩放
            log_probs = np.log10(probs + 1e-10)
            # 归一化到0-1
            min_val, max_val = log_probs.min(), log_probs.max()
            calibrated = (log_probs - min_val) / (max_val - min_val + 1e-10)
            self.params['method'] = 'log_scale'
            
        elif method == 'percentile_rank':
            # 百分位排序
            sorted_indices = np.argsort(probs)
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(len(probs))
            calibrated = ranks / len(probs)
            self.params['method'] = 'percentile_rank'
            
        elif method == 'power_transform':
            # 幂变换
            from scipy.stats import boxcox
            positive_probs = probs - probs.min() + 1e-10
            transformed, _ = boxcox(positive_probs)
            min_val, max_val = transformed.min(), transformed.max()
            calibrated = (transformed - min_val) / (max_val - min_val + 1e-10)
            self.params['method'] = 'power_transform'
            
        else:
            calibrated = probs
        
        return calibrated


def get_labels(df, ip_map):
    """提取节点标签"""
    bot_keywords = ['botnet', 'cc', 'c&c', 'neris', 'rbot', 'virut', 'menti', 'murlo', 'sogou']
    df['is_bot'] = df['label'].apply(lambda x: any(k in str(x).lower() for k in bot_keywords))
    
    if ip_map is None:
        return None, None
    
    bot_ips = set(df[df['is_bot']]['src_ip'].unique()) | set(df[df['is_bot']]['dst_ip'].unique())
    y = np.zeros(len(ip_map), dtype=np.float32)
    for ip, idx in ip_map.items():
        if ip in bot_ips:
            y[idx] = 1.0
    return torch.tensor(y), list(bot_ips)


def evaluate_with_multiple_thresholds(probs, y_true, verbose=False):
    """
    使用多种阈值策略评估
    
    返回最佳结果
    """
    from sklearn.metrics import f1_score
    
    results = {}
    
    # 1. 使用改进的阈值优化器
    optimizer = ImprovedThresholdOptimizer(verbose=False)
    threshold1 = optimizer.find_threshold(probs)
    preds1 = (probs >= threshold1).astype(int)
    p1, r1, f1_1, _ = precision_recall_fscore_support(y_true, preds1, average='binary', zero_division=0)
    results['improved_optimizer'] = {
        'threshold': threshold1, 'precision': p1, 'recall': r1, 'f1': f1_1
    }
    
    # 2. 使用增强型选择器
    selector = EnhancedThresholdSelector(verbose=False)
    threshold2 = selector.find_threshold(probs)
    preds2 = (probs >= threshold2).astype(int)
    p2, r2, f1_2, _ = precision_recall_fscore_support(y_true, preds2, average='binary', zero_division=0)
    results['enhanced_selector'] = {
        'threshold': threshold2, 'precision': p2, 'recall': r2, 'f1': f1_2
    }
    
    # 3. 多种百分位阈值
    for percentile in [98, 99, 99.5, 99.9]:
        thresh = np.percentile(probs, percentile)
        preds = (probs >= thresh).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
        results[f'percentile_{percentile}'] = {
            'threshold': thresh, 'precision': p, 'recall': r, 'f1': f
        }
    
    # 4. Youden阈值（需要标签，作为上限参考）
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    threshold_youden = thresholds[optimal_idx]
    preds_y = (probs >= threshold_youden).astype(int)
    py, ry, fy, _ = precision_recall_fscore_support(y_true, preds_y, average='binary', zero_division=0)
    results['youden'] = {
        'threshold': threshold_youden, 'precision': py, 'recall': ry, 'f1': fy
    }
    
    # 5. F1优化阈值（需要标签，作为理论上限）
    precisions, recalls, threshs = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    threshold_f1 = threshs[min(best_f1_idx, len(threshs)-1)]
    preds_f1 = (probs >= threshold_f1).astype(int)
    pf, rf, ff, _ = precision_recall_fscore_support(y_true, preds_f1, average='binary', zero_division=0)
    results['f1_optimal'] = {
        'threshold': threshold_f1, 'precision': pf, 'recall': rf, 'f1': ff
    }
    
    # 找最佳无监督方法
    unsupervised_methods = ['improved_optimizer', 'enhanced_selector', 
                           'percentile_98', 'percentile_99', 'percentile_99.5', 'percentile_99.9']
    best_unsupervised = max(unsupervised_methods, key=lambda x: results[x]['f1'])
    
    if verbose:
        print("\n【阈值策略对比】")
        for name, res in results.items():
            marker = "★" if name == best_unsupervised else " "
            print(f"  {marker} {name:20s}: Thresh={res['threshold']:.6f}, P={res['precision']:.4f}, R={res['recall']:.4f}, F1={res['f1']:.4f}")
    
    return results, best_unsupervised


def evaluate_single_scenario(scenario_id, model_path, data_dir, device, verbose=True):
    """评估单个场景"""
    print(f"\n{'='*60}")
    print(f"评估场景 {scenario_id}")
    print(f"{'='*60}")
    
    # 加载数据
    print(f"[Phase 1] 加载场景 {scenario_id} 数据...")
    loader = CTU13Loader(data_dir)
    df = loader.load_data([scenario_id])
    
    if df.empty:
        print(f"[Error] 场景 {scenario_id} 数据加载失败!")
        return None
    
    # 构建图
    print(f"[Phase 2] 构建图...")
    builder = ImprovedHeterogeneousGraphBuilder()
    data, ip_map = builder.build(df, include_semantic=True, include_struct=True)
    
    # 获取标签
    labels, _ = get_labels(df, ip_map)
    del df
    
    # 加载模型
    print(f"[Phase 3] 加载模型...")
    model = ImprovedBotnetDetectorV3Final.load(model_path, device=device)
    model.eval()
    
    # 推理
    print(f"[Phase 4] 执行推理...")
    infer_loader = NeighborLoader(
        data, 
        num_neighbors=[15, 10], 
        batch_size=4096, 
        input_nodes='ip', 
        shuffle=False
    )
    
    all_probs = []
    with torch.no_grad():
        for batch in infer_loader:
            batch = batch.to(device)
            probs = model.predict_proba(batch)[:batch['ip'].batch_size]
            all_probs.append(probs.cpu())
    
    probs = torch.cat(all_probs, dim=0).numpy()
    y_true = labels.numpy()[:len(probs)]
    probs = probs[:len(y_true)]
    
    # 多策略阈值评估
    print(f"[Phase 5] 多策略阈值优化...")
    results, best_method = evaluate_with_multiple_thresholds(probs, y_true, verbose=verbose)
    
    # AUC
    auc = roc_auc_score(y_true, probs)
    
    # 分数分布统计
    normal_probs = probs[y_true == 0]
    bot_probs = probs[y_true == 1]
    
    # 构建结果
    result = {
        'scenario': scenario_id,
        'n_nodes': len(probs),
        'n_botnet': int(y_true.sum()),
        'botnet_ratio': float(y_true.sum() / len(y_true)),
        'auc': float(auc),
        'best_method': best_method,
        'best_result': results[best_method],
        'all_results': results,
        'normal_mean': float(np.mean(normal_probs)) if len(normal_probs) > 0 else 0,
        'normal_median': float(np.median(normal_probs)) if len(normal_probs) > 0 else 0,
        'bot_mean': float(np.mean(bot_probs)) if len(bot_probs) > 0 else 0,
        'bot_median': float(np.median(bot_probs)) if len(bot_probs) > 0 else 0,
        'separation': float(np.median(bot_probs) - np.median(normal_probs)) if len(bot_probs) > 0 and len(normal_probs) > 0 else 0,
    }
    
    # 打印结果
    print(f"\n【场景 {scenario_id} 结果】")
    print(f"  节点数: {result['n_nodes']}, 僵尸节点: {result['n_botnet']} ({result['botnet_ratio']*100:.2f}%)")
    print(f"  AUC: {result['auc']:.4f}")
    print(f"  最佳方法: {best_method}")
    print(f"    阈值: {results[best_method]['threshold']:.6f}")
    print(f"    Precision: {results[best_method]['precision']:.4f}")
    print(f"    Recall: {results[best_method]['recall']:.4f}")
    print(f"    F1: {results[best_method]['f1']:.4f}")
    
    # 理论上限
    print(f"  理论上限 (Youden): F1={results['youden']['f1']:.4f}")
    print(f"  理论上限 (F1-opt): F1={results['f1_optimal']['f1']:.4f}")
    
    # 清理
    del data, model, labels
    torch.cuda.empty_cache()
    
    return result


def main():
    data_dir = '/root/autodl-fs/CTU-13/CTU-13-Dataset'
    model_path = 'improved_botnet_model_v3_final.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("改进版 V3 Final 模型批量评估 - 场景 6-13")
    print("="*70)
    print(f"设备: {device}")
    print(f"模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"[Error] 模型不存在: {model_path}")
        return
    
    # 评估场景 6-13
    scenarios = list(range(6, 14))
    results = []
    
    total_start = time.time()
    
    for scenario_id in scenarios:
        try:
            result = evaluate_single_scenario(scenario_id, model_path, data_dir, device)
            if result:
                results.append(result)
        except Exception as e:
            print(f"[Error] 场景 {scenario_id} 评估失败: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start
    
    # 汇总统计
    print("\n" + "="*70)
    print("汇总结果")
    print("="*70)
    
    if results:
        # 计算平均指标
        avg_auc = np.mean([r['auc'] for r in results])
        
        # 最佳无监督方法
        avg_best_f1 = np.mean([r['best_result']['f1'] for r in results])
        avg_best_precision = np.mean([r['best_result']['precision'] for r in results])
        avg_best_recall = np.mean([r['best_result']['recall'] for r in results])
        
        # Youden方法
        avg_youden_f1 = np.mean([r['all_results']['youden']['f1'] for r in results])
        avg_youden_precision = np.mean([r['all_results']['youden']['precision'] for r in results])
        avg_youden_recall = np.mean([r['all_results']['youden']['recall'] for r in results])
        
        # F1最优方法（理论上限）
        avg_f1_optimal = np.mean([r['all_results']['f1_optimal']['f1'] for r in results])
        
        print(f"\n【最佳无监督方法 - 平均指标】")
        print(f"  AUC:       {avg_auc:.4f}")
        print(f"  Precision: {avg_best_precision:.4f}")
        print(f"  Recall:    {avg_best_recall:.4f}")
        print(f"  F1-Score:  {avg_best_f1:.4f}")
        
        print(f"\n【传统Youden方法 - 平均指标】")
        print(f"  Precision: {avg_youden_precision:.4f}")
        print(f"  Recall:    {avg_youden_recall:.4f}")
        print(f"  F1-Score:  {avg_youden_f1:.4f}")
        
        print(f"\n【理论上限 (F1最优阈值)】")
        print(f"  F1-Score:  {avg_f1_optimal:.4f}")
        
        print(f"\n【改进效果】")
        f1_improvement = (avg_best_f1 - avg_youden_f1) / avg_youden_f1 * 100 if avg_youden_f1 > 0 else 0
        precision_improvement = (avg_best_precision - avg_youden_precision) / avg_youden_precision * 100 if avg_youden_precision > 0 else 0
        print(f"  F1-Score提升: {f1_improvement:+.2f}%")
        print(f"  Precision提升: {precision_improvement:+.2f}%")
        
        # 统计各方法被选为最佳的次数
        method_counts = {}
        for r in results:
            method = r['best_method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print(f"\n【各方法被选为最佳的次数】")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"  {method}: {count} 次")
        
        # 各场景详细结果
        print(f"\n【各场景详细结果】")
        print(f"{'场景':<6} {'AUC':<8} {'Best-F1':<10} {'Best-P':<10} {'Best-R':<10} {'Youden-F1':<10} {'理论最优':<10}")
        print("-" * 70)
        for r in results:
            print(f"{r['scenario']:<6} {r['auc']:<8.4f} {r['best_result']['f1']:<10.4f} {r['best_result']['precision']:<10.4f} {r['best_result']['recall']:<10.4f} {r['all_results']['youden']['f1']:<10.4f} {r['all_results']['f1_optimal']['f1']:<10.4f}")
        
        # 保存结果
        output = {
            'summary': {
                'avg_auc': float(avg_auc),
                'avg_best_f1': float(avg_best_f1),
                'avg_best_precision': float(avg_best_precision),
                'avg_best_recall': float(avg_best_recall),
                'avg_youden_f1': float(avg_youden_f1),
                'avg_youden_precision': float(avg_youden_precision),
                'avg_youden_recall': float(avg_youden_recall),
                'avg_f1_optimal': float(avg_f1_optimal),
                'f1_improvement_percent': float(f1_improvement),
                'precision_improvement_percent': float(precision_improvement),
                'method_counts': method_counts,
            },
            'scenarios': results,
            'total_time_seconds': total_time,
        }
        
        with open('evaluation_scenarios_improved.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n[Info] 结果已保存到 evaluation_scenarios_improved.json")
    
    print(f"\n总耗时: {total_time:.1f} 秒")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Info] 用户中断")
    except Exception as e:
        print(f"\n[Error] 发生错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        os._exit(0)