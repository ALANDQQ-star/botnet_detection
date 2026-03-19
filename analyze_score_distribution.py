"""
分析僵尸节点和非僵尸节点的分数分布
诊断 AUC 高但检测效果差的原因
"""

import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from data_loader import CTU13Loader
from improved_heterogeneous_graph import ImprovedHeterogeneousGraphBuilder
from improved_model import ImprovedBotnetDetector
from torch_geometric.loader import NeighborLoader
import json

# 尝试导入 matplotlib，如果失败则使用文本分析
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib 不可用，将使用文本分析模式")


def get_labels(df, ip_map):
    """提取节点标签"""
    bot_keywords = ['botnet', 'cc', 'c&c', 'neris', 'rbot', 'virut', 'menti', 'murlo', 'sogou']
    df['is_bot'] = df['label'].apply(lambda x: any(k in str(x).lower() for k in bot_keywords))
    
    bot_ips = set(df[df['is_bot']]['src_ip'].unique()) | set(df[df['is_bot']]['dst_ip'].unique())
    y = np.zeros(len(ip_map), dtype=np.float32)
    for ip, idx in ip_map.items():
        if ip in bot_ips:
            y[idx] = 1.0
    return torch.tensor(y), list(bot_ips)


def analyze_scenario(scenario, data_dir='/root/autodl-fs/CTU-13/CTU-13-Dataset', 
                     model_path='improved_botnet_model.pth'):
    """分析特定场景的分数分布"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"分析场景 {scenario}")
    print(f"{'='*70}")
    
    # 加载数据
    loader = CTU13Loader(data_dir)
    df = loader.load_data([scenario])
    
    if df.empty:
        print(f"[Error] 场景 {scenario} 数据加载失败!")
        return None
    
    # 构建图
    builder = ImprovedHeterogeneousGraphBuilder()
    data, ip_map = builder.build(df, include_semantic=True, include_struct=True)
    
    # 获取标签
    labels, bot_ips = get_labels(df, ip_map)
    
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
    
    # 分析分布
    bot_probs = probs[y_true == 1]
    normal_probs = probs[y_true == 0]
    
    print(f"\n[数据概览]")
    print(f"  总节点数：{len(probs)}")
    print(f"  僵尸节点数：{len(bot_probs)} ({len(bot_probs)/len(probs)*100:.2f}%)")
    print(f"  正常节点数：{len(normal_probs)}")
    
    print(f"\n[分数统计]")
    print(f"  僵尸节点分数:")
    print(f"    - 均值：{np.mean(bot_probs):.4f}")
    print(f"    - 中位数：{np.median(bot_probs):.4f}")
    print(f"    - 标准差：{np.std(bot_probs):.4f}")
    print(f"    - 最小值：{np.min(bot_probs):.4f}")
    print(f"    - 最大值：{np.max(bot_probs):.4f}")
    print(f"    - 25% 分位：{np.percentile(bot_probs, 25):.4f}")
    print(f"    - 75% 分位：{np.percentile(bot_probs, 75):.4f}")
    
    print(f"\n  正常节点分数:")
    print(f"    - 均值：{np.mean(normal_probs):.4f}")
    print(f"    - 中位数：{np.median(normal_probs):.4f}")
    print(f"    - 标准差：{np.std(normal_probs):.4f}")
    print(f"    - 最小值：{np.min(normal_probs):.4f}")
    print(f"    - 最大值：{np.max(normal_probs):.4f}")
    print(f"    - 25% 分位：{np.percentile(normal_probs, 25):.4f}")
    print(f"    - 75% 分位：{np.percentile(normal_probs, 75):.4f}")
    
    # 分析不同阈值下的性能
    print(f"\n[阈值分析]")
    print(f"  {'阈值':<10} {'预测僵尸':<10} {'真正例':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}")
    print(f"  {'-'*66}")
    
    thresholds_to_check = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for thresh in thresholds_to_check:
        preds = (probs >= thresh).astype(int)
        num_pred = preds.sum()
        tp = ((preds == 1) & (y_true == 1)).sum()
        
        precision = tp / num_pred if num_pred > 0 else 0
        recall = tp / len(bot_probs) if len(bot_probs) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {thresh:<10.2f} {num_pred:<10} {tp:<10} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
    
    # 找出最优阈值
    from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score
    
    auc = roc_auc_score(y_true, probs)
    print(f"\n[AUC 指标]: {auc:.4f}")
    
    # 找到 F1 最优的阈值
    best_f1 = 0
    best_thresh = 0
    for thresh in np.linspace(0.01, 0.99, 100):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    print(f"\n[F1 最优阈值]: {best_thresh:.4f}, F1 = {best_f1:.4f}")
    
    # 使用当前分类器的结果
    from enhanced_classifier import compute_botnet_metrics
    result = compute_botnet_metrics(y_true, probs)
    print(f"\n[当前分类器结果]:")
    print(f"  方法：{result['method']}")
    print(f"  阈值：{result['threshold']:.4f}")
    print(f"  预测僵尸：{result['num_predicted']}")
    print(f"  精确率：{result['precision']:.4f}")
    print(f"  召回率：{result['recall']:.4f}")
    print(f"  F1: {result['f1']:.4f}")
    
    # 计算僵尸节点超过阈值的比例
    bot_above_thresh = (bot_probs >= result['threshold']).sum()
    print(f"\n[关键发现]:")
    print(f"  僵尸节点中分数 >= 阈值的比例：{bot_above_thresh}/{len(bot_probs)} = {bot_above_thresh/len(bot_probs)*100:.1f}%")
    print(f"  这意味着 {len(bot_probs) - bot_above_thresh} 个僵尸节点的分数低于阈值")
    
    # 分析为什么分类器选择了这个阈值
    print(f"\n[分布重叠分析]:")
    print(f"  正常节点分数 > 阈值的比例：{(normal_probs >= result['threshold']).sum() / len(normal_probs) * 100:.2f}%")
    print(f"  僵尸节点分数 < 阈值的比例：{(bot_probs < result['threshold']).sum() / len(bot_probs) * 100:.2f}%")
    
    # 保存详细数据
    results = {
        'scenario': scenario,
        'total_nodes': len(probs),
        'num_bots': len(bot_probs),
        'num_normals': len(normal_probs),
        'bot_stats': {
            'mean': float(np.mean(bot_probs)),
            'median': float(np.median(bot_probs)),
            'std': float(np.std(bot_probs)),
            'min': float(np.min(bot_probs)),
            'max': float(np.max(bot_probs)),
            'q25': float(np.percentile(bot_probs, 25)),
            'q75': float(np.percentile(bot_probs, 75))
        },
        'normal_stats': {
            'mean': float(np.mean(normal_probs)),
            'median': float(np.median(normal_probs)),
            'std': float(np.std(normal_probs)),
            'min': float(np.min(normal_probs)),
            'max': float(np.max(normal_probs)),
            'q25': float(np.percentile(normal_probs, 25)),
            'q75': float(np.percentile(normal_probs, 75))
        },
        'auc': float(auc),
        'best_f1_threshold': float(best_thresh),
        'best_f1': float(best_f1),
        'classifier_result': result,
        'bot_scores': bot_probs.tolist(),
        'normal_scores': normal_probs.tolist()
    }
    
    return results


def plot_distribution(results_list, save_path='score_distribution.png'):
    """绘制分数分布图"""
    if not HAS_MATPLOTLIB or not results_list:
        print("\n[绘图] 跳过绘图（无 matplotlib 或无数据）")
        return
    
    fig, axes = plt.subplots(2, len(results_list), figsize=(7*len(results_list), 10))
    if len(results_list) == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, results in enumerate(results_list):
        ax_hist = axes[0, idx]
        ax_cdf = axes[1, idx]
        
        bot_probs = np.array(results['bot_scores'])
        normal_probs = np.array(results['normal_scores'])
        
        # 直方图
        ax_hist.hist(normal_probs, bins=50, alpha=0.5, label=f"Normal (n={len(normal_probs)})", 
                     color='blue', density=True)
        ax_hist.hist(bot_probs, bins=50, alpha=0.5, label=f"Botnet (n={len(bot_probs)})", 
                     color='red', density=True)
        
        # 标记阈值
        thresh = results['classifier_result']['threshold']
        ax_hist.axvline(thresh, color='black', linestyle='--', label=f"Threshold={thresh:.3f}")
        
        # 标记最优 F1 阈值
        best_thresh = results['best_f1_threshold']
        ax_hist.axvline(best_thresh, color='green', linestyle=':', label=f"Best F1={best_thresh:.3f}")
        
        ax_hist.set_xlabel('Probability Score')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title(f"Scenario {results['scenario']}\nAUC={results['auc']:.4f}, Best F1={results['best_f1']:.4f}")
        ax_hist.legend()
        
        # CDF 图
        sorted_normal = np.sort(normal_probs)
        sorted_bot = np.sort(bot_probs)
        cdf_normal = np.arange(1, len(sorted_normal)+1) / len(sorted_normal)
        cdf_bot = np.arange(1, len(sorted_bot)+1) / len(sorted_bot)
        
        ax_cdf.plot(sorted_normal, cdf_normal, label='Normal', color='blue')
        ax_cdf.plot(sorted_bot, cdf_bot, label='Botnet', color='red')
        ax_cdf.axvline(thresh, color='black', linestyle='--', label=f"Current={thresh:.3f}")
        ax_cdf.axvline(best_thresh, color='green', linestyle=':', label=f"Best={best_thresh:.3f}")
        
        ax_cdf.set_xlabel('Probability Score')
        ax_cdf.set_ylabel('CDF')
        ax_cdf.set_title('Cumulative Distribution')
        ax_cdf.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[绘图] 分布图已保存到：{save_path}")
    plt.close()


def text_plot_distribution(results_list):
    """使用文本绘制简单的分布示意"""
    print("\n" + "="*70)
    print("分数分布可视化 (文本版)")
    print("="*70)
    
    for results in results_list:
        bot_probs = np.array(results['bot_scores'])
        normal_probs = np.array(results['normal_scores'])
        thresh = results['classifier_result']['threshold']
        best_thresh = results['best_f1_threshold']
        
        print(f"\n场景 {results['scenario']} (AUC={results['auc']:.4f}):")
        print("-"*60)
        
        # 创建分数段直方图
        bins = np.linspace(0, 1, 21)
        normal_hist, _ = np.histogram(normal_probs, bins=bins)
        bot_hist, _ = np.histogram(bot_probs, bins=bins)
        
        # 归一化以便比较
        normal_max = normal_hist.max() if normal_hist.max() > 0 else 1
        bot_max = bot_hist.max() if bot_hist.max() > 0 else 1
        
        print(f"{'分数段':<10} {'正常节点':<30} {'僵尸节点':<30}")
        print("-"*60)
        
        for i in range(len(bins)-1):
            range_str = f"[{bins[i]:.1f}-{bins[i+1]:.1f})"
            normal_bar = "█" * int(normal_hist[i] / normal_max * 25)
            bot_bar = "█" * int(bot_hist[i] / bot_max * 25)
            print(f"{range_str:<10} {normal_bar:<30} {bot_bar:<30}")
        
        # 标记阈值位置
        thresh_bin = int(thresh * 20)
        print(f"\n当前阈值位置：{thresh:.3f} (在第 {thresh_bin} 个区间)")
        print(f"最优 F1 阈值：{best_thresh:.3f}")
        
        # 关键分析
        bot_below_thresh = (bot_probs < thresh).sum()
        normal_above_thresh = (normal_probs >= thresh).sum()
        
        print(f"\n[关键问题诊断]:")
        print(f"  - {bot_below_thresh} 个僵尸节点 ({bot_below_thresh/len(bot_probs)*100:.1f}%) 分数低于当前阈值")
        print(f"  - {normal_above_thresh} 个正常节点 ({normal_above_thresh/len(normal_probs)*100:.1f}%) 分数高于当前阈值")
        print(f"  - 僵尸节点最高分：{np.max(bot_probs):.4f}")
        print(f"  - 正常节点最高分：{np.max(normal_probs):.4f}")
        print(f"  - 僵尸节点中位数：{np.median(bot_probs):.4f}")
        print(f"  - 正常节点中位数：{np.median(normal_probs):.4f}")


def main():
    # 分析多个场景
    print("开始分析分数分布...")
    
    # 检查模型文件
    model_path = 'improved_botnet_model.pth'
    if not os.path.exists(model_path):
        print(f"[Error] 模型文件不存在：{model_path}")
        return
    
    results = []
    
    # 测试所有可用场景
    test_scenarios = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    for scenario in test_scenarios:
        try:
            print(f"\n{'='*70}")
            print(f"测试场景 {scenario}")
            print(f"{'='*70}")
            
            result = analyze_scenario(scenario, model_path=model_path)
            if result:
                results.append(result)
        except Exception as e:
            print(f"[Error] 场景 {scenario} 分析失败：{e}")
        
        # 清理内存
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 可视化
    if HAS_MATPLOTLIB and results:
        plot_distribution(results)
    
    # 文本可视化
    text_plot_distribution(results)
    
    # 保存详细结果
    with open('score_analysis.json', 'w') as f:
        # 移除大的分数列表以便阅读
        json_save = []
        for r in results:
            r_copy = r.copy()
            json_save.append(r_copy)
        json.dump(json_save, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    
    print(f"\n[Info] 详细分析结果已保存到 score_analysis.json")
    
    # 总结问题
    print("\n" + "="*70)
    print("问题诊断总结")
    print("="*70)
    
    for r in results:
        print(f"\n场景 {r['scenario']}:")
        thresh = r['classifier_result']['threshold']
        bot_below = (np.array(r['bot_scores']) < thresh).sum()
        bot_total = r['num_bots']
        
        if bot_below > bot_total * 0.5:
            print(f"  ⚠️ 问题：超过{bot_below/bot_total*100:.1f}%的僵尸节点分数低于阈值！")
            print(f"     原因：模型预测的僵尸节点分数普遍偏低")
            print(f"     建议：降低分类阈值或改进模型训练")
        else:
            print(f"  ✓ 僵尸节点分数分布相对合理")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[Error] 分析过程中出错：{e}")
        import traceback
        traceback.print_exc()