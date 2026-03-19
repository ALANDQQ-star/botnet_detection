"""
批量评估脚本 - 在场景6-13上评估V3 Final模型
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
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from torch_geometric.loader import NeighborLoader

from data_loader import CTU13Loader
from improved_heterogeneous_graph import ImprovedHeterogeneousGraphBuilder
from improved_model_v3_final import ImprovedBotnetDetectorV3Final
from smart_threshold_optimizer import SmartThresholdOptimizer


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


def evaluate_single_scenario(scenario_id, model_path, data_dir, device):
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
    
    # 智能阈值优化
    print(f"[Phase 5] 智能阈值优化...")
    optimizer = SmartThresholdOptimizer(verbose=False)
    smart_threshold = optimizer.find_threshold(probs)
    
    # 智能阈值预测
    smart_preds = (probs >= smart_threshold).astype(int)
    smart_precision, smart_recall, smart_f1, _ = precision_recall_fscore_support(
        y_true, smart_preds, average='binary', zero_division=0
    )
    
    # 传统方法 (ROC Youden指数)
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    youden_threshold = thresholds[optimal_idx]
    
    youden_preds = (probs >= youden_threshold).astype(int)
    youden_precision, youden_recall, youden_f1, _ = precision_recall_fscore_support(
        y_true, youden_preds, average='binary', zero_division=0
    )
    
    # AUC
    auc = roc_auc_score(y_true, probs)
    
    # 分数分布统计
    normal_probs = probs[y_true == 0]
    bot_probs = probs[y_true == 1]
    
    result = {
        'scenario': scenario_id,
        'n_nodes': len(probs),
        'n_botnet': int(y_true.sum()),
        'botnet_ratio': float(y_true.sum() / len(y_true)),
        'auc': float(auc),
        'smart_threshold': float(smart_threshold),
        'smart_precision': float(smart_precision),
        'smart_recall': float(smart_recall),
        'smart_f1': float(smart_f1),
        'youden_threshold': float(youden_threshold),
        'youden_precision': float(youden_precision),
        'youden_recall': float(youden_recall),
        'youden_f1': float(youden_f1),
        'normal_mean': float(np.mean(normal_probs)) if len(normal_probs) > 0 else 0,
        'normal_median': float(np.median(normal_probs)) if len(normal_probs) > 0 else 0,
        'bot_mean': float(np.mean(bot_probs)) if len(bot_probs) > 0 else 0,
        'bot_median': float(np.median(bot_probs)) if len(bot_probs) > 0 else 0,
        'separation': float(np.median(bot_probs) - np.median(normal_probs)) if len(bot_probs) > 0 and len(normal_probs) > 0 else 0,
        'estimated_ratio': optimizer.debug_info.get('estimated_ratio', 0),
    }
    
    print(f"\n【场景 {scenario_id} 结果】")
    print(f"  节点数: {result['n_nodes']}, 僵尸节点: {result['n_botnet']} ({result['botnet_ratio']*100:.2f}%)")
    print(f"  AUC: {result['auc']:.4f}")
    print(f"  智能阈值: {result['smart_threshold']:.6f}")
    print(f"    Precision: {result['smart_precision']:.4f}, Recall: {result['smart_recall']:.4f}, F1: {result['smart_f1']:.4f}")
    print(f"  Youden阈值: {result['youden_threshold']:.6f}")
    print(f"    Precision: {result['youden_precision']:.4f}, Recall: {result['youden_recall']:.4f}, F1: {result['youden_f1']:.4f}")
    
    # 清理
    del data, model, labels
    torch.cuda.empty_cache()
    
    return result


def main():
    data_dir = '/root/autodl-fs/CTU-13/CTU-13-Dataset'
    model_path = 'improved_botnet_model_v3_final.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("V3 Final 模型批量评估 - 场景 6-13")
    print("="*70)
    print(f"设备: {device}")
    print(f"模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"[Error] 模型不存在: {model_path}")
        return
    
    # 评估场景 6-13
    scenarios = list(range(4, 14))
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
        avg_smart_f1 = np.mean([r['smart_f1'] for r in results])
        avg_smart_precision = np.mean([r['smart_precision'] for r in results])
        avg_smart_recall = np.mean([r['smart_recall'] for r in results])
        avg_youden_f1 = np.mean([r['youden_f1'] for r in results])
        avg_youden_precision = np.mean([r['youden_precision'] for r in results])
        avg_youden_recall = np.mean([r['youden_recall'] for r in results])
        
        print(f"\n【智能阈值方法 - 平均指标】")
        print(f"  AUC:       {avg_auc:.4f}")
        print(f"  Precision: {avg_smart_precision:.4f}")
        print(f"  Recall:    {avg_smart_recall:.4f}")
        print(f"  F1-Score:  {avg_smart_f1:.4f}")
        
        print(f"\n【传统Youden方法 - 平均指标】")
        print(f"  Precision: {avg_youden_precision:.4f}")
        print(f"  Recall:    {avg_youden_recall:.4f}")
        print(f"  F1-Score:  {avg_youden_f1:.4f}")
        
        print(f"\n【改进效果】")
        f1_improvement = (avg_smart_f1 - avg_youden_f1) / avg_youden_f1 * 100 if avg_youden_f1 > 0 else 0
        precision_improvement = (avg_smart_precision - avg_youden_precision) / avg_youden_precision * 100 if avg_youden_precision > 0 else 0
        print(f"  F1-Score提升: {f1_improvement:+.2f}%")
        print(f"  Precision提升: {precision_improvement:+.2f}%")
        
        # 各场景详细结果
        print(f"\n【各场景详细结果】")
        print(f"{'场景':<6} {'AUC':<8} {'Smart-F1':<10} {'Smart-P':<10} {'Smart-R':<10} {'Youden-F1':<10}")
        print("-" * 60)
        for r in results:
            print(f"{r['scenario']:<6} {r['auc']:<8.4f} {r['smart_f1']:<10.4f} {r['smart_precision']:<10.4f} {r['smart_recall']:<10.4f} {r['youden_f1']:<10.4f}")
        
        # 保存结果
        output = {
            'summary': {
                'avg_auc': float(avg_auc),
                'avg_smart_f1': float(avg_smart_f1),
                'avg_smart_precision': float(avg_smart_precision),
                'avg_smart_recall': float(avg_smart_recall),
                'avg_youden_f1': float(avg_youden_f1),
                'avg_youden_precision': float(avg_youden_precision),
                'avg_youden_recall': float(avg_youden_recall),
                'f1_improvement_percent': float(f1_improvement),
                'precision_improvement_percent': float(precision_improvement),
            },
            'scenarios': results,
            'total_time_seconds': total_time,
        }
        
        with open('evaluation_scenarios_6_13.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n[Info] 结果已保存到 evaluation_scenarios_6_13.json")
    
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