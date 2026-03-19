"""快速分析场景12的分数分布"""
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
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_recall_curve
import json

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

def main():
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
    labels, bot_ips = get_labels(df, ip_map)
    
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
    
    # 分析分布
    bot_probs = probs[y_true == 1]
    normal_probs = probs[y_true == 0]
    
    print("\n" + "="*70)
    print("分数分布分析")
    print("="*70)
    
    print(f"\n[数据概览]")
    print(f"  总节点数：{len(probs)}")
    print(f"  僵尸节点数：{len(bot_probs)} ({len(bot_probs)/len(probs)*100:.2f}%)")
    print(f"  正常节点数：{len(normal_probs)}")
    
    print(f"\n[僵尸节点分数统计]")
    print(f"  均值：{np.mean(bot_probs):.6f}")
    print(f"  中位数：{np.median(bot_probs):.6f}")
    print(f"  标准差：{np.std(bot_probs):.6f}")
    print(f"  最小值：{np.min(bot_probs):.6f}")
    print(f"  最大值：{np.max(bot_probs):.6f}")
    print(f"  5%分位：{np.percentile(bot_probs, 5):.6f}")
    print(f"  25%分位：{np.percentile(bot_probs, 25):.6f}")
    print(f"  75%分位：{np.percentile(bot_probs, 75):.6f}")
    print(f"  95%分位：{np.percentile(bot_probs, 95):.6f}")
    
    print(f"\n[正常节点分数统计]")
    print(f"  均值：{np.mean(normal_probs):.6f}")
    print(f"  中位数：{np.median(normal_probs):.6f}")
    print(f"  标准差：{np.std(normal_probs):.6f}")
    print(f"  最小值：{np.min(normal_probs):.6f}")
    print(f"  最大值：{np.max(normal_probs):.6f}")
    print(f"  5%分位：{np.percentile(normal_probs, 5):.6f}")
    print(f"  25%分位：{np.percentile(normal_probs, 25):.6f}")
    print(f"  75%分位：{np.percentile(normal_probs, 75):.6f}")
    print(f"  95%分位：{np.percentile(normal_probs, 95):.6f}")
    
    # AUC
    auc = roc_auc_score(y_true, probs)
    print(f"\n[AUC]: {auc:.4f}")
    
    # 分析不同阈值下的表现
    print(f"\n[阈值分析]")
    print(f"  {'阈值':<12} {'预测数':<10} {'TP':<8} {'FP':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}")
    print(f"  {'-'*78}")
    
    best_f1 = 0
    best_thresh = 0
    best_result = None
    
    for thresh in np.concatenate([
        np.array([0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]),
        np.linspace(0.05, 0.5, 10)
    ]):
        preds = (probs >= thresh).astype(int)
        num_pred = preds.sum()
        if num_pred == 0:
            continue
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        
        precision = tp / num_pred
        recall = tp / len(bot_probs)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_result = (num_pred, tp, fp, precision, recall, f1)
        
        print(f"  {thresh:<12.6f} {num_pred:<10} {tp:<8} {fp:<10} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
    
    print(f"\n[最优F1阈值]: {best_thresh:.6f}")
    print(f"  预测数: {best_result[0]}, TP: {best_result[1]}, FP: {best_result[2]}")
    print(f"  精确率: {best_result[3]:.4f}, 召回率: {best_result[4]:.4f}, F1: {best_result[5]:.4f}")
    
    # 详细分析当前阈值为什么失败
    current_thresh = 0.0138  # 从原始输出获取
    print(f"\n[当前阈值分析]: {current_thresh}")
    preds_current = (probs >= current_thresh).astype(int)
    num_pred_current = preds_current.sum()
    tp_current = ((preds_current == 1) & (y_true == 1)).sum()
    
    print(f"  预测僵尸节点: {num_pred_current}")
    print(f"  真正例 (TP): {tp_current}")
    print(f"  假正例 (FP): {num_pred_current - tp_current}")
    print(f"  僵尸节点分数 >= 阈值的数量: {(bot_probs >= current_thresh).sum()}")
    print(f"  正常节点分数 >= 阈值的数量: {(normal_probs >= current_thresh).sum()}")
    
    # 分析分数分布的重叠情况
    print(f"\n[分布重叠分析]")
    print(f"  僵尸节点最高分: {np.max(bot_probs):.6f}")
    print(f"  正常节点最高分: {np.max(normal_probs):.6f}")
    print(f"  僵尸节点最低分: {np.min(bot_probs):.6f}")
    print(f"  正常节点最低分: {np.min(normal_probs):.6f}")
    
    # 使用 PR 曲线找最优阈值
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    
    print(f"\n[PR曲线最优阈值]")
    print(f"  阈值: {thresholds[best_idx]:.6f}")
    print(f"  精确率: {precisions[best_idx]:.4f}")
    print(f"  召回率: {recalls[best_idx]:.4f}")
    print(f"  F1: {f1_scores[best_idx]:.4f}")
    
    # 保存分数数据
    np.savez('s12_scores.npz', probs=probs, y_true=y_true, bot_probs=bot_probs, normal_probs=normal_probs)
    print(f"\n[Info] 分数数据已保存到 s12_scores.npz")

if __name__ == "__main__":
    main()