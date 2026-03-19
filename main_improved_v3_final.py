"""
V3 Final 训练脚本
核心改进：移除校准层和温度缩放，保持分数区分度
"""

import sys
import time
import os
import warnings
import argparse
import gc
import json
import numpy as np
import pandas as pd

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from torch_geometric.loader import NeighborLoader

from data_loader import CTU13Loader
from improved_heterogeneous_graph import ImprovedHeterogeneousGraphBuilder
from improved_model_v3_final import ImprovedBotnetDetectorV3Final, ImprovedTrainerV3Final
from smart_threshold_optimizer import SmartThresholdOptimizer


def parse_args():
    parser = argparse.ArgumentParser(description='V3 Final - 移除校准层，保持分数区分度')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-fs/CTU-13/CTU-13-Dataset')
    parser.add_argument('--train_scenarios', type=str, default='1,2,3')
    parser.add_argument('--test_scenarios', type=str, default='13')
    parser.add_argument('--model_path', type=str, default='improved_botnet_model_v3_final.pth')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--no_semantic', action='store_true', help='不使用语义特征')
    parser.add_argument('--no_struct', action='store_true', help='不使用结构特征')
    return parser.parse_args()


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


def analyze_score_distribution(probs, y_true, name=""):
    """分析分数分布并返回关键指标"""
    normal_probs = probs[y_true == 0]
    bot_probs = probs[y_true == 1]
    
    normal_mean = np.mean(normal_probs)
    normal_median = np.median(normal_probs)
    normal_std = np.std(normal_probs)
    bot_mean = np.mean(bot_probs)
    bot_median = np.median(bot_probs)
    bot_std = np.std(bot_probs)
    
    separation = bot_median - normal_median
    
    # 计算分布重叠
    from scipy.stats import gaussian_kde
    try:
        if len(normal_probs) > 1 and len(bot_probs) > 1:
            x_range = np.linspace(0, 1, 100)
            kde_normal = gaussian_kde(normal_probs)
            kde_bot = gaussian_kde(bot_probs)
            overlap = np.minimum(kde_normal(x_range), kde_bot(x_range)).sum() / 100
        else:
            overlap = 1.0
    except:
        overlap = 1.0
    
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    preds = (probs >= optimal_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average='binary', zero_division=0
    )
    
    auc = roc_auc_score(y_true, probs)
    
    return {
        'name': name,
        'auc': auc,
        'normal_mean': normal_mean,
        'normal_median': normal_median,
        'normal_std': normal_std,
        'bot_mean': bot_mean,
        'bot_median': bot_median,
        'bot_std': bot_std,
        'separation': separation,
        'overlap': overlap,
        'optimal_threshold': optimal_threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def train_model(args, device):
    """训练 V3 Final 模型，返回训练集先验信息"""
    print("\n" + "="*60)
    print("开始训练 V3 Final 僵尸网络检测模型")
    print("核心改进：移除校准层和温度缩放，保持分数区分度")
    print("="*60)

    train_prior = None
    prior_path = args.model_path.replace('.pth', '_prior.json')

    if os.path.exists(args.model_path) and not args.force_retrain:
        print(f"[Train] 模型已存在：{args.model_path}，跳过训练")
        # 尝试加载已有的先验
        if os.path.exists(prior_path):
            with open(prior_path, 'r') as f:
                train_prior = json.load(f)
            print(f"[Train] 已加载训练集先验阈值: {train_prior.get('threshold', 'N/A')}")
        return train_prior
    
    # 加载数据
    print("\n[Phase 1] 加载训练数据...")
    loader = CTU13Loader(args.data_dir)
    train_scenarios = [int(s) for s in args.train_scenarios.split(',')]
    df = loader.load_data(train_scenarios)
    
    if df.empty:
        print("[Error] 数据加载失败!")
        return
    
    # 构建图
    print("\n[Phase 2] 构建三模态异构图...")
    builder = ImprovedHeterogeneousGraphBuilder()
    data, ip_map = builder.build(
        df, 
        include_semantic=not args.no_semantic,
        include_struct=not args.no_struct
    )
    
    # 获取标签
    labels, _ = get_labels(df, ip_map)
    del df
    gc.collect()
    
    # 准备训练数据
    print("\n[Phase 3] 准备训练数据...")
    num_nodes = labels.size(0)
    indices = np.arange(num_nodes)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(num_nodes * 0.8)
    train_idx = indices[:split_idx]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    data['ip'].train_mask = train_mask
    data['ip'].y = labels
    
    # 创建数据加载器
    train_loader = NeighborLoader(
        data, 
        num_neighbors=[15, 10], 
        batch_size=args.batch_size,
        input_nodes=('ip', data['ip'].train_mask), 
        shuffle=True,
        num_workers=0
    )
    
    # 创建模型
    print("\n[Phase 4] 创建模型...")
    stat_dim = data['ip'].x.shape[1]
    semantic_dim = data['ip'].semantic_x.shape[1] if data['ip'].semantic_x is not None else 32
    struct_dim = data['ip'].struct_x.shape[1] if data['ip'].struct_x is not None else 16
    
    model = ImprovedBotnetDetectorV3Final(
        stat_dim=stat_dim,
        semantic_dim=semantic_dim,
        struct_dim=struct_dim,
        hidden_dim=args.hidden_dim
    )
    
    print(f"  - 统计特征维度：{stat_dim}")
    print(f"  - 语义特征维度：{semantic_dim}")
    print(f"  - 结构特征维度：{struct_dim}")
    print(f"  - 隐藏层维度：{args.hidden_dim}")
    print(f"  - 校准层：已移除")
    print(f"  - 温度缩放：已移除")
    
    trainer = ImprovedTrainerV3Final(model, lr=args.lr, device=device)
    
    # 训练
    print("\n[Phase 5] 开始训练...")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Batch Size: {args.batch_size}")
    print()
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        avg_loss = trainer.train_epoch(train_loader, epoch)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            model.save(args.model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Train] 早停触发，在第 {epoch+1} 轮")
                break
    
    print(f"\n[Train] 训练完成！最佳损失：{best_loss:.4f}")

    # ==============================
    # Phase 6: 计算训练集先验（用验证集部分）
    # ==============================
    print("\n[Phase 6] 计算训练集阈值先验...")
    model = ImprovedBotnetDetectorV3Final.load(args.model_path, device=device)
    model.eval()

    val_idx = indices[split_idx:]
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_idx] = True

    val_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        batch_size=4096,
        input_nodes=('ip', val_mask),
        shuffle=False,
        num_workers=0
    )

    val_probs = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            probs = model.predict_proba(batch)[:batch['ip'].batch_size]
            val_probs.append(probs.cpu())

    val_probs = torch.cat(val_probs, dim=0).numpy()
    val_labels = labels.numpy()[val_idx[:len(val_probs)]]
    val_probs = val_probs[:len(val_labels)]

    # 用验证集真实标签找最优阈值（Youden + PR曲线取较好者）
    from sklearn.metrics import precision_recall_curve
    fpr, tpr, roc_thresholds = roc_curve(val_labels, val_probs)
    youden_j = tpr - fpr
    youden_idx = np.argmax(youden_j)
    youden_thresh = roc_thresholds[youden_idx]

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(val_labels, val_probs)
    pr_f1 = 2 * pr_precision * pr_recall / (pr_precision + pr_recall + 1e-10)
    pr_best_idx = np.argmax(pr_f1)
    pr_thresh = pr_thresholds[pr_best_idx] if pr_best_idx < len(pr_thresholds) else youden_thresh

    # 选F1更高的阈值
    from sklearn.metrics import precision_recall_fscore_support
    for t_name, t_val in [('youden', youden_thresh), ('pr_curve', pr_thresh)]:
        preds = (val_probs >= t_val).astype(int)
        p, r, f, _ = precision_recall_fscore_support(val_labels, preds, average='binary', zero_division=0)
        print(f"  验证集 {t_name}: thresh={t_val:.6f}, P={p:.4f}, R={r:.4f}, F1={f:.4f}")

    youden_preds = (val_probs >= youden_thresh).astype(int)
    _, _, youden_f1, _ = precision_recall_fscore_support(val_labels, youden_preds, average='binary', zero_division=0)
    pr_preds = (val_probs >= pr_thresh).astype(int)
    _, _, pr_f1_val, _ = precision_recall_fscore_support(val_labels, pr_preds, average='binary', zero_division=0)

    best_thresh = pr_thresh if pr_f1_val >= youden_f1 else youden_thresh
    print(f"  选择阈值: {best_thresh:.6f} (方法: {'pr_curve' if pr_f1_val >= youden_f1 else 'youden'})")

    # 计算bot和normal的分数统计
    bot_scores = val_probs[val_labels == 1]
    normal_scores = val_probs[val_labels == 0]

    # 计算阈值在验证集中的分位数位置
    threshold_percentile = float(np.mean(val_probs < best_thresh) * 100)

    train_prior = {
        'threshold': float(best_thresh),
        'threshold_percentile': threshold_percentile,
        'bot_score_median': float(np.median(bot_scores)) if len(bot_scores) > 0 else None,
        'bot_score_mean': float(np.mean(bot_scores)) if len(bot_scores) > 0 else None,
        'normal_score_median': float(np.median(normal_scores)) if len(normal_scores) > 0 else None,
        'normal_score_mean': float(np.mean(normal_scores)) if len(normal_scores) > 0 else None,
        'bot_ratio': float(val_labels.sum() / len(val_labels)),
    }

    with open(prior_path, 'w') as f:
        json.dump(train_prior, f, indent=2)
    print(f"  训练集先验已保存至: {prior_path}")

    # 清理
    del data, model, labels
    torch.cuda.empty_cache()
    gc.collect()

    return train_prior


def evaluate_model(args, device, verbose=True, train_prior=None):
    """评估模型"""
    if verbose:
        print("\n" + "="*60)
        print("开始评估 V3 Final 模型")
        print("="*60)
    
    if not os.path.exists(args.model_path):
        if verbose:
            print(f"[Error] 模型不存在：{args.model_path}")
        return None
    
    # 加载测试数据
    if verbose:
        print("\n[Phase 1] 加载测试数据...")
    loader = CTU13Loader(args.data_dir)
    test_scenarios = [int(s) for s in args.test_scenarios.split(',')]
    df = loader.load_data(test_scenarios)
    
    if df.empty:
        if verbose:
            print("[Error] 数据加载失败!")
        return None
    
    # 构建图
    if verbose:
        print("\n[Phase 2] 构建测试图...")
    builder = ImprovedHeterogeneousGraphBuilder()
    data, ip_map = builder.build(
        df,
        include_semantic=not args.no_semantic,
        include_struct=not args.no_struct
    )
    
    # 获取标签
    labels, _ = get_labels(df, ip_map)
    del df
    gc.collect()
    
    # 加载模型
    if verbose:
        print("\n[Phase 3] 加载模型...")
    model = ImprovedBotnetDetectorV3Final.load(args.model_path, device=device)
    model.eval()
    
    # 推理
    if verbose:
        print("\n[Phase 4] 执行推理...")
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
    
    # 使用智能阈值优化器
    if verbose:
        print("\n[Phase 5] 智能阈值优化...")
    optimizer = SmartThresholdOptimizer(verbose=verbose, train_prior=train_prior)
    smart_threshold = optimizer.find_threshold(probs)
    
    # 使用智能阈值进行预测
    smart_preds = (probs >= smart_threshold).astype(int)
    smart_precision, smart_recall, smart_f1, _ = precision_recall_fscore_support(
        y_true, smart_preds, average='binary', zero_division=0
    )
    
    # 分析分布（用于对比）
    result = analyze_score_distribution(probs, y_true, name=f"Scenario_{args.test_scenarios}")
    
    # 添加智能阈值结果
    result['smart_threshold'] = smart_threshold
    result['smart_precision'] = smart_precision
    result['smart_recall'] = smart_recall
    result['smart_f1'] = smart_f1
    result['estimated_anomaly_ratio'] = optimizer.debug_info.get('estimated_ratio', 0)
    
    if verbose:
        print("\n" + "="*60)
        print("评估结果（V3 Final 模型 + 智能阈值优化）")
        print("="*60)
        
        print(f"\n【智能阈值方法】")
        print(f"  估计异常比例: {result['estimated_anomaly_ratio']:.4f}")
        print(f"  智能阈值:     {smart_threshold:.6f}")
        print(f"  AUC:          {result['auc']:.4f}")
        print(f"  Precision:    {smart_precision:.4f}")
        print(f"  Recall:       {smart_recall:.4f}")
        print(f"  F1-Score:     {smart_f1:.4f}")
        
        print(f"\n【传统方法对比（ROC Youden指数）】")
        print(f"  Optimal Thresh: {result['optimal_threshold']:.6f}")
        print(f"  Precision:    {result['precision']:.4f}")
        print(f"  Recall:       {result['recall']:.4f}")
        print(f"  F1-Score:     {result['f1']:.4f}")
        
        print(f"\n【分数分布特征】")
        print(f"  正常节点分数:")
        print(f"    - 均值：{result['normal_mean']:.6f}")
        print(f"    - 中位数：{result['normal_median']:.6f}")
        print(f"    - 标准差：{result['normal_std']:.6f}")
        print(f"  僵尸节点分数:")
        print(f"    - 均值：{result['bot_mean']:.6f}")
        print(f"    - 中位数：{result['bot_median']:.6f}")
        print(f"    - 标准差：{result['bot_std']:.6f}")
        
        print(f"\n【分离度指标】")
        print(f"  中位数分离度：{result['separation']:.4f}")
        print(f"  分布重叠面积：{result['overlap']:.4f}")
        
        print("="*60)
    
    return result, probs, y_true


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Config] 使用设备：{device}")
    print(f"[Config] 训练场景：{args.train_scenarios}")
    print(f"[Config] 测试场景：{args.test_scenarios}")
    print(f"[Config] 语义特征：{'禁用' if args.no_semantic else '启用'}")
    print(f"[Config] 结构特征：{'禁用' if args.no_struct else '启用'}")
    
    # 训练
    train_prior = train_model(args, device)

    # 评估
    result = evaluate_model(args, device, train_prior=train_prior)
    
    # 保存结果
    if result:
        result_dict, probs, y_true = result
        
        def to_python_type(val):
            if hasattr(val, 'item'):
                return val.item()
            elif isinstance(val, (np.int32, np.int64)):
                return int(val)
            elif isinstance(val, (np.float32, np.float64)):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            return val
        
        json_results = {k: to_python_type(v) for k, v in result_dict.items()}
        
        np.savez(
            'v3_final_score_distribution.npz',
            probs=probs,
            y_true=y_true,
            normal_probs=probs[y_true == 0],
            bot_probs=probs[y_true == 1]
        )
        
        with open('v3_final_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n[Info] 结果已保存到 v3_final_results.json 和 v3_final_score_distribution.npz")
    
    print("\n" + "="*60)
    print("所有任务完成！")
    print("="*60)


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