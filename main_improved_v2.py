"""
改进的僵尸网络检测训练脚本 V2
核心改进：增强分数区分度 + 分数校准
"""

import sys
import time
import math
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
from improved_heterogeneous_graph import ImprovedHeterogeneousGraphBuilder, MiniBatchGraphBuilder
from improved_model_v2 import ImprovedBotnetDetectorV2, ImprovedTrainerV2


def parse_args():
    parser = argparse.ArgumentParser(description='改进的僵尸网络检测 V2 - 增强分数区分度')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-fs/CTU-13/CTU-13-Dataset')
    parser.add_argument('--train_scenarios', type=str, default='1,2,3')
    parser.add_argument('--test_scenarios', type=str, default='13')
    parser.add_argument('--model_path', type=str, default='improved_botnet_model_v2.pth')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--no_semantic', action='store_true', help='不使用语义特征')
    parser.add_argument('--no_struct', action='store_true', help='不使用结构特征')
    parser.add_argument('--no_calibration', action='store_true', help='不使用校准层')
    parser.add_argument('--no_temperature', action='store_true', help='不使用温度缩放')
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
    
    # 基本统计
    normal_mean = np.mean(normal_probs)
    normal_median = np.median(normal_probs)
    bot_mean = np.mean(bot_probs)
    bot_median = np.median(bot_probs)
    
    # 分离度
    separation = bot_mean - normal_mean
    overlap = np.minimum(
        np.histogram(normal_probs, bins=100, range=(0,1), density=True)[0],
        np.histogram(bot_probs, bins=100, range=(0,1), density=True)[0]
    ).sum() / 100  # 归一化重叠面积
    
    # 找到最优阈值
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # 在最优阈值下的性能
    preds = (probs >= optimal_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average='binary', zero_division=0
    )
    
    # 计算 AUC
    auc = roc_auc_score(y_true, probs)
    
    return {
        'name': name,
        'auc': auc,
        'normal_mean': normal_mean,
        'normal_median': normal_median,
        'bot_mean': bot_mean,
        'bot_median': bot_median,
        'separation': separation,
        'overlap': overlap,
        'optimal_threshold': optimal_threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def train_model(args, device):
    """训练 V2 模型"""
    print("\n" + "="*60)
    print("开始训练 V2 改进的僵尸网络检测模型")
    print("核心改进：增强分数区分度 + 分数校准")
    print("="*60)
    
    if os.path.exists(args.model_path) and not args.force_retrain:
        print(f"[Train] 模型已存在：{args.model_path}，跳过训练")
        return
    
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
    MiniBatchGraphBuilder.prepare_for_training(data, labels, train_ratio=0.8)
    
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
    
    model = ImprovedBotnetDetectorV2(
        stat_dim=stat_dim,
        semantic_dim=semantic_dim,
        struct_dim=struct_dim,
        hidden_dim=args.hidden_dim,
        use_calibration=not args.no_calibration,
        use_temperature=not args.no_temperature
    )
    
    print(f"  - 统计特征维度：{stat_dim}")
    print(f"  - 语义特征维度：{semantic_dim}")
    print(f"  - 结构特征维度：{struct_dim}")
    print(f"  - 隐藏层维度：{args.hidden_dim}")
    print(f"  - 校准层：{'启用' if not args.no_calibration else '禁用'}")
    print(f"  - 温度缩放：{'启用' if not args.no_temperature else '禁用'}")
    
    trainer = ImprovedTrainerV2(model, lr=args.lr, device=device)
    
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
        
        # 早停检查
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
    
    # 清理
    del data, model, trainer, labels
    torch.cuda.empty_cache()
    gc.collect()


def evaluate_model(args, device, verbose=True):
    """评估模型"""
    if verbose:
        print("\n" + "="*60)
        print("开始评估模型")
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
    model = ImprovedBotnetDetectorV2.load(args.model_path, device=device)
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
    
    all_logits = []
    with torch.no_grad():
        for batch in infer_loader:
            batch = batch.to(device)
            out = model(batch)[:batch['ip'].batch_size]
            all_logits.append(out.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    probs = logits.squeeze().numpy()
    
    # 确保标签和预测对齐
    y_true = labels.numpy()[:len(probs)]
    probs = probs[:len(y_true)]
    
    # 分析分布
    result = analyze_score_distribution(probs, y_true, name=f"Scenario_{args.test_scenarios}")
    
    if verbose:
        # 打印结果
        print("\n" + "="*60)
        print("评估结果（V2 改进模型）")
        print("="*60)
        print(f"\n【整体性能】")
        print(f"  AUC:          {result['auc']:.4f}")
        print(f"  Precision:    {result['precision']:.4f}")
        print(f"  Recall:       {result['recall']:.4f}")
        print(f"  F1-Score:     {result['f1']:.4f}")
        print(f"  Optimal Thresh: {result['optimal_threshold']:.4f}")
        
        print(f"\n【分数分布特征】")
        print(f"  正常节点分数:")
        print(f"    - 均值：{result['normal_mean']:.6f}")
        print(f"    - 中位数：{result['normal_median']:.6f}")
        print(f"  僵尸节点分数:")
        print(f"    - 均值：{result['bot_mean']:.6f}")
        print(f"    - 中位数：{result['bot_median']:.6f}")
        
        print(f"\n【分离度指标】")
        print(f"  均值分离度：{result['separation']:.4f}")
        print(f"  分布重叠面积：{result['overlap']:.4f}")
        
        print("="*60)
    
    return result, probs, y_true


def compare_v1_v2(args, device):
    """对比 V1 和 V2 模型"""
    print("\n" + "="*60)
    print("对比实验：V1 vs V2")
    print("="*60)
    
    results = {}
    
    # 评估 V2 模型
    print("\n>>> 评估 V2 模型 <<<")
    original_path = args.model_path
    args.model_path = 'improved_botnet_model_v2.pth'
    v2_result = evaluate_model(args, device, verbose=False)
    if v2_result:
        results['v2'] = v2_result[0]
        print(f"  AUC: {v2_result[0]['auc']:.4f}")
        print(f"  分离度：{v2_result[0]['separation']:.4f}")
        print(f"  正常节点中位分数：{v2_result[0]['normal_median']:.6f}")
        print(f"  僵尸节点中位分数：{v2_result[0]['bot_median']:.6f}")
    
    # 评估 V1 模型
    print("\n>>> 评估 V1 模型 <<<")
    args.model_path = 'improved_botnet_model.pth'
    if os.path.exists(args.model_path):
        # 临时导入 V1 模型
        from improved_model import ImprovedBotnetDetector
        from improved_heterogeneous_graph import ImprovedHeterogeneousGraphBuilder
        from data_loader import CTU13Loader
        from torch_geometric.loader import NeighborLoader
        
        loader = CTU13Loader(args.data_dir)
        test_scenarios = [int(s) for s in args.test_scenarios.split(',')]
        df = loader.load_data(test_scenarios)
        
        builder = ImprovedHeterogeneousGraphBuilder()
        data, ip_map = builder.build(df, include_semantic=True, include_struct=True)
        labels, _ = get_labels(df, ip_map)
        
        model = ImprovedBotnetDetector.load(args.model_path, device=device)
        model.eval()
        
        infer_loader = NeighborLoader(data, num_neighbors=[15, 10], batch_size=4096, input_nodes='ip', shuffle=False)
        
        all_logits = []
        with torch.no_grad():
            for batch in infer_loader:
                batch = batch.to(device)
                out = model(batch)[:batch['ip'].batch_size]
                all_logits.append(out.cpu())
        
        probs = torch.cat(all_logits, dim=0).squeeze().numpy()
        y_true = labels.numpy()[:len(probs)]
        
        v1_result = analyze_score_distribution(probs[:len(y_true)], y_true, name="V1")
        results['v1'] = v1_result
        
        print(f"  AUC: {v1_result['auc']:.4f}")
        print(f"  分离度：{v1_result['separation']:.4f}")
        print(f"  正常节点中位分数：{v1_result['normal_median']:.6f}")
        print(f"  僵尸节点中位分数：{v1_result['bot_median']:.6f}")
    else:
        print("  V1 模型不存在，跳过对比")
    
    # 打印对比总结
    print("\n" + "="*60)
    print("对比总结")
    print("="*60)
    
    if 'v1' in results and 'v2' in results:
        print(f"{'指标':<15} {'V1':<15} {'V2':<15} {'改进':<15}")
        print("-" * 60)
        print(f"{'AUC':<15} {results['v1']['auc']:<15.4f} {results['v2']['auc']:<15.4f} "
              f"{(results['v2']['auc'] - results['v1']['auc'])*100:+.2f}%")
        print(f"{'分离度':<15} {results['v1']['separation']:<15.4f} {results['v2']['separation']:<15.4f} "
              f"{(results['v2']['separation'] - results['v1']['separation'])*100:+.2f}%")
        print(f"{'正常节点中位':<15} {results['v1']['normal_median']:<15.6f} {results['v2']['normal_median']:<15.6f}")
        print(f"{'僵尸节点中位':<15} {results['v1']['bot_median']:<15.6f} {results['v2']['bot_median']:<15.6f}")
        print(f"{'F1-Score':<15} {results['v1']['f1']:<15.4f} {results['v2']['f1']:<15.4f} "
              f"{(results['v2']['f1'] - results['v1']['f1'])*100:+.2f}%")
    
    args.model_path = original_path
    return results


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Config] 使用设备：{device}")
    print(f"[Config] 训练场景：{args.train_scenarios}")
    print(f"[Config] 测试场景：{args.test_scenarios}")
    print(f"[Config] 语义特征：{'禁用' if args.no_semantic else '启用'}")
    print(f"[Config] 结构特征：{'禁用' if args.no_struct else '启用'}")
    print(f"[Config] 校准层：{'禁用' if args.no_calibration else '启用'}")
    print(f"[Config] 温度缩放：{'禁用' if args.no_temperature else '启用'}")
    
    # 训练
    train_model(args, device)
    
    # 评估
    result = evaluate_model(args, device)
    
    # 保存结果
    if result:
        result_dict, probs, y_true = result
        
        # 将 numpy 类型转换为 Python 原生类型
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
        
        # 保存详细分布数据
        np.savez(
            'v2_score_distribution.npz',
            probs=probs,
            y_true=y_true,
            normal_probs=probs[y_true == 0],
            bot_probs=probs[y_true == 1]
        )
        
        with open('v2_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n[Info] 结果已保存到 v2_results.json 和 v2_score_distribution.npz")
    
    # 对比实验
    print("\n" + "="*60)
    print("运行对比实验...")
    compare_v1_v2(args, device)
    
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