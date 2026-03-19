"""
改进的僵尸网络检测主脚本
整合三模态特征融合 + 概念漂移感知对比学习
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

# 屏蔽警告
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# 导入必要库
import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split
from torch_geometric.loader import NeighborLoader

# 导入改进模块
from data_loader import CTU13Loader
from improved_heterogeneous_graph import ImprovedHeterogeneousGraphBuilder, MiniBatchGraphBuilder
from improved_model import ImprovedBotnetDetector, ImprovedTrainer
from final_botnet_classifier import compute_botnet_metrics_final


def parse_args():
    parser = argparse.ArgumentParser(description='改进的僵尸网络检测 - 三模态对比学习')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-fs/CTU-13/CTU-13-Dataset')
    parser.add_argument('--train_scenarios', type=str, default='1,2,3')
    parser.add_argument('--test_scenarios', type=str, default='13')
    parser.add_argument('--model_path', type=str, default='improved_botnet_model.pth')
    parser.add_argument('--epochs', type=int, default=10)
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


def find_optimal_threshold(y_true, probs):
    """使用Youden's J找到最优阈值"""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx]


def find_available_model():
    """查找当前目录可用的模型文件"""
    # 优先检查指定模型
    if os.path.exists('improved_botnet_model.pth'):
        return 'improved_botnet_model.pth'
    
    # 检查其他可能的模型文件
    model_patterns = [
        'improved_botnet_model.pth',
        'botnet_model.pth', 
        'model.pth',
        'best_model.pth'
    ]
    
    for pattern in model_patterns:
        if os.path.exists(pattern):
            return pattern
    
    # 检查 .pkl 文件
    import glob
    pkl_files = glob.glob('*.pkl')
    if pkl_files:
        return pkl_files[0]
    
    return None


def train_model(args, device):
    """训练改进模型"""
    print("\n" + "="*60)
    print("开始训练改进的僵尸网络检测模型")
    print("="*60)
    
    # 检查是否有可用的模型文件
    available_model = find_available_model()
    if available_model and not args.force_retrain:
        # 如果指定模型存在或其他模型存在，跳过训练
        if available_model == args.model_path or args.model_path == 'improved_botnet_model.pth':
            print(f"[Train] 模型已存在：{available_model}，跳过训练")
            args.model_path = available_model
            return
        else:
            # 有其他模型但路径不同，提示用户
            print(f"[Train] 发现已有模型：{available_model}")
            print(f"[Train] 如需使用该模型，请设置 --model_path {available_model}")
    
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
    
    model = ImprovedBotnetDetector(
        stat_dim=stat_dim,
        semantic_dim=semantic_dim,
        struct_dim=struct_dim,
        hidden_dim=args.hidden_dim
    )
    
    trainer = ImprovedTrainer(model, lr=args.lr, device=device)
    
    # 训练
    print("\n[Phase 5] 开始训练...")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Hidden Dim: {args.hidden_dim}")
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
    
    print(f"\n[Train] 训练完成！最佳损失: {best_loss:.4f}")
    
    # 清理
    del data, model, trainer, labels
    torch.cuda.empty_cache()
    gc.collect()


def evaluate_model(args, device):
    """评估模型"""
    print("\n" + "="*60)
    print("开始评估模型")
    print("="*60)
    
    if not os.path.exists(args.model_path):
        print(f"[Error] 模型不存在：{args.model_path}")
        return None
    
    # 加载测试数据
    print("\n[Phase 1] 加载测试数据...")
    loader = CTU13Loader(args.data_dir)
    test_scenarios = [int(s) for s in args.test_scenarios.split(',')]
    df = loader.load_data(test_scenarios)
    
    if df.empty:
        print("[Error] 数据加载失败!")
        return None
    
    # 构建图
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
    print("\n[Phase 3] 加载模型...")
    model = ImprovedBotnetDetector.load(args.model_path, device=device)
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
    
    # 确保标签和预测对齐
    y_true = labels.numpy()[:len(probs)]
    probs = probs[:len(y_true)]
    
    # 使用最终分类器
    print("\n[Classifier] 使用基于固定分位数的最终分类器...")
    result = compute_botnet_metrics_final(y_true, probs)
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果（最终分类器）")
    print("="*60)
    print(f"  AUC:       {result['auc']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1-Score:  {result['f1']:.4f}")
    print(f"  Threshold: {result['threshold']:.4f}")
    print("="*60)
    print(f"\n  预测为僵尸网络的节点：{result['num_predicted']}")
    print(f"  实际僵尸网络节点：{result['num_true']}")
    print(f"  总节点数：{len(y_true)}")
    
    return result


def compare_with_baseline(args, device):
    """与基线方法比较"""
    print("\n" + "="*60)
    print("对比实验：改进方法 vs 原始方法")
    print("="*60)
    
    results = {}
    
    # 评估改进方法
    print("\n>>> 评估改进方法 <<<")
    improved_results = evaluate_model(args, device)
    if improved_results:
        results['improved'] = improved_results
    
    # 评估原始方法
    print("\n>>> 评估原始方法 <<<")
    baseline_model_path = 'botnet_gnn_model.pth'
    if os.path.exists(baseline_model_path):
        # 临时修改模型路径
        original_path = args.model_path
        args.model_path = baseline_model_path
        
        # 这里需要用原始模型评估，暂时跳过
        print("[Info] 原始模型存在，但需要使用原始评估代码")
        
        args.model_path = original_path
    
    return results


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Config] 使用设备: {device}")
    print(f"[Config] 训练场景: {args.train_scenarios}")
    print(f"[Config] 测试场景: {args.test_scenarios}")
    print(f"[Config] 语义特征: {'禁用' if args.no_semantic else '启用'}")
    print(f"[Config] 结构特征: {'禁用' if args.no_struct else '启用'}")
    
    # 训练
    train_model(args, device)
    
    # 评估
    results = evaluate_model(args, device)
    
    # 保存结果（修复 JSON 序列化问题）
    if results:
        # 将 numpy 类型转换为 Python 原生类型
        def to_python_type(val):
            """安全转换为 Python 原生类型"""
            if hasattr(val, 'item'):
                try:
                    return val.item()
                except:
                    return float(val) if np.isscalar(val) else val.tolist()
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, (np.int32, np.int64)):
                return int(val)
            elif isinstance(val, (np.float32, np.float64)):
                return float(val)
            return val
        
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: to_python_type(v) for k, v in value.items()}
            else:
                json_results[key] = to_python_type(value)
        
        with open('improved_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n[Info] 结果已保存到 improved_results.json")
    
    print("\n" + "="*60)
    print("所有任务完成！")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Info] 用户中断")
    except Exception as e:
        print(f"\n[Error] 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os._exit(0)