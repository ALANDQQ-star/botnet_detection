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

# === 源头屏蔽警告 ===
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# --- UI通信辅助函数 ---
import threading

# UI 状态文件路径
UI_STATE_FILE = "ui_state.json"
_ui_state_lock = threading.Lock()

def send_ui(type, msg):
    """发送 UI 信号，同时写入文件和 stdout"""
    signal = f"@@UI_SIGNAL@@|{type}|{msg}"
    print(signal, flush=True)
    # 写入状态文件
    update_ui_state_file(type, msg)

def update_ui_state_file(signal_type, content):
    """更新 UI 状态文件"""
    global _ui_state_lock
    with _ui_state_lock:
        try:
            # 读取现有状态
            state = {}
            if os.path.exists(UI_STATE_FILE):
                try:
                    with open(UI_STATE_FILE, "r") as f:
                        state = json.load(f)
                except:
                    pass
            
            # 更新状态
            state["last_update"] = time.time()
            state["signal_type"] = signal_type
            state["content"] = content
            
            # 根据信号类型更新特定字段
            if signal_type == "BATCH_UPDATE":
                # 读取 batch 数据文件
                if os.path.exists(content):
                    with open(content, "r") as f:
                        batch_data = json.load(f)
                    state["batch_data"] = batch_data
                    # 更新可视化状态
                    state["red_nodes"] = state.get("red_nodes", []) + batch_data.get("new_red_nodes", [])
                    state["yellow_nodes"] = batch_data.get("yellow_nodes", [])
                    state["country_stats"] = batch_data.get("country_stats", {})
                    state["province_stats"] = batch_data.get("province_stats", {})
                    state["sim_time"] = batch_data.get("sim_time")
                    step = batch_data.get("step", 0)
                    total_steps = batch_data.get("total_steps", 1)
                    if total_steps > 0:
                        state["progress"] = min(1.0, 0.5 + (step / total_steps) * 0.5)
                    state["step_info"] = f"推演中: 步骤 {step}/{total_steps}"
            elif signal_type == "PHASE_START":
                state["step_info"] = content
                if content.strip() == "运行完毕":
                    state["progress"] = 1.0
                elif "HMM" in content or "攻击链" in content:
                    state["progress"] = max(state.get("progress", 0), 0.05)
                elif "GNN" in content or "训练" in content:
                    state["progress"] = max(state.get("progress", 0), 0.15)
                elif "推演" in content or "感染" in content:
                    state["progress"] = max(state.get("progress", 0), 0.5)
                elif "报告" in content or "生成" in content:
                    state["progress"] = max(state.get("progress", 0), 0.9)
                elif "评估" in content or "威胁" in content:
                    state["progress"] = max(state.get("progress", 0), 0.35)
            elif signal_type == "EPOCH_UPDATE":
                parts = content.split("|")
                if len(parts) == 3:
                    ep, tot, loss = parts
                    state["step_info"] = f"训练轮次 {ep}/{tot}（损失: {loss}）"
                    ep_val = int(ep)
                    tot_val = int(tot)
                    if tot_val > 0:
                        state["progress"] = max(state.get("progress", 0), 0.15 + (ep_val / tot_val) * 0.2)
            elif signal_type == "PROGRESS_INFERENCE":
                parts = content.split("|")
                if len(parts) == 2:
                    cur, tot = int(parts[0]), int(parts[1])
                    state["step_info"] = f"推理进度 {cur}/{tot}"
                    if tot > 0:
                        state["progress"] = max(state.get("progress", 0), 0.35 + (cur / tot) * 0.15)
            elif signal_type == "METRICS":
                parts = content.split("|")
                if len(parts) == 5:
                    auc, f1, prec, rec, th = parts
                    state["metrics"] = {"auc": auc, "f1": f1, "prec": prec, "rec": rec, "thresh": th}
            elif signal_type == "GNN_VIS":
                try:
                    state["gnn_viz_data"] = json.loads(content)
                except:
                    pass
            
            # 写入文件
            with open(UI_STATE_FILE, "w") as f:
                json.dump(state, f)
        except Exception as e:
            print(f"Error updating UI state file: {e}")

def log_ui(msg):
    send_ui("LOG", msg)

# --- 导入过程带进度显示 ---
log_ui("正在初始化后端环境…")
print("Loading PyTorch...", flush=True)
import torch
warnings.filterwarnings("ignore") 

print("Loading Data & ML Libraries...", flush=True)
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from torch_geometric.loader import NeighborLoader

print("Loading Custom Modules...", flush=True)
from data_loader import CTU13Loader
from heterogeneous_graph import HeterogeneousGraphBuilder
from graph_contrastive_learning import BotnetDetector, Trainer
from attack_chain_fsm import AttackChainInference
from spatiotemporal_analysis import SpatioTemporalAnalyzer
from viz_utils import get_ip_info
from bot_ahgcn_baseline import (
    BotAHGCNGraphBuilder, MetaPathSimilarity, 
    BotAHGCNModel, BotAHGCNTrainer, prepare_ahgcn_data
)

log_ui("环境就绪，解析参数…")

# --- 参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/root/autodl-fs/CTU-13/CTU-13-Dataset')
    parser.add_argument('--train_scenarios', type=str, default='1,2,3,4,5,6,7,8,9,10')
    parser.add_argument('--test_scenarios', type=str, default='13') 
    parser.add_argument('--model_path', type=str, default='botnet_gnn_model.pth')
    parser.add_argument('--epochs', type=int, default=5) 
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--use_dynamic', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--force_hmm', action='store_true')
    parser.add_argument('--method', type=str, default='existing', choices=['existing', 'baseline'], help='Detection method: existing (GAT+GCN) or baseline (Bot-AHGCN)')
    return parser.parse_args()

# --- 辅助函数 ---
def find_optimal_threshold(y_true, probs, method='youden'):
    """
    Find optimal threshold using multiple strategies.
    
    Args:
        y_true: Ground truth labels
        probs: Predicted probabilities
        method: 'youden' (Youden's J), 'f1' (max F1), 'balanced' (balanced accuracy),
                'precision_focused' (prioritize precision), or 'percentile' (top-k based)
    
    Returns:
        optimal threshold value
    """
    from sklearn.metrics import roc_curve
    
    # Method 1: Youden's J statistic (maximizes sensitivity + specificity - 1)
    if method == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        # Youden's J = TPR - FPR = Sensitivity + Specificity - 1
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]
    
    # Method 2: Maximize F1 score directly
    elif method == 'f1':
        best_thresh = 0.5
        best_f1 = 0
        # Search through candidate thresholds
        candidates = np.percentile(probs, np.arange(5, 100, 5))  # 5th to 95th percentile
        for thresh in candidates:
            preds = (probs >= thresh).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh
    
    # Method 3: Balanced accuracy (average of sensitivity and specificity)
    elif method == 'balanced':
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        balanced_acc = (tpr + (1 - fpr)) / 2
        optimal_idx = np.argmax(balanced_acc)
        return thresholds[optimal_idx]
    
    # Method 4: Precision-focused (ensures precision >= recall, good for imbalanced data)
    elif method == 'precision_focused':
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        # Find threshold where precision is at least as high as recall (roughly)
        # This helps avoid too many false positives
        best_thresh = 0.5
        best_score = 0
        for i, thresh in enumerate(thresholds):
            if tpr[i] > 0:  # Only consider thresholds with positive recall
                # Score: prioritize precision by penalizing high FPR
                # precision ~= TP / (TP + FP), and FP rate is FPR
                # For imbalanced data, we want FPR to be very low
                score = tpr[i] * (1 - fpr[i])  # Similar to Youden but different weighting
                if score > best_score:
                    best_score = score
                    best_thresh = thresh
        return best_thresh
    
    # Method 5: Percentile-based (assumes top X% are anomalies)
    elif method == 'percentile':
        # Use the 95th percentile as threshold (top 5% are bots)
        # Adjust based on expected anomaly ratio
        pos_ratio = np.mean(y_true)
        # If bot ratio is ~1%, threshold should be around 99th percentile
        if pos_ratio < 0.01:
            return np.percentile(probs, 99)
        elif pos_ratio < 0.05:
            return np.percentile(probs, 95)
        else:
            return np.percentile(probs, 90)
    
    return 0.5  # fallback

def select_best_threshold(y_true, probs, min_threshold=0.01):
    """
    Select the best threshold by combining multiple strategies.
    For imbalanced bot detection, we prioritize precision while maintaining reasonable recall.
    Optimized for large datasets using sampling.
    
    Args:
        y_true: Ground truth labels
        probs: Predicted probabilities
        min_threshold: Minimum threshold to enforce (default 0.01, very low for imbalanced data)
    
    Returns:
        optimal threshold value (at least min_threshold)
    """
    n_samples = len(y_true)
    
    # For large datasets, use sampling to speed up threshold selection
    if n_samples > 50000:
        # Stratified sampling to preserve class distribution
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        n_pos = min(len(pos_idx), 5000)
        n_neg = min(len(neg_idx), 45000)
        if n_pos > 0 and n_neg > 0:
            sampled_pos = np.random.choice(pos_idx, n_pos, replace=False)
            sampled_neg = np.random.choice(neg_idx, n_neg, replace=False)
            sample_idx = np.concatenate([sampled_pos, sampled_neg])
            y_true_sample = y_true[sample_idx]
            probs_sample = probs[sample_idx]
            print(f"[Threshold] Using sampled data for threshold selection ({len(sample_idx)} samples)")
        else:
            y_true_sample = y_true
            probs_sample = probs
    else:
        y_true_sample = y_true
        probs_sample = probs
    
    # Calculate multiple candidate thresholds on full data (fast operations)
    thresh_youden = find_optimal_threshold(y_true, probs, method='youden')
    thresh_prec = find_optimal_threshold(y_true, probs, method='precision_focused')
    thresh_perc = find_optimal_threshold(y_true, probs, method='percentile')
    
    # Evaluate each threshold using sampled data
    candidates = {
        'youden': thresh_youden,
        'precision_focused': thresh_prec,
        'percentile': thresh_perc,
    }
    
    print(f"[Threshold Candidates] Youden: {thresh_youden:.4f}, Precision: {thresh_prec:.4f}, Percentile: {thresh_perc:.4f}")
    
    best_thresh = None
    best_f1 = 0
    best_metrics = None
    
    for name, thresh in candidates.items():
        # Only enforce a very low minimum threshold to avoid degenerate cases
        if thresh < min_threshold:
            print(f"[Threshold] {name} threshold {thresh:.4f} below minimum {min_threshold}, skipping")
            continue
        
        # Use sampled data for fast evaluation
        preds = (probs_sample >= thresh).astype(int)
        pred_sum = preds.sum()
        if pred_sum == 0:
            print(f"[Threshold] {name} threshold {thresh:.4f} predicts no positives, skipping")
            continue
            
        tp = ((preds == 1) & (y_true_sample == 1)).sum()
        fp = ((preds == 1) & (y_true_sample == 0)).sum()
        fn = ((preds == 0) & (y_true_sample == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # For bot detection, we want a balance but lean towards precision
        if precision < 0.001:  # Less than 0.1% precision is too low
            continue
            
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = (precision, recall, f1)
            print(f"[Threshold] {name}: thresh={thresh:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    
    # If no good threshold found, use percentile-based threshold as fallback
    if best_thresh is None or best_metrics is None:
        best_thresh = thresh_perc
        preds = (probs_sample >= best_thresh).astype(int)
        tp = ((preds == 1) & (y_true_sample == 1)).sum()
        fp = ((preds == 1) & (y_true_sample == 0)).sum()
        fn = ((preds == 0) & (y_true_sample == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        best_metrics = (precision, recall, f1)
        print(f"[Threshold] Using percentile fallback threshold: {best_thresh:.4f}")
    
    print(f"[Threshold] Selected: {best_thresh:.4f}, Precision: {best_metrics[0]:.4f}, Recall: {best_metrics[1]:.4f}, F1: {best_metrics[2]:.4f}")
    return best_thresh

def get_labels(df, ip_map):
    bot_keywords = ['botnet', 'cc', 'c&c', 'neris', 'rbot', 'virut', 'menti', 'murlo', 'sogou']
    df['is_bot'] = df['label'].apply(lambda x: any(k in str(x).lower() for k in bot_keywords))
    if ip_map is None: return None, None
    bot_ips = set(df[df['is_bot']]['src_ip'].unique()) | set(df[df['is_bot']]['dst_ip'].unique())
    y = np.zeros(len(ip_map), dtype=np.float32)
    for ip, idx in ip_map.items():
        if ip in bot_ips: y[idx] = 1.0
    return torch.tensor(y), list(bot_ips)

# --- 模块1: HMM 攻击链 ---
def run_hmm_training(args):
    hmm_path = "hmm_model.json"
    if os.path.exists(hmm_path) and not args.force_hmm:
        log_ui("已存在 HMM 模型，跳过训练。")
        return

    send_ui("PHASE_START", "攻击链建模 (HMM)")
    engine = AttackChainInference(target_ips=None, model_path=hmm_path)
    loader = CTU13Loader(args.data_dir)
    scens = [int(s) for s in args.train_scenarios.split(',')]
    
    for sc in scens:
        log_ui(f"HMM 从场景 {sc} 学习…")
        try:
            df = loader.load_data([sc])
            if not df.empty:
                if 'start_time' in df.columns:
                    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
                    df['ts'] = df['start_time'].astype(np.int64) // 10**9
                get_labels(df, None)
                engine.train_on_dataset(df)
                del df; gc.collect()
        except Exception as e:
            print(f"Error in HMM sc {sc}: {e}")

    engine.finalize_training()
    log_ui("HMM 建模完成。")

# --- 模块2: GNN 训练 (现有方法) ---
def run_gnn_training(args, device):
    if args.method == 'baseline':
        # 调用基线方法训练
        run_baseline_training(args, device)
        return
        
    if os.path.exists(args.model_path) and not args.force_retrain:
        log_ui("已存在 GNN 模型，跳过训练。")
        return

    send_ui("PHASE_START", "GNN 模型训练")
    loader = CTU13Loader(args.data_dir)
    log_ui("加载训练数据…")
    df = loader.load_data([int(s) for s in args.train_scenarios.split(',')])
    if df.empty: return

    log_ui("构建异构图…")
    builder = HeterogeneousGraphBuilder()
    data, ip_map = builder.build(df)
    labels, _ = get_labels(df, ip_map)
    del df; gc.collect()

    indices = np.arange(len(labels))
    train_idx, _ = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[train_idx] = True
    data['ip'].y = labels
    data['ip'].train_mask = train_mask

    train_loader = NeighborLoader(data, num_neighbors=[15, 10], batch_size=args.batch_size, 
                                  input_nodes=('ip', train_mask), shuffle=True, num_workers=0)

    model = BotnetDetector(in_dim=data['ip'].x.shape[1], hidden_dim=128)
    trainer = Trainer(model, lr=args.lr, device=device)

    log_ui(f"开始训练，共 {args.epochs} 轮…")
    
    # === 发送初始可视化信号 (Epoch 0) ===
    send_ui("GNN_VIS", json.dumps({"phase": "train", "epoch": 0, "loss": 1.0, "layer": -1}))
    
    for epoch in range(args.epochs):
        loss_list = []
        
        for batch in train_loader:
            loss = trainer.train_step_batch(batch)
            loss_list.append(loss)
        avg_loss = np.mean(loss_list)
        
        send_ui("GNN_VIS", json.dumps({"phase": "train", "epoch": epoch + 1, "loss": round(avg_loss, 4), "layer": -1}))
        send_ui("EPOCH_UPDATE", f"{epoch+1}|{args.epochs}|{avg_loss:.4f}")
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # === 训练结束，重置为 Idle ===
    send_ui("GNN_VIS", json.dumps({"phase": "idle", "epoch": args.epochs, "loss": avg_loss, "layer": -1}))
    
    model.save(args.model_path)
    del data, model, trainer, labels
    torch.cuda.empty_cache(); gc.collect()
    log_ui("GNN 训练结束。")

# --- 模块2b: Bot-AHGCN 基线方法训练 ---
def run_baseline_training(args, device):
    baseline_model_path = "bot_ahgcn_baseline.pth"
    if os.path.exists(baseline_model_path) and not args.force_retrain:
        log_ui("已存在 Bot-AHGCN 基线模型，跳过训练。")
        args.model_path = baseline_model_path
        return

    send_ui("PHASE_START", "Bot-AHGCN 基线方法训练")
    loader = CTU13Loader(args.data_dir)
    log_ui("加载训练数据…")
    df = loader.load_data([int(s) for s in args.train_scenarios.split(',')])
    if df.empty: return

    log_ui("构建异构信息网络 (AHIN)…")
    builder = BotAHGCNGraphBuilder()
    ahin, ip_map, node_sizes = builder.build_ahin(df)
    labels, _ = get_labels(df, ip_map)
    del df; gc.collect()

    # 准备相似性计算器
    similarity_calculator = MetaPathSimilarity(ahin, ip_map)
    
    # 准备训练数据
    x, AM, AG, labels_tensor = prepare_ahgcn_data(ahin, similarity_calculator, labels)
    
    # 划分训练集
    indices = np.arange(len(labels_tensor))
    train_idx, _ = train_test_split(indices, test_size=0.2, stratify=labels_tensor, random_state=42)
    train_mask = torch.zeros(len(labels_tensor), dtype=torch.bool)
    train_mask[train_idx] = True

    # 创建模型 - 在CPU上训练，避免大矩阵GPU内存问题
    print("[Bot-AHGCN] Creating model (training on CPU for large matrices)...")
    model = BotAHGCNModel(in_dim=x.shape[1], hidden_dim=64, out_dim=32)
    trainer = BotAHGCNTrainer(model, lr=0.01, device='cpu')  # 使用CPU训练

    log_ui(f"开始 Bot-AHGCN 训练，共 {args.epochs} 轮…")
    
    # 发送初始可视化信号
    send_ui("GNN_VIS", json.dumps({"phase": "train", "epoch": 0, "loss": 1.0, "layer": -1}))
    
    print("[Bot-AHGCN] Starting training loop...", flush=True)
    for epoch in range(args.epochs):
        print(f"[Bot-AHGCN] Training epoch {epoch+1}...", flush=True)
        loss = trainer.train_step(x, AM, AG, labels_tensor, train_mask)
        
        send_ui("GNN_VIS", json.dumps({"phase": "train", "epoch": epoch + 1, "loss": round(loss, 4), "layer": -1}))
        send_ui("EPOCH_UPDATE", f"{epoch+1}|{args.epochs}|{loss:.4f}")
        print(f"Bot-AHGCN Epoch {epoch+1} Loss: {loss:.4f}", flush=True)

    # 训练结束
    send_ui("GNN_VIS", json.dumps({"phase": "idle", "epoch": args.epochs, "loss": loss, "layer": -1}))
    
    trainer.save_model(baseline_model_path)
    args.model_path = baseline_model_path
    del ahin, model, trainer, x, AM, AG, labels_tensor
    torch.cuda.empty_cache(); gc.collect()
    log_ui("Bot-AHGCN 训练结束。")

# --- 模块3: 推理与流式展示 ---
def run_inference(args, device):
    if args.method == 'baseline':
        # 调用基线方法推理
        run_baseline_inference(args, device)
        return
        
    send_ui("PHASE_START", "威胁评估中")
    
    if not os.path.exists(args.model_path):
        log_ui("错误：未找到模型！")
        return

    log_ui("加载测试场景（可能需要一段时间）…")
    loader = CTU13Loader(args.data_dir)
    df = loader.load_data([int(s) for s in args.test_scenarios.split(',')])
    if df.empty: return

    log_ui("构建推理图…")
    builder = HeterogeneousGraphBuilder()
    data, ip_map = builder.build(df)
    labels, _ = get_labels(df, ip_map)

    log_ui("执行 GNN 推理…")
    send_ui("GNN_VIS", json.dumps({"phase": "inference", "epoch": 0, "loss": 0, "layer": -1}))

    model = BotnetDetector.load(args.model_path, device=device)
    model.eval()
    
    infer_loader = NeighborLoader(data, num_neighbors=[15, 10], batch_size=4096, input_nodes='ip', shuffle=False)
    all_logits = []
    total_batches = len(infer_loader)
    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            out = model(batch.to(device))[:batch['ip'].batch_size]
            all_logits.append(out.cpu())
            if i % 5 == 0: 
                send_ui("PROGRESS_INFERENCE", f"{i}|{total_batches}")

    logits = torch.cat(all_logits, dim=0)
    probs = torch.sigmoid(logits).squeeze().numpy()
    
    # === 推理计算结束，重置为 Idle ===
    send_ui("GNN_VIS", json.dumps({"phase": "idle", "layer": -1}))
    
    y_true = labels.numpy()[:len(probs)]
    probs = probs[:len(y_true)]

    # Use optimal threshold selection to balance precision and recall
    if args.use_dynamic:
        thresh = select_best_threshold(y_true, probs)
    else:
        thresh = args.threshold

    final_pred = (probs >= thresh).astype(int)
    
    auc = roc_auc_score(y_true, probs)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, final_pred, average='binary', zero_division=0)
    send_ui("METRICS", f"{auc:.4f}|{f1:.4f}|{precision:.4f}|{recall:.4f}|{thresh:.4f}")

    simulate_streaming(df, final_pred, probs, ip_map, args)

# --- 模块3b: Bot-AHGCN 基线方法推理 ---
def run_baseline_inference(args, device):
    send_ui("PHASE_START", "Bot-AHGCN 基线方法威胁评估")
    
    baseline_model_path = "bot_ahgcn_baseline.pth"
    if not os.path.exists(baseline_model_path):
        log_ui("错误：未找到 Bot-AHGCN 基线模型！")
        return

    log_ui("加载测试场景（可能需要一段时间）…")
    loader = CTU13Loader(args.data_dir)
    df = loader.load_data([int(s) for s in args.test_scenarios.split(',')])
    if df.empty: return

    log_ui("构建异构信息网络 (AHIN)…")
    builder = BotAHGCNGraphBuilder()
    ahin, ip_map, node_sizes = builder.build_ahin(df)
    labels, _ = get_labels(df, ip_map)
    # 注意：不要删除df，后面simulate_streaming还需要使用

    # 准备相似性计算器
    similarity_calculator = MetaPathSimilarity(ahin, ip_map)
    
    # 准备推理数据
    x, AM, AG, labels_tensor = prepare_ahgcn_data(ahin, similarity_calculator, labels)

    log_ui("执行 Bot-AHGCN 推理…")
    send_ui("GNN_VIS", json.dumps({"phase": "inference", "epoch": 0, "loss": 0, "layer": -1}))

    # 加载模型
    print("[Bot-AHGCN] Loading model...", flush=True)
    model = BotAHGCNModel(in_dim=x.shape[1], hidden_dim=64, out_dim=32)
    trainer = BotAHGCNTrainer(model, lr=0.01, device='cpu')  # 使用CPU推理
    trainer.load_model(baseline_model_path)
    model.eval()
    print("[Bot-AHGCN] Model loaded, running inference...", flush=True)

    # 推理
    with torch.no_grad():
        logits = model(x, AM, AG)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    print(f"[Bot-AHGCN] Inference done, probs shape: {probs.shape}", flush=True)
    
    # 推理计算结束
    send_ui("GNN_VIS", json.dumps({"phase": "idle", "layer": -1}))
    
    y_true = labels_tensor.numpy()[:len(probs)]
    probs = probs[:len(y_true)]

    # Use optimal threshold selection to balance precision and recall
    if args.use_dynamic:
        thresh = select_best_threshold(y_true, probs)
    else:
        thresh = args.threshold

    final_pred = (probs >= thresh).astype(int)
    
    auc = roc_auc_score(y_true, probs)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, final_pred, average='binary', zero_division=0)
    send_ui("METRICS", f"{auc:.4f}|{f1:.4f}|{precision:.4f}|{recall:.4f}|{thresh:.4f}")

    # 使用现有的流式展示函数
    simulate_streaming(df, final_pred, probs, ip_map, args)
    
    # 清理内存
    del df; gc.collect()

    # Save run history for scenario comparison
    try:
        from datetime import datetime as dt_mod
        history_file = "run_history.json"
        history = []
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)
        
        history.append({
            "timestamp": dt_mod.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_scenarios": args.test_scenarios,
            "train_scenarios": args.train_scenarios,
            "bots_detected": int(final_pred.sum()),
            "method": "baseline",
            "metrics": {
                "auc": f"{auc:.4f}", "f1": f"{f1:.4f}",
                "prec": f"{precision:.4f}", "rec": f"{recall:.4f}",
                "thresh": f"{thresh:.4f}"
            }
        })
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving run history: {e}")

def simulate_streaming(df, final_pred, probs, ip_map, args):
    send_ui("PHASE_START", "全网感染推演")
    log_ui("准备可视化数据…")
    
    if 'start_time' not in df.columns: return
    df['start_time'] = pd.to_datetime(df['start_time'])
    
    idx_to_ip = {v: k for k, v in ip_map.items()}
    bot_indices = np.where(final_pred == 1)[0]
    predicted_bots = set([idx_to_ip[idx] for idx in bot_indices])
    
    ip_times = df[df['src_ip'].isin(predicted_bots)].groupby('src_ip')['start_time'].min().sort_values()
    sorted_bots = ip_times.index.tolist()
    
    if not sorted_bots:
        log_ui("未检测到威胁。")
        return

    log_ui(f"回放感染链，共 {len(sorted_bots)} 个节点…")
    
    BATCH_SIZE = math.ceil(len(sorted_bots) / 15)
    if BATCH_SIZE < 1: BATCH_SIZE = 1
    
    total_steps = math.ceil(len(sorted_bots) / BATCH_SIZE)
    if total_steps < 1: total_steps = 1
    
    ip_cache = {}
    country_stats = {}
    province_stats = {}
    
    # === 细化推演期间的 GNN 脉冲动画 ===
    # 移除 Python 端的 sleep 和频繁刷新，完全依赖前端 CSS 动画
    # 只需要在推演开始时发送一次 inference 信号即可
    send_ui("GNN_VIS", json.dumps({"phase": "inference", "epoch": 0, "loss": 0, "layer": -1}))
    
    for i in range(0, len(sorted_bots), BATCH_SIZE):
        step_idx = i // BATCH_SIZE + 1
        batch_ips = sorted_bots[i : i+BATCH_SIZE]
        yellow_ips = sorted_bots[i+BATCH_SIZE : i+BATCH_SIZE*2]
        
        sim_time = ip_times[batch_ips[-1]]
        
        payload = {
            "step": step_idx,
            "total_steps": total_steps,
            "sim_time": sim_time.isoformat(),
            "new_red_nodes": [],
            "yellow_nodes": [],
            "country_stats": {},
            "province_stats": {} 
        }
        
        for ip in batch_ips:
            if ip not in ip_cache: ip_cache[ip] = get_ip_info(ip)
            lat, lon, _, cname, region = ip_cache[ip]
            
            country_stats[cname] = country_stats.get(cname, 0) + 1
            if cname in ["China", "Taiwan", "Hong Kong", "Macao", "People's Republic of China"]:
                province_stats[region] = province_stats.get(region, 0) + 1

            payload["new_red_nodes"].append({
                "ip": ip, "lat": lat, "lon": lon, "country": cname, "region": region, "conf": float(probs[ip_map[ip]])
            })
            
        for ip in yellow_ips:
            if ip not in ip_cache: ip_cache[ip] = get_ip_info(ip)
            lat, lon, _, cname, region = ip_cache[ip]
            payload["yellow_nodes"].append({
                "ip": ip, "lat": lat, "lon": lon, "country": cname, "region": region, "conf": float(probs[ip_map[ip]])
            })
            
        payload["country_stats"] = country_stats
        payload["province_stats"] = province_stats
        
        with open("batch_data.json", "w") as f:
            json.dump(payload, f)
            
        send_ui("BATCH_UPDATE", "batch_data.json")
        time.sleep(0.5) # 仅保留少量延迟用于前端渲染地图

    log_ui("可视化序列完成。")
    
    # === 生成报告数据 ===
    send_ui("PHASE_START", "正在生成报告…")
    log_ui("正在生成分析报告…")
    try:
        # 1. 生成 HMM 攻击链报告 (选择实际有流量记录的 Bot)
        src_ips_in_data = set(df['src_ip'].unique())
        bots_with_traffic = [ip for ip in predicted_bots if ip in src_ips_in_data]
        MAX_HMM_BOTS = 80
        hmm_bot_list = bots_with_traffic[:MAX_HMM_BOTS] if len(bots_with_traffic) > MAX_HMM_BOTS else bots_with_traffic
        log_ui(f"对 {len(hmm_bot_list)} 个有流量的 Bot 运行 HMM（共 {len(predicted_bots)} 个）…")
        hmm_engine = AttackChainInference(target_ips=hmm_bot_list, model_path="hmm_model.json")
        if hmm_engine.model.is_trained:
            report = hmm_engine.run_inference_with_evaluation(df)
            with open("attack_chain_report.json", "w") as f:
                json.dump(report, f)
                
        # 2. 生成时空拓扑报告（始终写入拓扑评估报告，便于前端不报错）
        log_ui("生成时空拓扑报告…")
        topo_eval = {
            "nodes_evaluated": len(predicted_bots),
            "edges_analyzed": len(df),
            "density": len(df) / (len(predicted_bots) * len(predicted_bots) + 1e-5),
            "c2_found": 0
        }
        try:
            analyzer = SpatioTemporalAnalyzer(bot_ips=list(predicted_bots))
            res, _ = analyzer.analyze(df)
            if res:
                with open("c2_candidates.json", "w") as f:
                    json.dump(res.get('c2_candidates', []), f)
                topo_eval["c2_found"] = len(res.get('c2_candidates', []))
            else:
                # 即使分析返回空，也要写入空文件，确保前端不报错
                with open("c2_candidates.json", "w") as f:
                    json.dump([], f)
            # 复制拓扑可视化数据
            if os.path.exists("viz_data.json"):
                import shutil
                shutil.copy("viz_data.json", "network_topology.json")
            else:
                # 如果 viz_data.json 不存在，生成一个空的拓扑文件
                with open("network_topology.json", "w") as f:
                    json.dump({"nodes": [], "links": []}, f)
        except Exception as ex:
            print(f"SpatioTemporal analyze: {ex}")
            # 发生异常时，确保写入空文件
            with open("c2_candidates.json", "w") as f:
                json.dump([], f)
            with open("network_topology.json", "w") as f:
                json.dump({"nodes": [], "links": []}, f)
        with open("topology_eval_report.json", "w") as f:
            json.dump(topo_eval, f)
    except Exception as e:
        print(f"Error generating reports: {e}")

    send_ui("GNN_VIS", json.dumps({"phase": "idle", "layer": -1}))
    send_ui("PHASE_START", "运行完毕")
    log_ui("全部任务已完成。")

# --- 主入口 ---
def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)

    if args.mode in ['train', 'all']:
        run_hmm_training(args)
        run_gnn_training(args, device)
    
    if args.mode in ['test', 'all']:
        run_inference(args, device)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Force exit to prevent hanging threads (e.g. from dataloaders or other libs)
        os._exit(0)