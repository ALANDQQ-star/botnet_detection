"""批量评估所有场景 4-13，汇总对比"""
import sys
import os
import json
import warnings
import numpy as np

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve

from data_loader import CTU13Loader
from improved_heterogeneous_graph import ImprovedHeterogeneousGraphBuilder
from improved_model_v3_final import ImprovedBotnetDetectorV3Final
from smart_threshold_optimizer import SmartThresholdOptimizer
from torch_geometric.loader import NeighborLoader
import gc

DATA_DIR = '/root/autodl-fs/CTU-13/CTU-13-Dataset'
MODEL_PATH = 'improved_botnet_model_v3_final.pth'
PRIOR_PATH = 'improved_botnet_model_v3_final_prior.json'

def get_labels(df, ip_map):
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


def evaluate_scenario(scenario_id, device, model, train_prior):
    """评估单个场景"""
    loader = CTU13Loader(DATA_DIR)
    df = loader.load_data([scenario_id])
    if df.empty:
        return None

    builder = ImprovedHeterogeneousGraphBuilder()
    data, ip_map = builder.build(df)
    labels, _ = get_labels(df, ip_map)
    del df
    gc.collect()

    infer_loader = NeighborLoader(
        data, num_neighbors=[15, 10], batch_size=4096,
        input_nodes='ip', shuffle=False
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

    n_bot = int(y_true.sum())
    bot_ratio = n_bot / len(y_true) * 100

    auc = roc_auc_score(y_true, probs)

    # 智能阈值（改进版）
    opt = SmartThresholdOptimizer(verbose=False, train_prior=train_prior)
    smart_thresh = opt.find_threshold(probs)
    smart_preds = (probs >= smart_thresh).astype(int)
    sp, sr, sf, _ = precision_recall_fscore_support(y_true, smart_preds, average='binary', zero_division=0)

    # Youden（有标签参考）
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    youden_idx = np.argmax(tpr - fpr)
    youden_thresh = thresholds[youden_idx]
    youden_preds = (probs >= youden_thresh).astype(int)
    yp, yr, yf, _ = precision_recall_fscore_support(y_true, youden_preds, average='binary', zero_division=0)

    del data, labels
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'scenario': scenario_id,
        'bot_ratio': bot_ratio,
        'auc': auc,
        'smart_f1': sf, 'smart_p': sp, 'smart_r': sr, 'smart_thresh': smart_thresh,
        'smart_method': opt.debug_info.get('selection_method', ''),
        'youden_f1': yf, 'youden_p': yp, 'youden_r': yr, 'youden_thresh': youden_thresh,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    model = ImprovedBotnetDetectorV3Final.load(MODEL_PATH, device=device)
    model.eval()

    # 加载先验
    train_prior = None
    if os.path.exists(PRIOR_PATH):
        with open(PRIOR_PATH, 'r') as f:
            train_prior = json.load(f)

    results = []
    for s in range(4, 14):
        print(f"\n>>> 评估场景 {s} ...")
        r = evaluate_scenario(s, device, model, train_prior)
        if r:
            results.append(r)
            print(f"  AUC={r['auc']:.4f} | Smart F1={r['smart_f1']:.4f} ({r['smart_method']}) | Youden F1={r['youden_f1']:.4f}")

    # 汇总表格
    print("\n" + "=" * 100)
    print(f"{'场景':>6} | {'Bot%':>6} | {'AUC':>6} | {'Smart F1':>9} | {'Smart P':>8} | {'Smart R':>8} | {'Youden F1':>9} | {'Youden P':>8} | {'Youden R':>8}")
    print("-" * 100)
    for r in results:
        print(f"  S{r['scenario']:>3}  | {r['bot_ratio']:>5.2f}% | {r['auc']:.4f} | {r['smart_f1']:>9.4f} | {r['smart_p']:>8.4f} | {r['smart_r']:>8.4f} | {r['youden_f1']:>9.4f} | {r['youden_p']:>8.4f} | {r['youden_r']:>8.4f}")

    smart_f1s = [r['smart_f1'] for r in results]
    youden_f1s = [r['youden_f1'] for r in results]
    aucs = [r['auc'] for r in results]
    print("-" * 100)
    print(f"  平均  |       | {np.mean(aucs):.4f} | {np.mean(smart_f1s):>9.4f} |          |          | {np.mean(youden_f1s):>9.4f} |")
    print("=" * 100)


if __name__ == "__main__":
    main()
