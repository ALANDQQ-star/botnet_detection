import os
import time
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, silhouette_samples
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

from data_loader import CTU13Loader
from improved_heterogeneous_graph import ImprovedHeterogeneousGraphBuilder
from improved_model_v3_final import ImprovedBotnetDetectorV3Final as ImprovedBotnetDetector
from intelligent_threshold_optimizer import IntelligentThresholdOptimizer

# ==========================================
# 1. TRUSTED Algorithm Implementation
# ==========================================
class TRUSTED_NodeClassifier:
    def __init__(self, window_size=1000, distance_threshold=0.5, similarity_threshold=0.0025):
        self.window_size = window_size
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        
    def fit_predict_batch(self, X):
        if len(X) < 2:
            return np.zeros(len(X))
            
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # In sklearn > 1.2, affinity is deprecated, use metric
        clusterer = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=self.distance_threshold, 
            linkage='average',
            metric='euclidean' # changed from affinity
        )
        labels = clusterer.fit_predict(X_scaled)
        
        # If only 1 cluster is found, silhouette_samples throws error
        if len(set(labels)) <= 1:
            return np.zeros(len(X))
            
        sil_scores = silhouette_samples(X_scaled, labels, metric='euclidean')
        
        # Calculate mean silhouette per cluster
        pred_labels = np.zeros(len(X))
        unique_clusters = np.unique(labels)
        
        for c in unique_clusters:
            c_mask = (labels == c)
            mean_sil = np.mean(sil_scores[c_mask])
            if mean_sil <= self.similarity_threshold:
                pred_labels[c_mask] = 1 # Anomaly / Botnet
                
        return pred_labels
        
    def predict(self, X):
        predictions = []
        n_samples = len(X)
        for i in range(0, n_samples, self.window_size):
            end_idx = min(i + self.window_size, n_samples)
            X_batch = X[i:end_idx]
            preds = self.fit_predict_batch(X_batch)
            predictions.extend(preds)
        return np.array(predictions)

# ==========================================
# 2. Evaluation Setup
# ==========================================
def extract_node_data(scenarios):
    # Find dataset path used by previous scripts
    loader = CTU13Loader('/root/autodl-fs/CTU-13/CTU-13-Dataset')
    df = loader.load_data(scenarios)
    
    if df.empty:
        raise ValueError(f"No data for scenarios {scenarios}")
        
    from main_improved_v3_final import get_labels
    
    graph_builder = ImprovedHeterogeneousGraphBuilder()
    data, ip_map = graph_builder.build(df, include_semantic=True, include_struct=True)
    labels, _ = get_labels(df, ip_map)
    labels = labels.numpy()
    
    # Concatenate features
    stat = data['ip'].x.numpy()
    sem = data['ip'].semantic_x.numpy()
    struc = data['ip'].struct_x.numpy()
    X = np.concatenate([stat, sem, struc], axis=1)
    
    return X, labels

def main():
    print("============================================================")
    print("对比评估：TRUSTED 算法 vs V3 Final 异构图模型")
    print("训练场景：1, 2, 3 | 测试场景：13")
    print("============================================================\n")
    
    # 1. 加载数据
    print("[Phase 1] 加载训练和测试数据...")
    X_train, y_train = extract_node_data([1, 2, 3])
    X_test, y_test = extract_node_data([13])
    
    print(f"训练集节点数: {len(X_train)}, 僵尸节点占比: {np.mean(y_train)*100:.2f}%")
    print(f"测试集节点数: {len(X_test)}, 僵尸节点占比: {np.mean(y_test)*100:.2f}%\n")
    
    # 2. TRUSTED 算法评估
    print("[Phase 2] 评估 TRUSTED 算法 (基于窗口的无监督聚集聚类)...")
    # "Train" TRUSTED: Grid search on a subset of train data to find best thresholds
    print("  正在训练集上搜索最佳参数...")
    best_f1 = -1
    best_params = {}
    
    # Use a subset of training data for faster parameter tuning
    sample_size = min(50000, len(X_train))
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_val, y_val = X_train[idx], y_train[idx]
    
    distances = [0.1, 0.5, 1.0, 1.5]
    for dist in distances:
        sim_thresh = dist * 0.005 # original logic from trusted.py
        clf = TRUSTED_NodeClassifier(window_size=1000, distance_threshold=dist, similarity_threshold=sim_thresh)
        preds = clf.predict(X_val)
        f1 = f1_score(y_val, preds, zero_division=0)
        print(f"    参数 dist={dist}, sim_thresh={sim_thresh:.5f} -> F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_params = {'dist': dist, 'sim_thresh': sim_thresh}
            
    print(f"  最佳参数: {best_params} (Val F1: {best_f1:.4f})")
    
    print("  在测试集上应用 TRUSTED...")
    start_time = time.time()
    trusted_clf = TRUSTED_NodeClassifier(window_size=1000, distance_threshold=best_params['dist'], similarity_threshold=best_params['sim_thresh'])
    trusted_preds = trusted_clf.predict(X_test)
    trusted_time = time.time() - start_time
    
    trusted_prec = precision_score(y_test, trusted_preds, zero_division=0)
    trusted_rec = recall_score(y_test, trusted_preds, zero_division=0)
    trusted_f1 = f1_score(y_test, trusted_preds, zero_division=0)
    
    try:
        trusted_auc = roc_auc_score(y_test, trusted_preds)
    except:
        trusted_auc = 0.5
        
    print(f"  TRUSTED 耗时: {trusted_time:.2f}s")
    print(f"  TRUSTED 结果 - AUC: {trusted_auc:.4f}, Precision: {trusted_prec:.4f}, Recall: {trusted_rec:.4f}, F1: {trusted_f1:.4f}\n")
    
    # 3. 运行 V3 Final 模型原代码
    print("[Phase 3] 运行 V3 Final 模型...")
    import subprocess
    import re
    
    v3_time = 0.0
    v3_auc = 0.0
    v3_prec = 0.0
    v3_rec = 0.0
    v3_f1 = 0.0
    
    try:
        # 运行 main_improved_v3_final.py 并捕获输出
        start_time = time.time()
        result = subprocess.run(['python', 'main_improved_v3_final.py'], capture_output=True, text=True)
        v3_time = time.time() - start_time
        
        output = result.stdout
        print(output)
        
        # 解析输出获取指标
        # 寻找类似于：
        # AUC:          0.9922
        # Precision:    0.2503
        # Recall:       0.2268
        # F1-Score:     0.2380
        
        # For the new metrics layout, we want to grab the metrics specifically under the "传统方法对比（ROC Youden指数）" section
        # Or under the "智能阈值方法" section depending on preference. Let's get the Smart Threshold ones:
        auc_match = re.search(r'AUC:\s+([0-9.]+)', output)
        
        # Get metrics under 智能阈值方法
        smart_section = re.search(r'【智能阈值方法】(.*?)【', output, re.DOTALL)
        if smart_section:
            sec_text = smart_section.group(1)
            prec_match = re.search(r'Precision:\s+([0-9.]+)', sec_text)
            rec_match = re.search(r'Recall:\s+([0-9.]+)', sec_text)
            f1_match = re.search(r'F1-Score:\s+([0-9.]+)', sec_text)
        else:
            prec_match = re.search(r'Precision:\s+([0-9.]+)', output)
            rec_match = re.search(r'Recall:\s+([0-9.]+)', output)
            f1_match = re.search(r'F1-Score:\s+([0-9.]+)', output)
            
        # If output is missing, could be due to process failure. Let's dump output in case of error
        if not auc_match and not f1_match:
            print("Failed to parse output! Raw output below:")
            print("="*40)
            print(output[-1000:] if len(output) > 1000 else output)
            print("="*40)
        
        if auc_match: v3_auc = float(auc_match.group(1))
        if prec_match: v3_prec = float(prec_match.group(1))
        if rec_match: v3_rec = float(rec_match.group(1))
        if f1_match: v3_f1 = float(f1_match.group(1))
        
        print(f"  V3 Final 解析结果 - AUC: {v3_auc:.4f}, Precision: {v3_prec:.4f}, Recall: {v3_rec:.4f}, F1: {v3_f1:.4f}\n")
    except Exception as e:
        print(f"  运行 V3 Final 失败: {e}")
    
    # 4. 生成对比报告
    print("[Phase 4] 生成对比结果...")
    
    results = {
        "TRUSTED": {
            "AUC": float(trusted_auc),
            "Precision": float(trusted_prec),
            "Recall": float(trusted_rec),
            "F1-Score": float(trusted_f1),
            "Time (s)": float(trusted_time)
        },
        "V3_Final": {
            "AUC": float(v3_auc),
            "Precision": float(v3_prec),
            "Recall": float(v3_rec),
            "F1-Score": float(v3_f1),
            "Time (s)": float(v3_time)
        }
    }
    
    with open('trusted_vs_v3_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("============================================================")
    print("对比总结：")
    print(f"{'Metric':<15} | {'TRUSTED':<15} | {'V3 Final':<15} | {'提升/差异':<15}")
    print("-" * 65)
    print(f"{'AUC':<15} | {trusted_auc:<15.4f} | {v3_auc:<15.4f} | {v3_auc-trusted_auc:<15.4f}")
    print(f"{'Precision':<15} | {trusted_prec:<15.4f} | {v3_prec:<15.4f} | {v3_prec-trusted_prec:<15.4f}")
    print(f"{'Recall':<15} | {trusted_rec:<15.4f} | {v3_rec:<15.4f} | {v3_rec-trusted_rec:<15.4f}")
    print(f"{'F1-Score':<15} | {trusted_f1:<15.4f} | {v3_f1:<15.4f} | {v3_f1-trusted_f1:<15.4f}")
    print(f"{'Inference Time':<15} | {trusted_time:<15.2f}s | {v3_time:<15.2f}s | -")
    print("============================================================")

if __name__ == "__main__":
    main()
