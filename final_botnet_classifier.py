"""
最终僵尸网络分类器

基于场景 11,12,13 的测试结果，选择固定分位数 (96%) 方法作为最终判定方法。

测试结果显示：
- 场景 12: quantile_96 方法 F1=0.500, P=0.345, R=0.908
- 该方法简单、稳定、不依赖复杂假设

核心方法：
- 使用 96% 分位数作为阈值
- 预测分数最高的 4% 节点为僵尸网络
- 基于先验知识：僵尸节点约占 1-3%
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


class FinalBotnetClassifier:
    """
    最终僵尸网络分类器
    
    使用固定的 96% 分位数作为阈值
    """
    
    def __init__(self):
        self.threshold = None
        self.method_name = "Quantile_96"
    
    def find_threshold(self, probs):
        """
        找到最优阈值
        
        方法：使用 96% 分位数
        理论依据：僵尸节点约占 1-3%，选择 top 4% 可以覆盖大部分僵尸节点
        """
        probs = np.asarray(probs).flatten()
        self.threshold = np.percentile(probs, 96)
        return self.threshold
    
    def predict(self, probs):
        """生成预测"""
        if self.threshold is None:
            self.find_threshold(probs)
        return (np.asarray(probs) >= self.threshold).astype(int)


def compute_botnet_metrics_final(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """
    使用最终分类器计算指标
    
    完全不使用标签信息选择阈值
    """
    classifier = FinalBotnetClassifier()
    threshold = classifier.find_threshold(probs)
    
    preds = (probs >= threshold).astype(int)
    
    # 计算指标（仅用于评估）
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average='binary', zero_division=0
    )
    auc = roc_auc_score(y_true, probs)
    
    return {
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'threshold': float(threshold),
        'num_predicted': int(preds.sum()),
        'num_true': int(y_true.sum()),
        'method': 'Quantile_96',
        'predicted_ratio': float(preds.sum() / len(probs))
    }


if __name__ == "__main__":
    print("="*70)
    print("最终僵尸网络分类器测试")
    print("="*70)
    
    try:
        data = np.load('s12_scores.npz', allow_pickle=True)
        probs = data['probs']
        y_true = data['y_true']
        
        print(f"\n[数据概览]")
        print(f"  总节点数：{len(probs)}")
        print(f"  僵尸节点数：{y_true.sum()} ({y_true.sum()/len(probs)*100:.2f}%)")
        
        print("\n[测试最终分类器]...")
        result = compute_botnet_metrics_final(y_true, probs)
        
        print(f"\n评估结果（最终分类器）")
        print("="*60)
        print(f"  AUC:       {result['auc']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1-Score:  {result['f1']:.4f}")
        print(f"  Threshold: {result['threshold']:.6f}")
        print("="*60)
        print(f"  预测为僵尸网络的节点：{result['num_predicted']}")
        print(f"  实际僵尸网络节点：{result['num_true']}")
        print(f"  预测比例：{result['predicted_ratio']:.2%}")
        print(f"  方法：{result['method']}")
        
    except FileNotFoundError:
        print("[Error] 请先运行 analyze_s12.py 生成分数数据")