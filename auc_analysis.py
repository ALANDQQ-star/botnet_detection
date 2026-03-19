"""
AUC 指标分析

AUC（Area Under ROC Curve）详解
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score


def explain_auc():
    """解释 AUC 的概念"""
    
    print("="*70)
    print("AUC（Area Under ROC Curve）详解")
    print("="*70)
    
    # 1. AUC 的定义
    print("\n【1. AUC 的定义】")
    print("""
AUC = Area Under the ROC Curve（ROC 曲线下面积）

ROC 曲线（Receiver Operating Characteristic Curve）:
- X 轴：假阳性率（FPR）= FP / (FP + TN)
- Y 轴：真阳性率（TPR）= TP / (TP + FN) = 召回率

AUC 值范围：0.5 ~ 1.0
- 0.5：随机猜测（对角线）
- 1.0：完美分类器
- >0.9：优秀
- 0.7-0.9：良好
- <0.7：一般
""")
    
    # 2. AUC 的计算方法
    print("\n【2. AUC 的计算方法】")
    print("""
方法 1：梯形积分法
    将 ROC 曲线下的区域分成多个梯形，求面积和

方法 2：概率解释（更直观）
    AUC = 随机选一个正样本和一个负样本
          正样本得分 > 负样本得分的概率
""")
    
    # 3. 示例计算
    print("\n【3. 示例计算】")
    
    # 模拟一个简单的例子
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_value = auc(fpr, tpr)
    
    print(f"真实标签：{y_true}")
    print(f"预测分数：{y_scores}")
    print(f"AUC 值：{auc_value:.4f}")
    
    # 绘制 ROC 曲线
    plt.figure(figsize=(10, 5))
    
    # 子图 1: ROC 曲线
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess (AUC = 0.5)')
    plt.fill_between(fpr, tpr, alpha=0.3, label=f'Area = {auc_value:.2f}')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve Example')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图 2: 分数分布
    plt.subplot(1, 2, 2)
    neg_scores = y_scores[y_true == 0]
    pos_scores = y_scores[y_true == 1]
    plt.hist(neg_scores, bins=4, alpha=0.5, label='Negative', color='red', density=True)
    plt.hist(pos_scores, bins=4, alpha=0.5, label='Positive', color='blue', density=True)
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('auc_explanation.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存：auc_explanation.png")
    
    # 4. AUC 高的含义
    print("\n【4. AUC 数值高的含义】")
    print("""
AUC = 0.95 意味着:
✓ 随机选一个僵尸节点和一个正常节点，95% 的概率僵尸节点的分数更高
✓ 模型有很好的「排序」能力
✓ 正负样本的分数分布有较大程度的分离

AUC 高 BUT F1 低的原因:
⚠️ AUC 只衡量排序质量，不衡量绝对分数
⚠️ 即使 AUC 很高，如果所有分数都集中在很小范围（如 0-0.1），
   很难选择合适的阈值进行分类
⚠️ F1 依赖于具体阈值的选择，而 AUC 是积分所有可能阈值的结果
""")
    
    # 5. 当前项目的 AUC 分析
    print("\n【5. 当前项目的 AUC 分析】")
    
    # 模拟低分场景
    np.random.seed(42)
    n_neg = 1000
    n_pos = 50
    
    # 正常节点：分数在 0-0.05 之间
    neg_scores_low = np.random.beta(1, 50, n_neg) * 0.1
    # 僵尸节点：分数略高，但仍然很低
    pos_scores_low = 0.01 + np.random.beta(2, 20, n_pos) * 0.1
    
    y_true_sim = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    y_scores_sim = np.concatenate([neg_scores_low, pos_scores_low])
    
    auc_sim = roc_auc_score(y_true_sim, y_scores_sim)
    
    print(f"""
模拟低分场景（类似当前项目场景 12）:
- 正常节点分数范围：[{neg_scores_low.min():.4f}, {neg_scores_low.max():.4f}]
- 僵尸节点分数范围：[{pos_scores_low.min():.4f}, {pos_scores_low.max():.4f}]
- AUC = {auc_sim:.4f}

解释:
虽然所有分数都很低（<0.15），但僵尸节点的分数「相对」高于正常节点
这导致 AUC 很高，但在实际分类时:
- 阈值设高（如 0.1）：大部分僵尸节点被漏检 → 召回率低
- 阈值设低（如 0.01）：大量正常节点被误判 → 精确率低

这就是「AUC 高但 F1 低」的典型情况!
""")
    
    # 绘制低分场景
    plt.figure(figsize=(15, 5))
    
    # 分数分布
    plt.subplot(1, 3, 1)
    plt.hist(neg_scores_low, bins=30, alpha=0.5, label='Normal', color='red', density=True)
    plt.hist(pos_scores_low, bins=30, alpha=0.5, label='Botnet', color='blue', density=True)
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title(f'Low Score Distribution\nAUC = {auc_sim:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 排序后的分数
    plt.subplot(1, 3, 2)
    sorted_indices = np.argsort(y_scores_sim)[::-1]
    sorted_true = y_true_sim[sorted_indices]
    sorted_scores = y_scores_sim[sorted_indices]
    
    plt.scatter(range(len(sorted_scores)), sorted_scores, c=sorted_true, cmap='bwr', s=5)
    plt.xlabel('Rank (sorted by score)')
    plt.ylabel('Score')
    plt.title('Sorted Scores (High to Low)')
    plt.colorbar(label='True Label (0=Normal, 1=Botnet)')
    plt.grid(True, alpha=0.3)
    
    # ROC 曲线
    plt.subplot(1, 3, 3)
    fpr_sim, tpr_sim, _ = roc_curve(y_true_sim, y_scores_sim)
    plt.plot(fpr_sim, tpr_sim, 'b-', linewidth=2, label=f'AUC = {auc_sim:.4f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('low_score_auc.png', dpi=150, bbox_inches='tight')
    print("\n低分场景图表已保存：low_score_auc.png")
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("""
1. AUC 衡量的是「排序质量」，不是「分类质量」
   
2. AUC 高 = 正样本得分倾向于高于负样本
   - 这是「相对」关系，与绝对分数无关
   
3. AUC 高但 F1 低 = 模型能排序但难以设置阈值
   - 解决方法：
     a) 使用更好的阈值选择策略（如基于聚类的自适应阈值）
     b) 改进模型训练，使分数分布更均匀
     c) 使用温度缩放（Temperature Scaling）校准输出分数
     d) 使用多特征融合的异常检测方法
""")


if __name__ == "__main__":
    explain_auc()