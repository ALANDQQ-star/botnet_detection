"""
分数分布的统计学分析

目标：
1. 分析僵尸节点和正常节点的分数分布特征
2. 使用统计学方法拟合分布
3. 基于分布特性设计最优分类器
"""

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

# 统计分析和分布拟合
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, f1_score, precision_recall_curve, roc_curve


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


def analyze_distribution(probs, y_true, scenario):
    """详细分析分数分布"""
    normal_probs = probs[y_true == 0]
    bot_probs = probs[y_true == 1]
    
    print(f"\n{'='*70}")
    print(f"场景 {scenario} 分布分析")
    print(f"{'='*70}")
    
    # 基本统计量
    print(f"\n【基本统计量】")
    print(f"正常节点数：{len(normal_probs)}")
    print(f"僵尸节点数：{len(bot_probs)}")
    print(f"类别比例：{len(bot_probs)/len(probs)*100:.4f}%")
    
    print(f"\n正常节点分数:")
    print(f"  均值：{np.mean(normal_probs):.6f}")
    print(f"  中位数：{np.median(normal_probs):.6f}")
    print(f"  标准差：{np.std(normal_probs):.6f}")
    print(f"  偏度：{stats.skew(normal_probs):.4f}")
    print(f"  峰度：{stats.kurtosis(normal_probs):.4f}")
    print(f"  最小值：{np.min(normal_probs):.6f}")
    print(f"  最大值：{np.max(normal_probs):.6f}")
    print(f"  Q1(25%): {np.percentile(normal_probs, 25):.6f}")
    print(f"  Q3(75%): {np.percentile(normal_probs, 75):.6f}")
    
    print(f"\n僵尸节点分数:")
    print(f"  均值：{np.mean(bot_probs):.6f}")
    print(f"  中位数：{np.median(bot_probs):.6f}")
    print(f"  标准差：{np.std(bot_probs):.6f}")
    print(f"  偏度：{stats.skew(bot_probs):.4f}")
    print(f"  峰度：{stats.kurtosis(bot_probs):.4f}")
    print(f"  最小值：{np.min(bot_probs):.6f}")
    print(f"  最大值：{np.max(bot_probs):.6f}")
    print(f"  Q1(25%): {np.percentile(bot_probs, 25):.6f}")
    print(f"  Q3(75%): {np.percentile(bot_probs, 75):.6f}")
    
    # 分布拟合
    print(f"\n【分布拟合】")
    
    # 尝试多种分布
    distributions = {
        'beta': stats.beta,
        'gamma': stats.gamma,
        'expon': stats.expon,
        'weibull_min': stats.weibull_min,
        'lognorm': stats.lognorm,
        'johnsonsb': stats.johnsonsb,  # Johnson 分布（有界）
        'johnsonsu': stats.johnsonsu,  # Johnson 无界
    }
    
    # 由于分数在 [0, 1] 范围，使用有界分布
    results = []
    for name, dist in distributions.items():
        try:
            if name in ['beta', 'johnsonsb']:
                # 有界分布需要特殊处理
                params = dist.fit(probs, floc=0, scale=1)
            else:
                # 添加小偏移避免 0 值问题
                probs_pos = probs + 1e-10
                params = dist.fit(probs_pos)
            
            # 计算 KS 统计量
            if name in ['beta', 'johnsonsb']:
                ks_stat, p_value = stats.kstest(probs, dist.cdf, args=params, alternative='two-sided')
            else:
                ks_stat, p_value = stats.kstest(probs_pos, dist.cdf, args=params, alternative='two-sided')
            
            results.append({
                'name': name,
                'params': params,
                'ks_stat': ks_stat,
                'p_value': p_value
            })
        except Exception as e:
            pass
    
    # 按 KS 统计量排序
    results.sort(key=lambda x: x['ks_stat'])
    
    print(f"\n最佳拟合分布（KS 检验）:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. {r['name']}: KS={r['ks_stat']:.4f}, p={r['p_value']:.4f}, params={r['params']}")
    
    # 分别拟合正常节点和僵尸节点
    print(f"\n分别拟合正常节点和僵尸节点分布:")
    
    # Beta 分布拟合（最适合 [0,1] 范围）
    try:
        normal_params = stats.beta.fit(normal_probs, floc=0, scale=1)
        bot_params = stats.beta.fit(bot_probs, floc=0, scale=1)
        print(f"  正常节点 Beta 分布参数：a={normal_params[0]:.4f}, b={normal_params[1]:.4f}")
        print(f"  僵尸节点 Beta 分布参数：a={bot_params[0]:.4f}, b={bot_params[1]:.4f}")
    except:
        pass
    
    # 分离度分析
    print(f"\n【分离度分析】")
    
    # 重叠区域
    normal_hist, bin_edges = np.histogram(normal_probs, bins=100, range=(0, 1), density=True)
    bot_hist, _ = np.histogram(bot_probs, bins=100, range=(0, 1), density=True)
    
    # 归一化
    normal_hist = normal_hist / normal_hist.sum()
    bot_hist = bot_hist / bot_hist.sum()
    
    # 重叠面积
    overlap = np.minimum(normal_hist, bot_hist).sum()
    print(f"  分布重叠面积：{overlap:.4f}")
    
    # Bhattacharyya 距离
    bhattacharyya_dist = -np.log(np.sqrt(normal_hist * bot_hist).sum() + 1e-10)
    print(f"  Bhattacharyya 距离：{bhattacharyya_dist:.4f}")
    
    # Cohen's d (效应量)
    pooled_std = np.sqrt((np.std(normal_probs)**2 + np.std(bot_probs)**2) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(bot_probs) - np.mean(normal_probs)) / pooled_std
        print(f"  Cohen's d (效应量): {cohens_d:.4f}")
    else:
        cohens_d = 0
        print(f"  Cohen's d: 无法计算（标准差为 0）")
    
    # 使用 Youden's J 找最优阈值
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    preds = (probs >= optimal_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
    
    print(f"\n【最优阈值（Youden's J）】")
    print(f"  阈值：{optimal_threshold:.6f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    
    # 使用 P-R 曲线找 F1 最优阈值
    prec_curve, rec_curve, thresholds_pr = precision_recall_curve(y_true, probs)
    f1_curve = 2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-10)
    best_f1_idx = np.argmax(f1_curve[:-1])  # 最后一个是 0
    best_f1_threshold = thresholds_pr[best_f1_idx]
    best_f1 = f1_curve[best_f1_idx]
    
    preds_f1 = (probs >= best_f1_threshold).astype(int)
    precision_f1, recall_f1, f1_f1, _ = precision_recall_fscore_support(y_true, preds_f1, average='binary', zero_division=0)
    
    print(f"\n【F1 最优阈值】")
    print(f"  阈值：{best_f1_threshold:.6f}")
    print(f"  Precision: {precision_f1:.4f}")
    print(f"  Recall: {recall_f1:.4f}")
    print(f"  F1: {f1_f1:.4f}")
    
    return {
        'scenario': scenario,
        'normal_stats': {
            'mean': np.mean(normal_probs),
            'median': np.median(normal_probs),
            'std': np.std(normal_probs),
            'skew': stats.skew(normal_probs),
            'kurtosis': stats.kurtosis(normal_probs),
            'q25': np.percentile(normal_probs, 25),
            'q75': np.percentile(normal_probs, 75),
        },
        'bot_stats': {
            'mean': np.mean(bot_probs),
            'median': np.median(bot_probs),
            'std': np.std(bot_probs),
            'skew': stats.skew(bot_probs),
            'kurtosis': stats.kurtosis(bot_probs),
            'q25': np.percentile(bot_probs, 25),
            'q75': np.percentile(bot_probs, 75),
        },
        'overlap': overlap,
        'bhattacharyya': bhattacharyya_dist,
        'cohens_d': cohens_d,
        'optimal_threshold_youden': optimal_threshold,
        'optimal_f1_threshold': best_f1_threshold,
        'optimal_f1': best_f1,
        'num_normal': len(normal_probs),
        'num_bot': len(bot_probs),
    }


def find_optimal_threshold_stats(normal_probs, bot_probs, target_fpr=0.01):
    """
    基于统计学方法找最优阈值
    
    方法 1: 使用正态假设（如果数据近似正态）
    方法 2: 使用百分位数
    方法 3: 使用分布拟合
    """
    print(f"\n【基于统计学的阈值选择】")
    
    # 方法 1: 控制 FPR
    # 假设正常节点服从某分布，找到使 FPR <= target_fpr 的阈值
    normal_threshold = np.percentile(normal_probs, (1 - target_fpr) * 100)
    print(f"  方法 1 (控制 FPR={target_fpr}): 阈值 = {normal_threshold:.6f}")
    
    # 在这个阈值下的召回率
    recall_at_threshold = (bot_probs >= normal_threshold).mean()
    print(f"  对应召回率：{recall_at_threshold:.4f}")
    
    # 方法 2: 使用正态假设
    normal_mean = np.mean(normal_probs)
    normal_std = np.std(normal_probs)
    
    # 找到使正常节点只有 1% 超过的阈值
    z_score = stats.norm.ppf(1 - target_fpr)
    normal_approx_threshold = normal_mean + z_score * normal_std
    print(f"  方法 2 (正态假设): 阈值 = {normal_approx_threshold:.6f}")
    
    # 方法 3: 使用两分布的交集
    # 找到两分布密度相等的点
    from scipy.optimize import brentq
    
    try:
        # 拟合 Beta 分布
        normal_params = stats.beta.fit(normal_probs, floc=0, scale=1)
        bot_params = stats.beta.fit(bot_probs, floc=0, scale=1)
        
        # 找到密度相等的点
        def density_diff(x):
            return stats.beta.pdf(x, *normal_params) - stats.beta.pdf(x, *bot_params)
        
        # 在均值之间找交点
        bracket = [max(0.001, min(normal_params[0], normal_params[1]) / 10), 
                   min(0.999, max(normal_params[0], normal_params[1]) / 10 + 0.1)]
        
        try:
            intersection = brentq(density_diff, *bracket)
            print(f"  方法 3 (分布交点): 阈值 = {intersection:.6f}")
        except:
            intersection = None
            print(f"  方法 3: 无法找到交点")
    except:
        intersection = None
        print(f"  方法 3: 拟合失败")
    
    return {
        'fpr_control_threshold': normal_threshold,
        'normal_approx_threshold': normal_approx_threshold,
        'intersection_threshold': intersection
    }


def main():
    print("="*70)
    print("僵尸网络检测分数分布的统计学分析")
    print("="*70)
    
    model_path = 'improved_botnet_model.pth'
    if not os.path.exists(model_path):
        print(f"[Error] 模型文件不存在：{model_path}")
        return
    
    # 分析场景 12（主要问题场景）
    data_dir = '/root/autodl-fs/CTU-13/CTU-13-Dataset'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    loader = CTU13Loader(data_dir)
    df = loader.load_data([12])
    
    if df.empty:
        print("[Error] 数据加载失败!")
        return
    
    # 构建图
    builder = ImprovedHeterogeneousGraphBuilder()
    data, ip_map = builder.build(df, include_semantic=True, include_struct=True)
    
    # 获取标签
    labels, _ = get_labels(df, ip_map)
    
    # 加载模型
    model = ImprovedBotnetDetector.load(model_path, device=device)
    model.eval()
    
    # 推理
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
    
    y_true = labels.numpy()[:len(probs)]
    probs = probs[:len(y_true)]
    
    # 分析分布
    results = analyze_distribution(probs, y_true, 12)
    
    # 分割数据
    normal_probs = probs[y_true == 0]
    bot_probs = probs[y_true == 1]
    
    # 找最优阈值
    thresholds = find_optimal_threshold_stats(normal_probs, bot_probs, target_fpr=0.01)
    
    # 测试各阈值
    print(f"\n【各阈值方法对比】")
    methods = {
        'Youden J': results['optimal_threshold_youden'],
        'F1 Optimal': results['optimal_f1_threshold'],
        'FPR Control (1%)': thresholds['fpr_control_threshold'],
        'Normal Approx': thresholds['normal_approx_threshold'],
    }
    if thresholds['intersection_threshold']:
        methods['Distribution Intersection'] = thresholds['intersection_threshold']
    
    for name, thresh in methods.items():
        preds = (probs >= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
        print(f"  {name}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[Error] 分析过程中出错：{e}")
        import traceback
        traceback.print_exc()