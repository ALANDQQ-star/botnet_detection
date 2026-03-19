"""
测试多维分类器在真实场景上的效果
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

# 导入两种分类器
from enhanced_classifier import compute_botnet_metrics as clustering_classifier
from multidimensional_classifier import compute_botnet_metrics_multidim


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


def test_scenario(scenario, data_dir='/root/autodl-fs/CTU-13/CTU-13-Dataset', 
                  model_path='improved_botnet_model.pth'):
    """测试单个场景"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载数据
    loader = CTU13Loader(data_dir)
    df = loader.load_data([scenario])
    
    if df.empty:
        return None
    
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
    
    # 确保对齐
    y_true = labels.numpy()[:len(probs)]
    probs = probs[:len(y_true)]
    features = data['ip'].x.numpy()
    
    # 测试两种分类器
    result_clustering = clustering_classifier(y_true, probs, features, use_multidim=False)
    result_multidim = compute_botnet_metrics_multidim(y_true, probs, features)
    
    return {
        'scenario': scenario,
        'num_nodes': len(probs),
        'num_bots': int(y_true.sum()),
        'auc': result_clustering['auc'],
        'clustering': {
            'precision': result_clustering['precision'],
            'recall': result_clustering['recall'],
            'f1': result_clustering['f1'],
            'num_predicted': result_clustering['num_predicted'],
            'method': result_clustering['method']
        },
        'multidim': {
            'precision': result_multidim['precision'],
            'recall': result_multidim['recall'],
            'f1': result_multidim['f1'],
            'num_predicted': result_multidim['num_predicted'],
            'method': result_multidim['method']
        }
    }


def main():
    print("="*80)
    print("多维分类器 vs 聚类分类器 - 对比测试")
    print("="*80)
    
    model_path = 'improved_botnet_model.pth'
    if not os.path.exists(model_path):
        print(f"[Error] 模型文件不存在：{model_path}")
        return
    
    results = []
    test_scenarios = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    
    for scenario in test_scenarios:
        print(f"\n测试场景 {scenario}...", end=" ", flush=True)
        result = test_scenario(scenario)
        if result:
            results.append(result)
            print(f"完成 (AUC={result['auc']:.4f}, 僵尸节点={result['num_bots']})")
        else:
            print(f"失败")
        
        # 清理内存
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("结果汇总")
    print("="*80)
    
    header = f"{'场景':<6} {'AUC':<8} {'方法':<12} {'P':<8} {'R':<8} {'F1':<8} {'预测/实际':<12}"
    print(header)
    print("-"*80)
    
    for r in results:
        scenario = r['scenario']
        auc = r['auc']
        num_bots = r['num_bots']
        
        # 聚类分类器
        c = r['clustering']
        print(f"{scenario:<6} {auc:<8.4f} {'Clustering':<12} {c['precision']:<8.4f} {c['recall']:<8.4f} {c['f1']:<8.4f} {c['num_predicted']}/{num_bots:<8}")
        
        # 多维分类器
        m = r['multidim']
        print(f"{'':<6} {'':<8} {'MultiDim':<12} {m['precision']:<8.4f} {m['recall']:<8.4f} {m['f1']:<8.4f} {m['num_predicted']}/{num_bots:<8}")
        print("-"*80)
    
    # 计算平均提升
    print("\n" + "="*80)
    print("性能对比")
    print("="*80)
    
    avg_auc = np.mean([r['auc'] for r in results])
    avg_p_clustering = np.mean([r['clustering']['precision'] for r in results])
    avg_r_clustering = np.mean([r['clustering']['recall'] for r in results])
    avg_f1_clustering = np.mean([r['clustering']['f1'] for r in results])
    
    avg_p_multidim = np.mean([r['multidim']['precision'] for r in results])
    avg_r_multidim = np.mean([r['multidim']['recall'] for r in results])
    avg_f1_multidim = np.mean([r['multidim']['f1'] for r in results])
    
    print(f"{'':<18} AUC: {avg_auc:.4f}")
    print(f"Clustering - P: {avg_p_clustering:.4f}, R: {avg_r_clustering:.4f}, F1: {avg_f1_clustering:.4f}")
    print(f"MultiDim   - P: {avg_p_multidim:.4f}, R: {avg_r_multidim:.4f}, F1: {avg_f1_multidim:.4f}")
    
    f1_improvement = (avg_f1_multidim - avg_f1_clustering) / avg_f1_clustering * 100 if avg_f1_clustering > 0 else 0
    print(f"\nF1 提升：{f1_improvement:+.1f}%")
    
    # 统计获胜场景
    multidim_better = sum(1 for r in results if r['multidim']['f1'] > r['clustering']['f1'])
    clustering_better = len(results) - multidim_better
    
    print(f"F1 优胜场景：MultiDim {multidim_better}/{len(results)}, Clustering {clustering_better}/{len(results)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[Error] 测试过程中出错：{e}")
        import traceback
        traceback.print_exc()