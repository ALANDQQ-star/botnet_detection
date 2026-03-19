"""
快速测试多维分类器效果 - 只测试场景 11, 12（数据量适中）
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
    print(f"  运行聚类分类器...", end=" ", flush=True)
    result_clustering = clustering_classifier(y_true, probs, features, use_multidim=False)
    print(f"完成 (F1={result_clustering['f1']:.4f})")
    
    print(f"  运行多维分类器...", end=" ", flush=True)
    result_multidim = compute_botnet_metrics_multidim(y_true, probs, features)
    print(f"完成 (F1={result_multidim['f1']:.4f})")
    
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
    print("快速测试：多维分类器 vs 聚类分类器")
    print("="*80)
    
    model_path = 'improved_botnet_model.pth'
    if not os.path.exists(model_path):
        print(f"[Error] 模型文件不存在：{model_path}")
        return
    
    results = []
    # 只测试场景 11 和 12（数据量适中，有代表性）
    test_scenarios = [11, 12]
    
    for scenario in test_scenarios:
        print(f"\n测试场景 {scenario}...")
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
    
    for r in results:
        scenario = r['scenario']
        auc = r['auc']
        num_bots = r['num_bots']
        
        print(f"\n场景 {scenario} (AUC={auc:.4f}, 僵尸节点={num_bots}):")
        print("-"*70)
        
        c = r['clustering']
        print(f"  聚类分类器：P={c['precision']:.4f}, R={c['recall']:.4f}, F1={c['f1']:.4f}, "
              f"预测={c['num_predicted']}, 方法={c['method']}")
        
        m = r['multidim']
        print(f"  多维分类器：P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, "
              f"预测={m['num_predicted']}, 方法={m['method']}")
        
        # 计算提升
        f1_improve = (m['f1'] - c['f1']) / c['f1'] * 100 if c['f1'] > 0 else 0
        print(f"  F1 提升：{f1_improve:+.1f}%")
    
    # 总体统计
    if results:
        print("\n" + "="*80)
        print("总体统计")
        print("="*80)
        
        avg_f1_clustering = np.mean([r['clustering']['f1'] for r in results])
        avg_f1_multidim = np.mean([r['multidim']['f1'] for r in results])
        overall_improve = (avg_f1_multidim - avg_f1_clustering) / avg_f1_clustering * 100 if avg_f1_clustering > 0 else 0
        
        print(f"平均 F1 (聚类): {avg_f1_clustering:.4f}")
        print(f"平均 F1 (多维): {avg_f1_multidim:.4f}")
        print(f"平均 F1 提升：{overall_improve:+.1f}%")
        
        multidim_better = sum(1 for r in results if r['multidim']['f1'] > r['clustering']['f1'])
        print(f"F1 优胜场景：多维分类器 {multidim_better}/{len(results)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[Error] 测试过程中出错：{e}")
        import traceback
        traceback.print_exc()