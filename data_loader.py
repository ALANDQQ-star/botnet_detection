import pandas as pd
import os
import glob
import warnings
warnings.filterwarnings('ignore')

class CTU13Loader:
    """支持多场景加载的数据加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def _find_files(self, scenarios: list):
        files = []
        for sc in scenarios:
            # 兼容不同的目录结构
            path1 = os.path.join(self.data_dir, str(sc), "*.binetflow")
            found = glob.glob(path1)
            
            if not found:
                path2 = os.path.join(self.data_dir, f"*Scenario{sc}-*.binetflow")
                found = glob.glob(path2)
                
            if found:
                files.extend(found)
                print(f"[Loader] 场景 {sc} 找到文件: {len(found)} 个")
            else:
                print(f"[Loader] 警告: 未找到场景 {sc} 的文件")
        return files
    
    def load_data(self, scenarios: list) -> pd.DataFrame:
        target_files = self._find_files(scenarios)
        if not target_files:
            print("[Loader] 未找到任何数据文件！")
            return pd.DataFrame()
            
        dfs = []
        cols_map = {
            'StartTime': 'start_time', 'Dur': 'duration', 'Proto': 'protocol',
            'SrcAddr': 'src_ip', 'Sport': 'src_port', 'DstAddr': 'dst_ip', 
            'Dport': 'dst_port', 'TotPkts': 'packets', 'TotBytes': 'bytes',
            'Label': 'label'
        }
        
        print(f"[Loader] 开始加载 {len(target_files)} 个文件...")
        for f_path in target_files:
            try:
                # 优化读取：只读取有用列
                df = pd.read_csv(f_path, sep=',', low_memory=False, usecols=lambda x: x in cols_map.keys())
                df.rename(columns=cols_map, inplace=True)
                
                # 基础清洗
                df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)
                df['packets'] = pd.to_numeric(df['packets'], errors='coerce').fillna(0)
                df['bytes'] = pd.to_numeric(df['bytes'], errors='coerce').fillna(0)
                
                # 过滤掉无法确定方向的流
                df = df.dropna(subset=['src_ip', 'dst_ip'])
                dfs.append(df)
            except Exception as e:
                print(f"[Loader] 读取错误 {f_path}: {e}")
                
        if not dfs: return pd.DataFrame()
        
        full_df = pd.concat(dfs, ignore_index=True)
        print(f"[Loader] 数据合并完成。总记录数: {len(full_df)}")
        return full_df