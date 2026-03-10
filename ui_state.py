import streamlit as st
from collections import deque
from datetime import datetime

def init_state():
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "activity_stream" not in st.session_state:
        st.session_state.activity_stream = deque(maxlen=200)
    if "viz_state" not in st.session_state:
        st.session_state.viz_state = {
            "red_nodes": [],
            "yellow_nodes": [],
            "country_stats": {},
            "province_stats": {},
            "step_info": "就绪",
            "metrics": None,
            "sim_time": None,
            "progress": 0.0,
        }
    if "gnn_viz_data" not in st.session_state:
        st.session_state.gnn_viz_data = {
            "phase": "idle",
            "epoch": 0,
            "loss": 0.0
        }
    if "config" not in st.session_state:
        st.session_state.config = {
            "mode_ui": "完整流程（训练+测试）",
            "train_scens": "1,2,9",
            "test_scens": "13",
            "epochs": 6,
            "lr": 0.003,
            "threshold": 0.01,
            "data_dir": "/root/autodl-fs/CTU-13/CTU-13-Dataset",
            "use_dynamic": True,
            "force_retrain": False,
            "force_hmm": False,
            "method": "existing",  # Method selection: "existing" or "baseline"
        }
    if "run_requested" not in st.session_state:
        st.session_state.run_requested = False
    if "show_china_map" not in st.session_state:
        st.session_state.show_china_map = False


def reset_viz_state():
    st.session_state.viz_state = {
        "red_nodes": [],
        "yellow_nodes": [],
        "country_stats": {},
        "province_stats": {},
        "step_info": "初始化",
        "metrics": None,
        "sim_time": None,
        "progress": 0.0
    }
    # 重置可视化状态
    st.session_state.gnn_viz_data = {"phase": "idle", "epoch": 0, "loss": 0.0, "layer": -1}
    
    # 清理旧的报告文件
    import os
    for f in ["network_topology.json", "c2_candidates.json", "attack_chain_report.json", "topology_eval_report.json", "batch_data.json"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass


def update_log(msg: str):
    st.session_state.activity_stream.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    )