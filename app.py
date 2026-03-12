import streamlit as st
import pandas as pd
import json
import os
import subprocess
import sys
import time
import requests
import pydeck as pdk
import streamlit.components.v1 as components 
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from viz_utils import (
    load_ctf_style, get_gnn_viz_html, render_stat_card, 
    render_status_badge, render_progress_ring, render_glass_card,
    render_header_bar, render_map_legend, render_time_display
)
from ui_state import init_state, reset_viz_state, update_log

try:
    import folium
    from streamlit_folium import st_folium
    from shapely.geometry import Point, shape
    CHINA_CLICK_ENABLED = True
except Exception:
    CHINA_CLICK_ENABLED = False
    Point = None
    shape = None

st.set_page_config(layout="wide", page_title="BOTNET HUNTER", page_icon="B", initial_sidebar_state="expanded")
load_ctf_style()
init_state()

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.cache_data.clear()
    data_files = ["viz_state_persist.json", "network_topology.json", "c2_candidates.json", "attack_chain_report.json", "topology_eval_report.json", "batch_data.json"]
    for f in data_files:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

PERSISTENCE_FILE = "viz_state_persist.json"

def load_persisted_state():
    if os.path.exists(PERSISTENCE_FILE):
        try:
            with open(PERSISTENCE_FILE, "r") as f:
                data = json.load(f)
                if st.session_state.viz_state.get("progress", 0) > 0:
                    for key in ["red_nodes", "yellow_nodes", "country_stats", "province_stats", "progress", "step_info", "sim_time", "metrics"]:
                        if key in data:
                            st.session_state.viz_state[key] = data[key]
                return True
        except:
            pass
    return False

def save_persisted_state():
    data = {
        "red_nodes": st.session_state.viz_state.get("red_nodes", []),
        "yellow_nodes": st.session_state.viz_state.get("yellow_nodes", []),
        "country_stats": st.session_state.viz_state.get("country_stats", {}),
        "province_stats": st.session_state.viz_state.get("province_stats", {}),
        "progress": st.session_state.viz_state.get("progress", 0.0),
        "step_info": st.session_state.viz_state.get("step_info", "就绪"),
        "sim_time": st.session_state.viz_state.get("sim_time"),
        "metrics": st.session_state.viz_state.get("metrics"),
    }
    try:
        with open(PERSISTENCE_FILE, "w") as f:
            json.dump(data, f)
    except:
        pass

mode_map = {"完整流程（训练+测试）": "all", "仅训练": "train", "仅测试": "test"}

@st.cache_data
def get_world_geojson():
    local_path = "world.geo.json"
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f: return json.load(f)
        except: pass
    return None

@st.cache_data
def get_china_standard_geojson():
    try:
        url = "https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

@st.cache_data
def load_topology_subset(max_neighbors=40):
    topo_path = "network_topology.json"
    c2_path = "c2_candidates.json"
    if not os.path.exists(topo_path) or not os.path.exists(c2_path): return set(), [], []
    with open(c2_path, "r") as f: c2_list = json.load(f)
    c2_set = set([c["ip"] for c in c2_list])
    with open(topo_path, "r") as f: data = json.load(f)
    nodes = data.get("nodes", [])
    node_ids = [n.get("id") or n.get("name") for n in nodes]
    def resolve(node):
        if isinstance(node, int): return node_ids[node] if node < len(node_ids) else None
        return node.get("id") or node.get("name") if isinstance(node, dict) else node
    subset_nodes = set(c2_set)
    subset_links = []
    neighbor_counts = {c2: 0 for c2 in c2_set}
    for link in data.get("links", []):
        src, tgt = resolve(link.get("source")), resolve(link.get("target"))
        if not src or not tgt: continue
        if src in c2_set or tgt in c2_set:
            c2 = src if src in c2_set else tgt
            if neighbor_counts[c2] >= max_neighbors: continue
            subset_nodes.update([src, tgt])
            subset_links.append((src, tgt))
            neighbor_counts[c2] += 1
    return c2_set, list(subset_nodes), subset_links

world_geojson = get_world_geojson()
china_standard_geojson = get_china_standard_geojson()

cfg = st.session_state.config
mode_label = cfg.get("mode_ui", "完整流程（训练+测试）")
data_dir = cfg.get("data_dir", "N/A")
threshold = cfg.get("threshold", "N/A")
progress_overall = st.session_state.viz_state.get("progress", 0.0)
step_info = st.session_state.viz_state.get("step_info", "就绪")

if st.session_state.is_running:
    system_status = "running"
elif progress_overall >= 1.0 or step_info == "运行完毕":
    system_status = "completed"
else:
    system_status = "ready"

st.markdown(render_header_bar(title="BOTNET HUNTER", subtitle="僵尸网络威胁分析平台 - 学术实验环境", status=system_status, version="v2.0"), unsafe_allow_html=True)

red_nodes = st.session_state.viz_state.get("red_nodes", [])
yellow_nodes = st.session_state.viz_state.get("yellow_nodes", [])
country_stats = st.session_state.viz_state.get("country_stats", {})
c2_set, topo_nodes, topo_links = load_topology_subset(max_neighbors=25)

threat_count = len(red_nodes)
suspect_count = len(yellow_nodes)
c2_count = len(c2_set)
risk_score = min(100, int(threat_count * 2 + c2_count * 10 + suspect_count * 0.5))

stat_row = st.columns(4)
stat_containers = [stat_row[i].empty() for i in range(4)]
with stat_containers[0]:
    st.markdown(render_stat_card("已识别威胁节点", f"{threat_count:,}", "确认的僵尸网络节点", color="red"), unsafe_allow_html=True)
with stat_containers[1]:
    st.markdown(render_stat_card("C2 服务器", f"{c2_count:,}", "命令与控制服务器", color="purple"), unsafe_allow_html=True)
with stat_containers[2]:
    st.markdown(render_stat_card("疑似风险节点", f"{suspect_count:,}", "待进一步分析", color="yellow"), unsafe_allow_html=True)
with stat_containers[3]:
    st.markdown(render_stat_card("风险评分", f"{risk_score}", "综合威胁指数", color="cyan"), unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

col_main_left, col_main_right = st.columns([3, 7])

def render_maps(map_view, container):
    stats = st.session_state.viz_state["country_stats"]
    prov_stats = st.session_state.viz_state.get("province_stats", {})
    stats_lower = {k.lower().strip(): v for k, v in stats.items() if k != "Unknown"}
    def get_color_by_count(count):
        if count > 0:
            intensity = min(count + 120, 255)
            return [255, 0, 0, intensity] if count >= 100 else [255, 69, 0, 200] if count >= 50 else [255, 140, 0, 180] if count >= 20 else [255, 215, 0, 160] if count >= 5 else [0, 200, 0, 140]
        return [20, 20, 20, 100]
    scatter_layers = []
    for node_type, color in [("red_nodes", [255, 50, 50, 200]), ("yellow_nodes", [255, 215, 0, 255])]:
        data = st.session_state.viz_state.get(node_type)
        if data:
            scatter_layers.append(pdk.Layer("ScatterplotLayer", data=pd.DataFrame(data), get_position=["lon", "lat"], get_color=color, get_radius=30000, radius_min_pixels=6, radius_max_pixels=30, pickable=True))
    with container:
        if map_view == "全球":
            world_layers = []
            if world_geojson:
                geo_data = json.loads(json.dumps(world_geojson))
                for feature in geo_data["features"]:
                    c_name = feature["properties"].get("name", "Unknown").lower().strip()
                    count = stats_lower.get(c_name, 0)
                    if count == 0:
                        for k, v in stats_lower.items():
                            if k in c_name or c_name in k: count = v; break
                    if c_name == "taiwan":
                        china_count = stats_lower.get("china", 0) + stats_lower.get("people's republic of china", 0)
                        count = max(count, china_count)
                    feature["properties"]["fillColor"] = get_color_by_count(count)
                    feature["properties"]["lineColor"] = [50, 50, 50, 100]
                world_layers.append(pdk.Layer("GeoJsonLayer", data=geo_data, pickable=False, stroked=True, filled=True, line_width_min_pixels=1, get_fill_color="properties.fillColor", get_line_color="properties.lineColor"))
            deck_world = pdk.Deck(map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json", initial_view_state=pdk.ViewState(latitude=20, longitude=10, zoom=1.1), layers=world_layers + scatter_layers, tooltip={"html": "<b>IP:</b> {ip}<br/><b>国家:</b> {country}<br/><b>风险:</b> {conf}"})
            st.pydeck_chart(deck_world)
        else:
            if china_standard_geojson:
                cn_data = json.loads(json.dumps(china_standard_geojson))
                for feature in cn_data["features"]:
                    p_name = feature["properties"].get("name", "")
                    count = 0
                    if p_name in prov_stats: count = prov_stats[p_name]
                    else:
                        for stat_name, stat_val in prov_stats.items():
                            if stat_name and (stat_name in p_name or p_name in stat_name): count += stat_val
                    feature["properties"]["fillColor"] = get_color_by_count(count)
                    feature["properties"]["lineColor"] = [0, 255, 65, 200]
                china_layers = [pdk.Layer("GeoJsonLayer", data=cn_data, pickable=False, stroked=True, filled=True, line_width_min_pixels=1.5, get_fill_color="properties.fillColor", get_line_color="properties.lineColor")]
                deck_china = pdk.Deck(map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json", initial_view_state=pdk.ViewState(latitude=35, longitude=105, zoom=3.2), layers=china_layers + scatter_layers, tooltip={"html": "<b>IP:</b> {ip}<br/><b>省份/地区:</b> {region}<br/><b>风险:</b> {conf}"})
                st.pydeck_chart(deck_china)
            else:
                st.warning("等待标准地图数据...")

def render_botnet_topology(container):
    c2_set, nodes, links = load_topology_subset(max_neighbors=25)
    if not nodes:
        container.info("暂无拓扑数据。")
        return
    dot = ["graph G {", "layout=neato;", "overlap=false;", "splines=curved;", "bgcolor=\"#00000000\";", "outputorder=edgesfirst;", "node [shape=circle, style=filled, fixedsize=true, width=0.10, height=0.10, fillcolor=\"#334155\", color=\"#38bdf8\", penwidth=1.0, label=\"\", fontsize=0];", "edge [color=\"#475569\", penwidth=0.6, arrowsize=0.4];"]
    for n in nodes:
        if n in c2_set: dot.append(f"\"{n}\" [shape=diamond, width=0.25, height=0.25, color=\"#f43f5e\", fillcolor=\"#881337\", penwidth=1.5, tooltip=\"C2 服务器: {n}\"];")
        else: dot.append(f"\"{n}\" [tooltip=\"僵尸节点: {n}\"];")
    for s, t in links: dot.append(f"\"{s}\" -- \"{t}\";")
    dot.append("}")
    container.graphviz_chart("\n".join(dot))

def update_ui_elements(progress_container, progress_ring_container, stat_containers, time_container=None):
    red_nodes = st.session_state.viz_state.get("red_nodes", [])
    yellow_nodes = st.session_state.viz_state.get("yellow_nodes", [])
    c2_set, _, _ = load_topology_subset(max_neighbors=25)
    threat_count = len(red_nodes)
    suspect_count = len(yellow_nodes)
    c2_count = len(c2_set)
    risk_score = min(100, int(threat_count * 2 + c2_count * 10 + suspect_count * 0.5))
    current_progress = st.session_state.viz_state.get("progress", 0.0)
    step_info = st.session_state.viz_state.get("step_info", "就绪")
    with stat_containers[0]:
        st.markdown(render_stat_card("已识别威胁节点", f"{threat_count:,}", "确认的僵尸网络节点", color="red"), unsafe_allow_html=True)
    with stat_containers[1]:
        st.markdown(render_stat_card("C2 服务器", f"{c2_count:,}", "命令与控制服务器", color="purple"), unsafe_allow_html=True)
    with stat_containers[2]:
        st.markdown(render_stat_card("疑似风险节点", f"{suspect_count:,}", "待进一步分析", color="yellow"), unsafe_allow_html=True)
    with stat_containers[3]:
        st.markdown(render_stat_card("风险评分", f"{risk_score}", "综合威胁指数", color="cyan"), unsafe_allow_html=True)
    with progress_ring_container.container():
        st.markdown(render_progress_ring(current_progress, size=70, stroke_width=5), unsafe_allow_html=True)
    with progress_container.container():
        st.progress(current_progress)
        if step_info == "运行完毕" or current_progress >= 1.0:
            st.success("运行完毕")
        else:
            st.info(f"{step_info}")
    # 更新时间显示
    if time_container is not None:
        st_ts = st.session_state.viz_state.get("sim_time")
        sim_str = "N/A"
        if st_ts:
            try: sim_str = datetime.fromisoformat(str(st_ts)).strftime("%Y-%m-%d %H:%M:%S")
            except: sim_str = str(st_ts)
        real_str = datetime.now().strftime('%H:%M:%S')
        with time_container.container():
            st.markdown(render_time_display(real_str, sim_str), unsafe_allow_html=True)

def set_progress_monotonic(new_progress):
    current = st.session_state.viz_state.get("progress", 0.0)
    if new_progress > current:
        st.session_state.viz_state["progress"] = new_progress
        return True
    return False

with col_main_left:
    st_ts = st.session_state.viz_state.get("sim_time")
    sim_str = "N/A"
    if st_ts:
        try: sim_str = datetime.fromisoformat(str(st_ts)).strftime("%Y-%m-%d %H:%M:%S")
        except: sim_str = str(st_ts)
    real_str = datetime.now().strftime('%H:%M:%S')
    time_container = st.empty()
    with time_container.container():
        st.markdown(render_time_display(real_str, sim_str), unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div style="background:rgba(15,23,42,0.7);backdrop-filter:blur(12px);border:1px solid rgba(51,65,85,0.5);border-radius:16px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.3);"><div style="padding:1rem 1.25rem;border-bottom:1px solid rgba(51,65,85,0.3);display:flex;align-items:center;"><span style="color:#38bdf8;font-weight:600;font-size:1rem;">分析控制台</span></div><div style="padding:1.25rem;">', unsafe_allow_html=True)
    st.caption("一键启动完整扫描，实时跟踪训练与推演进度。")
    
    # 方法选择
    method_label = {
        "existing": "现有方法 (GAT+GCN)",
        "baseline": "基线方法 (Bot-AHGCN)"
    }
    selected_method = st.selectbox(
        "检测方法",
        ["existing", "baseline"],
        format_func=lambda x: method_label.get(x, x),
        index=0 if st.session_state.config.get("method", "existing") == "existing" else 1,
        key="method_selector"
    )
    st.session_state.config["method"] = selected_method
    
    col_btn = st.columns([3, 2])
    with col_btn[0]:
        if not st.session_state.is_running:
            if st.button("启动分析", type="primary", use_container_width=True):
                st.session_state.run_requested = True
        else:
            st.button("运行中...", disabled=True, use_container_width=True)
    with col_btn[1]:
        if st.button("配置", type="secondary", use_container_width=True):
            st.switch_page("pages/07_参数配置.py")
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    current_progress = st.session_state.viz_state.get("progress", 0.0)
    step_info = st.session_state.viz_state.get("step_info", "就绪")
    progress_ring_container = st.empty()
    progress_container = st.empty()
    with progress_ring_container.container():
        st.markdown(render_progress_ring(current_progress, size=70, stroke_width=5), unsafe_allow_html=True)
    with progress_container.container():
        st.progress(current_progress)
        if step_info == "运行完毕" or current_progress >= 1.0:
            st.success("运行完毕")
        else:
            st.info(f"{step_info}")
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    topo_expander = st.expander("网络拓扑概要", expanded=False)
    with topo_expander:
        st.caption("展示当前识别到的 C2 服务器及其控制的僵尸节点之间的连接关系。")
        topo_ph = st.container()
    if not st.session_state.is_running:
        render_botnet_topology(topo_ph)
    data_name = os.path.basename(str(data_dir)) or data_dir
    st.markdown(f'<div style="background:rgba(15,23,42,0.5);border:1px solid rgba(51,65,85,0.3);border-radius:10px;padding:12px;margin-top:0.5rem;"><div style="display:flex;flex-wrap:wrap;gap:8px;font-size:0.8rem;"><span style="background:#1e293b;padding:4px 10px;border-radius:6px;color:#94a3b8;">数据: <span style="color:#f1f5f9;">{data_name}</span></span><span style="background:#1e293b;padding:4px 10px;border-radius:6px;color:#94a3b8;">阈值: <span style="color:#f1f5f9;">{threshold}</span></span><span style="background:#1e293b;padding:4px 10px;border-radius:6px;color:#94a3b8;">模式: <span style="color:#f1f5f9;">{mode_label}</span></span></div></div>', unsafe_allow_html=True)

with col_main_right:
    st.markdown('<div style="background:rgba(15,23,42,0.7);backdrop-filter:blur(12px);border:1px solid rgba(51,65,85,0.5);border-radius:16px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.3);"><div style="padding:1rem 1.25rem;border-bottom:1px solid rgba(51,65,85,0.3);display:flex;justify-content:space-between;align-items:center;"><div style="display:flex;align-items:center;gap:8px;"><span style="color:#f1f5f9;font-weight:600;font-size:1rem;">全球僵尸网络态势</span></div></div><div style="padding:1rem;">', unsafe_allow_html=True)
    st.caption("通过地理热力与散点标记，直观展示已识别僵尸节点与疑似高危主机的分布。")
    map_view = st.radio("地图视图", ["全球", "中国"], horizontal=True, label_visibility="collapsed")
    map_container = st.empty()
    st.markdown(render_map_legend(), unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)
    render_maps("全球" if "全球" in map_view else "中国", map_container)

st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
st.markdown('<div style="background:rgba(15,23,42,0.7);backdrop-filter:blur(12px);border:1px solid rgba(51,65,85,0.5);border-radius:16px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.3);"><div style="padding:1rem 1.25rem;border-bottom:1px solid rgba(51,65,85,0.3);display:flex;align-items:center;"><span style="color:#f1f5f9;font-weight:600;font-size:1rem;">图神经网络推理过程可视化</span></div><div style="padding:0.5rem 1.25rem 1rem;"><p style="color:#94a3b8;font-size:0.85rem;margin:0 0 0.5rem 0;"> </p></div></div>', unsafe_allow_html=True)

gnn_viz_container = st.empty()

if st.session_state.run_requested and not st.session_state.is_running:
    st.session_state.is_running = True
    st.session_state.run_requested = False
    reset_viz_state()
    st.session_state.viz_state["progress"] = 0.0
    st.session_state.viz_state["step_info"] = "初始化中..."
    cfg = st.session_state.config
    mode_arg = mode_map.get(cfg.get("mode_ui"), "all")
    cmd = [sys.executable, "-u", "main.py", "--data_dir", cfg.get("data_dir"), "--mode", mode_arg, "--train_scenarios", cfg.get("train_scens"), "--test_scenarios", cfg.get("test_scens"), "--epochs", str(cfg.get("epochs")), "--lr", str(cfg.get("lr")), "--threshold", str(cfg.get("threshold")), "--method", cfg.get("method", "existing")]
    if cfg.get("use_dynamic"): cmd.append("--use_dynamic")
    if cfg.get("force_retrain"): cmd.append("--force_retrain")
    if cfg.get("force_hmm"): cmd.append("--force_hmm")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        update_log("子进程已启动，正在初始化...")
        update_ui_elements(progress_container, progress_ring_container, stat_containers)
        while True:
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None: break
                time.sleep(0.1); continue
            sys.stdout.write(line)
            sys.stdout.flush()
            line = line.strip()
            if "warning" in line.lower() and "@@UI_SIGNAL@@" not in line: continue
            if "@@UI_SIGNAL@@" in line:
                try:
                    _, sig, content = line.split("|", 2)
                    if sig == "BATCH_UPDATE":
                        with open(content, "r") as f: data = json.load(f)
                        st.session_state.viz_state["red_nodes"].extend(data["new_red_nodes"])
                        st.session_state.viz_state["yellow_nodes"] = data["yellow_nodes"]
                        st.session_state.viz_state["country_stats"] = data["country_stats"]
                        if "province_stats" in data: st.session_state.viz_state["province_stats"] = data["province_stats"]
                        st.session_state.viz_state["sim_time"] = data.get("sim_time")
                        save_persisted_state()
                        render_maps("全球" if "全球" in map_view else "中国", map_container)
                        update_log(f"感染扩散: +{len(data['new_red_nodes'])} 节点")
                        step = data.get("step", 0)
                        total_steps = data.get("total_steps", 1)
                        if total_steps > 0:
                            prog = min(1.0, 0.5 + (step / total_steps) * 0.5)
                            set_progress_monotonic(prog)
                            st.session_state.viz_state["step_info"] = f"推演中: 步骤 {step}/{total_steps}"
                            save_persisted_state()
                        update_ui_elements(progress_container, progress_ring_container, stat_containers, time_container)
                    elif sig == "LOG":
                        update_log(content)
                    elif sig == "PHASE_START":
                        st.session_state.viz_state["step_info"] = content
                        if content.strip() == "运行完毕":
                            set_progress_monotonic(1.0)
                        else:
                            if "HMM" in content or "攻击链" in content:
                                set_progress_monotonic(0.05)
                            elif "GNN" in content or "训练" in content:
                                set_progress_monotonic(0.15)
                            elif "推演" in content or "感染" in content:
                                set_progress_monotonic(0.5)
                            elif "报告" in content or "生成" in content:
                                set_progress_monotonic(0.9)
                            elif "评估" in content or "威胁" in content:
                                set_progress_monotonic(0.35)
                        save_persisted_state()
                        update_ui_elements(progress_container, progress_ring_container, stat_containers)
                    elif sig == "PROGRESS_HMM":
                        parts = content.split("|")
                        if len(parts) == 2:
                            prog_val = float(parts[0])
                            msg = parts[1]
                            mapped_prog = 0.05 + prog_val * 0.1
                            set_progress_monotonic(mapped_prog)
                            st.session_state.viz_state["step_info"] = msg
                            save_persisted_state()
                            update_ui_elements(progress_container, progress_ring_container, stat_containers)
                    elif sig == "EPOCH_UPDATE":
                        ep, tot, loss = content.split("|")
                        st.session_state.viz_state["step_info"] = f"训练轮次 {ep}/{tot}（损失: {loss}）"
                        ep_val = int(ep)
                        tot_val = int(tot)
                        if tot_val > 0:
                            mapped_prog = 0.15 + (ep_val / tot_val) * 0.2
                            set_progress_monotonic(mapped_prog)
                        save_persisted_state()
                        update_ui_elements(progress_container, progress_ring_container, stat_containers)
                    elif sig == "PROGRESS_INFERENCE":
                        parts = content.split("|")
                        if len(parts) == 2:
                            cur, tot = int(parts[0]), int(parts[1])
                            st.session_state.viz_state["step_info"] = f"推理进度 {cur}/{tot}"
                            if tot > 0:
                                mapped_prog = 0.35 + (cur / tot) * 0.15
                                set_progress_monotonic(mapped_prog)
                            save_persisted_state()
                            update_ui_elements(progress_container, progress_ring_container, stat_containers)
                    elif sig == "METRICS":
                        auc, f1, prec, rec, th = content.split("|")
                        st.session_state.viz_state["metrics"] = {"auc": auc, "f1": f1, "prec": prec, "rec": rec, "thresh": th}
                        update_log(f"指标更新: AUC={auc}, F1={f1}")
                        save_persisted_state()
                    elif sig == "GNN_VIS":
                        payload = json.loads(content)
                        st.session_state.gnn_viz_data = payload
                        with gnn_viz_container:
                            html_code = get_gnn_viz_html(phase=payload.get("phase", "idle"), epoch=payload.get("epoch", 0), loss=payload.get("loss", 0.0), layer=payload.get("layer", -1))
                            components.html(html_code, height=520, scrolling=False)
                except Exception as e: print(f"Signal Parse Error: {e}")
            elif line.startswith("[Loader]") or line.startswith("[Graph]") or line.startswith("[Feature]"):
                update_log(line)
                current_prog = st.session_state.viz_state.get("progress", 0.0)
                if line.startswith("[Loader]"):
                    new_prog = min(current_prog + 0.01, 0.05)
                    layer_idx = -1
                    st.session_state.viz_state["step_info"] = "加载数据中..."
                elif line.startswith("[Graph]"):
                    new_prog = min(current_prog + 0.02, 0.35)
                    layer_idx = 0
                    st.session_state.viz_state["step_info"] = "构建推理图中..."
                elif line.startswith("[Feature]"):
                    new_prog = min(current_prog + 0.01, 0.35)
                    layer_idx = 0 if "提取" in line or "聚合" in line else 1
                    st.session_state.viz_state["step_info"] = "提取特征中..."
                else:
                    new_prog = current_prog
                    layer_idx = -1
                set_progress_monotonic(new_prog)
                save_persisted_state()
                update_ui_elements(progress_container, progress_ring_container, stat_containers)
                if layer_idx >= 0:
                    payload = {"phase": "inference", "epoch": 0, "loss": 0.0, "layer": layer_idx}
                    st.session_state.gnn_viz_data = payload
                    with gnn_viz_container:
                        html_code = get_gnn_viz_html(phase=payload.get("phase", "idle"), epoch=payload.get("epoch", 0), loss=payload.get("loss", 0.0), layer=payload.get("layer", -1))
                        components.html(html_code, height=520, scrolling=False)
            elif line and not line.startswith("@@"): pass
        if process.returncode != 0:
            st.error("后端进程异常，请查看终端详情。")
            set_progress_monotonic(1.0)
        else:
            set_progress_monotonic(1.0)
            st.session_state.viz_state["step_info"] = "运行完毕"
            save_persisted_state()
            st.cache_data.clear()
            update_ui_elements(progress_container, progress_ring_container, stat_containers, time_container)
            with topo_expander:
                topo_ph.empty()
                render_botnet_topology(topo_ph)
            st.balloons()
            st.markdown('<div style="background:linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.8) 100%);border:1px solid rgba(56,189,248,0.3);border-radius:16px;padding:2rem;margin-top:1.5rem;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.4);"><h3 style="color:#38bdf8;margin:0 0 0.5rem 0;font-family:Inter,sans-serif;font-weight:700;">威胁扫描与推演已完成</h3><p style="color:#94a3b8;font-size:1rem;margin:0 0 1.5rem 0;">分析报告已生成，点击下方按钮查看详细结果</p></div>', unsafe_allow_html=True)
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
            with btn_col1:
                if st.button("📊 流量态势", key="btn_flow_status", use_container_width=True):
                    st.switch_page("pages/01_流量态势.py")
                st.caption("攻击链与专家建议")
            with btn_col2:
                if st.button("📄 报告导出", key="btn_report", use_container_width=True):
                    st.switch_page("pages/09_报告导出.py")
                st.caption("下载分析报告")
    except Exception as e: st.error(f"启动进程失败：{e}")
    st.session_state.is_running = False
    save_persisted_state()

if not st.session_state.is_running and progress_overall > 0 and progress_overall < 1.0:
    time.sleep(1)
    st.rerun()
