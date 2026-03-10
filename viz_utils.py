import streamlit as st
import os
import re
import geoip2.database
import requests
import json
import random
import math

STATE_CN = {
    "BOOT": "启动",
    "DNS_QUERY": "DNS 查询",
    "C2_SETUP": "C2 建立",
    "C2_MAINTAIN": "C2 维持",
    "SCANNING": "扫描",
    "ATTACKING": "攻击",
    "DATA_EXFIL": "数据外泄",
    "DEAD": "僵死",
}

def state_to_cn(name):
    return STATE_CN.get(str(name).strip(), str(name))

OBS_CN = {
    "SILENCE": "静默",
    "DNS_PKT": "DNS",
    "TCP_SYN_LOW": "TCP 低包数",
    "TCP_SYN_HIGH": "TCP 高包数",
    "UDP_HIGH": "UDP 高包",
    "HIGH_BPP": "高字节/包",
    "LONG_CONN": "长连接",
    "UNKNOWN": "未知",
}

def obs_to_cn(name):
    return OBS_CN.get(str(name).strip(), str(name))

def ip_to_int(ip_str):
    if not ip_str or not isinstance(ip_str, str):
        return 0
    try:
        parts = ip_str.strip().split(".")
        if len(parts) != 4:
            return 0
        return int(parts[0]) * 16777216 + int(parts[1]) * 65536 + int(parts[2]) * 256 + int(parts[3])
    except (ValueError, TypeError):
        return 0

def format_port_display(port_val):
    if port_val is None or (isinstance(port_val, str) and port_val.strip() in ("", "N/A")):
        return "N/A"
    if isinstance(port_val, (list, tuple)):
        nums = []
        for p in port_val:
            try:
                n = int(p)
                if n >= 0:
                    nums.append(str(n))
            except (TypeError, ValueError):
                continue
        return ", ".join(nums) if nums else "N/A"
    if isinstance(port_val, (int, float)):
        try:
            n = int(port_val)
            return str(n) if n >= 0 else "N/A"
        except (TypeError, ValueError):
            return "N/A"
    s = str(port_val).strip()
    nums = re.findall(r"-?\d+", s)
    if not nums:
        return "N/A"
    valid = [n for n in nums if int(n) >= 0]
    return ", ".join(valid) if valid else "N/A"

_GEOIP_READER = None

def get_geoip_reader():
    global _GEOIP_READER
    db_path = 'GeoLite2-City.mmdb'
    if _GEOIP_READER is None:
        if os.path.exists(db_path):
            try:
                _GEOIP_READER = geoip2.database.Reader(db_path)
            except Exception:
                pass
    return _GEOIP_READER

def get_ip_info(ip_str):
    if not ip_str or ip_str.startswith(('127.', '192.168.', '10.', 'localhost', '0.0', '0.')):
        return 0.0, 0.0, "UNK", "Unknown", "Unknown"

    lat, lon, c_code, c_name, r_name = 0.0, 0.0, "UNK", "Unknown", "Unknown"
    found = False

    reader = get_geoip_reader()
    if reader:
        try:
            response = reader.city(ip_str)
            if response.location.latitude and response.location.longitude:
                lat = response.location.latitude
                lon = response.location.longitude
                c_code = response.country.iso_code if response.country.iso_code else "UNK"
                c_name = response.country.name if response.country.name else "Unknown"
                subdiv = response.subdivisions.most_specific
                r_name = subdiv.names.get('zh-CN', subdiv.name) if subdiv.name else "Unknown"
                found = True
        except:
            pass

    if not found:
        try:
            url = f"http://ip-api.com/json/{ip_str}?lang=zh-CN"
            resp = requests.get(url, timeout=0.5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('status') == 'success':
                    lat = data.get('lat')
                    lon = data.get('lon')
                    c_code = data.get('countryCode', 'UNK')
                    c_name = data.get('country', 'Unknown')
                    r_name = data.get('regionName', 'Unknown')
                    found = True
        except:
            pass

    if c_name in ['Taiwan', 'Republic of China', 'Taiwan (Province of China)']:
        r_name = "台湾省"
    elif c_name == 'Hong Kong':
        r_name = "香港特别行政区"
    elif c_name == 'Macao':
        r_name = "澳门特别行政区"

    if found:
        return lat, lon, c_code, c_name, r_name
    return 0.0, 0.0, "UNK", "Unknown", "Unknown"

def load_ctf_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --bg-primary: #030712;
            --bg-secondary: #0f172a;
            --bg-tertiary: #1e293b;
            --bg-card: rgba(15, 23, 42, 0.8);
            --bg-glass: rgba(30, 41, 59, 0.6);
            --border-primary: #334155;
            --border-accent: #38bdf8;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-blue: #38bdf8;
            --accent-cyan: #22d3ee;
            --accent-green: #10b981;
            --accent-yellow: #fbbf24;
            --accent-orange: #f97316;
            --accent-red: #ef4444;
            --accent-pink: #ec4899;
            --accent-purple: #a78bfa;
            --gradient-primary: linear-gradient(135deg, #38bdf8 0%, #22d3ee 100%);
            --gradient-danger: linear-gradient(135deg, #ef4444 0%, #f97316 100%);
            --gradient-success: linear-gradient(135deg, #10b981 0%, #22d3ee 100%);
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -2px rgba(0, 0, 0, 0.3);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -4px rgba(0, 0, 0, 0.4);
            --shadow-glow: 0 0 20px rgba(56, 189, 248, 0.3);
        }
        
        .stApp {
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(ellipse at 20% 0%, rgba(56, 189, 248, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, var(--bg-secondary) 0%, var(--bg-primary) 100%);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        
        h1 { font-size: 1.75rem !important; }
        h2 { font-size: 1.35rem !important; }
        h3 { font-size: 1.1rem !important; }
        
        .stMarkdown { font-size: 0.95rem; }
        
        div[data-testid="stMetricValue"] { 
            color: var(--accent-blue); 
            font-weight: 700; 
            font-size: 1.75rem;
            font-family: 'JetBrains Mono', monospace;
        }
        div[data-testid="stMetricLabel"] { 
            color: var(--text-secondary); 
            font-size: 0.85rem;
            font-weight: 500;
        }
        div[data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', monospace;
        }
        
        .stButton > button {
            background: var(--gradient-primary) !important;
            border: none !important;
            color: var(--bg-primary) !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.5rem !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            box-shadow: var(--shadow-md) !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-glow) !important;
        }
        .stButton > button:disabled {
            background: var(--bg-tertiary) !important;
            color: var(--text-muted) !important;
            box-shadow: none !important;
            transform: none !important;
        }
        .stButton > button[kind="secondary"] {
            background: var(--bg-glass) !important;
            border: 1px solid var(--border-primary) !important;
            color: var(--text-primary) !important;
        }
        
        .stProgress > div > div {
            background: var(--bg-tertiary);
            border-radius: 10px;
            overflow: hidden;
        }
        .stProgress > div > div > div {
            background: var(--gradient-primary);
            border-radius: 10px;
        }
        
        .stRadio > label, .stCheckbox > label {
            color: var(--text-primary) !important;
        }
        .stRadio [role="radiogroup"] {
            gap: 0.5rem;
        }
        .stRadio [role="radio"] {
            background: var(--bg-glass);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            padding: 0.4rem 1rem;
            transition: all 0.2s ease;
        }
        .stRadio [role="radio"]:hover {
            border-color: var(--accent-blue);
        }
        .stRadio [role="radio"][aria-checked="true"] {
            background: rgba(56, 189, 248, 0.15);
            border-color: var(--accent-blue);
        }
        
        .stExpander {
            background: var(--bg-glass);
            border: 1px solid var(--border-primary);
            border-radius: 12px;
            overflow: hidden;
            backdrop-filter: blur(10px);
        }
        .stExpander > summary {
            background: transparent;
            color: var(--text-primary);
            font-weight: 500;
            padding: 1rem 1.25rem;
        }
        .stExpander > summary:hover {
            background: rgba(56, 189, 248, 0.05);
        }
        .stExpander > div {
            background: transparent;
            padding: 0 1.25rem 1rem;
        }
        
        .stDataFrame {
            background: var(--bg-glass);
            border: 1px solid var(--border-primary);
            border-radius: 12px;
            overflow: hidden;
        }
        .stDataFrame thead th {
            background: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }
        .stDataFrame tbody td {
            background: transparent !important;
            color: var(--text-secondary) !important;
        }
        .stDataFrame tbody tr:hover td {
            background: rgba(56, 189, 248, 0.05) !important;
        }
        
        .stSelectbox > div > div {
            background: var(--bg-glass);
            border: 1px solid var(--border-primary);
            border-radius: 8px;
        }
        .stSelectbox > div > div:hover {
            border-color: var(--accent-blue);
        }
        
        .stTextInput > div > div {
            background: var(--bg-glass);
            border: 1px solid var(--border-primary);
            border-radius: 8px;
        }
        .stTextInput > div > div:hover {
            border-color: var(--accent-blue);
        }
        .stTextInput input {
            background: transparent !important;
            color: var(--text-primary) !important;
        }
        
        .stSlider > div > div > div {
            background: var(--bg-tertiary);
        }
        .stSlider [role="slider"] {
            background: var(--accent-blue);
        }
        
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-secondary); }
        ::-webkit-scrollbar-thumb { 
            background: var(--border-primary); 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
        
        [data-testid="stSidebar"] {
            background: var(--bg-secondary) !important;
            border-right: 1px solid var(--border-primary);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-secondary);
        }
        
        [data-testid="stHeader"] {
            background: transparent !important;
        }
        
        .element-container { margin-bottom: 0.75rem; }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 5px rgba(56, 189, 248, 0.5); }
            50% { box-shadow: 0 0 20px rgba(56, 189, 248, 0.8); }
        }
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        
        .animate-fade-in { animation: fadeIn 0.5s ease-out forwards; }
        .animate-slide-left { animation: slideInLeft 0.5s ease-out forwards; }
        .animate-slide-right { animation: slideInRight 0.5s ease-out forwards; }
        .animate-pulse { animation: pulse 2s ease-in-out infinite; }
        .animate-glow { animation: glow 2s ease-in-out infinite; }
        
        @media (prefers-reduced-motion: reduce) {
            * { animation: none !important; transition: none !important; }
        }
        </style>
    """, unsafe_allow_html=True)

def render_stat_card(title, value, subtitle="", color="blue"):
    colors = {"blue": ("#38bdf8", "#0ea5e9"), "green": ("#10b981", "#059669"), "red": ("#ef4444", "#dc2626"), "yellow": ("#fbbf24", "#f59e0b"), "purple": ("#a78bfa", "#8b5cf6"), "cyan": ("#22d3ee", "#06b6d4"), "orange": ("#f97316", "#ea580c"), "pink": ("#ec4899", "#db2777")}
    c1, c2 = colors.get(color, colors["blue"])
    subtitle_html = f'<div style="color:#64748b;font-size:0.8rem;margin-top:0.35rem;">{subtitle}</div>' if subtitle else ""
    return f'<div style="background:linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);border:1px solid rgba(51, 65, 85, 0.5);border-radius:16px;padding:1.25rem;position:relative;overflow:hidden;transition:all 0.3s ease;" onmouseover="this.style.transform=\'translateY(-2px)\';this.style.boxShadow=\'0 8px 25px rgba(0,0,0,0.3)\';" onmouseout="this.style.transform=\'none\';this.style.boxShadow=\'none\';"><div style="position:absolute;top:0;left:0;width:4px;height:100%;background:linear-gradient(180deg,{c1},{c2});"></div><div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;"><span style="color:#94a3b8;font-size:0.85rem;font-weight:500;letter-spacing:0.02em;">{title}</span></div><div style="display:flex;align-items:baseline;"><span style="color:{c1};font-size:2rem;font-weight:700;font-family:\'JetBrains Mono\',monospace;letter-spacing:-0.02em;">{value}</span></div>{subtitle_html}</div>'

def render_status_badge(status, size="medium"):
    sizes = {"small": "0.7rem", "medium": "0.8rem", "large": "0.9rem"}
    padding = {"small": "4px 10px", "medium": "6px 14px", "large": "8px 18px"}
    configs = {"running": ("#fbbf24", "#f59e0b", "运行中", ""), "ready": ("#10b981", "#059669", "就绪", ""), "completed": ("#38bdf8", "#0ea5e9", "已完成", ""), "error": ("#ef4444", "#dc2626", "错误", ""), "warning": ("#f97316", "#ea580c", "警告", ""), "idle": ("#64748b", "#475569", "待机", "")}
    bg, fg, text, icon = configs.get(status, configs["idle"])
    font_size = sizes.get(size, sizes["medium"])
    pad = padding.get(size, padding["medium"])
    return f'<span style="display:inline-flex;align-items:center;gap:6px;background:{bg}15;border:1px solid {bg}40;color:{bg};font-size:{font_size};font-weight:600;padding:{pad};border-radius:20px;font-family:Inter,sans-serif;">{text}</span>'

def render_progress_ring(progress, size=80, stroke_width=6, color="#38bdf8"):
    radius = (size - stroke_width) / 2
    circumference = 2 * 3.14159 * radius
    dashoffset = circumference * (1 - progress)
    
    return f'<div style="position:relative;display:inline-flex;align-items:center;justify-content:center;"><svg width="{size}" height="{size}" style="transform:rotate(-90deg);"><circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none" stroke="#1e293b" stroke-width="{stroke_width}"/><circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-dasharray="{circumference}" stroke-dashoffset="{dashoffset}" stroke-linecap="round" style="transition:stroke-dashoffset 0.5s ease;"/></svg><span style="position:absolute;color:{color};font-size:1.1rem;font-weight:700;font-family:\'JetBrains Mono\',monospace;">{int(progress * 100)}%</span></div>'
    
def render_glass_card(title, content, icon="", collapsible=False, expanded=True):
    icon_html = f'<span style="font-size:1.25rem;margin-right:8px;">{icon}</span>' if icon else ""
    
    return f'<div style="background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(51, 65, 85, 0.5); border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);"><div style="padding: 1rem 1.25rem; border-bottom: 1px solid rgba(51, 65, 85, 0.3); display: flex; align-items: center;">{icon_html}<span style="color:#f1f5f9;font-weight:600;font-size:1rem;">{title}</span></div><div style="padding: 1rem 1.25rem;">{content}</div></div>'
    
def render_header_bar(title, subtitle, status="idle", version="v2.0"):
    status_badge = render_status_badge(status, "medium")
    
    return f'<div style="background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(15, 23, 42, 0.8) 100%); backdrop-filter: blur(12px); border-bottom: 1px solid rgba(51, 65, 85, 0.5); padding: 1rem 1.5rem; margin: -1rem -1rem 1rem -1rem; position: sticky; top: 0; z-index: 100;"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;"><div style="display:flex;align-items:center;gap:1rem;"><div style="background: linear-gradient(135deg, #38bdf8 0%, #22d3ee 100%); width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.25rem; color: #030712; box-shadow: 0 4px 15px rgba(56, 189, 248, 0.4);">B</div><div><h1 style="margin:0;font-size:1.5rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.02em;">{title}</h1><p style="margin:0;font-size:0.85rem;color:#64748b;">{subtitle}</p></div></div><div style="display:flex;align-items:center;gap:1rem;"><span style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); color: #38bdf8; font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 6px; font-family: \'JetBrains Mono\', monospace;">{version}</span>{status_badge}</div></div></div>'

def render_map_legend():
    return '<div style="background:rgba(15,23,42,0.85);backdrop-filter:blur(8px);border:1px solid rgba(51,65,85,0.5);border-radius:10px;padding:10px 16px;display:inline-flex;align-items:center;gap:16px;font-size:0.85rem;"><span style="display:flex;align-items:center;gap:6px;"><span style="width:10px;height:10px;border-radius:50%;background:#ef4444;box-shadow:0 0 8px #ef4444;"></span><span style="color:#94a3b8;">已确认威胁</span></span><span style="display:flex;align-items:center;gap:6px;"><span style="width:10px;height:10px;border-radius:50%;background:#fbbf24;box-shadow:0 0 8px #fbbf24;"></span><span style="color:#94a3b8;">疑似风险</span></span><span style="color:#64748b;">|</span><span style="color:#94a3b8;">热力: <span style="color:#ef4444;">红</span> &gt; <span style="color:#f97316;">橙</span> &gt; <span style="color:#fbbf24;">黄</span> &gt; <span style="color:#10b981;">绿</span></span></div>'
    
def render_time_display(real_time, sim_time):
    return f'<div style="background:linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(30,41,59,0.7) 100%);border:1px solid rgba(51,65,85,0.5);border-left:4px solid #38bdf8;border-radius:10px;padding:12px 16px;display:flex;align-items:center;gap:20px;font-family:JetBrains Mono,monospace;"><div><span style="color:#64748b;font-size:0.75rem;display:block;margin-bottom:2px;">实时</span><span style="color:#f1f5f9;font-size:1rem;font-weight:600;">{real_time}</span></div><span style="color:#334155;font-size:1.25rem;">|</span><div><span style="color:#64748b;font-size:0.75rem;display:block;margin-bottom:2px;">推演时间</span><span style="color:#38bdf8;font-size:1rem;font-weight:600;">{sim_time}</span></div></div>'

def get_gnn_viz_html(phase="idle", epoch=0, loss=0.0, layer=-1):
    if phase == "train":
        status_text = f"训练中 &nbsp;|&nbsp; 轮次: {epoch} &nbsp;|&nbsp; 损失: {loss:.4f}"
        status_color = "#38bdf8"
        particle_color = "#7dd3fc"
        anim_dur = "2.5s"
        show_flow = True
    elif phase == "inference":
        status_text = "推理与威胁扫描进行中…"
        status_color = "#f43f5e"
        particle_color = "#fda4af"
        anim_dur = "1.8s"
        show_flow = True
    else:
        status_text = "模型待机 &nbsp;/&nbsp; 等待数据"
        status_color = "#475569"
        particle_color = "transparent"
        anim_dur = "0s"
        show_flow = False

    W, H = 1000, 400
    label_y = H - 25
    label_sub_y = H - 8
    layer_x = [100, 300, 500, 700, 900]
    layer_nodes = [8, 12, 10, 6, 2]
    layer_names = ["输入特征", "GAT 层 1", "GAT 层 2", "全局池化", "分类器"]
    layer_sub = ["(64 维)", "(多头注意力)", "(高层表示)", "(图摘要)", "(威胁概率)"]
    layer_stroke = ["#38bdf8", "#10b981", "#10b981", "#a78bfa", "#f43f5e"]

    max_nodes = max(layer_nodes)
    spacing = min(30, (H - 100) / max(max_nodes, 1))

    node_coords = []
    svg_nodes = []
    for l_idx in range(5):
        coords = []
        n = layer_nodes[l_idx]
        cx = layer_x[l_idx]
        y_start = (H - 60) / 2 - (n - 1) * spacing / 2
        for i in range(n):
            cy = y_start + i * spacing
            coords.append((cx, cy))
            stroke = layer_stroke[l_idx] if show_flow else "#334155"
            svg_nodes.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="#0f172a" stroke="{stroke}" stroke-width="1.8"/>')
        node_coords.append(coords)

    svg_lines = []
    svg_particles = []
    if show_flow:
        path_idx = 0
        random.seed(42)
        for l_idx in range(4):
            for sx, sy in node_coords[l_idx]:
                n_targets = min(3, len(node_coords[l_idx + 1]))
                targets = random.sample(node_coords[l_idx + 1], n_targets)
                for dx, dy in targets:
                    pid = f"p{path_idx}"
                    mx = (sx + dx) / 2
                    svg_lines.append(f'<path id="{pid}" d="M{sx},{sy} C{mx},{sy} {mx},{dy} {dx},{dy}" fill="none" stroke="#334155" stroke-width="1" opacity="0.35"/>')
                    delay = f"{random.uniform(0, float(anim_dur[:-1])):.2f}s"
                    svg_particles.append(f'<circle r="3.5" fill="{particle_color}" opacity="0.85"><animateMotion dur="{anim_dur}" begin="{delay}" repeatCount="indefinite"><mpath href="#{pid}"/></animateMotion></circle>')
                    path_idx += 1

    svg_labels = []
    for l_idx in range(5):
        x = layer_x[l_idx]
        svg_labels.append(f'<text x="{x}" y="{label_y}" fill="#94a3b8" font-size="13" font-family="Arial,sans-serif" font-weight="600" text-anchor="middle">{layer_names[l_idx]}</text>')
        svg_labels.append(f'<text x="{x}" y="{label_sub_y}" fill="#64748b" font-size="11" font-family="Arial,sans-serif" text-anchor="middle">{layer_sub[l_idx]}</text>')

    return f"""<!DOCTYPE html><html><head><style>
body{{margin:0;background:#020617;overflow:hidden;}}
</style></head><body>
<div style="text-align:center;padding:12px 0 6px;font:bold 14px Arial,sans-serif;color:{status_color};letter-spacing:1.5px;">{status_text}</div>
<svg viewBox="0 0 {W} {H}" width="100%" height="440">
{''.join(svg_lines)}
{''.join(svg_particles)}
{''.join(svg_nodes)}
{''.join(svg_labels)}
</svg></body></html>"""
