"""
后台进程管理器 - 支持跨页面推演持续运行
"""
import os
import json
import time
import subprocess
import sys
from datetime import datetime

PROCESS_STATE_FILE = "process_state.json"

def get_process_state():
    """获取进程状态"""
    if os.path.exists(PROCESS_STATE_FILE):
        try:
            with open(PROCESS_STATE_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return None

def save_process_state(pid, start_time, config, status="running"):
    """保存进程状态"""
    state = {
        "pid": pid,
        "start_time": start_time,
        "status": status,
        "config": config,
        "last_update": datetime.now().isoformat()
    }
    try:
        with open(PROCESS_STATE_FILE, "w") as f:
            json.dump(state, f)
    except:
        pass

def update_process_status(status):
    """更新进程状态"""
    state = get_process_state()
    if state:
        state["status"] = status
        state["last_update"] = datetime.now().isoformat()
        try:
            with open(PROCESS_STATE_FILE, "w") as f:
                json.dump(state, f)
        except:
            pass

def clear_process_state():
    """清除进程状态"""
    if os.path.exists(PROCESS_STATE_FILE):
        try:
            os.remove(PROCESS_STATE_FILE)
        except:
            pass

def is_process_alive(pid):
    """检查进程是否存活"""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def get_running_process():
    """获取正在运行的进程信息"""
    state = get_process_state()
    if state and state.get("status") == "running":
        pid = state.get("pid")
        if pid and is_process_alive(pid):
            return pid, state
        else:
            clear_process_state()
    return None, None

def start_background_process(config, mode_map):
    """启动后台进程"""
    mode_arg = mode_map.get(config.get("mode_ui"), "all")
    cmd = [
        sys.executable, "-u", "main.py",
        "--data_dir", config.get("data_dir"),
        "--mode", mode_arg,
        "--train_scenarios", config.get("train_scens"),
        "--test_scenarios", config.get("test_scens"),
        "--epochs", str(config.get("epochs")),
        "--lr", str(config.get("lr")),
        "--threshold", str(config.get("threshold")),
        "--method", config.get("method", "existing")
    ]
    if config.get("use_dynamic"):
        cmd.append("--use_dynamic")
    if config.get("force_retrain"):
        cmd.append("--force_retrain")
    if config.get("force_hmm"):
        cmd.append("--force_hmm")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    
    save_process_state(process.pid, datetime.now().isoformat(), config, "running")
    
    return process

def stop_background_process():
    """停止后台进程"""
    pid, state = get_running_process()
    if pid:
        try:
            os.kill(pid, 9)
        except:
            pass
    clear_process_state()
