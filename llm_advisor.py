import os
import json
import requests
import pandas as pd


def generate_expert_advice(df: pd.DataFrame, metrics: dict = None) -> str:
    """
    Call DeepSeek API to generate dynamic expert recommendations
    based on actual analysis data. Returns HTML string.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        except Exception:
            pass

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set in environment or st.secrets")

    # Build data summary from actual results
    total_ips = len(df)
    state_dist = df["final_state"].value_counts().head(5).to_dict()
    max_chain = int(df["chain_len"].max())
    avg_chain = round(df["chain_len"].mean(), 2)
    top_ips = df.sort_values("chain_len", ascending=False).head(5)["ip"].tolist()

    metrics_str = "N/A"
    if metrics:
        metrics_str = f"AUC={metrics.get('auc','?')}, F1={metrics.get('f1','?')}, Precision={metrics.get('prec','?')}, Recall={metrics.get('rec','?')}"

    prompt = f"""You are a senior cybersecurity analyst specializing in botnet detection and network threat intelligence.
You are analyzing the output of a GNN + HMM based botnet detection system on real network traffic data.

Analysis Results Summary:
- Total tracked bot IPs: {total_ips}
- Attack chain state distribution: {json.dumps(state_dist)}
- Maximum attack chain length: {max_chain} hops
- Average attack chain length: {avg_chain} hops
- Top 5 high-risk IPs (by chain length): {', '.join(top_ips)}
- Model performance metrics: {metrics_str}

Based on these results, provide a professional analysis in the following structure.
Write in Chinese. Use HTML formatting (h4, p, ul, ol, li, b, code tags). Do NOT use markdown.

1. Situation Assessment (综合态势研判): Interpret the state distribution and chain lengths. What phase of botnet lifecycle is dominant?
2. Key Risk Indicators (关键风险指标): Identify specific high-risk patterns from the data.
3. Actionable Recommendations (处置建议): Provide 4-5 specific, prioritized actions referencing actual IPs and states from the data.
4. Confidence Assessment (置信度评估): Comment on the model's detection reliability based on the metrics.

Wrap everything in a single <div> with style='background-color:#0f172a; border-left: 4px solid #38bdf8; padding:20px; border-radius:4px;'.
Use color:#cbd5e1 for body text, color:#38bdf8 for h4 assessment title, color:#f43f5e for risk title, color:#10b981 for recommendation title."""

    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000
            },
            timeout=30
        )
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        
        # Strip markdown code fences if the model wraps output in them
        content = content.strip()
        if content.startswith("```html"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        return content.strip()
    except Exception as e:
        raise RuntimeError(f"DeepSeek API call failed: {e}")
