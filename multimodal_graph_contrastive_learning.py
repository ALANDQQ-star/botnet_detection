"""
多模态异构图对比学习模型
融合 Bot-DM 双模态思想和 TRUSTED 概念漂移感知机制

改进点:
1. Payload 语义编码 (Transformer-based)
2. 流量图像编码 (CNN-based)
3. Cross-Attention 多模态融合
4. 概念漂移感知的对比学习
5. 互信息增强的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, GCNConv, GINConv
from torch_geometric.utils import to_dense_adj
import math
import warnings
warnings.filterwarnings('ignore')


# ==================== 1. Payload 语义编码器 (Botflow-token inspired) ====================

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PayloadSemanticEncoder(nn.Module):
    """
    Payload 语义编码器
    基于 Bot-DM 的 Botflow-token 思想，使用 Transformer 编码 payload 隐式语义
    """
    def __init__(self, vocab_size=256, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(embed_dim, 32)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, payload_tokens):
        """
        Args:
            payload_tokens: (batch, seq_len) 或 (num_nodes, seq_len) 的 token 序列
        Returns:
            semantic_features: (batch/output_dim) 语义特征向量
        """
        # Token 嵌入
        x = self.token_embedding(payload_tokens)  # (N, seq_len, embed_dim)
        x = self.pos_encoder(x)
        
        # Transformer 编码
        x = self.transformer_encoder(x)  # (N, seq_len, embed_dim)
        x = self.layer_norm(x)
        
        # 全局池化 (使用 [CLS] 位置或平均池化)
        cls_token = x[:, 0, :]  # 取第一个位置作为 [CLS]
        semantic_features = self.output_proj(cls_token)  # (N, 32)
        
        return semantic_features


# ==================== 2. 流量图像编码器 (Botflow-image inspired) ====================

class TrafficImageEncoder(nn.Module):
    """
    流量图像编码器
    基于 Bot-DM 的 Botflow-image 思想，使用 CNN 提取流量时空特征
    """
    def __init__(self, input_channels=1, embed_dim=32):
        super().__init__()
        
        # CNN 特征提取 (类似 Bot-DM 的结构)
        self.conv_layers = nn.Sequential(
            # Conv1: 输入 -> 32 通道
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # Conv2: 32 -> 64 通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
            
            # Conv3: 64 -> 128 通道
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 全局池化
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, embed_dim)
        )
        
    def forward(self, traffic_images):
        """
        Args:
            traffic_images: (N, 1, 32, 32) 流量图像
        Returns:
            image_features: (N, embed_dim) 图像特征
        """
        x = self.conv_layers(traffic_images)  # (N, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 128)
        image_features = self.fc(x)  # (N, embed_dim)
        return image_features


# ==================== 3. Cross-Attention 多模态融合层 ====================

class CrossModalAttention(nn.Module):
    """
    Cross-Attention 多模态融合层
    实现统计特征、语义特征和图像特征的交互融合
    """
    def __init__(self, stat_dim=64, semantic_dim=32, image_dim=32, hidden_dim=128, num_heads=4):
        super().__init__()
        
        # 投影到统一维度
        self.stat_proj = nn.Linear(stat_dim, hidden_dim)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        # Cross-Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, 
                                                dropout=0.1, batch_first=True)
        
        # 融合输出
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, stat_features, semantic_features=None, image_features=None):
        """
        Args:
            stat_features: (N, stat_dim) 统计特征
            semantic_features: (N, semantic_dim) 语义特征 (可选)
            image_features: (N, image_dim) 图像特征 (可选)
        Returns:
            fused_features: (N, hidden_dim//2) 融合特征
        """
        # 投影
        stat_emb = self.stat_proj(stat_features)  # (N, hidden_dim)
        
        # 如果没有多模态特征，直接返回统计特征
        if semantic_features is None and image_features is None:
            return self.output_proj(stat_emb)
        
        # 构建序列 (N, 3, hidden_dim)
        batch_size = stat_features.size(0)
        modalities = [stat_emb]
        
        if semantic_features is not None:
            semantic_emb = self.semantic_proj(semantic_features)
            modalities.append(semantic_emb)
        else:
            modalities.append(torch.zeros_like(stat_emb))
            
        if image_features is not None:
            image_emb = self.image_proj(image_features)
            modalities.append(image_emb)
        else:
            modalities.append(torch.zeros_like(stat_emb))
        
        # Stack: (N, 3, hidden_dim)
        modal_seq = torch.stack(modalities, dim=1)
        
        # Self-Attention 融合
        fused, _ = self.attention(modal_seq, modal_seq, modal_seq)
        
        # 全局池化 + 门控融合
        pooled = fused.mean(dim=1)  # (N, hidden_dim)
        gate = self.fusion_gate(pooled)
        
        # 残差连接
        output = gate * pooled + (1 - gate) * stat_emb
        output = self.output_proj(output)
        
        return output


# ==================== 4. 概念漂移检测器 (D3N inspired) ====================

class ConceptDriftDetector(nn.Module):
    """
    概念漂移检测器
    基于 TRUSTED 论文的 D3N 思想，检测数据分布变化
    """
    def __init__(self, input_dim, hidden_dim=64, threshold=0.7):
        super().__init__()
        
        self.threshold = threshold
        
        # 判别分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # 旧数据 (0) vs 新数据 (1)
        )
        
    def forward(self, features):
        """
        Args:
            features: (N, input_dim) 特征
        Returns:
            logits: (N, 2) 新旧数据分类 logits
        """
        return self.classifier(features)
    
    def detect_drift(self, old_features, new_features):
        """
        检测是否发生概念漂移
        Args:
            old_features: (N_old, input_dim) 旧数据特征
            new_features: (N_new, input_dim) 新数据特征
        Returns:
            drift_detected: bool 是否发生漂移
            auc_score: float AUC 分数
        """
        from sklearn.metrics import roc_auc_score
        
        # 合并数据
        combined = torch.cat([old_features, new_features], dim=0)
        labels = torch.cat([
            torch.zeros(old_features.size(0)),
            torch.ones(new_features.size(0))
        ])
        
        # 预测
        self.eval()
        with torch.no_grad():
            logits = self.forward(combined)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        
        # 计算 AUC
        try:
            auc = roc_auc_score(labels.numpy(), probs)
        except:
            auc = 0.5
        
        # AUC > threshold 表示发生漂移
        drift_detected = auc > self.threshold
        
        return drift_detected, auc


# ==================== 5. 漂移感知的图编码器 ====================

class DriftAwareGraphEncoder(nn.Module):
    """
    概念漂移感知的图编码器
    融合 GAT 和 GCN，加入漂移权重调整
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4):
        super().__init__()
        
        self.num_heads = num_heads
        
        # GAT 层 (捕获注意力权重)
        self.gat1 = GATConv(in_dim, hidden_dim, heads=num_heads, concat=True, dropout=0.3)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=False, dropout=0.3)
        
        # GCN 层 (平滑传播)
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        
        # 归一化和激活
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
