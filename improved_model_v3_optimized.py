"""
优化版V3模型 - 使用混合损失函数
结合对比学习(Triplet Loss)和监督学习(BCE Loss)的优势
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FeatureAttention(nn.Module):
    """特征注意力层"""
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)

class ImprovedModelV3Optimized(nn.Module):
    """
    优化版V3模型
    - 使用Transformer编码器
    - 混合损失：Triplet Loss + BCE Loss
    - 更好的嵌入学习
    """
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 特征注意力
        self.feature_attention = FeatureAttention(hidden_dim, num_heads)
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 嵌入头（用于对比学习）
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64)
        )
        
        # 分类头（用于监督学习）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 场景嵌入
        self.scenario_embedding = nn.Embedding(20, 32)
        self.scenario_proj = nn.Linear(32, hidden_dim)
        
        self.hidden_dim = hidden_dim
        
    def forward(self, x, scenario_ids=None, return_embedding=False):
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_proj(x)
        
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 特征注意力
        x = self.feature_attention(x)
        
        # 全局池化 [batch, hidden_dim]
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        
        # 添加场景信息
        if scenario_ids is not None:
            scenario_emb = self.scenario_embedding(scenario_ids)
            scenario_feat = self.scenario_proj(scenario_emb)
            x = x + scenario_feat
        
        # 归一化特征
        x = F.normalize(x, p=2, dim=-1)
        
        if return_embedding:
            # 返回嵌入向量
            return self.embedding_head(x)
        else:
            # 返回分类logits
            return self.classifier(x)
    
    def get_embedding(self, x, scenario_ids=None):
        """获取归一化的嵌入向量"""
        with torch.no_grad():
            emb = self.forward(x, scenario_ids, return_embedding=True)
            return F.normalize(emb, p=2, dim=-1)
    
    def predict(self, x, scenario_ids=None):
        """预测胜负概率"""
        with torch.no_grad():
            logits = self.forward(x, scenario_ids, return_embedding=False)
            return torch.sigmoid(logits)


class HybridLoss(nn.Module):
    """
    混合损失函数
    结合Triplet Loss和BCE Loss
    """
    def __init__(self, triplet_weight=0.3, bce_weight=0.7, margin=1.0):
        super().__init__()
        self.triplet_weight = triplet_weight
        self.bce_weight = bce_weight
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, embeddings, logits, labels, winner_emb=None, loser_emb=None):
        """
        Args:
            embeddings: 当前batch的嵌入 [batch, emb_dim]
            logits: 分类logits [batch, 1]
            labels: 二分类标签 [batch]
            winner_emb: 获胜玩家嵌入（可选）
            loser_emb: 失败玩家嵌入（可选）
        """
        # BCE损失
        bce = self.bce_loss(logits.squeeze(), labels.float())
        
        # Triplet损失（如果提供了正负样本嵌入）
        if winner_emb is not None and loser_emb is not None:
            triplet = self.triplet_loss(embeddings, winner_emb, loser_emb)
        else:
            # 使用batch内的样本构建triplet
            triplet = torch.tensor(0.0, device=embeddings.device)
            if labels.sum() > 0 and (1 - labels).sum() > 0:
                # 获取正负样本索引
                pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
                neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
                
                if len(pos_idx) > 0 and len(neg_idx) > 0:
                    # 构建triplet
                    anchor = embeddings[pos_idx[0]:pos_idx[0]+1]
                    positive = embeddings[pos_idx] if len(pos_idx) > 1 else embeddings[pos_idx[0]:pos_idx[0]+1]
                    negative = embeddings[neg_idx[0]:neg_idx[0]+1]
                    
                    if positive.shape[0] > 0:
                        # 扩展anchor以匹配positive数量
                        anchor = anchor.expand(positive.shape[0], -1)
                        triplet = self.triplet_loss(anchor, positive, negative.expand(positive.shape[0], -1))
        
        return self.bce_weight * bce + self.triplet_weight * triplet


def create_model_v3_optimized(input_dim, config=None):
    """创建优化版V3模型"""
    if config is None:
        config = {
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.3
        }
    
    model = ImprovedModelV3Optimized(
        input_dim=input_dim,
        hidden_dim=config.get('hidden_dim', 128),
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3)
    )
    
    return model