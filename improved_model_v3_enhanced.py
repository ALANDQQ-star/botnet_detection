"""
V3 增强版 - 大幅增加分数分离度 + 加快训练速度
核心改进：
1. Margin-based Contrastive Loss - 强制拉大正负样本距离
2. 分布对齐损失 - 让正样本分数趋向高值，负样本趋向低值  
3. 保持网络维度和深度不变
4. 优化训练流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
import torch.cuda.amp as amp
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class MarginContrastiveLoss(nn.Module):
    """
    边界对比损失 - 强制正负样本之间有明显的分数差距
    目标：正样本分数 > pos_margin, 负样本分数 < neg_margin
    """
    def __init__(self, pos_margin=0.7, neg_margin=0.3, margin_gap=0.4):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.margin_gap = margin_gap
    
    def forward(self, probs, labels):
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        loss = 0.0
        count = 0
        
        # 正样本损失：希望分数接近1
        if pos_mask.sum() > 0:
            pos_probs = probs[pos_mask]
            # 惩罚分数低于pos_margin的正样本
            pos_loss = F.relu(self.pos_margin - pos_probs).mean()
            # 额外：鼓励分数接近1
            pos_target_loss = F.mse_loss(pos_probs, torch.ones_like(pos_probs))
            loss += pos_loss + 0.5 * pos_target_loss
            count += 1
        
        # 负样本损失：希望分数接近0
        if neg_mask.sum() > 0:
            neg_probs = probs[neg_mask]
            # 惩罚分数高于neg_margin的负样本
            neg_loss = F.relu(neg_probs - self.neg_margin).mean()
            # 额外：鼓励分数接近0
            neg_target_loss = F.mse_loss(neg_probs, torch.zeros_like(neg_probs))
            loss += neg_loss + 0.5 * neg_target_loss
            count += 1
        
        # 间隔损失：确保正负样本均值差距足够大
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_mean = probs[pos_mask].mean()
            neg_mean = probs[neg_mask].mean()
            margin_loss = F.relu(self.margin_gap - (pos_mean - neg_mean))
            loss += 2.0 * margin_loss  # 加大权重
            count += 1
        
        return loss / max(count, 1)


class FocalLoss(nn.Module):
    """Focal Loss: 专注于难分类样本"""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class StatisticalEncoder(nn.Module):
    """统计特征编码器 (GAT-based)"""
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(0.3)
        self.prelu = nn.PReLU()
        
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        return x


class SemanticEncoder(nn.Module):
    """语义特征编码器 (Transformer-based)"""
    def __init__(self, in_dim, hidden_dim, out_dim, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*2,
            dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.output_proj(x)
        return x


class StructuralEncoder(nn.Module):
    """结构特征编码器 (GCN-based)"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        return x


class CrossModalAttention(nn.Module):
    """跨模态注意力融合"""
    def __init__(self, hidden_dim, num_modalities=3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, h_list):
        stacked = torch.stack(h_list, dim=1)
        attn_scores = self.attention(stacked)
        attn_weights = F.softmax(attn_scores, dim=1)
        weighted = (stacked * attn_weights).sum(dim=1)
        output = self.fusion(weighted)
        return output, attn_weights.squeeze(-1)


class ImprovedBotnetDetectorV3Enhanced(nn.Module):
    """
    V3 增强版
    - 保持网络维度和深度
    - 增加分离度的损失函数
    - 直接输出logits保持分数范围
    """
    def __init__(self, stat_dim=64, semantic_dim=32, struct_dim=16, hidden_dim=128):
        super().__init__()
        self.stat_dim = stat_dim
        self.semantic_dim = semantic_dim
        self.struct_dim = struct_dim
        self.hidden_dim = hidden_dim
        
        # 三模态编码器
        self.stat_encoder = StatisticalEncoder(stat_dim, hidden_dim // 2, hidden_dim)
        self.semantic_encoder = SemanticEncoder(semantic_dim, hidden_dim // 2, hidden_dim)
        self.struct_encoder = StructuralEncoder(struct_dim, hidden_dim // 2, hidden_dim)
        
        # 跨模态融合
        self.cross_modal_attention = CrossModalAttention(hidden_dim, num_modalities=3)
        
        # 分类器 - 直接输出logits
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.PReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 辅助生成器
        self.semantic_fallback = nn.Sequential(
            nn.Linear(stat_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim)
        )
        
        self.struct_fallback = nn.Sequential(
            nn.Linear(stat_dim, struct_dim),
            nn.ReLU(),
            nn.Linear(struct_dim, struct_dim)
        )
        
        # 损失函数
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
        self.contrastive_loss = MarginContrastiveLoss(pos_margin=0.7, neg_margin=0.3, margin_gap=0.4)
        
    def forward(self, data, return_all=False):
        """前向传播 - 直接输出logits"""
        stat_x = data['ip'].x
        edge_index = data['ip', 'flow', 'ip'].edge_index
        
        # 语义特征
        if hasattr(data['ip'], 'semantic_x') and data['ip'].semantic_x is not None:
            semantic_x = data['ip'].semantic_x
        else:
            semantic_x = self.semantic_fallback(stat_x)
            
        # 结构特征
        if hasattr(data['ip'], 'struct_x') and data['ip'].struct_x is not None:
            struct_x = data['ip'].struct_x
        else:
            struct_x = self.struct_fallback(stat_x)
        
        # 三模态编码
        h_stat = self.stat_encoder(stat_x, edge_index)
        h_sem = self.semantic_encoder(semantic_x)
        h_struct = self.struct_encoder(struct_x, edge_index)
        
        # 跨模态融合
        h_fused, attn_weights = self.cross_modal_attention([h_stat, h_sem, h_struct])
        
        # 分类 - 输出logits
        logits = self.classifier(h_fused)
        
        if return_all:
            return logits, h_fused, [h_stat, h_sem, h_struct], attn_weights
        return logits
    
    def compute_loss(self, data, labels, return_components=False):
        """计算损失 - Focal Loss + Margin Contrastive Loss"""
        logits, h_fused, h_views, attn_weights = self.forward(data, return_all=True)
        
        # 获取实际的 batch_size
        if hasattr(data['ip'], 'batch_size'):
            actual_batch_size = data['ip'].batch_size
        else:
            actual_batch_size = logits.size(0)
        
        logits = logits[:actual_batch_size].squeeze()
        labels = labels[:actual_batch_size]
        
        # Focal Loss (基于logits)
        focal_loss = self.focal_loss(logits, labels.float())
        
        # Margin Contrastive Loss (基于概率)
        probs = torch.sigmoid(logits)
        contrastive_loss = self.contrastive_loss(probs, labels)
        
        # 总损失：平衡两种损失
        total_loss = focal_loss + 2.0 * contrastive_loss
        
        if return_components:
            return total_loss, {
                'focal_loss': focal_loss.item(),
                'contrastive_loss': contrastive_loss.item()
            }
        return total_loss
    
    def predict_proba(self, data):
        """预测概率"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            return torch.sigmoid(logits).squeeze()
    
    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'stat_dim': self.stat_dim,
                'semantic_dim': self.semantic_dim,
                'struct_dim': self.struct_dim,
                'hidden_dim': self.hidden_dim
            }
        }, path)
        print(f"[Model] V3 Enhanced 模型已保存至：{path}")
        
    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print(f"[Model] V3 Enhanced 模型已加载")
        return model


class ImprovedTrainerV3Enhanced:
    """V3 Enhanced 模型的训练器"""
    def __init__(self, model, lr=0.002, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=15)
        self.scaler = amp.GradScaler()
        self.best_loss = float('inf')
        
    def train_step_batch(self, batch):
        """单步训练"""
        self.model.train()
        batch = batch.to(self.device)
        
        self.optimizer.zero_grad()
        
        with amp.autocast():
            loss, components = self.model.compute_loss(
                batch, batch['ip'].y, return_components=True
            )
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item(), components
    
    def train_epoch(self, train_loader, epoch):
        """训练一个 epoch"""
        total_loss = 0
        total_components = {'focal_loss': 0, 'contrastive_loss': 0}
        num_batches = 0
        
        for batch in train_loader:
            loss, components = self.train_step_batch(batch)
            total_loss += loss
            for k, v in components.items():
                if k in total_components:
                    total_components[k] += v
            num_batches += 1
            
        self.scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        
        if epoch % 2 == 0 or epoch == 0:
            print(f"  损失：Focal={total_components['focal_loss']/num_batches:.4f}, "
                  f"Contrastive={total_components['contrastive_loss']/num_batches:.4f}")
        
        return avg_loss
    
    def evaluate(self, data, labels):
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            probs = self.model.predict_proba(data)
            
            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            auc = roc_auc_score(labels_np, probs_np)
            
            from sklearn.metrics import precision_recall_fscore_support, roc_curve
            fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            best_thresh = thresholds[best_idx]
            
            preds = (probs_np >= best_thresh).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_np, preds, average='binary', zero_division=0
            )
            
        return {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': best_thresh,
            'probs': probs_np,
            'preds': preds
        }