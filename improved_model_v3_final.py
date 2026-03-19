"""
V3 最终优化版 - 修复分数区分度问题
核心改进：
1. 移除破坏性的校准层和温度缩放
2. 直接输出logits，保持分数区分度
3. 使用简化的损失函数
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


class FocalLoss(nn.Module):
    """Focal Loss: 专注于难分类样本"""
    def __init__(self, alpha=0.95, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class DynamicSeparationLoss(nn.Module):
    """动态正负样本分离损失，避免硬性边界导致的训练崩塌"""
    def __init__(self, margin=5.0, hard_mining=True):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        
    def forward(self, logits, labels):
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]
        
        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return torch.tensor(0.0, device=logits.device)
            
        if self.hard_mining:
            # 难例挖掘：只让分数较低的正样本和分数较高的负样本参与推离
            pos_k = max(1, len(pos_logits) // 2)
            neg_k = max(1, len(neg_logits) // 10) # 负样本更多，只取 top 10%
            hard_pos = torch.topk(pos_logits, k=pos_k, largest=False)[0]
            hard_neg = torch.topk(neg_logits, k=neg_k, largest=True)[0]
            
            pos_mean = hard_pos.mean()
            neg_mean = hard_neg.mean()
        else:
            pos_mean = pos_logits.mean()
            neg_mean = neg_logits.mean()
            
        # 动态 push: 如果分离度不足 margin，则惩罚；否则为0
        diff = pos_mean - neg_mean
        loss = F.relu(self.margin - diff)
        
        return loss


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


class ImprovedBotnetDetectorV3Final(nn.Module):
    """
    V3 最终优化版
    - 移除校准层和温度缩放
    - 直接输出logits
    - 保持分数区分度
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
        
        # 分类器 - 移除 LayerNorm，保留原始的 LogitSeparationLoss 来扩展概率差异
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
        self.focal_loss = FocalLoss(alpha=0.95, gamma=2.0)
        self.separation_loss = DynamicSeparationLoss(margin=5.0, hard_mining=True)
        
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
        """计算损失"""
        logits, h_fused, h_views, attn_weights = self.forward(data, return_all=True)
        
        if hasattr(data['ip'], 'batch_size'):
            actual_batch_size = data['ip'].batch_size
        else:
            actual_batch_size = logits.size(0)
            
        logits = logits[:actual_batch_size].squeeze(-1) if logits.dim() > 1 else logits[:actual_batch_size]
        labels = labels[:actual_batch_size]
        
        # 主分类损失
        focal_loss = self.focal_loss(logits, labels.float())
        
        # 确保正负样本的 logits 有明显的间隔
        sep_loss = self.separation_loss(logits, labels)
        
        # 强制分数的有效分离，动态惩罚避免过拟合
        total_loss = focal_loss + 0.3 * sep_loss
        
        if return_components:
            return total_loss, {
                'focal_loss': focal_loss.item(),
                'sep_loss': sep_loss.item(),
                'total_loss': total_loss.item()
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
        print(f"[Model] V3 Final 模型已保存至：{path}")
        
    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print(f"[Model] V3 Final 模型已加载")
        return model


class ImprovedTrainerV3Final:
    """V3 Final 模型的训练器"""
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
        total_components = {'focal_loss': 0, 'sep_loss': 0, 'total_loss': 0}
        num_batches = 0
        
        for batch in train_loader:
            loss, components = self.train_step_batch(batch)
            total_loss += loss
            for k, v in components.items():
                if k in total_components:
                    total_components[k] += v
                else:
                    total_components[k] = v
            num_batches += 1
            
        self.scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        
        if epoch % 2 == 0 or epoch == 0:
            print(f"  损失：Total={total_components.get('total_loss', total_loss)/max(num_batches, 1):.4f} "
                  f"| Focal={total_components.get('focal_loss', 0)/max(num_batches, 1):.4f} "
                  f"| Sep={total_components.get('sep_loss', 0)/max(num_batches, 1):.4f}")
        
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