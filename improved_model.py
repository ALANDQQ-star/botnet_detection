"""
改进的异构图对比学习僵尸网络检测模型
整合三模态特征融合 + 概念漂移感知对比学习 + 动态数据增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
import torch.cuda.amp as amp
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    """Focal Loss: 专注于难分类样本"""
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


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
        # x: [batch_size, feat_dim]
        # 添加序列维度用于Transformer
        x = self.input_proj(x)  # [batch, hidden]
        x = x.unsqueeze(1)  # [batch, 1, hidden]
        x = self.transformer(x)  # [batch, 1, hidden]
        x = x.squeeze(1)  # [batch, hidden]
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
        self.num_modalities = num_modalities
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, h_list):
        # h_list: list of [batch, hidden_dim] tensors
        stacked = torch.stack(h_list, dim=1)  # [batch, num_mod, hidden]
        
        # 计算注意力权重
        attn_scores = self.attention(stacked)  # [batch, num_mod, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch, num_mod, 1]
        
        # 加权融合
        weighted = (stacked * attn_weights).sum(dim=1)  # [batch, hidden]
        output = self.fusion(weighted)
        
        return output, attn_weights.squeeze(-1)


class ConceptDriftDetector(nn.Module):
    """概念漂移检测器 (简化版D3N)"""
    def __init__(self, hidden_dim, history_size=500):
        super().__init__()
        self.history_size = history_size
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.register_buffer('feature_history', None)
        
    def update_history(self, features):
        """更新特征历史"""
        if self.feature_history is None:
            self.feature_history = features[:min(len(features), self.history_size)].detach()
        else:
            new_history = torch.cat([
                features[:min(len(features), self.history_size)].detach(),
                self.feature_history
            ], dim=0)
            self.feature_history = new_history[:self.history_size]
            
    def compute_mmd(self, x, y, gamma=1.0):
        """计算最大均值差异(MMD)"""
        if x is None or y is None:
            return 0.0
        if len(x) < 2 or len(y) < 2:
            return 0.0
            
        # 使用RBF核
        def rbf_kernel(a, b):
            dist = torch.cdist(a, b, p=2)
            return torch.exp(-gamma * dist ** 2)
        
        Kxx = rbf_kernel(x, x)
        Kyy = rbf_kernel(y, y)
        Kxy = rbf_kernel(x, y)
        
        mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
        return max(0, mmd.item())
        
    def detect_drift(self, current_features):
        """检测概念漂移"""
        if self.feature_history is None or len(self.feature_history) < 10:
            self.update_history(current_features)
            return 0.0
            
        # 计算MMD距离
        old_features = self.feature_history
        new_features = current_features[:min(len(current_features), len(old_features))]
        
        mmd_distance = self.compute_mmd(old_features, new_features)
        
        # 更新历史
        self.update_history(current_features)
        
        # 归一化漂移分数到[0,1]
        drift_score = min(1.0, mmd_distance * 10)
        return drift_score


class ContrastiveLearningModule(nn.Module):
    """概念漂移感知的对比学习模块"""
    def __init__(self, hidden_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        
    def info_nce_loss(self, z1, z2, labels=None):
        """InfoNCE对比损失"""
        batch_size = z1.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=z1.device)
            
        # 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 正样本相似度
        pos_sim = (z1 * z2).sum(dim=1) / self.temperature
        
        # 负样本相似度矩阵
        neg_sim = torch.mm(z1, z2.t()) / self.temperature
        
        # 对角线是正样本，需要排除
        mask = torch.eye(batch_size, device=z1.device).bool()
        
        # 构建logits: [batch_size, batch_size]
        # 正样本放第一列
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.masked_fill(mask, -1e9)], dim=1)
        
        # 目标：正样本在第0位
        targets = torch.zeros(batch_size, dtype=torch.long, device=z1.device)
        
        loss = F.cross_entropy(logits, targets)
        return loss
    
    def forward(self, h_views, labels=None):
        """
        计算多视图对比损失
        h_views: list of view representations
        """
        if len(h_views) < 2:
            return torch.tensor(0.0, device=h_views[0].device)
            
        # 投影到对比空间
        z_views = [self.projector(h) for h in h_views]
        
        total_loss = 0
        num_pairs = 0
        
        for i in range(len(z_views)):
            for j in range(i + 1, len(z_views)):
                loss_ij = self.info_nce_loss(z_views[i], z_views[j], labels)
                total_loss += loss_ij
                num_pairs += 1
                
        return total_loss / max(num_pairs, 1)


class ImprovedBotnetDetector(nn.Module):
    """
    改进的僵尸网络检测器
    整合：三模态特征融合 + 概念漂移感知对比学习
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
        
        # 概念漂移检测
        self.drift_detector = ConceptDriftDetector(hidden_dim)
        
        # 对比学习模块
        self.contrastive_module = ContrastiveLearningModule(hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.PReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 辅助：语义特征生成器（当没有语义特征时）
        self.semantic_fallback = nn.Sequential(
            nn.Linear(stat_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim)
        )
        
        # 辅助：结构特征生成器（当没有结构特征时）
        self.struct_fallback = nn.Sequential(
            nn.Linear(stat_dim, struct_dim),
            nn.ReLU(),
            nn.Linear(struct_dim, struct_dim)
        )
        
    def forward(self, data, return_all=False):
        """
        前向传播
        data: HeteroData对象，需要包含:
            - data['ip'].x: 统计特征 [num_nodes, stat_dim]
            - data['ip'].semantic_x: 语义特征 [num_nodes, semantic_dim] (可选)
            - data['ip'].struct_x: 结构特征 [num_nodes, struct_dim] (可选)
            - data['ip', 'flow', 'ip'].edge_index: 边索引
        """
        # 提取输入
        stat_x = data['ip'].x
        edge_index = data['ip', 'flow', 'ip'].edge_index
        
        # 语义特征（如果没有则从统计特征生成）
        if hasattr(data['ip'], 'semantic_x') and data['ip'].semantic_x is not None:
            semantic_x = data['ip'].semantic_x
        else:
            semantic_x = self.semantic_fallback(stat_x)
            
        # 结构特征（如果没有则从统计特征生成）
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
        
        # 分类
        logits = self.classifier(h_fused)
        
        if return_all:
            return logits, h_fused, [h_stat, h_sem, h_struct], attn_weights
        return logits
    
    def compute_loss(self, data, labels, return_components=False):
        """计算总损失"""
        # 前向传播
        logits, h_fused, h_views, attn_weights = self.forward(data, return_all=True)
        
        # 分类损失 (Focal Loss)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(), labels.float(), reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = (0.75 * (1-pt)**2.0 * bce_loss).mean()
        
        # 概念漂移检测
        with torch.no_grad():
            drift_score = self.drift_detector.detect_drift(h_fused)
        
        # 对比学习损失
        con_loss = self.contrastive_module(h_views, labels)
        
        # 漂移自适应权重：漂移越大，对比学习权重越高
        con_weight = 0.3 * (1.0 + drift_score)
        
        # 总损失
        total_loss = focal_loss + con_weight * con_loss
        
        if return_components:
            return total_loss, {
                'focal_loss': focal_loss.item(),
                'con_loss': con_loss.item(),
                'drift_score': drift_score,
                'con_weight': con_weight
            }
        return total_loss
    
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
        print(f"[Model] 改进模型已保存至: {path}")
        
    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print(f"[Model] 改进模型已加载")
        return model


class ImprovedTrainer:
    """改进模型的训练器"""
    def __init__(self, model, lr=0.002, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        self.scaler = amp.GradScaler()
        self.best_loss = float('inf')
        
    def train_step_batch(self, batch):
        """单步训练"""
        self.model.train()
        batch = batch.to(self.device)
        
        self.optimizer.zero_grad()
        
        with amp.autocast():
            # NeighborLoader 返回的子图中，需要计算损失的节点数由 batch_size 指定
            # logits 输出是所有采样节点的，但只有前 batch_size 个是目标节点
            out = self.model(batch)
            
            target = batch['ip'].y
            
            # 获取实际的 batch_size（目标节点数量）
            if hasattr(batch['ip'], 'batch_size'):
                actual_batch_size = batch['ip'].batch_size
            else:
                actual_batch_size = out.size(0)
            
            # 确保 logits 和 target 尺寸匹配
            logits = out[:actual_batch_size].squeeze()
            targets = target[:actual_batch_size]
            
            # Focal Loss
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = (0.75 * (1-pt)**2.0 * bce_loss).mean()
            
            loss = focal_loss
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            loss = self.train_step_batch(batch)
            total_loss += loss
            num_batches += 1
            
        self.scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def evaluate(self, data, labels):
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(data)
            probs = torch.sigmoid(logits).squeeze()
            
            # 计算各种指标
            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            auc = roc_auc_score(labels_np, probs_np)
            
            # 找最优阈值
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