"""
改进的异构图对比学习僵尸网络检测模型 V2
核心改进：增强分数区分度 + 分数校准

主要改进：
1. 对比损失增强 - 强制拉大类间分数距离
2. 分数校准层 - 学习分数的非线性变换扩展动态范围
3. 温度缩放 - 后处理调整分数动态范围
4. AUC 损失 + 校准约束 - 直接优化 AUC 同时约束分数范围
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
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, 1e-7, 1-1e-7)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class ContrastiveMarginLoss(nn.Module):
    """
    对比边界损失
    强制正类分数 > pos_target, 负类分数 < neg_target
    并在两类之间创建 margin
    """
    def __init__(self, pos_target=0.7, neg_target=0.1, margin=0.4):
        super().__init__()
        self.pos_target = pos_target
        self.neg_target = neg_target
        self.margin = margin
    
    def forward(self, probs, labels):
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        total_loss = 0
        count = 0
        
        # 正类损失：拉向 pos_target
        if pos_mask.sum() > 0:
            pos_probs = probs[pos_mask]
            # 希望正类分数接近 pos_target
            pos_loss = F.mse_loss(pos_probs, torch.ones_like(pos_probs) * self.pos_target)
            total_loss += pos_loss
            count += 1
        
        # 负类损失：拉向 neg_target
        if neg_mask.sum() > 0:
            neg_probs = probs[neg_mask]
            # 希望负类分数接近 neg_target
            neg_loss = F.mse_loss(neg_probs, torch.ones_like(neg_probs) * self.neg_target)
            total_loss += neg_loss
            count += 1
        
        # Margin 损失：确保正类分数 - 负类分数 > margin
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_mean = probs[pos_mask].mean()
            neg_mean = probs[neg_mask].mean()
            margin_loss = F.relu(self.margin - (pos_mean - neg_mean))
            total_loss += margin_loss
            count += 1
        
        return total_loss / max(count, 1)


class DistributionLoss(nn.Module):
    """
    分布分离损失
    基于 Bhattacharyya 距离的近似，最大化两类分布的分离度
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, probs, labels):
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        if pos_mask.sum() < 2 or neg_mask.sum() < 2:
            return torch.tensor(0.0, device=probs.device)
        
        pos_probs = probs[pos_mask]
        neg_probs = probs[neg_mask]
        
        # 计算均值和方差
        pos_mean = pos_probs.mean()
        neg_mean = neg_probs.mean()
        pos_var = pos_probs.var() + 1e-6
        neg_var = neg_probs.var() + 1e-6
        
        # Bhattacharyya 距离近似
        # 最大化这个距离
        bc_dist = 0.25 * ((pos_mean - neg_mean)**2 / (pos_var + neg_var + 1e-6))
        bc_dist += 0.5 * torch.log((pos_var + neg_var) / (2 * torch.sqrt(pos_var * neg_var) + 1e-6))
        
        # 返回负值作为损失（因为要最大化距离）
        return 1.0 / (bc_dist + self.temperature)


class AUCMLoss(nn.Module):
    """
    AUC  Margin Loss
    直接优化 AUC，基于配对比较
    """
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin
    
    def forward(self, probs, labels):
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=probs.device)
        
        pos_probs = probs[pos_mask]  # [N_pos]
        neg_probs = probs[neg_mask]  # [N_neg]
        
        # 配对比较：pos - neg + margin > 0
        # 使用 outer operation 计算所有配对
        pos_expanded = pos_probs.unsqueeze(1)  # [N_pos, 1]
        neg_expanded = neg_probs.unsqueeze(0)  # [1, N_neg]
        
        # 计算 violations
        violations = F.relu(self.margin - (pos_expanded - neg_expanded))
        
        return violations.mean()


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
        self.num_modalities = num_modalities
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


class CalibrationLayer(nn.Module):
    """
    分数校准层
    学习一个单调非线性变换，扩展分数的动态范围
    """
    def __init__(self, hidden_size=16):
        super().__init__()
        # 使用单调网络确保变换是单调的
        self.layer1 = nn.Linear(1, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        
        # 正权重约束（确保单调性）
        self._constrain_weights()
        
    def _constrain_weights(self):
        """约束权重为正，确保单调递增"""
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data = torch.abs(m.weight.data)
    
    def forward(self, probs):
        # probs: [batch, 1] or [batch]
        if probs.dim() == 1:
            probs = probs.unsqueeze(-1)
        
        # 确保输入在 (0, 1) 范围
        probs = torch.clamp(probs, 1e-7, 1-1e-7)
        
        # 使用 logit 变换扩展范围
        logits = torch.log(probs / (1 - probs))
        
        # 通过校准网络
        x = self.layer1(logits)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        
        # 缩放回 [0, 1]
        calibrated = torch.sigmoid(x + logits)  # 残差连接
        
        return calibrated


class TemperatureScaling(nn.Module):
    """
    温度缩放层
    学习一个温度参数来调整分数的动态范围
    """
    def __init__(self, init_temp=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)
    
    def forward(self, probs):
        probs = torch.clamp(probs, 1e-7, 1-1e-7)
        logits = torch.log(probs / (1 - probs))
        scaled_logits = logits / self.temperature
        return torch.sigmoid(scaled_logits)
    
    def get_temperature(self):
        return self.temperature.item()


class ImprovedBotnetDetectorV2(nn.Module):
    """
    改进的僵尸网络检测器 V2
    核心改进：增强分数区分度
    """
    def __init__(self, stat_dim=64, semantic_dim=32, struct_dim=16, hidden_dim=128,
                 use_calibration=True, use_temperature=True):
        super().__init__()
        self.stat_dim = stat_dim
        self.semantic_dim = semantic_dim
        self.struct_dim = struct_dim
        self.hidden_dim = hidden_dim
        self.use_calibration = use_calibration
        self.use_temperature = use_temperature
        
        # 三模态编码器
        self.stat_encoder = StatisticalEncoder(stat_dim, hidden_dim // 2, hidden_dim)
        self.semantic_encoder = SemanticEncoder(semantic_dim, hidden_dim // 2, hidden_dim)
        self.struct_encoder = StructuralEncoder(struct_dim, hidden_dim // 2, hidden_dim)
        
        # 跨模态融合
        self.cross_modal_attention = CrossModalAttention(hidden_dim, num_modalities=3)
        
        # 分类器（返回 logits）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.PReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 校准层
        if use_calibration:
            self.calibration = CalibrationLayer()
        else:
            self.calibration = None
        
        # 温度缩放
        if use_temperature:
            self.temperature_scaling = TemperatureScaling(init_temp=2.0)
        else:
            self.temperature_scaling = None
        
        # 辅助：语义特征生成器
        self.semantic_fallback = nn.Sequential(
            nn.Linear(stat_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim)
        )
        
        # 辅助：结构特征生成器
        self.struct_fallback = nn.Sequential(
            nn.Linear(stat_dim, struct_dim),
            nn.ReLU(),
            nn.Linear(struct_dim, struct_dim)
        )
        
        # 损失函数
        self.contrastive_loss = ContrastiveMarginLoss(pos_target=0.75, neg_target=0.15, margin=0.5)
        self.auc_loss = AUCMLoss(margin=0.1)
        self.distribution_loss = DistributionLoss()
        
    def forward(self, data, return_all=False):
        """前向传播"""
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
        
        # 分类（返回 logits）
        logits = self.classifier(h_fused)
        base_probs = torch.sigmoid(logits)
        
        # 校准
        if self.calibration is not None:
            calibrated_probs = self.calibration(base_probs)
        else:
            calibrated_probs = base_probs
        
        # 温度缩放
        if self.temperature_scaling is not None:
            final_probs = self.temperature_scaling(calibrated_probs)
        else:
            final_probs = calibrated_probs
        
        if return_all:
            return final_probs, base_probs, logits, h_fused, [h_stat, h_sem, h_struct], attn_weights
        return final_probs
    
    def compute_loss(self, data, labels, return_components=False):
        """
        计算综合损失
        包含：Focal Loss + 对比损失 + AUC 损失 + 分布损失
        
        注意：处理 NeighborLoader 的 batch 问题
        - logits 输出是所有采样节点的
        - 但只有前 batch_size 个是目标节点
        """
        # 前向传播
        probs, base_probs, logits, h_fused, h_views, attn_weights = self.forward(data, return_all=True)
        
        # 获取实际的 batch_size（目标节点数量）
        if hasattr(data['ip'], 'batch_size'):
            actual_batch_size = data['ip'].batch_size
        else:
            actual_batch_size = probs.size(0)
        
        # 只取前 batch_size 个节点的输出用于计算损失
        logits = logits[:actual_batch_size].squeeze()
        probs = probs[:actual_batch_size].squeeze()
        labels = labels[:actual_batch_size]
        
        # 基础分类损失 (Focal Loss) - 使用 BCEWithLogits 以支持 autocast
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = (0.75 * (1-pt)**2.0 * bce_loss).mean()
        
        # 对比损失 - 拉大类间距离（使用 probs）
        con_loss = self.contrastive_loss(probs, labels)
        
        # AUC 损失 - 直接优化 AUC
        auc_loss = self.auc_loss(probs, labels)
        
        # 分布损失 - 增加分布分离度
        dist_loss = self.distribution_loss(probs, labels)
        
        # 权重配置
        total_loss = (
            1.0 * focal_loss +
            0.8 * con_loss +
            0.3 * auc_loss +
            0.2 * dist_loss
        )
        
        if return_components:
            return total_loss, {
                'focal_loss': focal_loss.item(),
                'con_loss': con_loss.item(),
                'auc_loss': auc_loss.item(),
                'dist_loss': dist_loss.item(),
                'temp': self.temperature_scaling.get_temperature() if self.temperature_scaling else 1.0
            }
        return total_loss
    
    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'stat_dim': self.stat_dim,
                'semantic_dim': self.semantic_dim,
                'struct_dim': self.struct_dim,
                'hidden_dim': self.hidden_dim,
                'use_calibration': self.use_calibration,
                'use_temperature': self.use_temperature
            }
        }, path)
        print(f"[Model] V2 模型已保存至：{path}")
        
    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print(f"[Model] V2 模型已加载")
        return model


class ImprovedTrainerV2:
    """V2 模型的训练器"""
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
            out = self.model(batch)
            
            target = batch['ip'].y
            
            if hasattr(batch['ip'], 'batch_size'):
                actual_batch_size = batch['ip'].batch_size
            else:
                actual_batch_size = out.size(0)
            
            probs = out[:actual_batch_size].squeeze()
            targets = target[:actual_batch_size]
            
            # 计算综合损失
            loss, components = self.model.compute_loss(batch, targets, return_components=True)
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item(), components
    
    def train_epoch(self, train_loader, epoch):
        """训练一个 epoch"""
        total_loss = 0
        total_components = {'focal_loss': 0, 'con_loss': 0, 'auc_loss': 0, 'dist_loss': 0, 'temp': 0}
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
            print(f"  损失分解：Focal={total_components['focal_loss']/num_batches:.4f}, "
                  f"Contrastive={total_components['con_loss']/num_batches:.4f}, "
                  f"AUC={total_components['auc_loss']/num_batches:.4f}, "
                  f"Dist={total_components['dist_loss']/num_batches:.4f}, "
                  f"Temp={total_components['temp']/num_batches:.2f}")
        
        return avg_loss
    
    def evaluate(self, data, labels):
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            probs = self.model(data)
            
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