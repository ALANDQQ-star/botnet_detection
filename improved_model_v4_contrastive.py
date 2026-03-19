"""
V4 对比学习增强版模型

核心改进：
1. 对比学习预训练 - 拉大正负样本间距
2. ArcFace Loss - 角度间隔损失
3. 困难样本挖掘 - 关注边界样本
4. 类别平衡采样 - 解决数据不平衡
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
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss: 角度间隔损失
    
    在角度空间中增加类别间隔，增大正负样本分离度
    """
    def __init__(self, in_features, out_features=2, s=30.0, m=0.5):
        super().__init__()
        self.s = s  # 缩放因子
        self.m = m  # 角度间隔
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings, labels):
        # 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # 余弦相似度
        cosine = F.linear(embeddings, weight)
        
        # 角度间隔
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # 对正样本添加角度间隔
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        target_logits = torch.where(one_hot == 1, torch.cos(theta + self.m), torch.cos(theta))
        
        # 缩放
        output = self.s * target_logits
        
        return output


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    
    拉大正负样本在嵌入空间的距离
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, feat_dim]
            labels: [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 归一化特征
        features = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)
        
        # 创建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除对角线（自己和自己）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算对比损失
        exp_logits = torch.exp(similarity_matrix / self.temperature) * logits_mask
        log_prob = similarity_matrix / self.temperature - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        # 计算正样本对的平均对数概率
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        
        # 损失
        loss = -(self.base_temperature / self.temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class MarginLoss(nn.Module):
    """
    间隔最大化损失
    
    直接优化正负样本嵌入的中心距离
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        """
        最小化类内距离，最大化类间距离
        """
        # 正样本和负样本的嵌入
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        pos_embeddings = embeddings[pos_mask]
        neg_embeddings = embeddings[neg_mask]
        
        # 计算类中心
        pos_center = pos_embeddings.mean(dim=0)
        neg_center = neg_embeddings.mean(dim=0)
        
        # 类间距离（希望最大化）
        inter_dist = torch.norm(pos_center - neg_center, p=2)
        
        # 类内距离（希望最小化）
        pos_intra = torch.mean(torch.norm(pos_embeddings - pos_center, p=2, dim=1))
        neg_intra = torch.mean(torch.norm(neg_embeddings - neg_center, p=2, dim=1))
        
