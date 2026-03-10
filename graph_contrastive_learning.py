import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
import torch.cuda.amp as amp  # [新增] 混合精度训练模块

class FocalLoss(nn.Module):
    """
    Focal Loss: 专注于难分类样本
    """
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

class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # [保持] 用户要求保持 heads=4 以维持性能
        self.heads = 4 
        
        self.gat = GATConv(in_dim, hidden_dim, heads=self.heads, concat=True)
        # GAT 输出维度 = hidden_dim * heads
        self.gcn = GCNConv(hidden_dim * self.heads, out_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim * self.heads)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.PReLU()

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gcn(x, edge_index)
        x = self.bn2(x)
        return x

class BotnetDetector(nn.Module):
    def __init__(self, in_dim=32, hidden_dim=128):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = GraphEncoder(in_dim, hidden_dim // 2, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, data):
        # 兼容 PyG 的 HeteroData 结构
        x, edge_index = data['ip'].x, data['ip', 'flow', 'ip'].edge_index
        emb = self.encoder(x, edge_index)
        logits = self.classifier(emb)
        return logits
    
    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'config': {'in_dim': self.in_dim, 'hidden_dim': self.hidden_dim}
        }, path)
        print(f"[Model] 模型已保存至: {path}")
        
    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print(f"[Model] 模型已加载 (In: {config['in_dim']}, Hidden: {config['hidden_dim']})")
        return model

class Trainer:
    def __init__(self, model, lr=0.002, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        
        # Focal Loss
        self.criterion = FocalLoss(alpha=0.80, gamma=2.0)
        
        # [新增] 梯度缩放器，用于混合精度训练 (AMP)
        self.scaler = amp.GradScaler()

    def train_step_batch(self, batch):
        """
        [修改] 专门用于 Mini-batch 的训练步
        """
        self.model.train()
        batch = batch.to(self.device)
        
        self.optimizer.zero_grad()
        
        # [新增] 开启混合精度上下文
        with amp.autocast():
            logits = self.model(batch)
            
            # NeighborLoader 会对 batch 中的节点进行切片
            # batch['ip'].y 是当前采样到的子图中节点的标签
            # batch['ip'].train_mask 是标记哪些节点是“中心节点”（即我们需要计算Loss的节点）
            target = batch['ip'].y
            
            # 这里的 batch_size 是 loader 设置的大小，通常位于 logits 的前部
            # 我们只计算前 batch_size 个节点的损失（这些是 Target Nodes）
            # 或者使用 train_mask (如果 Loader 传递了 mask)
            if hasattr(batch['ip'], 'train_mask'):
                 mask = batch['ip'].train_mask
                 loss = self.criterion(logits[mask].squeeze(), target[mask])
            else:
                 # 兜底：假设前 batch_size 个是目标节点
                 # 但 NeighborLoader 的 train_mask 方式最稳妥
                 loss = self.criterion(logits.squeeze(), target)

        # [新增] 使用 Scaler 进行反向传播
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()