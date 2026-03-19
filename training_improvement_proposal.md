# 训练端改进方案：提升分数区分度

## 问题诊断

### 当前状态
- **分数分布**：正常节点和僵尸节点分数都集中在 [0, 0.01] 区间
- **AUC**：约 0.958（排序能力好）
- **问题**：分数缺乏校准，难以设定阈值

### 根本原因
1. **类别不平衡**：僵尸节点仅占 1-3%
2. **模型倾向**：GNN 倾向于输出保守分数
3. **损失函数**：标准 BCE 损失对分数校准无约束

---

## 改进方案

### 方案 1：对比学习增强（推荐）

**核心思想**：在训练时加入对比损失，强制同类型节点分数接近，不同类型节点分数分离。

```python
# 改进的损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3, temperature=0.1):
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, probs, labels):
        # 标准 BCE 损失
        bce_loss = F.binary_cross_entropy(probs, labels)
        
        # 对比损失：拉大类间距离
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_probs = probs[pos_mask]
            neg_probs = probs[neg_mask]
            
            # 计算类间距离
            pos_mean = pos_probs.mean()
            neg_mean = neg_probs.mean()
            
            # 对比损失：希望 pos_mean - neg_mean > margin
            contrastive_loss = torch.relu(self.margin - (pos_mean - neg_mean))
        else:
            contrastive_loss = 0
        
        return bce_loss + 0.5 * contrastive_loss
```

**优点**：
- 直接优化分数分离度
- 与 AUC 优化目标一致
- 实现简单

**风险**：
- 可能略微降低 AUC（需要调参）

---

### 方案 2：Focal Loss + 类别权重

**核心思想**：降低易分样本的权重，让模型关注难分样本。

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=10.0):
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, probs, labels):
        probs = torch.clamp(probs, 1e-7, 1-1e-7)
        
        # Focal Loss 权重
        pt = torch.where(labels == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # BCE 损失
        bce = F.binary_cross_entropy(probs, labels, reduction='none')
        
        # 类别权重
        class_weight = torch.where(labels == 1, self.pos_weight, 1.0)
        
        loss = (focal_weight * bce * class_weight).mean()
        return loss
```

**优点**：
- 处理类别不平衡
- 提升难分样本的区分度

---

### 方案 3：分数校准层（Calibration Layer）

**核心思想**：在模型输出后添加校准层，学习分数的非线性变换。

```python
class CalibratedBotnetDetector(nn.Module):
    def __init__(self, base_model):
        self.base_model = base_model
        # 校准层：学习单调变换
        self.calibration = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data):
        # 基础模型输出
        base_logits = self.base_model(data)
        base_probs = torch.sigmoid(base_logits)
        
        # 校准变换
        calibrated_probs = self.calibration(base_probs.unsqueeze(-1))
        
        return calibrated_probs.squeeze(-1)
```

**优点**：
- 灵活学习分数变换
- 可以扩展分数动态范围

**风险**：
- 需要额外训练数据
- 可能过拟合

---

### 方案 4：AUC 损失 + 校准损失

**核心思想**：直接优化 AUC，同时加入校准约束。

```python
class AUCWithCalibrationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, probs, labels):
        # AUC 损失（基于配对比较）
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        pos_probs = probs[pos_mask].unsqueeze(1)  # [N_pos, 1]
        neg_probs = probs[neg_mask].unsqueeze(0)  # [1, N_neg]
        
        # 计算正确配对比例
        auc_loss = (torch.relu(neg_probs - pos_probs + 0.1)).mean()
        
        # 校准损失：希望正类分数>0.5，负类分数<0.1
        if pos_mask.sum() > 0:
            calibration_pos = torch.relu(0.5 - probs[pos_mask]).mean()
        else:
            calibration_pos = 0
        
        if neg_mask.sum() > 0:
            calibration_neg = torch.relu(probs[neg_mask] - 0.1).mean()
        else:
            calibration_neg = 0
        
        total_loss = auc_loss + 0.5 * calibration_pos + 0.5 * calibration_neg
        return total_loss
```

**优点**：
- 直接优化 AUC
- 强制分数校准

---

### 方案 5：温度缩放（后处理，最简单）

**核心思想**：训练后学习一个温度参数，调整分数的动态范围。

```python
class TemperatureScaling:
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1))
    
    def fit(self, probs, labels):
        # 使用验证数据学习温度
        def nll_loss(temp):
            scaled_probs = torch.sigmoid(torch.log(probs / (1-probs + 1e-7)) / temp)
            return F.binary_cross_entropy(scaled_probs, labels)
        
        # 优化温度
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(
            lambda t: nll_loss(torch.tensor(t)).item(),
            bounds=(0.1, 10),
            method='bounded'
        )
        self.temperature = torch.tensor(result.x)
    
    def transform(self, probs):
        logits = torch.log(probs / (1 - probs + 1e-7))
        scaled_probs = torch.sigmoid(logits / self.temperature)
        return scaled_probs
```

**优点**：
- 无需重新训练
- 简单有效

---

## 推荐实施路径

### 阶段 1：后处理校准（1-2 小时）
1. 使用温度缩放松驰分数分布
2. 在验证集上优化温度参数
3. 评估效果

### 阶段 2：对比学习训练（半天）
1. 在训练损失中加入对比损失项
2. 调整 margin 参数
3. 重新训练模型

### 阶段 3：AUC + 校准联合优化（如需进一步优化）
1. 实现 AUC+ 校准联合损失
2. 训练模型
3. 综合评估

---

## 评估指标

| 指标 | 当前 | 目标 |
|------|------|------|
| AUC | 0.958 | > 0.95 |
| 分数范围 | [0, 0.01] | [0, 1] |
| 正常节点中位分数 | 0.000002 | < 0.05 |
| 僵尸节点中位分数 | 0.001 | > 0.5 |
| 分数分离度 | 0.001 | > 0.3 |

---

## 预期效果

通过对比学习 + 校准损失，预期可以：
1. 保持 AUC > 0.95
2. 僵尸节点分数提升到 0.5-0.9 区间
3. 正常节点分数保持在 0-0.2 区间
4. 分界点清晰（约 0.3-0.4）