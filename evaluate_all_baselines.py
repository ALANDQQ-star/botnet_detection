import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, silhouette_samples
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

from data_loader import CTU13Loader
from improved_heterogeneous_graph import ImprovedHeterogeneousGraphBuilder
from bot_ahgcn_baseline import BotAHGCNGraphBuilder, MetaPathSimilarity, BotAHGCNModel, prepare_ahgcn_data

# ==========================================
# 1. TRUSTED Algorithm Implementation
# ==========================================
class TRUSTED_NodeClassifier:
    def __init__(self, window_size=1000, distance_threshold=1.0, similarity_threshold=0.005):
        self.window_size = window_size
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        
    def fit_predict_batch(self, X):
        if len(X) < 2:
            return np.zeros(len(X))
            
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        clusterer = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=self.distance_threshold, 
            linkage='average',
            metric='euclidean'
        )
        labels = clusterer.fit_predict(X_scaled)
        
        if len(set(labels)) <= 1:
            return np.zeros(len(X))
            
        sil_scores = silhouette_samples(X_scaled, labels, metric='euclidean')
        
        pred_labels = np.zeros(len(X))
        unique_clusters = np.unique(labels)
        
        for c in unique_clusters:
            c_mask = (labels == c)
            mean_sil = np.mean(sil_scores[c_mask])
            if mean_sil <= self.similarity_threshold:
                pred_labels[c_mask] = 1 # Anomaly / Botnet
                
        return pred_labels
        
    def predict(self, X):
        predictions = []
        n_samples = len(X)
        for i in range(0, n_samples, self.window_size):
            end_idx = min(i + self.window_size, n_samples)
            X_batch = X[i:end_idx]
            preds = self.fit_predict_batch(X_batch)
            predictions.extend(preds)
        return np.array(predictions)

# ==========================================
# 2. Bot-DM Baseline (Rigorous Implementation)
# ==========================================

def mutual_information_loss(z, z_prime, num_classes=2):
    """Implement Eq(6): Maximizing Mutual Information"""
    proj_z = F.softmax(nn.Linear(z.size(1), num_classes).to(z.device)(z), dim=-1)
    proj_zp = F.softmax(nn.Linear(z_prime.size(1), num_classes).to(z_prime.device)(z_prime), dim=-1)
    P = torch.mm(proj_z.t(), proj_zp) / z.size(0)
    P = (P + P.t()) / 2.0
    P_c = P.sum(dim=1, keepdim=True)
    P_c_prime = P.sum(dim=0, keepdim=True)
    eps = 1e-8
    joint_entropy = P * torch.log(P / (torch.mm(P_c, P_c_prime) + eps) + eps)
    MI = joint_entropy.sum()
    return -MI

class BotDM_Strict(nn.Module):
    """
    Rigorous implementation of Bot-DM architecture:
    Botflow-image: CNN -> MaxPool -> CNN -> MaxPool -> Self-Attention -> Bi-LSTM
    Botflow-token: Token/Pos Embeddings -> Multi-layer Transformer Encoder
    Fusion: Mutual Information maximization & Cross-modal prediction
    """
    def __init__(self, token_dim=32, image_feat_dim=64, out_dim=64):
        super().__init__()
        
        # 1. Project input semantic features to emulate token sequence [B, 16, 16]
        self.token_proj = nn.Linear(token_dim, 16 * 16)
        
        # Transformer-based Token Model
        encoder_layers = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.token_out = nn.Linear(16, out_dim)
        
        # 2. Project input statistical/structural features to emulate 32x32 image
        self.img_proj = nn.Linear(image_feat_dim, 32 * 32)
        
        # CNN + Bi-LSTM Image Model
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.attn_query = nn.Linear(32, 4)
        self.attn_key = nn.Linear(32, 4)
        self.attn_value = nn.Linear(32, 32)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        self.image_out = nn.Linear(64, out_dim)
        
        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_dim, 1)
        )
        
    def _forward_image(self, x):
        # x is (B, image_feat_dim)
        x = self.img_proj(x).view(-1, 1, 32, 32)
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Self-attention
        B, C, W, H = x.size()
        x_flat = x.view(B, C, -1).permute(0, 2, 1) # B x N x C
        q = self.attn_query(x_flat)
        k = self.attn_key(x_flat)
        v = self.attn_value(x_flat)
        
        energy = torch.bmm(q, k.permute(0, 2, 1))
        attention = F.softmax(energy, dim=-1)
        
        out = torch.bmm(attention, v)
        out = self.gamma * out + x_flat
        
        # Bi-LSTM
        lstm_out, _ = self.lstm(out)
        final_state = lstm_out[:, -1, :] # Take last hidden state
        
        return self.image_out(final_state)
        
    def _forward_token(self, x):
        # x is (B, token_dim)
        x = self.token_proj(x).view(-1, 16, 16) # Emulate sequence of length 16, dim 16
        
        # Transformer
        x = self.transformer_encoder(x)
        cls_token = x[:, 0, :] # Use first token representation
        
        return self.token_out(cls_token)

    def forward(self, token_x, img_x):
        z_prime = self._forward_image(img_x)
        z = self._forward_token(token_x)
        
        fused = torch.cat([z_prime, z], dim=1)
        logits = self.classifier(fused)
        
        return logits, z_prime, z

def train_botdm(X_sem, X_graph, y, epochs=10, batch_size=512, device='cuda'):
    model = BotDM_Strict(token_dim=X_sem.shape[1], image_feat_dim=X_graph.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_sem, dtype=torch.float32),
        torch.tensor(X_graph, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        start_t = time.time()
        for step, (b_sem, b_graph, b_y) in enumerate(dataloader):
            b_sem, b_graph, b_y = b_sem.to(device), b_graph.to(device), b_y.to(device)
            optimizer.zero_grad()
            
            logits, z_prime, z = model(b_sem, b_graph)
            
            # Loss = BCE + MI (Mutual Information from dual modalities)
            loss_bce = criterion(logits.squeeze(), b_y)
            loss_mi = mutual_information_loss(z, z_prime)
            loss = loss_bce + 0.1 * loss_mi
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (step + 1) % 500 == 0:
                print(f"    [Bot-DM] Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(dataloader)}] Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"    [Bot-DM] Epoch {epoch+1}/{epochs} completed in {time.time()-start_t:.1f}s | Avg Loss: {avg_loss:.4f}")
        
    return model

def evaluate_botdm(model, X_sem, X_graph, y, device='cuda'):
    model.eval()
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_sem, dtype=torch.float32),
        torch.tensor(X_graph, dtype=torch.float32)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
    
    scores = []
    with torch.no_grad():
        for b_sem, b_graph in dataloader:
            b_sem, b_graph = b_sem.to(device), b_graph.to(device)
            logits, _, _ = model(b_sem, b_graph)
            logits = logits.squeeze()
            if logits.dim() == 0: logits = logits.unsqueeze(0)
            probs = torch.sigmoid(logits)
            scores.extend(probs.cpu().numpy().tolist())
            
    scores = np.array(scores)
    
    try:
        auc = roc_auc_score(y, scores)
    except:
        auc = 0.5
        
    # Use Youden's J statistic for threshold
    from sklearn.metrics import roc_curve
    try:
        fpr, tpr, thresholds = roc_curve(y, scores)
        j_scores = tpr - fpr
        best_thresh = thresholds[np.argmax(j_scores)]
    except:
        best_thresh = 0.5
        
    preds = (scores > best_thresh).astype(int)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    
    return {'auc': auc, 'precision': prec, 'recall': rec, 'f1': f1, 'scores': scores}

# ==========================================
# 3. Helpers
# ==========================================
def extract_features_v3(scenarios):
    loader = CTU13Loader('/root/autodl-fs/CTU-13/CTU-13-Dataset')
    df = loader.load_data(scenarios)
    if df.empty: return None, None, None
    
    from main_improved_v3_final import get_labels
    graph_builder = ImprovedHeterogeneousGraphBuilder()
    data, ip_map = graph_builder.build(df, include_semantic=True, include_struct=True)
    labels, _ = get_labels(df, ip_map)
    labels = labels.numpy()
    
    # Extract features for Bot-DM & TRUSTED
    # Bot-DM semantic = semantic_x + struct_x, graph = x
    X_sem = np.concatenate([data['ip'].semantic_x.numpy(), data['ip'].struct_x.numpy()], axis=1)
    X_graph = data['ip'].x.numpy()
    
    X_all = np.concatenate([X_graph, X_sem], axis=1)
    
    return df, X_all, X_sem, X_graph, labels

# ==========================================
# 4. Main Evaluation Loop
# ==========================================
def evaluate_all_scenarios():
    print("============================================================")
    print("Baseline Comparison Evaluation: Scenarios 4-13")
    print("============================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train Bot-DM & BotHGCN on Scenarios 1-3
    print("\n[Phase 1] Training Baselines on Scenarios 1-3...")
    df_train, X_train_all, X_train_sem, X_train_graph, y_train = extract_features_v3([1, 2, 3])
    
    print(f"  Training Bot-DM...")
    botdm_model = train_botdm(X_train_sem, X_train_graph, y_train, device=device)
    
    print(f"  Training BotHGCN...")
    ahgcn_builder = BotAHGCNGraphBuilder()
    ahin_train, ip_map_train, _ = ahgcn_builder.build_ahin(df_train)
    sim_calc_train = MetaPathSimilarity(ahin_train, ip_map_train)
    # Using dummy labels for BotHGCN builder alignment
    y_train_ahgcn = torch.zeros(len(ip_map_train))
    from main_improved_v3_final import get_labels
    labels_h, _ = get_labels(df_train, ip_map_train)
    x_h, am_h, ag_h, y_h = prepare_ahgcn_data(ahin_train, sim_calc_train, labels_h)
    
    ahgcn_net = BotAHGCNModel(in_dim=x_h.shape[1], hidden_dim=64, out_dim=32).to(device)
    ahgcn_optimizer = torch.optim.Adam(ahgcn_net.parameters(), lr=0.01)
    ahgcn_criterion = nn.BCEWithLogitsLoss()
    
    x_h = x_h.to(device)
    am_h = am_h.to(device)
    ag_h = ag_h.to(device)
    y_h = y_h.to(device)
    
    ahgcn_net.train()
    for ep in range(10):
        ahgcn_optimizer.zero_grad()
        logits = ahgcn_net(x_h, am_h, ag_h).squeeze()
        loss = ahgcn_criterion(logits, y_h.float())
        loss.backward()
        ahgcn_optimizer.step()
    print("  BotHGCN training complete.")
    del df_train, ahin_train, sim_calc_train, x_h, am_h, ag_h, y_h
    
    # Note: TRUSTED is an online unsupervised clustering algorithm, so it doesn't need to be pre-trained on 1-3.
    # It evaluates directly per scenario.
    # We will use distance_threshold=1.0, similarity_threshold=0.005 which performed well in early tests.
    
    results = {}
    
    test_scenarios = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    for scen in test_scenarios:
        print(f"\n============================================================")
        print(f"Evaluating Scenario {scen}")
        print(f"============================================================")
        
        df_test, X_test_all, X_test_sem, X_test_graph, y_test = extract_features_v3([scen])
        if df_test is None:
            continue
            
        print(f"  Nodes: {len(y_test)}, Bots: {np.sum(y_test)} ({np.mean(y_test)*100:.2f}%)")
        results[scen] = {}
        
        # 1. TRUSTED
        print("  Running TRUSTED...")
        t_start = time.time()
        trusted_clf = TRUSTED_NodeClassifier(window_size=1000, distance_threshold=1.0, similarity_threshold=0.005)
        trusted_preds = trusted_clf.predict(X_test_all)
        t_time = time.time() - t_start
        
        t_prec = precision_score(y_test, trusted_preds, zero_division=0)
        t_rec = recall_score(y_test, trusted_preds, zero_division=0)
        t_f1 = f1_score(y_test, trusted_preds, zero_division=0)
        try: t_auc = roc_auc_score(y_test, trusted_preds)
        except: t_auc = 0.5
        
        results[scen]['TRUSTED'] = {'auc': t_auc, 'prec': t_prec, 'rec': t_rec, 'f1': t_f1, 'time': t_time}
        print(f"    TRUSTED - AUC: {t_auc:.4f}, F1: {t_f1:.4f}, Time: {t_time:.1f}s")
        
        # 2. Bot-DM
        print("  Running Bot-DM...")
        t_start = time.time()
        dm_metrics = evaluate_botdm(botdm_model, X_test_sem, X_test_graph, y_test, device)
        dm_time = time.time() - t_start
        results[scen]['Bot-DM'] = {
            'auc': dm_metrics['auc'], 'prec': dm_metrics['precision'],
            'rec': dm_metrics['recall'], 'f1': dm_metrics['f1'], 'time': dm_time
        }
        print(f"    Bot-DM - AUC: {dm_metrics['auc']:.4f}, F1: {dm_metrics['f1']:.4f}, Time: {dm_time:.1f}s")
        
        # 3. BotHGCN
        print("  Running BotHGCN...")
        t_start = time.time()
        try:
            ahgcn_builder_t = BotAHGCNGraphBuilder()
            ahin_test, ip_map_test, _ = ahgcn_builder_t.build_ahin(df_test)
            sim_calc_test = MetaPathSimilarity(ahin_test, ip_map_test)
            labels_ht, _ = get_labels(df_test, ip_map_test)
            x_ht, am_ht, ag_ht, y_ht = prepare_ahgcn_data(ahin_test, sim_calc_test, labels_ht)
            
            ahgcn_net.eval()
            with torch.no_grad():
                logits = ahgcn_net(x_ht.to(device), am_ht.to(device), ag_ht.to(device)).squeeze()
                if logits.dim() == 0: logits = logits.unsqueeze(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                
            try: hgcn_auc = roc_auc_score(y_ht.numpy(), probs)
            except: hgcn_auc = 0.5
            
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_ht.numpy(), probs)
            best_thresh = thresholds[np.argmax(tpr - fpr)]
            hgcn_preds = (probs > best_thresh).astype(int)
            hgcn_prec = precision_score(y_ht.numpy(), hgcn_preds, zero_division=0)
            hgcn_rec = recall_score(y_ht.numpy(), hgcn_preds, zero_division=0)
            hgcn_f1 = f1_score(y_ht.numpy(), hgcn_preds, zero_division=0)
        except Exception as e:
            print(f"    BotHGCN failed: {e}")
            hgcn_auc, hgcn_prec, hgcn_rec, hgcn_f1 = 0, 0, 0, 0
            
        hgcn_time = time.time() - t_start
        results[scen]['BotHGCN'] = {'auc': hgcn_auc, 'prec': hgcn_prec, 'rec': hgcn_rec, 'f1': hgcn_f1, 'time': hgcn_time}
        print(f"    BotHGCN - AUC: {hgcn_auc:.4f}, F1: {hgcn_f1:.4f}, Time: {hgcn_time:.1f}s")
        
        # V3 Final (Using our own improved main script results which are already pre-computed or we can just grab from evaluation_scenarios_6_13.json)
        results[scen]['V3_Final'] = {} # We'll populate this from evaluation file next
        
        del df_test, X_test_all, X_test_sem, X_test_graph
        
    # Read V3 Final metrics from evaluation file
    try:
        with open('evaluation_scenarios_6_13.json', 'r') as f:
            v3_results = json.load(f)
            
        for item in v3_results.get('scenarios', []):
            scen = item['scenario']
            if scen in results:
                # Store the metrics from smart threshold
                results[scen]['V3_Final'] = {
                    'auc': item.get('auc', 0),
                    'prec': item.get('smart_precision', 0),
                    'rec': item.get('smart_recall', 0),
                    'f1': item.get('smart_f1', 0),
                    'time': item.get('inference_time', 0)
                }
    except Exception as e:
        print(f"Warning: Could not load V3 Final results: {e}")
    
    # Generate Comparison Table Output
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS (AUC / F1-Score)")
    print("="*80)
    print(f"{'Scenario':<10} | {'TRUSTED':<15} | {'Bot-DM':<15} | {'BotHGCN':<15} | {'V3 Final (Ours)':<15}")
    print("-" * 80)
    
    for scen in test_scenarios:
        res = results.get(scen, {})
        t_res = res.get('TRUSTED', {})
        dm_res = res.get('Bot-DM', {})
        h_res = res.get('BotHGCN', {})
        v3_res = res.get('V3_Final', {})
        
        t_str = f"{t_res.get('auc',0):.4f} / {t_res.get('f1',0):.4f}" if t_res else "N/A"
        dm_str = f"{dm_res.get('auc',0):.4f} / {dm_res.get('f1',0):.4f}" if dm_res else "N/A"
        h_str = f"{h_res.get('auc',0):.4f} / {h_res.get('f1',0):.4f}" if h_res else "N/A"
        v3_str = f"{v3_res.get('auc',0):.4f} / {v3_res.get('f1',0):.4f}" if v3_res else "N/A"
        
        print(f"Scen {scen:<7} | {t_str:<15} | {dm_str:<15} | {h_str:<15} | {v3_str:<15}")
    
    print("=" * 80)
    
    # Save results
    with open('baseline_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nEvaluations complete. Results saved to baseline_comparison_results.json")

if __name__ == "__main__":
    evaluate_all_scenarios()
