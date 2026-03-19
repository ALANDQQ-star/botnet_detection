import numpy as np

data = np.load('v3_final_score_distribution.npz', allow_pickle=True)
probs = data['probs']
y_true = data['y_true']

print(f"Total: {len(probs)}, Bots: {y_true.sum()}")
best_f1 = 0
best_p = 0
best_r = 0
best_t = 0

for t in np.unique(probs):
    preds = (probs >= t).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    p = tp / preds.sum() if preds.sum() > 0 else 0
    r = tp / y_true.sum() if y_true.sum() > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    if f1 > best_f1:
        best_f1 = f1
        best_p = p
        best_r = r
        best_t = t

print(f"Best F1: {best_f1:.4f} at Threshold: {best_t:.5f}, P: {best_p:.4f}, R: {best_r:.4f}")

# Also let's check top N
for percent in [0.5, 0.6, 0.8, 1.0, 1.2, 1.5]:
    t = np.percentile(probs, 100 - percent)
    preds = (probs >= t).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    p = tp / preds.sum() if preds.sum() > 0 else 0
    r = tp / y_true.sum() if y_true.sum() > 0 else 0
    print(f"Top {percent}% -> Threshold: {t:.5f}, P: {p:.4f}, R: {r:.4f}")

