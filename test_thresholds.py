import numpy as np

data = np.load('v3_final_score_distribution.npz', allow_pickle=True)
probs = data['probs']
y_true = data['y_true']

print(f"Total: {len(probs)}, Bots: {y_true.sum()}")
best_p, best_r, best_t = 0, 0, 0

for t in np.linspace(0.001, 0.01, 100):
    preds = (probs >= t).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    p = tp / preds.sum() if preds.sum() > 0 else 0
    r = tp / y_true.sum() if y_true.sum() > 0 else 0
    if p > 0.5 or r > 0.5:
        print(f"Threshold: {t:.5f}, P: {p:.4f}, R: {r:.4f}")

