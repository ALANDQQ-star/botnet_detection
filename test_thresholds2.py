import numpy as np

data = np.load('v3_final_score_distribution.npz', allow_pickle=True)
probs = data['probs']
y_true = data['y_true']

print(f"Total: {len(probs)}, Bots: {y_true.sum()}")

for t in np.linspace(0.01, 0.5, 100):
    preds = (probs >= t).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    p = tp / preds.sum() if preds.sum() > 0 else 0
    r = tp / y_true.sum() if y_true.sum() > 0 else 0
    if p > 0.7 or r > 0.7:
        print(f"Threshold: {t:.5f}, P: {p:.4f}, R: {r:.4f}")

