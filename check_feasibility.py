
import numpy as np
from sklearn.metrics import precision_recall_curve

try:
    data = np.load('v3_final_score_distribution.npz', allow_pickle=True)
    probs = data['probs']
    y_true = data['y_true']
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    
    # Check if there is any point where P > 0.8 and R > 0.8
    qualified = (precisions[:-1] > 0.8) & (recalls[:-1] > 0.8)
    
    if np.any(qualified):
        print("Feasible: YES")
        indices = np.where(qualified)[0]
        for idx in indices[::len(indices)//10 + 1]: # Sample a few
            print(f"Threshold: {thresholds[idx]:.6f}, P: {precisions[idx]:.4f}, R: {recalls[idx]:.4f}")
    else:
        print("Feasible: NO")
        # Find closest
        f1 = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1)
        print(f"Best F1: {f1[best_idx]:.4f} at Threshold: {thresholds[best_idx]:.6f}")
        print(f"P: {precisions[best_idx]:.4f}, R: {recalls[best_idx]:.4f}")

        # Check max precision for recall > 0.8
        high_recall_idx = recalls[:-1] > 0.8
        if np.any(high_recall_idx):
            max_p_at_r80 = np.max(precisions[:-1][high_recall_idx])
            print(f"Max Precision at Recall > 0.8: {max_p_at_r80:.4f}")
            
except Exception as e:
    print(f"Error: {e}")
