import pandas as pd
import numpy as np
from ensemble_eval import compute_eer

df_base = pd.read_csv('checkpoints/attention_best_scores.csv')
df_new = pd.read_csv('checkpoints/attention_boundary_best_scores.csv')

s_base = df_base['score'].values
s_new = df_new['score'].values

# z-normalize
z_base = (s_base - np.mean(s_base)) / np.std(s_base)
z_new = (s_new - np.mean(s_new)) / np.std(s_new)

y_true = df_base['true_label'].values
attacks = df_base['attack'].values

print('Testing weight combinations between Baseline Attention and New Boundary Model:')
print(f'{"Weight (Base / New)":<25} | {"Overall EER":<12} | {"EER w/o A12":<12} | {"A12 Acc":<10} | {"Bonafide Acc":<12}')
print('-' * 80)

for w_new in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    w_base = 1.0 - w_new
    fused = w_base * z_base + w_new * z_new
    eer_all, thresh = compute_eer(y_true, fused)
    preds = (fused >= thresh).astype(int)
    
    mask_no_a12 = (attacks != 'A12')
    eer_no_a12, _ = compute_eer(y_true[mask_no_a12], fused[mask_no_a12])
    
    a12_acc = (preds[attacks == 'A12'] == 1).mean() * 100
    bon_acc = (preds[attacks == 'bonafide'] == 0).mean() * 100
    print(f'{w_base:4.2f} / {w_new:4.2f}                | {eer_all:9.3f}%   | {eer_no_a12:9.3f}%   | {a12_acc:8.2f}% | {bon_acc:10.2f}%')
