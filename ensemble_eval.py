import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def compute_eer(y_true, y_scores):
    y_scores = np.nan_to_num(np.array(y_scores), nan=0.0, posinf=100.0, neginf=-100.0)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except:
        fnr = 1 - tpr
        idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = fpr[idx]

    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    thresh = thresholds[idx]
    return eer * 100, thresh


def min_dcf(y_true, y_scores):
    """
    Normalized ASVspoof 5 Track 1 minDCF.

    Assumptions:
        y_true = 0: bona fide
        y_true = 1: spoof
        Higher input scores indicate spoof.

    Official parameters:
        C_miss = 1
        C_fa = 10
        pi_spf = 0.05
        beta = 1.90
    """

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    y_scores = np.nan_to_num(
        y_scores,
        nan=0.0,
        posinf=100.0,
        neginf=-100.0
    )

    # Convert scores so that higher scores indicate bona fide.
    bonafide_scores = -y_scores[y_true == 0]
    spoof_scores = -y_scores[y_true == 1]

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        raise ValueError("Both bona fide and spoof samples must be present.")

    # Positive class = bona fide
    labels = np.concatenate([
        np.ones(len(bonafide_scores)),
        np.zeros(len(spoof_scores))
    ])

    scores = np.concatenate([
        bonafide_scores,
        spoof_scores
    ])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    # Official ASVspoof 5 error rates
    p_miss = 1.0 - tpr
    p_fa = fpr

    # Normalized minDCF:
    # beta = (C_miss / C_fa) * ((1 - pi_spf) / pi_spf)
    beta = (1.0 / 10.0) * (0.95 / 0.05)  # 1.90

    normalized_dcf = beta * p_miss + p_fa

    return np.min(normalized_dcf)


def analyze_models():
    checkpoint_dir = Path("checkpoints")
    models = ['attention', 'attention_boundary', 'concat', 'mlp', 'mfcc', 'lfcc', 'waveform']
    dfs = {}

    print("=" * 105)
    print(
        f"{'Model':<18} | {'Overall EER':<12} | {'EER w/o A12':<12} | "
        f"{'minDCF':<10} | {'A12 Acc':<10} | {'Bonafide Acc':<12}"
    )
    print("-" * 105)

    for m in models:
        path = checkpoint_dir / f"{m}_best_scores.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        dfs[m] = df

        eer_all, thresh = compute_eer(df['true_label'], df['score'])
        dcf = min_dcf(df['true_label'], df['score'])

        preds = (df['score'] >= thresh).astype(int)
        df['pred'] = preds

        df_no_a12 = df[df['attack'] != 'A12']
        eer_no_a12, _ = compute_eer(
            df_no_a12['true_label'],
            df_no_a12['score']
        )

        a12_acc = (
            df[df['attack'] == 'A12']['pred'] == 1
        ).mean() * 100

        bon_acc = (
            df[df['attack'] == 'bonafide']['pred'] == 0
        ).mean() * 100

        print(
            f"{m:<18} | {eer_all:9.3f}%   | {eer_no_a12:9.3f}%   | "
            f"{dcf:8.4f} | {a12_acc:8.2f}% | {bon_acc:10.2f}%"
        )

    print("=" * 105)

    # Try Score Fusion with Z-score normalization
    print("\nEvaluating Multi-Model Score Fusion (Ensemble)...")

    z_scores = {}
    for m, df in dfs.items():
        s = df['score'].values
        z_scores[m] = (s - np.mean(s)) / (np.std(s) + 1e-6)

    y_true = dfs['attention']['true_label'].values
    attacks = dfs['attention']['attack'].values

    # Test combinations
    candidates = [
        ("Attention (95%) + Boundary (5%)", 0.95 * z_scores['attention'] + 0.05 * z_scores['attention_boundary']),
        ("Attention (Base) + Attention (Boundary)", 0.5 * z_scores['attention'] + 0.5 * z_scores['attention_boundary']),
        ("Attention (60%) + Boundary (40%)", 0.6 * z_scores['attention'] + 0.4 * z_scores['attention_boundary']),
        ("Attention (70%) + Boundary (30%)", 0.7 * z_scores['attention'] + 0.3 * z_scores['attention_boundary']),
        ("MFCC + Attention (Boundary)", 0.5 * z_scores['mfcc'] + 0.5 * z_scores['attention_boundary']),
        ("Attention + MFCC + Boundary", 0.4 * z_scores['attention'] + 0.3 * z_scores['mfcc'] + 0.3 * z_scores['attention_boundary']),
        ("Attention + Concat + Boundary", 0.4 * z_scores['attention'] + 0.3 * z_scores['concat'] + 0.3 * z_scores['attention_boundary']),
        ("Attention + MFCC (No Boundary)", 0.6 * z_scores['attention'] + 0.4 * z_scores['mfcc']),
    ]
    
    print(
        f"{'Ensemble Combo':<42} | {'Overall EER':<12} | "
        f"{'EER w/o A12':<12} | {'minDCF':<10} | {'A12 Acc':<10}"
    )
    print("-" * 100)

    for name, fused in candidates:
        eer_all, thresh = compute_eer(y_true, fused)
        dcf = min_dcf(y_true, fused)

        preds = (fused >= thresh).astype(int)

        mask_no_a12 = (attacks != 'A12')
        eer_no_a12, _ = compute_eer(
            y_true[mask_no_a12],
            fused[mask_no_a12]
        )

        mask_a12 = (attacks == 'A12')
        a12_acc = (preds[mask_a12] == 1).mean() * 100

        print(
            f"{name:<42} | {eer_all:9.3f}%   | "
            f"{eer_no_a12:9.3f}%   | {dcf:8.4f} | {a12_acc:8.2f}%"
        )

    print("=" * 100)

    return dfs


if __name__ == '__main__':
    dfs = analyze_models()