import argparse
import time
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from thop import profile

from dataset import ASVspoofDataset, apply_splicing_augmentation
import models

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

def compute_min_dcf(y_true, y_scores, p_target=0.05, c_miss=1.0, c_fa=1.0):
    y_scores = np.nan_to_num(np.array(y_scores), nan=0.0, posinf=100.0, neginf=-100.0)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    return np.min(dcf)

def mixup_data(x, y, alpha=0.5, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index

def mixup_criterion(criterion, pred_a, pred_b, y_a, y_b, lam):
    return lam * criterion(pred_a, y_a) + (1 - lam) * criterion(pred_b, y_b)

TRAIN_TSV = r"D:\ASVspoof5.train.tsv"
DEV_TSV = r"D:\ASVspoof5.dev.track_1.tsv"
FEATURES_DIR = r"D:\ASVspoof5_Features_New"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss).clamp(min=1e-7, max=1.0)
        focal_loss = ((1.0 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='attention', choices=['lfcc', 'mfcc', 'concat', 'mlp', 'attention', 'waveform'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pool_type', type=str, default='boundary', choices=['stats', 'dual', 'boundary'],
                        help='Pooling type: boundary (multi-scale with temporal max & delta for A12), dual (ASP), or stats')
    parser.add_argument('--use_asp', action='store_true', help='Legacy flag: aliases pool_type to dual if specified')
    parser.add_argument('--splice_prob', type=float, default=0.10,
                        help='Probability of synthetic splicing augmentation on bonafide speech (recommended: 0.10)')
    parser.add_argument('--freeze_backbones', action='store_true',
                        help='Freeze spectral backbones when training waveform fusion model')
    args = parser.parse_args()

    if args.use_asp and args.pool_type == 'stats':
        args.pool_type = 'dual'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"🚀 Using Device: {device.type.upper()}")
    print(f"🧠 Selected Model: {args.model.upper()}")
    print(f"🔬 Pooling Mechanism: {args.pool_type.upper()}")
    print(f"✂️ Synthetic Splicing Augmentation Prob: {args.splice_prob * 100:.1f}%")

    load_mfcc = args.model in ['mfcc', 'concat', 'mlp', 'attention', 'waveform']
    load_lfcc = args.model in ['lfcc', 'concat', 'mlp', 'attention', 'waveform']

    print("\nLoading Datasets...")
    train_dataset = ASVspoofDataset(TRAIN_TSV, FEATURES_DIR, "train", 400, is_training=True, load_mfcc=load_mfcc, load_lfcc=load_lfcc, load_waveform=(args.model == "waveform"))
    dev_dataset = ASVspoofDataset(DEV_TSV, FEATURES_DIR, "dev", 400, is_training=False, load_mfcc=load_mfcc, load_lfcc=load_lfcc, load_waveform=(args.model == "waveform"))

    print("Balancing Train Batches (50% Bonafide, 50% Spoof balanced across training attacks)...")
    train_labels = [s["label"] for s in train_dataset.samples]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    
    attack_counts = {}
    for s in train_dataset.samples:
        if s["label"] == 1:
            atk = s["attack"].upper()
            attack_counts[atk] = attack_counts.get(atk, 0) + 1
    num_attacks = max(len(attack_counts), 1)

    sample_weights = []
    for s in train_dataset.samples:
        if s["label"] == 0:
            # 50% mass to bonafide
            sample_weights.append(0.5 / class_counts[0])
        else:
            # 50% mass distributed evenly across all training attacks (A01-A08)
            atk = s["attack"].upper()
            sample_weights.append(0.5 / (num_attacks * attack_counts.get(atk, 1)))
        
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, persistent_workers=True if args.num_workers > 0 else False)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True if args.num_workers > 0 else False)

    # Instantiate model with specified pool_type
    pool_choice = args.pool_type
    if args.model == 'lfcc':
        model = models.LFCCModel(pool_type=pool_choice).to(device)
    elif args.model == 'mfcc':
        model = models.MFCCModel(pool_type=pool_choice).to(device)
    elif args.model in ['concat', 'mlp', 'attention', 'waveform']:
        if args.model == 'concat':
            model = models.ConcatFusionModel(pool_type=pool_choice).to(device)
        elif args.model == 'mlp':
            model = models.MLPFusionModel(pool_type=pool_choice).to(device)
        elif args.model == 'attention':
            model = models.AttentionFusionModel(pool_type=pool_choice).to(device)
        elif args.model == 'waveform':
            model = models.WaveformFusionModel(pool_type=pool_choice, freeze_backbones=args.freeze_backbones).to(device)
            
        print(f"\n🔄 Attempting to load pre-trained backbones for {args.model.upper()} fusion...")
        suffix = f"_{pool_choice}" if pool_choice != 'stats' else ""
        mfcc_ckpt = CHECKPOINT_DIR / f"mfcc{suffix}_best.pth"
        lfcc_ckpt = CHECKPOINT_DIR / f"lfcc{suffix}_best.pth"
        # Fallback to standard best checkpoints if pool-specific doesn't exist
        if not mfcc_ckpt.exists():
            mfcc_ckpt = CHECKPOINT_DIR / "mfcc_best.pth"
        if not lfcc_ckpt.exists():
            lfcc_ckpt = CHECKPOINT_DIR / "lfcc_best.pth"
        
        if mfcc_ckpt.exists():
            print(f"Loading MFCC backbone from {mfcc_ckpt}")
            mfcc_state = torch.load(mfcc_ckpt, map_location=device, weights_only=True)
            mfcc_net_state = {k.replace('mfcc_net.', ''): v for k, v in mfcc_state.items() if k.startswith('mfcc_net.')}
            model.mfcc_net.load_state_dict(mfcc_net_state, strict=False)
            
        if lfcc_ckpt.exists():
            print(f"Loading LFCC backbone from {lfcc_ckpt}")
            lfcc_state = torch.load(lfcc_ckpt, map_location=device, weights_only=True)
            lfcc_net_state = {k.replace('lfcc_net.', ''): v for k, v in lfcc_state.items() if k.startswith('lfcc_net.')}
            model.lfcc_net.load_state_dict(lfcc_net_state, strict=False)

    criterion = FocalLoss(gamma=2.0)
    
    # Optimizer configuration
    if args.model == 'waveform' and not args.freeze_backbones:
        print("\n🔥 Training Waveform branch & Fusion layers with pretrained backbones...")
        for param in model.mfcc_net.parameters():
            param.requires_grad = True
        for param in model.lfcc_net.parameters():
            param.requires_grad = True

        mfcc_params = list(model.mfcc_net.parameters())
        lfcc_params = list(model.lfcc_net.parameters())
        mfcc_lfcc_ids = set(id(p) for p in mfcc_params + lfcc_params)
        other_params = [p for p in model.parameters() if id(p) not in mfcc_lfcc_ids and p.requires_grad]

        optimizer = optim.AdamW([
            {'params': mfcc_params, 'lr': args.lr * 0.1},
            {'params': lfcc_params, 'lr': args.lr * 0.1}, 
            {'params': other_params, 'lr': args.lr}
        ], lr=args.lr, weight_decay=5e-4)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    # Calculate FLOPs and Params
    dummy_mfcc = torch.randn(1, 1, 20, 400).to(device) if load_mfcc else torch.empty(0).to(device)
    dummy_lfcc = torch.randn(1, 1, 20, 400).to(device) if load_lfcc else torch.empty(0).to(device)
    flops, params = 0, 0
    try:
        if args.model == 'waveform':
            dummy_waveform = torch.randn(1, 64000).to(device)
            flops, params = profile(
                model,
                inputs=(dummy_mfcc, dummy_lfcc, dummy_waveform),
                verbose=False
            )
        else:
            flops, params = profile(
                model,
                inputs=(dummy_mfcc, dummy_lfcc),
                verbose=False
            )
        print(f"\n🧠 Model Specs: {params/1e6:.2f} M Params | {flops/1e9:.2f} GFLOPs")
    except Exception as e:
        print(f"\n⚠️ Could not calculate FLOPs: {e}")

    best_eer = float('inf')
    history_log = []

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", bar_format='{l_bar}{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

        optimizer.zero_grad(set_to_none=True)
        for i, (mfcc, lfcc, cqcc, waveform, labels, _, acs, attacks) in enumerate(train_bar):
            # Synthetic Splicing Augmentation (Audio CutPaste) to learn concatenation boundaries for A12
            if args.splice_prob > 0:
                mfcc, lfcc, labels, attacks, waveform = apply_splicing_augmentation(
                    mfcc, lfcc, labels, attacks=attacks,
                    waveforms=waveform if args.model == 'waveform' else None,
                    splice_prob=args.splice_prob
                )

            mfcc = mfcc.to(device, non_blocking=True)
            lfcc = lfcc.to(device, non_blocking=True)
            if args.model == 'waveform':
                waveform = waveform.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            model_labels = labels

            with torch.amp.autocast('cuda'):
                if args.model == 'waveform':
                    outputs = model(mfcc, lfcc, waveform, model_labels)
                else:
                    outputs = model(mfcc, lfcc, model_labels)
                loss = criterion(outputs, labels)

            scaled_loss = loss / args.accum_steps
            scaler.scale(scaled_loss).backward()

            if (i + 1) % args.accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        all_preds, all_labels, all_scores, all_acs, all_attacks = [], [], [], [], []
        total_inference_time = 0.0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            val_bar = tqdm(dev_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]  ", bar_format='{l_bar}{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

            for mfcc, lfcc, cqcc, waveform, labels, _, acs, attacks in val_bar:
                mfcc = mfcc.to(device, non_blocking=True)
                lfcc = lfcc.to(device, non_blocking=True)
                if args.model == 'waveform':
                    waveform = waveform.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                start_time = time.time()
                with torch.amp.autocast('cuda'):
                    if args.model == 'waveform':
                        outputs = model(mfcc, lfcc, waveform, label=None)
                    else:
                        outputs = model(mfcc, lfcc, label=None)
                total_inference_time += (time.time() - start_time)

                loss = criterion(outputs.float(), labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                probs = outputs[:, 1] - outputs[:, 0]
                probs = torch.nan_to_num(probs, nan=0.0)
                all_scores.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_acs.extend(acs)
                all_attacks.extend(attacks)

                val_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(dev_loader)
        
        current_eer, eer_thresh = compute_eer(all_labels, all_scores)
        
        all_preds = (np.array(all_scores) >= eer_thresh).astype(int)
        val_acc = 100 * np.mean(all_preds == np.array(all_labels))
        
        current_min_dcf = compute_min_dcf(all_labels, all_scores)

        latency_ms = (total_inference_time / total_val) * 1000
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

        print(f"\n📊 EPOCH {epoch+1} METRICS:")
        print(f"🔹 Train Acc: {train_acc:.2f}% | Train Loss: {avg_train_loss:.4f}")
        print(f"🔹 Val Acc:   {val_acc:.2f}% | Val Loss:   {avg_val_loss:.4f}")
        print(f"🎯 VAL EER:   {current_eer:.3f}%")
        print(f"🎯 min-DCF:   {current_min_dcf:.4f}")
        print(f"⏱️ Latency:   {latency_ms:.2f} ms/sample")
        print(f"💾 Peak VRAM: {peak_vram:.2f} MB")

        history_log.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'train_loss': avg_train_loss,
            'val_acc': val_acc,
            'val_loss': avg_val_loss,
            'val_eer': current_eer,
            'min_dcf': current_min_dcf
        })
        pool_suffix = f"_{args.pool_type}" if args.pool_type != 'stats' else ""
        model_save_name = f"{args.model}{pool_suffix}"
        pd.DataFrame(history_log).to_csv(CHECKPOINT_DIR / f"{model_save_name}_history.csv", index=False)

        print("\n🔍 Analyzing Degradation Robustness...")
        df_results = pd.DataFrame({'pred': all_preds, 'true': all_labels, 'ac': all_acs, 'attack': all_attacks})
        
        print("-" * 40)
        print("ACOUSTIC CONDITION (AC) BREAKDOWN")
        print("-" * 40)
        ac_stats = df_results.groupby('ac')[['pred', 'true']].apply(lambda x: pd.Series({'acc': (x['pred'] == x['true']).mean() * 100, 'count': len(x)})).reset_index()
        for _, row in ac_stats.iterrows():
            print(f"{row['ac']:<10} | Acc: {row['acc']:.1f}% | N={int(row['count'])}")
            
        print("-" * 40)
        print("ATTACK TYPE (A) BREAKDOWN")
        print("-" * 40)
        attack_stats = df_results.groupby('attack')[['pred', 'true']].apply(lambda x: pd.Series({'acc': (x['pred'] == x['true']).mean() * 100, 'count': len(x)})).reset_index()
        for _, row in attack_stats.iterrows():
            if row['attack'] != '-':
                print(f"{row['attack']:<10} | Acc: {row['acc']:.1f}% | N={int(row['count'])}")
        print("-" * 40)

        if current_eer < best_eer:
            print(f"\n⭐ NEW BEST MODEL! EER dropped from {best_eer:.3f}% to {current_eer:.3f}%")
            best_eer = current_eer
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"{model_save_name}_best.pth")
            
            df_scores = pd.DataFrame({'true_label': all_labels, 'score': all_scores, 'ac': all_acs, 'attack': all_attacks})
            df_scores.to_csv(CHECKPOINT_DIR / f"{model_save_name}_best_scores.csv", index=False)
            
            with open(CHECKPOINT_DIR / f"{model_save_name}_best_breakdown.txt", "w", encoding="utf-8") as f:
                f.write(f"Best EER: {current_eer:.3f}%\n")
                f.write(f"min-DCF: {current_min_dcf:.4f}\n\n")
                
                f.write("HARDWARE & PERFORMANCE METRICS\n")
                f.write(f"Parameters: {params/1e6:.2f} M\n")
                f.write(f"FLOPs: {flops/1e9:.2f} G\n")
                f.write(f"Latency: {latency_ms:.2f} ms/sample\n")
                f.write(f"Peak VRAM: {peak_vram:.2f} MB\n\n")

                f.write("ACOUSTIC CONDITION (AC) BREAKDOWN\n")
                f.write(ac_stats.to_string(index=False) + "\n\n")
                f.write("ATTACK TYPE (A) BREAKDOWN\n")
                f.write(attack_stats.to_string(index=False) + "\n")
        else:
            print(f"\n📉 EER did not improve (Best is {best_eer:.3f}%). Skipping save.\n")