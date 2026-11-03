import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import random
import torchaudio
import torchaudio.transforms as T


class ASVspoofDataset(Dataset):
    def __init__(self, tsv_path, features_dir, split_name, max_frames=400,
                 is_training=True, load_mfcc=False, load_lfcc=True, load_cqcc=False,
                 load_waveform=False):

        self.features_dir = Path(features_dir)
        self.max_frames = max_frames
        self.is_training = is_training
        self.load_mfcc = load_mfcc
        self.load_lfcc = load_lfcc
        self.load_cqcc = load_cqcc
        self.load_waveform = load_waveform
        self.split_name = split_name

        # Original audio directories from extract.py
        self.audio_dir = (
            Path(r"D:\flac_t\flac_t")
            if split_name == "train"
            else Path(r"D:\flac_d")
        )

        self.sample_rate = 16000

        df = pd.read_csv(tsv_path, sep=r'\s+', header=None)
        self.samples = []

        print(f"[{split_name.upper()}] Validating file paths...")

        for row in df.itertuples(index=False):
            file_stem = str(row[1])
            label_str = str(row[8]).lower()

            label = 0 if 'bonafide' in label_str else 1

            ac = str(row[6])
            attack = str(row[7])

            sub_folder = file_stem[:2]

            mfcc_path = (
                self.features_dir
                / split_name
                / "MFCC"
                / sub_folder
                / f"{file_stem}.pt"
            )

            lfcc_path = (
                self.features_dir
                / split_name
                / "LFCC"
                / sub_folder
                / f"{file_stem}.pt"
            )

            cqcc_path = (
                self.features_dir
                / split_name
                / "CQCC"
                / sub_folder
                / f"{file_stem}.pt"
            )

            waveform_path = (
                self.audio_dir
                / f"{file_stem}.flac"
            )

            self.samples.append({
                "mfcc_path": mfcc_path,
                "lfcc_path": lfcc_path,
                "cqcc_path": cqcc_path,
                "waveform_path": waveform_path,
                "label": label,
                "file_stem": file_stem,
                "ac": ac,
                "attack": attack
            })

        print(
            f"[{split_name.upper()}] Ready! "
            f"Found {len(self.samples)} valid audio samples."
        )

        del df

        # Augmentation
        self.freq_mask = T.FrequencyMasking(freq_mask_param=12)
        self.time_mask = T.TimeMasking(time_mask_param=80)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_feature(path):
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True
        ).float()

    @staticmethod
    def _load_waveform(path, target_sr=16000):
        waveform, sr = torchaudio.load(path)

        # Convert stereo -> mono
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != target_sr:
            waveform = T.Resample(
                orig_freq=sr,
                new_freq=target_sr
            )(waveform)

        waveform = waveform.squeeze(0).float()

        return waveform

    @staticmethod
    def _fix_waveform_length(waveform, target_length=64000, is_training=False):
        current_length = waveform.size(0)

        if current_length > target_length:

            if is_training:
                start = random.randint(
                    0,
                    current_length - target_length
                )
            else:
                start = (
                    current_length - target_length
                ) // 2

            waveform = waveform[
                start:start + target_length
            ]

        elif current_length < target_length:

            repeats = (
                target_length // current_length
            ) + 1

            waveform = waveform.repeat(
                repeats
            )[:target_length]

        return waveform

    def __getitem__(self, idx):

        while True:
            try:
                sample = self.samples[idx]

                lfcc_feat = (
                    self._load_feature(
                        sample["lfcc_path"]
                    )
                    if self.load_lfcc
                    else torch.empty(0)
                )

                mfcc_feat = (
                    self._load_feature(
                        sample["mfcc_path"]
                    )
                    if self.load_mfcc
                    else torch.empty(0)
                )

                cqcc_feat = (
                    self._load_feature(
                        sample["cqcc_path"]
                    )
                    if self.load_cqcc
                    else torch.empty(0)
                )

                waveform = (
                    self._load_waveform(
                        sample["waveform_path"]
                    )
                    if self.load_waveform
                    else torch.empty(0)
                )

                break

            except Exception:
                import numpy as np
                idx = np.random.randint(
                    0,
                    len(self.samples)
                )

        ref_feat = (
            lfcc_feat
            if self.load_lfcc
            else (
                mfcc_feat
                if self.load_mfcc
                else cqcc_feat
            )
        )

        c, f, t = ref_feat.shape

        if t > self.max_frames:

            if self.is_training:
                start = random.randint(
                    0,
                    t - self.max_frames
                )
            else:
                start = (
                    t - self.max_frames
                ) // 2

            if self.load_lfcc:
                lfcc_feat = lfcc_feat[
                    :, :, start:start + self.max_frames
                ]

            if self.load_mfcc:
                mfcc_feat = mfcc_feat[
                    :, :, start:start + self.max_frames
                ]

            if self.load_cqcc:
                cqcc_feat = cqcc_feat[
                    :, :, start:start + self.max_frames
                ]

        elif t < self.max_frames:

            repeats = (
                self.max_frames // t
            ) + 1

            if self.load_lfcc:
                lfcc_feat = lfcc_feat.repeat(
                    1, 1, repeats
                )[:, :, :self.max_frames]

            if self.load_mfcc:
                mfcc_feat = mfcc_feat.repeat(
                    1, 1, repeats
                )[:, :, :self.max_frames]

            if self.load_cqcc:
                cqcc_feat = cqcc_feat.repeat(
                    1, 1, repeats
                )[:, :, :self.max_frames]

        # CMVN
        if self.load_lfcc:

            mean = lfcc_feat.mean(
                dim=2,
                keepdim=True
            )

            std = lfcc_feat.std(
                dim=2,
                keepdim=True
            )

            lfcc_feat = (
                lfcc_feat - mean
            ) / (
                std + 1e-6
            )

        if self.load_mfcc:

            mean = mfcc_feat.mean(
                dim=2,
                keepdim=True
            )

            std = mfcc_feat.std(
                dim=2,
                keepdim=True
            )

            mfcc_feat = (
                mfcc_feat - mean
            ) / (
                std + 1e-6
            )

        if self.load_cqcc:

            mean = cqcc_feat.mean(
                dim=2,
                keepdim=True
            )

            std = cqcc_feat.std(
                dim=2,
                keepdim=True
            )

            cqcc_feat = (
                cqcc_feat - mean
            ) / (
                std + 1e-6
            )

        # Keep existing behavior
        if self.is_training:
            pass

        # 4 seconds at 16 kHz
        if self.load_waveform:
            waveform = self._fix_waveform_length(
                waveform,
                target_length=64000,
                is_training=self.is_training
            )

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return (
            mfcc_feat,
            lfcc_feat,
            cqcc_feat,
            waveform,
            label,
            sample["file_stem"],
            sample["ac"],
            sample["attack"]
        )


def apply_splicing_augmentation(mfcc, lfcc, labels, attacks=None, waveforms=None, splice_prob=0.10):
    """
    On-the-fly Synthetic Splicing Augmentation (Audio CutPaste) to counter
    unit-selection spoofing attacks (such as ASVspoof 5 Attack A12).

    Selects bona fide utterances in the batch and synthetically splices segments
    from other bona fide utterances. Flips the target label to 1 (spoof).
    Forces the network to detect concatenation boundaries and localized spectral jumps
    rather than only relying on global neural vocoder artifacts.
    """
    if splice_prob <= 0 or mfcc.size(0) < 2:
        return mfcc, lfcc, labels, attacks, waveforms

    # Indices of bonafide samples in the batch (label == 0)
    bonafide_indices = (labels == 0).nonzero(as_tuple=True)[0]
    if len(bonafide_indices) < 2:
        return mfcc, lfcc, labels, attacks, waveforms

    mfcc_aug = mfcc.clone()
    lfcc_aug = lfcc.clone()
    labels_aug = labels.clone()
    waveforms_aug = waveforms.clone() if waveforms is not None and waveforms.numel() > 0 else None
    attacks_aug = list(attacks) if attacks is not None else None

    # For each bonafide sample, with probability splice_prob, splice from another bonafide
    for idx in bonafide_indices:
        if torch.rand(1).item() > splice_prob:
            continue

        other_indices = bonafide_indices[bonafide_indices != idx]
        if len(other_indices) == 0:
            continue
        donor_idx = other_indices[torch.randint(len(other_indices), (1,)).item()]

        # Determine splice region along time axis T (dim=-1)
        T_dim = mfcc.size(-1) if mfcc.numel() > 0 else lfcc.size(-1)
        # Unit length between 30 and 120 frames (approx 300ms to 1.2s)
        unit_len = torch.randint(30, min(120, max(31, T_dim // 2)), (1,)).item()
        start = torch.randint(5, max(6, T_dim - unit_len - 5), (1,)).item()
        donor_start = torch.randint(0, max(1, T_dim - unit_len), (1,)).item()

        # Splice feature frames
        if mfcc_aug.numel() > 0:
            mfcc_aug[idx, ..., start:start + unit_len] = mfcc[donor_idx, ..., donor_start:donor_start + unit_len]
        if lfcc_aug.numel() > 0:
            lfcc_aug[idx, ..., start:start + unit_len] = lfcc[donor_idx, ..., donor_start:donor_start + unit_len]

        # Splice waveform if present (160 samples per frame)
        if waveforms_aug is not None:
            w_start = start * 160
            w_len = unit_len * 160
            w_donor_start = donor_start * 160
            waveforms_aug[idx, ..., w_start:w_start + w_len] = waveforms[donor_idx, ..., w_donor_start:w_donor_start + w_len]

        # Flip label to spoof (1)
        labels_aug[idx] = 1
        if attacks_aug is not None:
            attacks_aug[idx] = "SPLICED_SYNTHETIC"

    return mfcc_aug, lfcc_aug, labels_aug, attacks_aug, waveforms_aug