import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import torchaudio
import torchaudio.transforms as T
import pandas as pd
import logging
import numpy as np
from scipy.fft import dct
from nnAudio.features import CQT
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==========================================
# 1. LOGGING SETUP
# ==========================================
logging.basicConfig(
    filename='extraction_pipeline.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==========================================
# 2. DIRECTORY CONFIGURATION
# ==========================================
TRAIN_AUDIO_DIR = Path(r"D:\flac_t\flac_t")
DEV_AUDIO_DIR = Path(r"D:\flac_d")

TRAIN_TSV_PATH = Path(r"D:\ASVspoof5.train.tsv")
DEV_TSV_PATH = Path(r"D:\ASVspoof5.dev.track_1.tsv")
OUTPUT_BASE = Path(r"D:\ASVspoof5_Features_New")

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 320
HOP_LENGTH = 160

# ==========================================
# 3. WORKER FUNCTION
# ==========================================
local_transforms = {}

N_CQCC = 60       # Number of CQCC coefficients (match n_mfcc and n_lfcc)
N_BINS_CQT = 84   # CQT bins (7 octaves * 12 bins per octave)
BINS_PER_OCTAVE = 12

def get_transforms():
    if "mfcc" not in local_transforms:
        local_transforms["mfcc"] = T.MFCC(
            sample_rate=SAMPLE_RATE, n_mfcc=60,
            melkwargs={"n_fft": N_FFT, "n_mels": 128, "hop_length": HOP_LENGTH, "win_length": WIN_LENGTH}
        )
        local_transforms["lfcc"] = T.LFCC(
            sample_rate=SAMPLE_RATE, n_filter=128, n_lfcc=60,
            speckwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "win_length": WIN_LENGTH}
        )
        local_transforms["cqt"] = CQT(
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            fmin=32.7,           # C1 note
            n_bins=N_BINS_CQT,
            bins_per_octave=BINS_PER_OCTAVE,
            output_format='Magnitude'
        )
    return local_transforms["mfcc"], local_transforms["lfcc"], local_transforms["cqt"]

def process_audio_wrapper(args):
    row, split_name = args
    file_stem = str(row[1]) 
    
    input_dir = TRAIN_AUDIO_DIR if split_name == "train" else DEV_AUDIO_DIR
    audio_path = input_dir / f"{file_stem}.flac"

    if not audio_path.exists():
        return False, file_stem, "File not found on disk."

    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

        mfcc_transform, lfcc_transform, cqt_transform = get_transforms()

        mfcc_feat = mfcc_transform(waveform).half()
        lfcc_feat = lfcc_transform(waveform).half()

        # CQCC: CQT magnitude -> log -> DCT along frequency axis
        cqt_mag = cqt_transform(waveform)          # (1, N_BINS_CQT, T)
        log_cqt = torch.log(cqt_mag.clamp(min=1e-8)).numpy()  # log-compress
        # Apply DCT along the frequency axis (axis=1) to get cepstral coefficients
        cqcc_np = dct(log_cqt, type=2, axis=1, norm='ortho')[:, :N_CQCC, :]  # keep first 60 coeffs
        cqcc_feat = torch.from_numpy(cqcc_np).half()  # (1, 60, T)

        sub_folder = file_stem[:2] 
        mfcc_out_dir = OUTPUT_BASE / split_name / "MFCC" / sub_folder
        lfcc_out_dir = OUTPUT_BASE / split_name / "LFCC" / sub_folder
        cqcc_out_dir = OUTPUT_BASE / split_name / "CQCC" / sub_folder
        
        mfcc_out_dir.mkdir(parents=True, exist_ok=True)
        lfcc_out_dir.mkdir(parents=True, exist_ok=True)
        cqcc_out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(mfcc_feat, mfcc_out_dir / f"{file_stem}.pt")
        torch.save(lfcc_feat, lfcc_out_dir / f"{file_stem}.pt")
        torch.save(cqcc_feat, cqcc_out_dir / f"{file_stem}.pt")

        return True, file_stem, "Success"

    except Exception as e:
        return False, file_stem, str(e)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def run_extraction(tsv_path, split_name):
    print(f"\n[{split_name.upper()}] Loading TSV...")
    
    # THE FIX: sep=r'\s+' mathematically splits by spaces OR tabs, completely avoiding the trap
    df = pd.read_csv(tsv_path, sep=r'\s+', header=None)
    
    args_list = [(tuple(row), split_name) for row in df.to_numpy()]
    total_files = len(args_list)
    
    print(f"[{split_name.upper()}] Running Sanity Check on 1st file...")
    sanity_success, sanity_stem, sanity_msg = process_audio_wrapper(args_list[0])
    
    if not sanity_success:
        print(f"\nCRITICAL ERROR on Sanity Check! The script cannot proceed.")
        print(f"File: {sanity_stem}")
        print(f"Error: {sanity_msg}")
        return
        
    print(f"[{split_name.upper()}] Sanity Check Passed! {sanity_stem}.pt was saved. Firing up CPU Cores...")

    processed_count = 0
    error_count = 0
    chunk_size = 100 
    
    with Pool(processes=cpu_count()) as pool:
        iterator = pool.imap_unordered(process_audio_wrapper, args_list, chunksize=chunk_size)
        
        for success, file_stem, msg in tqdm(iterator, total=total_files, desc=f"Processing {split_name}"):
            if success:
                processed_count += 1
            else:
                error_count += 1
                tqdm.write(f"[{split_name.upper()}] ERROR on {file_stem}: {msg}")

    print(f"\n[{split_name.upper()}] Completed. Saved: {processed_count} | Errors: {error_count}")

def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    if TRAIN_TSV_PATH.exists():
        run_extraction(TRAIN_TSV_PATH, "train")
    else:
        print(f"Error: Could not find {TRAIN_TSV_PATH}")

    if DEV_TSV_PATH.exists():
        run_extraction(DEV_TSV_PATH, "dev")
    else:
        print(f"Error: Could not find {DEV_TSV_PATH}")

if __name__ == '__main__':
    main()