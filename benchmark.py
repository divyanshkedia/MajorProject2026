import torch
from models import MFCCModel, LFCCModel, ConcatFusionModel, MLPFusionModel, AttentionFusionModel, measure_computational_cost

if __name__ == "__main__":
    # Automatically detect if a GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmarks on: {device.upper()}\n")
    print("==========================================")

    # Example shape: 1 batch, 400 frames, 64 Mel-bins
    input_size = (1, 400, 64) 

    models_to_test = [
        MFCCModel(),
        LFCCModel(),
        ConcatFusionModel(),
        MLPFusionModel(),
        AttentionFusionModel()
    ]

    for model in models_to_test:
        measure_computational_cost(model, input_size=input_size, device=device)
