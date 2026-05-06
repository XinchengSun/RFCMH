# RFCMH

Reference implementation for Robust Fuzzy Cross-Modal Hashing (RFCMH).

## Quick Start

```powershell
Set-Location "D:\桌面\RFCMH"

python main.py `
  --dataset INRIA-Websearch `
  --data_path "D:\path\to\INRIA-Websearch.mat" `
  --noisy_ratio 0.8 `
  --noise_mode sym `
  --bit 128 `
  --batch_size 64 `
  --MAX_EPOCH 150 `
  --lr 0.0001 `
  --seed 10 `
  --loss_type FARLoss `
  --tp 3 `
  --q 0.01 `
  --tau 0.7 `
  --comp_topk 4 `
  --GPU 0
```

## Files

- `main.py`: experiment entry point.
- `model.py`: image/text hash encoders and fixed category prototype matrix.
- `losses.py`: FAR loss, intra-instance alignment, and inter-instance alignment.
- `train_model.py`: training and validation loop.
- `load_data.py`: dataset loading and noisy-label generation.
