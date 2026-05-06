
from __future__ import print_function, division

import argparse

import torch
import torch.optim as optim

import torchvision

from model import IDCM_NN
from train_model import train_model
from load_data import get_loader
from evaluate import fx_calc_map_multilabel
from to_seed import to_seed
from losses import FARLoss
from lr_scheduler import CosineAnnealingRestartLR

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


# =========================
# Args
# =========================
parser = argparse.ArgumentParser(description='DFH minimal ablation runner')

parser.add_argument("--dataset", type=str, default="INRIA-Websearch")
parser.add_argument("--data_path", type=str, default="")
parser.add_argument("--seed", type=int, default=10)

parser.add_argument("--alpha", type=float, default=0.1) # Xmedia 0.5
parser.add_argument("--MAX_EPOCH", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--bit", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)

parser.add_argument("--noisy_ratio", type=float, default=0.8) #0.0 0.2 0.4 0.6 0.8 1.0
parser.add_argument("--noise_mode", type=str, default='sym')

parser.add_argument("--loss_type", type=str, default='FARLoss')
parser.add_argument("--tp", type=int, default=3)

parser.add_argument("--q", type=float, default=0.01)
parser.add_argument("--tau", type=float, default=0.7)

parser.add_argument("--GPU", type=int, default=0)

parser.add_argument("--comp_topk", type=int, default=4, help="competitor top-k for necessity/credibility")

args = parser.parse_args()
print(args)

if __name__ == '__main__':
    device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
    to_seed(args.seed)

    print('...Data loading is beginning...')
    data_path = args.data_path.strip() or None
    data_loader, input_data_par = get_loader(
        args.dataset, args.batch_size, args.noisy_ratio, args.noise_mode, data_path=data_path
    )
    print('...Data loading is completed...')

    args.data_class = input_data_par['num_class']
    args.warmup_epochs = args.tp

    model_ft = IDCM_NN(
        img_input_dim=input_data_par['img_dim'],
        text_input_dim=input_data_par['text_dim'],
        output_dim=args.bit,
        num_class=input_data_par['num_class'],
        member_tau=0.5,
        member_gamma=2.0
    ).to(device)

    optimizer = optim.Adam(model_ft.parameters(), lr=args.lr)

    scheduler = CosineAnnealingRestartLR(
        optimizer,
        periods=[50, 50, 50],
        restart_weights=[1.0, 0.7, 0.5],
        eta_min=1e-7
    )

    if args.loss_type != 'FARLoss':
        raise ValueError(f"Unsupported loss_type: {args.loss_type}")

    criterion = FARLoss(
        num_classes=input_data_par['num_class'],
        comp_topk=args.comp_topk
    ).to(device)
    print(f'Using FARLoss (args.comp_topk={args.comp_topk})')
    print("[DBG] criterion.comp_topk =", getattr(criterion, "comp_topk", None), flush=True)

    print('...Training is beginning...')
    model_ft, _ = train_model(model_ft, data_loader, criterion, optimizer, args, scheduler)
    print('...Training finished...')

    # =========================
    # Test
    # =========================
    print('...Evaluation on testing data...')
    with torch.no_grad():
        v1, v2 = model_ft(
            torch.tensor(input_data_par['img_test']).to(device),
            torch.tensor(input_data_par['text_test']).to(device)
        )
        h1 = v1.sign().detach().cpu().numpy()
        h2 = v2.sign().detach().cpu().numpy()

    label = input_data_par['label_test']

    img_to_txt = float(fx_calc_map_multilabel(h1, h2, label, metric='hamming'))
    txt_to_img = float(fx_calc_map_multilabel(h2, h1, label, metric='hamming'))
    avg_map = float((img_to_txt + txt_to_img) / 2.0)

    print(f'[TEST] I2T={img_to_txt:.6f}  T2I={txt_to_img:.6f}  MEAN={avg_map:.6f}')
