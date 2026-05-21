"""
Ablation Study Runner

Trains the full ECGNet plus 5 leave-one-block-out variants, evaluates each
on the held-out test set, and writes a results CSV.

Run after data.py / train.py / ecg_net.py are all in the same folder:
    python ablation.py
"""

import os
import torch
import pandas as pd

from ecg_net import ECGNet
from train  import train_model, evaluate
from data   import get_dataloaders


ABLATIONS = [
    {'name': 'full',         'flags': {}},
    {'name': 'no_cnn',       'flags': {'use_cnn':       False}},
    {'name': 'no_ae',        'flags': {'use_ae':        False}},
    {'name': 'no_bilstm',    'flags': {'use_bilstm':    False}},
    {'name': 'no_gru',       'flags': {'use_gru':       False}},
    {'name': 'no_attention', 'flags': {'use_attention': False}},
]


def run_ablations(data_root, output_dir='./ablation_results',
                  epochs=20, batch_size=64, lr=1e-3, sampling_rate=100):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root, sampling_rate, batch_size)

    rows = []
    for cfg in ABLATIONS:
        name  = cfg['name']
        flags = cfg['flags']
        print(f"\n{'='*60}\n Ablation: {name}  flags={flags}\n{'='*60}")

        model = ECGNet(in_channels=12, num_classes=5, **flags).to(device)
        ckpt  = os.path.join(output_dir, f'{name}.pt')

        _, best_val_f1 = train_model(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=lr, ckpt_path=ckpt, verbose=True)

        # reload best checkpoint for fair test eval
        state = torch.load(ckpt)
        model.load_state_dict(state['model_state'])
        test_acc, test_f1, test_auc = evaluate(model, test_loader, device)

        rows.append({
            'ablation':  name,
            'val_f1':    best_val_f1,
            'test_acc':  test_acc,
            'test_f1':   test_f1,
            'test_auc':  test_auc,
            'params':    sum(p.numel() for p in model.parameters()),
        })

        # persist after every variant so a crash doesn't lose everything
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, 'results.csv'), index=False)

    print("\n=== Ablation Results ===")
    print(pd.DataFrame(rows).to_string(index=False))
    return rows


if __name__ == "__main__":
    DATA_ROOT = os.environ.get('PTBXL_ROOT', '/content/drive/MyDrive/ptbxl')
    EPOCHS    = int(os.environ.get('EPOCHS', 20))
    run_ablations(DATA_ROOT, epochs=EPOCHS)
