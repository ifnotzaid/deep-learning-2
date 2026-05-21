"""
Training and evaluation utilities for ECGNet on PTB-XL.

Use this as a module (import train_model, evaluate) or run directly to
train the full model.
"""

import os
import time
import json
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR

from ecg_net import ECGNet, compute_loss


def train_one_epoch(model, loader, optimizer, device,
                    ce_weight=1.0, recon_weight=0.1):
    model.train()
    tot_loss = tot_ce = tot_rec = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, rt, rp = model(x)
        loss, ce, rec = compute_loss(logits, y, rt, rp, ce_weight, recon_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        bs = x.size(0)
        tot_loss += loss.item() * bs
        tot_ce   += ce * bs
        tot_rec  += rec * bs
        n += bs
    return tot_loss / n, tot_ce / n, tot_rec / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits, _, _ = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    preds  = logits.argmax(dim=1).numpy()
    probs  = F.softmax(logits, dim=1).numpy()

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='macro')
    try:
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')
    return acc, f1, auc


def train_model(model, train_loader, val_loader, device,
                epochs=30, lr=1e-3, weight_decay=1e-4,
                ce_weight=1.0, recon_weight=0.1,
                ckpt_path='best_model.pt', verbose=True,
                patience=7):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = -1.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, ce, rec = train_one_epoch(
            model, train_loader, optimizer, device, ce_weight, recon_weight)
        val_acc, val_f1, val_auc = evaluate(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        history.append(dict(
            epoch=epoch, train_loss=train_loss, ce=ce, recon=rec,
            val_acc=val_acc, val_f1=val_f1, val_auc=val_auc, time=dt,
        ))

        if verbose:
            print(f"[{epoch:02d}/{epochs}] "
                  f"loss={train_loss:.4f} ce={ce:.4f} rec={rec:.4f} | "
                  f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_auc={val_auc:.4f} | "
                  f"{dt:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1,
            }, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  ⏸ Early stopping at epoch {epoch} (best val_f1={best_f1:.4f}, "
                      f"no improvement for {patience} epochs)")
                break

    # save full history alongside the checkpoint
    with open(ckpt_path.replace('.pt', '_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return history, best_f1


if __name__ == "__main__":
    from data import get_dataloaders

    DATA_ROOT  = os.environ.get('PTBXL_ROOT', '/content/drive/MyDrive/ptbxl')
    EPOCHS     = int(os.environ.get('EPOCHS', 30))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))
    LR         = float(os.environ.get('LR', 1e-3))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        DATA_ROOT, sampling_rate=100, batch_size=BATCH_SIZE)

    model = ECGNet(in_channels=12, num_classes=5).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    history, best_val_f1 = train_model(
        model, train_loader, val_loader, device,
        epochs=EPOCHS, lr=LR, ckpt_path='best_full.pt')

    ckpt = torch.load('best_full.pt')
    model.load_state_dict(ckpt['model_state'])
    test_acc, test_f1, test_auc = evaluate(model, test_loader, device)
    print(f"\nFinal test — acc={test_acc:.4f}  f1={test_f1:.4f}  auc={test_auc:.4f}")
