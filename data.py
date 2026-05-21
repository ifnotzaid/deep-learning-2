"""
PTB-XL Dataset Loader

Loads the PTB-XL ECG dataset from PhysioNet and maps records to 5 diagnostic
superclasses for single-label classification:
    NORM (normal), MI (myocardial infarction), STTC (ST/T change),
    CD (conduction disturbance), HYP (hypertrophy).

Records with zero or multiple superclasses are filtered out so the task
becomes clean single-label multi-class classification.

Standard PTB-XL splits (recommended by Wagner et al.):
    - train: stratified folds 1-8
    - val:   stratified fold  9
    - test:  stratified fold 10
"""

import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import wfdb
except ImportError:
    raise ImportError("Install wfdb first:  pip install wfdb")


SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_TO_IDX = {c: i for i, c in enumerate(SUPERCLASSES)}


def _aggregate_superclasses(scp_codes, agg_df):
    """Map an ECG's SCP codes to its set of diagnostic superclasses."""
    out = set()
    for code in scp_codes.keys():
        if code in agg_df.index:
            sc = agg_df.loc[code, 'diagnostic_class']
            if isinstance(sc, str):
                out.add(sc)
    return list(out)


class PTBXLDataset(Dataset):
    """In-memory cached PTB-XL dataset.

    On first construction we read every WFDB record once, normalize, and
    save (signals, labels) as .npy files in `cache_dir`. Subsequent runs
    skip the slow wfdb path entirely. With cache_dir on Colab's local
    `/content` disk, epochs become an order of magnitude faster.
    """
    def __init__(self, data_root, sampling_rate=100, split='train',
                 cache_dir='/content/ptbxl_cache',
                 train_folds=range(1, 9), val_folds=(9,), test_folds=(10,)):
        self.split = split
        os.makedirs(cache_dir, exist_ok=True)
        sig_cache = os.path.join(cache_dir, f'{split}_signals_{sampling_rate}.npy')
        lab_cache = os.path.join(cache_dir, f'{split}_labels_{sampling_rate}.npy')

        if os.path.exists(sig_cache) and os.path.exists(lab_cache):
            self.signals = np.load(sig_cache)
            self.labels  = np.load(lab_cache)
            print(f"[PTBXL/{split}] loaded {len(self.labels)} cached records")
            return

        # ---- cache miss: build from WFDB ----
        df = pd.read_csv(os.path.join(data_root, 'ptbxl_database.csv'),
                         index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(ast.literal_eval)
        agg = pd.read_csv(os.path.join(data_root, 'scp_statements.csv'),
                          index_col=0)
        agg = agg[agg.diagnostic == 1]
        df['superclass'] = df.scp_codes.apply(
            lambda x: _aggregate_superclasses(x, agg))
        df = df[df.superclass.apply(len) == 1].copy()
        df['label'] = df.superclass.apply(lambda x: CLASS_TO_IDX[x[0]])

        if split == 'train':
            df = df[df.strat_fold.isin(list(train_folds))]
        elif split == 'val':
            df = df[df.strat_fold.isin(list(val_folds))]
        elif split == 'test':
            df = df[df.strat_fold.isin(list(test_folds))]
        else:
            raise ValueError(f"Unknown split: {split}")

        fn_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
        df = df.reset_index()

        signals, labels = [], []
        print(f"[PTBXL/{split}] caching {len(df)} records...")
        for i, row in df.iterrows():
            sig, _ = wfdb.rdsamp(os.path.join(data_root, row[fn_col]))
            sig = sig.T.astype(np.float32)                # (12, T)
            sig = (sig - sig.mean(axis=1, keepdims=True)) / \
                  (sig.std(axis=1, keepdims=True) + 1e-8)
            signals.append(sig)
            labels.append(int(row['label']))
            if (i + 1) % 500 == 0:
                print(f"  ...{i + 1}/{len(df)}")

        self.signals = np.stack(signals).astype(np.float32)
        self.labels  = np.asarray(labels, dtype=np.int64)
        np.save(sig_cache, self.signals)
        np.save(lab_cache, self.labels)
        print(f"[PTBXL/{split}] cached to {cache_dir}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.signals[idx]), int(self.labels[idx])


def get_dataloaders(data_root, sampling_rate=100, batch_size=64, num_workers=2,
                    cache_dir='/content/ptbxl_cache'):
    train_ds = PTBXLDataset(data_root, sampling_rate, 'train', cache_dir=cache_dir)
    val_ds   = PTBXLDataset(data_root, sampling_rate, 'val',   cache_dir=cache_dir)
    test_ds  = PTBXLDataset(data_root, sampling_rate, 'test',  cache_dir=cache_dir)
    common = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **common),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **common),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **common),
    )


def download_ptbxl(target_dir):
    """One-time download of PTB-XL 1.0.3 from PhysioNet (~2 GB).
    Call this once, store the extracted folder on Google Drive."""
    import zipfile
    import urllib.request

    url = ("https://physionet.org/static/published-projects/ptb-xl/"
           "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")
    zip_path = os.path.join(target_dir, "ptbxl.zip")
    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        print("Downloading PTB-XL (~2 GB)...")
        urllib.request.urlretrieve(url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(target_dir)
    print("Done. Set DATA_ROOT to the extracted folder.")
