#!/usr/bin/env python3
"""
Train a baseline IC50 regressor on the hetero graph created by build_graph.py.

- Loads: data/processed/hetero_graph.pkl  (PyG HeteroData)
- Uses learned Embedding for 'cell' and 'drug' nodes
- Predicts ln(IC50) from concatenated [cell_embed || drug_embed]
- Drug-aware random split of ('cell','treated_with','drug') edges
- Mixed precision (AMP), gradient clipping, cosine LR schedule
- Reports MAE / RMSE / R^2; saves best model to checkpoints/

CLI:
  python scripts/train_baseline.py \
    --graph data/processed/hetero_graph.pkl \
    --epochs 20 \
    --bs 8192 \
    --emb_dim 128 \
    --lr 3e-4
"""

import os
import math
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from torch_geometric.data import HeteroData
from torch_geometric.utils import mask_to_index

# -----------------------------
# Utils
# -----------------------------
def select_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA available: {name}")
        return torch.device("cuda")
    print("⚠️ CUDA not detected. Using CPU.")
    return torch.device("cpu")

def seed_all(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))

# -----------------------------
# Dataset from cell-drug edges
# -----------------------------
class CellDrugEdgeDataset(Dataset):
    def __init__(self, edge_index, ln_ic50):
        super().__init__()
        self.edge_index = edge_index  # [2, E]
        self.labels = ln_ic50         # [E]
    def __len__(self):
        return self.edge_index.size(1)
    def __getitem__(self, idx):
        ci = self.edge_index[0, idx].item()
        di = self.edge_index[1, idx].item()
        y  = self.labels[idx].item()
        return ci, di, y

# -----------------------------
# Model
# -----------------------------
class IC50Regressor(nn.Module):
    def __init__(self, num_cells, num_drugs, emb_dim=128, hidden=256, p_drop=0.1):
        super().__init__()
        self.cell_emb = nn.Embedding(num_cells, emb_dim)
        self.drug_emb = nn.Embedding(num_drugs, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2*emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )
        # Kaiming-like init
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cell_idx, drug_idx):
        c = self.cell_emb(cell_idx)  # [B, D]
        d = self.drug_emb(drug_idx)  # [B, D]
        x = torch.cat([c, d], dim=-1)
        y = self.mlp(x).squeeze(-1)  # [B]
        return y

# -----------------------------
# Splitting utilities
# -----------------------------
def drug_aware_split(edge_index, ln_ic50, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Splits by edge instances but tries to avoid leakage by shuffling within each drug
    so that each split contains different (cell,drug) pairs.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    di = edge_index[1].cpu().numpy()  # drug indices
    E = edge_index.size(1)
    all_idx = []
    val_idx = []
    test_idx = []

    # Group edges by drug
    from collections import defaultdict
    by_drug = defaultdict(list)
    for idx in range(E):
        by_drug[di[idx]].append(idx)

    for _, idxs in by_drug.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(test_ratio * n))
        n_val  = int(round(val_ratio  * n))
        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test:n_test+n_val].tolist())
        all_idx.extend(idxs[n_test+n_val:].tolist())

    train_idx = torch.tensor(all_idx, dtype=torch.long)
    val_idx   = torch.tensor(val_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx, dtype=torch.long)

    return train_idx, val_idx, test_idx

# -----------------------------
# Training / Eval loops
# -----------------------------
@dataclass
class Config:
    graph_path: str
    out_dir: str
    epochs: int = 20
    batch_size: int = 8192
    lr: float = 3e-4
    emb_dim: int = 128
    hidden: int = 256
    dropout: float = 0.10
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    amp: bool = True
    compile: bool = True
    seed: int = 42

def run(cfg: Config):
    seed_all(cfg.seed)
    device = select_device()

    # Load graph
    data: HeteroData = torch.load(cfg.graph_path, map_location="cpu")
    cd = data[("cell", "treated_with", "drug")]
    if not hasattr(cd, "edge_ln_ic50"):
        raise RuntimeError("Expected 'edge_ln_ic50' on ('cell','treated_with','drug') edges. Re-run preprocess_gdsc/build_graph.")

    edge_index = cd.edge_index  # [2, E]
    ln_ic50 = cd.edge_ln_ic50   # [E]

    num_cells = data["cell"].num_nodes
    num_drugs = data["drug"].num_nodes

    # Splits
    train_idx, val_idx, test_idx = drug_aware_split(edge_index, ln_ic50, cfg.val_ratio, cfg.test_ratio, cfg.seed)

    def subset_loader(idxs, shuffle):
        ds = CellDrugEdgeDataset(edge_index[:, idxs], ln_ic50[idxs])
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=False, num_workers=0, pin_memory=True)

    train_loader = subset_loader(train_idx, shuffle=True)
    val_loader   = subset_loader(val_idx, shuffle=False)
    test_loader  = subset_loader(test_idx, shuffle=False)

    # Model
    model = IC50Regressor(num_cells=num_cells, num_drugs=num_drugs,
                          emb_dim=cfg.emb_dim, hidden=cfg.hidden, p_drop=cfg.dropout).to(device)
    if cfg.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("✅ torch.compile enabled")
        except Exception as e:
            print("⚠️ torch.compile failed (continuing without):", e)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Cosine LR schedule
    total_steps = cfg.epochs * math.ceil(len(train_loader))
    lr_lambda = lambda step: 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    os.makedirs(cfg.out_dir, exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(cfg.out_dir, "baseline_best.pt")

    def loop(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()

        total_loss, total_mae, total_mse, total_n = 0.0, 0.0, 0.0, 0
        all_true = []
        all_pred = []

        for (cell_i, drug_i, y) in loader:
            cell_i = cell_i.to(device, non_blocking=True)
            drug_i = drug_i.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True, dtype=torch.float32)

            with (torch.cuda.amp.autocast(enabled=cfg.amp) if train else torch.no_grad()):
                yhat = model(cell_i, drug_i)
                loss = F.smooth_l1_loss(yhat, y)  # Huber is robust

            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()

            # stats
            with torch.no_grad():
                n = y.numel()
                total_n += n
                total_loss += float(loss) * n
                mae = torch.mean(torch.abs(yhat - y))
                mse = torch.mean((yhat - y) ** 2)
                total_mae += float(mae) * n
                total_mse += float(mse) * n
                all_true.append(y.detach().cpu())
                all_pred.append(yhat.detach().cpu())

        import torch as _t
        y_true = _t.cat(all_true, dim=0)
        y_pred = _t.cat(all_pred, dim=0)
        r2 = r2_score(y_true, y_pred)

        avg_loss = total_loss / max(1, total_n)
        avg_mae  = total_mae  / max(1, total_n)
        avg_rmse = math.sqrt(total_mse / max(1, total_n))
        return avg_loss, avg_mae, avg_rmse, r2

    # Train
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_mae, tr_rmse, tr_r2 = loop(train_loader, train=True)
        va_loss, va_mae, va_rmse, va_r2 = loop(val_loader, train=False)
        print(f"[{epoch:03d}/{cfg.epochs}] "
              f"train: loss={tr_loss:.4f} MAE={tr_mae:.4f} RMSE={tr_rmse:.4f} R2={tr_r2:.4f} | "
              f"val: loss={va_loss:.4f} MAE={va_mae:.4f} RMSE={va_rmse:.4f} R2={va_r2:.4f}")

        if va_mae < best_val:
            best_val = va_mae
            torch.save({"model": model.state_dict(),
                        "cfg": vars(cfg),
                        "val_mae": best_val}, best_path)
            print(f"💾 Saved best to {best_path} (val MAE={best_val:.4f})")

    # Final test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_mae, te_rmse, te_r2 = loop(test_loader, train=False)
    print("\n=== Final Test ===")
    print(f"MAE={te_mae:.4f} | RMSE={te_rmse:.4f} | R2={te_r2:.4f}")
    print(f"Best checkpoint: {best_path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/processed/hetero_graph.pkl", type=str)
    ap.add_argument("--out_dir", default="checkpoints", type=str)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--bs", dest="batch_size", default=8192, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--emb_dim", default=128, type=int)
    ap.add_argument("--hidden", default=256, type=int)
    ap.add_argument("--dropout", default=0.10, type=float)
    ap.add_argument("--weight_decay", default=1e-4, type=float)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    return Config(
        graph_path=args.graph,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        grad_clip=1.0,
        val_ratio=0.10,
        test_ratio=0.10,
        amp=not args.no_amp,
        compile=not args.no_compile,
        seed=args.seed,
    )

if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
