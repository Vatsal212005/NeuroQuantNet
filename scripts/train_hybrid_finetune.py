#!/usr/bin/env python3
"""
Fine-tune a hybrid quantum head using embeddings from the trained baseline.

Loads:
  - Graph: data/processed/hetero_graph.pkl
  - Baseline checkpoint: checkpoints/baseline_best.pt

Copies baseline embeddings into the hybrid model, freezes them by default,
and trains only the quantum head and the final regressor.

Run:
  python scripts/train_hybrid_finetune.py ^
    --graph data/processed/hetero_graph.pkl ^
    --baseline_ckpt checkpoints/baseline_best.pt ^
    --out_dir checkpoints_hybrid_ft ^
    --epochs 20 ^
    --bs 8192 ^
    --emb_dim 256 ^
    --n_qubits 8 ^
    --q_depth 3 ^
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
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import HeteroData

# ---------- utils ----------
def select_device():
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("CUDA not detected. Using CPU.")
    return torch.device("cpu")

def seed_all(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.detach(); y_pred = y_pred.detach()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))

# ---------- dataset ----------
class CellDrugEdgeDataset(Dataset):
    def __init__(self, edge_index, ln_ic50):
        self.edge_index = edge_index  # [2, E]
        self.labels = ln_ic50         # [E]
    def __len__(self): return self.edge_index.size(1)
    def __getitem__(self, i):
        return (self.edge_index[0, i].item(),
                self.edge_index[1, i].item(),
                self.labels[i].item())

# ---------- quantum layer ----------
def make_quantum_layer(n_qubits: int, q_depth: int):
    import pennylane as qml
    try:
        dev = qml.device("default.qubit", wires=n_qubits)
        print("Using PennyLane device: default.qubit (safe mode)")
    except Exception as e:
        print(f"Failed to init default.qubit: {e}")
        raise

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(inputs, weights):
        # inputs shape: [n_qubits]
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # For BasicEntanglerLayers with q_depth repetitions
    weight_shapes = {"weights": (q_depth, n_qubits)}
    return qml.qnn.TorchLayer(qnode, weight_shapes)

# ---------- models ----------
class HybridHead(nn.Module):
    def __init__(self, emb_dim=128, n_qubits=8, q_depth=3, hidden=256, p_drop=0.1):
        super().__init__()
        self.pre_q = nn.Sequential(
            nn.Linear(emb_dim * 2, n_qubits),
            nn.LayerNorm(n_qubits),
            nn.Tanh(),
        )
        self.qlayer = make_quantum_layer(n_qubits=n_qubits, q_depth=q_depth)
        self.post = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )
        for m in self.pre_q:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        for m in self.post:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, cat_embed):
        # cat_embed on CUDA, quantum layer runs on CPU for stability then returns to CUDA
        x = self.pre_q(cat_embed)
        q_out = self.qlayer(x.cpu()).to(cat_embed.device)
        y = self.post(q_out).squeeze(-1)
        return y

class HybridFinetuneModel(nn.Module):
    def __init__(self, num_cells, num_drugs, emb_dim=128, n_qubits=8, q_depth=3, hidden=256, p_drop=0.1):
        super().__init__()
        self.cell_emb = nn.Embedding(num_cells, emb_dim)
        self.drug_emb = nn.Embedding(num_drugs, emb_dim)
        self.head = HybridHead(emb_dim=emb_dim, n_qubits=n_qubits, q_depth=q_depth, hidden=hidden, p_drop=p_drop)

    def load_from_baseline(self, baseline_state_dict):
        copied = 0
        for k, v in baseline_state_dict.items():
            if k.startswith("cell_emb.") and v.shape == self.cell_emb.state_dict()[k].shape:
                self.cell_emb.state_dict()[k].copy_(v); copied += 1
            if k.startswith("drug_emb.") and v.shape == self.drug_emb.state_dict()[k].shape:
                self.drug_emb.state_dict()[k].copy_(v); copied += 1
        print(f"Copied {copied} embedding tensors from baseline")

    def freeze_embeddings(self):
        for p in self.cell_emb.parameters(): p.requires_grad = False
        for p in self.drug_emb.parameters(): p.requires_grad = False
        print("Embeddings frozen")

    def forward(self, cell_idx, drug_idx):
        c = self.cell_emb(cell_idx)
        d = self.drug_emb(drug_idx)
        x = torch.cat([c, d], dim=-1)
        return self.head(x)

# ---------- splitting ----------
def drug_aware_split(edge_index, ln_ic50, val_ratio=0.1, test_ratio=0.1, seed=42):
    import numpy as np
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    di = edge_index[1].cpu().numpy()
    E = edge_index.size(1)
    train_idx, val_idx, test_idx = [], [], []
    by_drug = defaultdict(list)
    for idx in range(E):
        by_drug[di[idx]].append(idx)
    for _, idxs in by_drug.items():
        idxs = np.array(idxs); rng.shuffle(idxs)
        n = len(idxs); n_test = int(round(test_ratio*n)); n_val = int(round(val_ratio*n))
        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test:n_test+n_val].tolist())
        train_idx.extend(idxs[n_test+n_val:].tolist())
    return (torch.tensor(train_idx, dtype=torch.long),
            torch.tensor(val_idx, dtype=torch.long),
            torch.tensor(test_idx, dtype=torch.long))

# ---------- config ----------
@dataclass
class Config:
    graph: str
    baseline_ckpt: str
    out_dir: str = "checkpoints_hybrid_ft"
    epochs: int = 20
    batch_size: int = 8192
    lr: float = 3e-4
    emb_dim: int = 128
    n_qubits: int = 8
    q_depth: int = 3
    hidden: int = 256
    dropout: float = 0.10
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    unfreeze_embeddings: bool = False
    seed: int = 42
    safe_load: bool = False

# ---------- run ----------
def run(cfg: Config):
    seed_all(cfg.seed)
    device = select_device()

    # Load graph
    data: HeteroData = torch.load(cfg.graph, map_location="cpu")
    cd = data[("cell","treated_with","drug")]
    if not hasattr(cd, "edge_ln_ic50"):
        raise RuntimeError("Expected edge_ln_ic50 on ('cell','treated_with','drug') edges.")

    edge_index = cd.edge_index
    ln_ic50 = cd.edge_ln_ic50

    num_cells = data["cell"].num_nodes
    num_drugs = data["drug"].num_nodes

    # Splits
    train_idx, val_idx, test_idx = drug_aware_split(edge_index, ln_ic50,
                                                    cfg.val_ratio, cfg.test_ratio, cfg.seed)

    def make_loader(idxs, shuffle):
        ds = CellDrugEdgeDataset(edge_index[:, idxs], ln_ic50[idxs])
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)

    train_loader = make_loader(train_idx, True)
    val_loader   = make_loader(val_idx, False)
    test_loader  = make_loader(test_idx, False)

    # Model
    model = HybridFinetuneModel(num_cells, num_drugs,
                                emb_dim=cfg.emb_dim,
                                n_qubits=cfg.n_qubits,
                                q_depth=cfg.q_depth,
                                hidden=cfg.hidden,
                                p_drop=cfg.dropout).to(device)

    # Load baseline embeddings
    ckpt = torch.load(cfg.baseline_ckpt, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    if cfg.safe_load:
        print("⚠️  Safe-load enabled: ignoring missing/unexpected keys from baseline checkpoint")
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(filtered)
        model.load_state_dict(model_state)
    else:
        model.load_state_dict(state_dict, strict=False)

    # Freeze or unfreeze embeddings
    if not cfg.unfreeze_embeddings:
        model.freeze_embeddings()
    else:
        print("Embeddings will be fine-tuned")

    # Optimizer for trainable params only
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * max(1, math.ceil(len(train_loader)))
    lr_lambda = lambda step: 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    os.makedirs(cfg.out_dir, exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(cfg.out_dir, "hybrid_ft_best.pt")

    def run_epoch(loader, train: bool):
        if train: model.train()
        else: model.eval()

        total_loss = total_mae = total_mse = 0.0
        total_n = 0
        ys, yhats = [], []

        for cell_i, drug_i, y in loader:
            cell_i = cell_i.to(device, non_blocking=True)
            drug_i = drug_i.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True, dtype=torch.float32)

            with (torch.enable_grad() if train else torch.no_grad()):
                yhat = model(cell_i, drug_i)
                loss = F.smooth_l1_loss(yhat, y)

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                sched.step()

            n = y.numel()
            total_n += n
            total_loss += float(loss) * n
            mae = torch.mean(torch.abs(yhat - y))
            mse = torch.mean((yhat - y) ** 2)
            total_mae += float(mae) * n
            total_mse += float(mse) * n
            ys.append(y.detach().cpu()); yhats.append(yhat.detach().cpu())

        y_true = torch.cat(ys, dim=0)
        y_pred = torch.cat(yhats, dim=0)
        r2 = r2_score(y_true, y_pred)
        avg_loss = total_loss / max(1, total_n)
        avg_mae = total_mae / max(1, total_n)
        import math as _math
        avg_rmse = _math.sqrt(total_mse / max(1, total_n))
        return avg_loss, avg_mae, avg_rmse, r2

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_mae, tr_rmse, tr_r2 = run_epoch(train_loader, True)
        va_loss, va_mae, va_rmse, va_r2 = run_epoch(val_loader, False)
        print(f"[{epoch:03d}/{cfg.epochs}] "
              f"train: loss={tr_loss:.4f} MAE={tr_mae:.4f} RMSE={tr_rmse:.4f} R2={tr_r2:.4f} | "
              f"val: loss={va_loss:.4f} MAE={va_mae:.4f} RMSE={va_rmse:.4f} R2={va_r2:.4f}")

        if va_mae < best_val:
            best_val = va_mae
            torch.save({"model": model.state_dict(),
                        "cfg": vars(cfg),
                        "val_mae": best_val}, best_path)
            print(f"Saved best to {best_path} (val MAE={best_val:.4f})")

    ckpt_ft = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt_ft["model"])
    te_loss, te_mae, te_rmse, te_r2 = run_epoch(test_loader, False)
    print("\n=== Hybrid Fine-tune Final Test ===")
    print(f"MAE={te_mae:.4f} | RMSE={te_rmse:.4f} | R2={te_r2:.4f}")
    print(f"Best checkpoint: {best_path}")

# ---------- args ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, type=str)
    ap.add_argument("--baseline_ckpt", required=True, type=str)
    ap.add_argument("--safe_load", action="store_true", help="Ignore missing keys when loading baseline checkpoint")
    ap.add_argument("--out_dir", default="checkpoints_hybrid_ft", type=str)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--bs", dest="batch_size", default=8192, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--emb_dim", default=128, type=int)
    ap.add_argument("--n_qubits", default=8, type=int)
    ap.add_argument("--q_depth", default=3, type=int)
    ap.add_argument("--hidden", default=256, type=int)
    ap.add_argument("--dropout", default=0.10, type=float)
    ap.add_argument("--weight_decay", default=1e-4, type=float)
    ap.add_argument("--unfreeze_embeddings", action="store_true")
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    return Config(
        graph=args.graph,
        baseline_ckpt=args.baseline_ckpt,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_dim=args.emb_dim,
        n_qubits=args.n_qubits,
        q_depth=args.q_depth,
        hidden=args.hidden,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        grad_clip=1.0,
        val_ratio=0.10,
        test_ratio=0.10,
        unfreeze_embeddings=args.unfreeze_embeddings,
        seed=args.seed,
        safe_load=args.safe_load,
    )

if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
