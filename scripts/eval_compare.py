#!/usr/bin/env python3
"""
Evaluate and plot classical baseline vs hybrid (non-finetuned).

Usage (PowerShell one-liner):
  python scripts/eval_compare.py --graph data/processed/hetero_graph.pkl ^
    --baseline_ckpt checkpoints_ft_ready/baseline_best.pt ^
    --hybrid_ckpt checkpoints_hybrid/hybrid_best.pt ^
    --out_dir reports
"""

import os
import math
import argparse
import json
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt
import numpy as np
import csv

# ========== utils ==========
def seed_all(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def r2_score_t(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))

# ========== dataset ==========
class CellDrugEdgeDataset(Dataset):
    def __init__(self, edge_index, ln_ic50):
        self.edge_index = edge_index  # [2, E]
        self.labels = ln_ic50
    def __len__(self): return self.edge_index.size(1)
    def __getitem__(self, i):
        return (self.edge_index[0, i].item(),
                self.edge_index[1, i].item(),
                self.labels[i].item())

def drug_aware_split(edge_index, ln_ic50, val_ratio=0.1, test_ratio=0.1, seed=42):
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

# ========== models ==========
class BaselineModel(nn.Module):
    """
    Simple baseline used earlier: cell and drug embeddings -> MLP regressor.
    """
    def __init__(self, num_cells, num_drugs, emb_dim=128, hidden=256, dropout=0.1):
        super().__init__()
        self.cell_emb = nn.Embedding(num_cells, emb_dim)
        self.drug_emb = nn.Embedding(num_drugs, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim*2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # init
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, cell_idx, drug_idx):
        c = self.cell_emb(cell_idx)
        d = self.drug_emb(drug_idx)
        x = torch.cat([c, d], dim=-1)
        y = self.mlp(x).squeeze(-1)
        return y

def make_quantum_layer(n_qubits: int, q_depth: int):
    import pennylane as qml
    # Windows safe: prefer default.qubit
    try:
        dev = qml.device("default.qubit", wires=n_qubits)
        print("Using PennyLane device for eval: default.qubit")
    except Exception as e:
        raise RuntimeError(f"Failed to init PennyLane device: {e}")

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (q_depth, n_qubits)}
    return qml.qnn.TorchLayer(qnode, weight_shapes)

class HybridModel(nn.Module):
    """
    Matches train_hybrid_qgnn.py structure: embeddings -> Linear -> QLayer -> regressor
    """
    def __init__(self, num_cells, num_drugs, emb_dim=128, n_qubits=8, q_depth=3, hidden=256, p_drop=0.1):
        super().__init__()
        self.cell_emb = nn.Embedding(num_cells, emb_dim)
        self.drug_emb = nn.Embedding(num_drugs, emb_dim)
        self.pre_q = nn.Sequential(
            nn.Linear(emb_dim*2, n_qubits),
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
        # init
        for m in self.pre_q:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        for m in self.post:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, cell_idx, drug_idx):
        # keep QNode on CPU for stability, then bring back to device
        device = cell_idx.device
        c = self.cell_emb(cell_idx)
        d = self.drug_emb(drug_idx)
        x = torch.cat([c, d], dim=-1)
        x = self.pre_q(x)
        q_out = self.qlayer(x.cpu()).to(device)
        y = self.post(q_out).squeeze(-1)
        return y

# ========== eval helpers ==========
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, yhats = [], []
    for cell_i, drug_i, y in loader:
        cell_i = cell_i.to(device)
        drug_i = drug_i.to(device)
        y = y.to(device, dtype=torch.float32)
        pred = model(cell_i, drug_i)
        ys.append(y.detach().cpu())
        yhats.append(pred.detach().cpu())
    y_true = torch.cat(ys, dim=0)
    y_pred = torch.cat(yhats, dim=0)
    mae = float(torch.mean(torch.abs(y_pred - y_true)))
    rmse = float(torch.sqrt(torch.mean((y_pred - y_true) ** 2)))
    r2 = r2_score_t(y_true, y_pred)
    return y_true.numpy(), y_pred.numpy(), {"MAE": mae, "RMSE": rmse, "R2": r2}

def scatter_plot(y_true, y_pred, title, out_path):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=6, alpha=0.4)
    mn = min(float(np.min(y_true)), float(np.min(y_pred)))
    mx = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True ln(IC50)")
    plt.ylabel("Pred ln(IC50)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def residual_plot(y_true, y_pred, title, out_path):
    res = y_pred - y_true
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=60)
    plt.xlabel("Residual (pred - true)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def bar_plot(metrics_a, metrics_b, labels, out_path):
    # metrics_a and metrics_b dicts with MAE, RMSE, R2
    names = ["MAE", "RMSE", "R2"]
    a = [metrics_a[n] for n in names]
    b = [metrics_b[n] for n in names]
    x = np.arange(len(names))
    w = 0.35
    plt.figure(figsize=(6.5,4))
    plt.bar(x - w/2, a, width=w, label=labels[0])
    plt.bar(x + w/2, b, width=w, label=labels[1])
    plt.xticks(x, names)
    plt.ylabel("Score")
    plt.title("Model comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ========== main ==========
@dataclass
class Config:
    graph: str
    baseline_ckpt: str
    hybrid_ckpt: str
    out_dir: str = "reports"
    batch_size: int = 8192
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42
    # default sizes if checkpoint lacks cfg
    emb_dim: int = 128
    hidden: int = 256
    dropout: float = 0.10
    n_qubits: int = 8
    q_depth: int = 3

def main(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    # load graph
    data: HeteroData = torch.load(cfg.graph, map_location="cpu")
    cd = data[("cell", "treated_with", "drug")]
    if not hasattr(cd, "edge_ln_ic50"):
        raise RuntimeError("edge_ln_ic50 not found on ('cell','treated_with','drug') edges")
    edge_index = cd.edge_index
    ln_ic50 = cd.edge_ln_ic50

    num_cells = data["cell"].num_nodes
    num_drugs = data["drug"].num_nodes

    # split; prefer checkpoint cfg if present
    def read_cfg(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return ckpt.get("cfg", {}), ckpt.get("model", ckpt)

    base_cfg, base_state = read_cfg(cfg.baseline_ckpt)
    hy_cfg, hy_state = read_cfg(cfg.hybrid_ckpt)

    val_ratio = float(base_cfg.get("val_ratio", cfg.val_ratio))
    test_ratio = float(base_cfg.get("test_ratio", cfg.test_ratio))
    seed = int(base_cfg.get("seed", cfg.seed))
    seed_all(seed)

    train_idx, val_idx, test_idx = drug_aware_split(edge_index, ln_ic50, val_ratio, test_ratio, seed)
    test_loader = DataLoader(CellDrugEdgeDataset(edge_index[:, test_idx], ln_ic50[test_idx]),
                             batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eval device: {device}")

    # reconstruct baseline model
    emb_dim_b = int(base_cfg.get("emb_dim", cfg.emb_dim))
    hidden_b = int(base_cfg.get("hidden", cfg.hidden))
    dropout_b = float(base_cfg.get("dropout", cfg.dropout))
    baseline = BaselineModel(num_cells, num_drugs, emb_dim=emb_dim_b, hidden=hidden_b, dropout=dropout_b).to(device)
    missing, unexpected = baseline.load_state_dict(base_state if isinstance(base_state, dict) else base_state.state_dict(), strict=False)
    if missing or unexpected:
        print(f"Baseline load warning. Missing: {missing}, Unexpected: {unexpected}")

    # reconstruct hybrid model with sizes from its checkpoint cfg if present
    emb_dim_h = int(hy_cfg.get("emb_dim", cfg.emb_dim))
    n_qubits = int(hy_cfg.get("n_qubits", cfg.n_qubits))
    q_depth = int(hy_cfg.get("q_depth", cfg.q_depth))
    hidden_h = int(hy_cfg.get("hidden", cfg.hidden))
    dropout_h = float(hy_cfg.get("dropout", cfg.dropout))
    hybrid = HybridModel(num_cells, num_drugs,
                         emb_dim=emb_dim_h, n_qubits=n_qubits, q_depth=q_depth,
                         hidden=hidden_h, p_drop=dropout_h).to(device)
    missing_h, unexpected_h = hybrid.load_state_dict(hy_state if isinstance(hy_state, dict) else hy_state.state_dict(), strict=False)
    if missing_h or unexpected_h:
        print(f"Hybrid load warning. Missing: {missing_h}, Unexpected: {unexpected_h}")

    # evaluate
    y_true_b, y_pred_b, m_base = evaluate(baseline, test_loader, device)
    y_true_h, y_pred_h, m_hyb  = evaluate(hybrid, test_loader, device)

    # save metrics table
    metrics_path = os.path.join(cfg.out_dir, "compare_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "MAE", "RMSE", "R2"])
        w.writerow(["Baseline", f"{m_base['MAE']:.4f}", f"{m_base['RMSE']:.4f}", f"{m_base['R2']:.4f}"])
        w.writerow(["Hybrid",   f"{m_hyb['MAE']:.4f}",  f"{m_hyb['RMSE']:.4f}",  f"{m_hyb['R2']:.4f}"])
    print(f"Wrote metrics: {metrics_path}")

    # plots
    scatter_plot(y_true_b, y_pred_b, "Baseline: predicted vs true", os.path.join(cfg.out_dir, "scatter_baseline.png"))
    scatter_plot(y_true_h, y_pred_h, "Hybrid: predicted vs true",   os.path.join(cfg.out_dir, "scatter_hybrid.png"))
    residual_plot(y_true_b, y_pred_b, "Baseline residuals", os.path.join(cfg.out_dir, "residuals_baseline.png"))
    residual_plot(y_true_h, y_pred_h, "Hybrid residuals",   os.path.join(cfg.out_dir, "residuals_hybrid.png"))
    bar_plot(m_base, m_hyb, ["Baseline", "Hybrid"], os.path.join(cfg.out_dir, "metrics_bar.png"))

    print("Saved:")
    for name in ["scatter_baseline.png", "scatter_hybrid.png", "residuals_baseline.png", "residuals_hybrid.png", "metrics_bar.png"]:
        print(" -", os.path.join(cfg.out_dir, name))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, type=str)
    ap.add_argument("--baseline_ckpt", required=True, type=str)
    ap.add_argument("--hybrid_ckpt", required=True, type=str)
    ap.add_argument("--out_dir", default="reports", type=str)
    ap.add_argument("--bs", dest="batch_size", default=8192, type=int)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()
    return Config(
        graph=args.graph,
        baseline_ckpt=args.baseline_ckpt,
        hybrid_ckpt=args.hybrid_ckpt,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        seed=args.seed
    )

if __name__ == "__main__":
    main(parse_args())
