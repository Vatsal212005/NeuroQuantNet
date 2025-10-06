#!/usr/bin/env python3
"""
Build a heterogeneous graph (PyTorch Geometric) from:
- STRING edges: data/processed/string_edges.csv  (gene_a,gene_b,weight)
- GDSC IC50:    data/processed/gdsc_ic50.csv     (cell_line,drug_name,ln_ic50,auc,...)

Creates node types: gene, drug, cell
Edges:
  ('gene','interacts','gene')         with edge_weight in [0,1]
  ('cell','treated_with','drug')      with edge_ic50 (float), edge_auc (float)

Outputs:
  data/processed/hetero_graph.pkl

CLI:
  python scripts/build_graph.py \
    --string data/processed/string_edges.csv \
    --gdsc   data/processed/gdsc_ic50.csv \
    --out    data/processed/hetero_graph.pkl \
    --min_gene_degree 2 \
    --max_genes 0
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
from collections import Counter

import torch

# PyG imports
try:
    from torch_geometric.data import HeteroData
except Exception as e:
    print("❌ PyTorch Geometric not found. Install with:\n"
          "   pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv")
    raise

def select_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA available: {name}")
        return d
    else:
        print("⚠️ CUDA not detected. Using CPU.")
        return torch.device("cpu")

def load_string_edges(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    # Expect columns: gene_a, gene_b, weight
    need = {"gene_a", "gene_b", "weight"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"[STRING] Missing columns in {path_csv}: {miss}")
    # Ensure str
    df["gene_a"] = df["gene_a"].astype(str)
    df["gene_b"] = df["gene_b"].astype(str)
    df["weight"] = df["weight"].astype(float)
    return df

def load_gdsc(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    need_any = {"cell_line", "drug_name"}
    if not need_any.issubset(df.columns):
        raise ValueError(f"[GDSC] Expected columns 'cell_line','drug_name' not found in {path_csv}")
    # ln_ic50 & auc can have NaNs; drop rows that miss both
    if "ln_ic50" not in df.columns and "auc" not in df.columns:
        raise ValueError(f"[GDSC] Need at least one of ln_ic50 or auc in {path_csv}")
    df["cell_line"] = df["cell_line"].astype(str).str.strip()
    df["drug_name"] = df["drug_name"].astype(str).str.strip()
    # Keep rows with either ln_ic50 or auc
    keep = (~df.get("ln_ic50", pd.Series([np.nan]*len(df))).isna()) | (~df.get("auc", pd.Series([np.nan]*len(df))).isna())
    df = df[keep].copy()
    # Basic clean
    if "ln_ic50" in df.columns:
        df["ln_ic50"] = pd.to_numeric(df["ln_ic50"], errors="coerce")
    if "auc" in df.columns:
        df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    # Drop exact duplicates
    df = df.drop_duplicates(subset=["cell_line", "drug_name"], keep="first")
    return df

def prune_genes_by_degree(df_edges: pd.DataFrame, min_degree: int, max_genes: int) -> pd.DataFrame:
    # Compute degrees
    deg = Counter(df_edges["gene_a"].tolist() + df_edges["gene_b"].tolist())
    if min_degree > 0:
        keep_nodes = {g for g, d in deg.items() if d >= min_degree}
        df_edges = df_edges[df_edges["gene_a"].isin(keep_nodes) & df_edges["gene_b"].isin(keep_nodes)].copy()
        print(f"[STRING] Pruned by min_degree={min_degree}: nodes≈{len(keep_nodes):,}, edges={len(df_edges):,}")

    if max_genes and max_genes > 0 and len(deg) > max_genes:
        # Keep top-k by degree
        top_nodes = set([g for g, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:max_genes]])
        df_edges = df_edges[df_edges["gene_a"].isin(top_nodes) & df_edges["gene_b"].isin(top_nodes)].copy()
        print(f"[STRING] Limited to top {max_genes} genes by degree: edges={len(df_edges):,}")

    return df_edges

def build_indices(gene_edges: pd.DataFrame, gdsc: pd.DataFrame):
    genes = sorted(set(gene_edges["gene_a"]).union(set(gene_edges["gene_b"])))
    drugs = sorted(gdsc["drug_name"].unique().tolist())
    cells = sorted(gdsc["cell_line"].unique().tolist())

    gene2i = {g:i for i,g in enumerate(genes)}
    drug2i = {d:i for i,d in enumerate(drugs)}
    cell2i = {c:i for i,c in enumerate(cells)}

    print(f"[IDX] Genes: {len(genes):,} | Drugs: {len(drugs):,} | Cells: {len(cells):,}")
    return gene2i, drug2i, cell2i

def to_edge_index_undirected(df_edges: pd.DataFrame, gene2i: dict):
    u = df_edges["gene_a"].map(gene2i).to_numpy(dtype=np.int64)
    v = df_edges["gene_b"].map(gene2i).to_numpy(dtype=np.int64)
    # both directions for PyG (unless you plan to use to_undirected later)
    edge_index = np.vstack([np.concatenate([u, v]), np.concatenate([v, u])])  # [2, 2E]
    weight = df_edges["weight"].to_numpy(dtype=np.float32)
    weight = np.concatenate([weight, weight])  # mirror
    return torch.from_numpy(edge_index), torch.from_numpy(weight)

def to_edge_index_cell_drug(gdsc: pd.DataFrame, cell2i: dict, drug2i: dict):
    # Only rows that map
    gdsc = gdsc[gdsc["cell_line"].isin(cell2i.keys()) & gdsc["drug_name"].isin(drug2i.keys())].copy()
    ci = gdsc["cell_line"].map(cell2i).to_numpy(dtype=np.int64)
    di = gdsc["drug_name"].map(drug2i).to_numpy(dtype=np.int64)
    edge_index = torch.from_numpy(np.vstack([ci, di]))  # [2, E]
    # Edge attributes
    ln_ic50 = torch.tensor(gdsc["ln_ic50"].to_numpy(dtype=np.float32), dtype=torch.float32) if "ln_ic50" in gdsc.columns else None
    auc = torch.tensor(gdsc["auc"].to_numpy(dtype=np.float32), dtype=torch.float32) if "auc" in gdsc.columns else None
    return edge_index, ln_ic50, auc, len(gdsc)

def build_features(num_genes: int, num_drugs: int, num_cells: int, gene_edges: pd.DataFrame, gene2i: dict):
    # Gene features: simple degree (normalized); shape [num_genes, 1]
    deg = Counter(gene_edges["gene_a"].tolist() + gene_edges["gene_b"].tolist())
    gdeg = np.zeros((num_genes,), dtype=np.float32)
    for g, d in deg.items():
        gdeg[gene2i[g]] = d
    if gdeg.max() > 0:
        gdeg = gdeg / gdeg.max()
    x_gene = torch.from_numpy(gdeg.reshape(-1, 1))  # [N_g, 1]

    # Drug / Cell: no features yet → placeholder zeros (models can use Embeddings later)
    x_drug = torch.zeros((num_drugs, 1), dtype=torch.float32)
    x_cell = torch.zeros((num_cells, 1), dtype=torch.float32)
    return x_gene, x_drug, x_cell

def main(args):
    device = select_device()

    # Load
    df_s = load_string_edges(args.string)
    df_g = load_gdsc(args.gdsc)

    # Prune by degree / cap gene count (optional)
    df_s = prune_genes_by_degree(df_s, args.min_gene_degree, args.max_genes)

    # Build indices
    gene2i, drug2i, cell2i = build_indices(df_s, df_g)

    # Edge: gene-gene
    gg_index, gg_weight = to_edge_index_undirected(df_s, gene2i)
    # Edge: cell-drug
    cd_index, cd_ic50, cd_auc, n_cd = to_edge_index_cell_drug(df_g, cell2i, drug2i)

    # Features
    x_gene, x_drug, x_cell = build_features(
        num_genes=len(gene2i),
        num_drugs=len(drug2i),
        num_cells=len(cell2i),
        gene_edges=df_s,
        gene2i=gene2i,
    )

    # HeteroData
    data = HeteroData()

    data["gene"].num_nodes = len(gene2i)
    data["drug"].num_nodes = len(drug2i)
    data["cell"].num_nodes = len(cell2i)

    data["gene"].x = x_gene
    data["drug"].x = x_drug
    data["cell"].x = x_cell

    data[("gene", "interacts", "gene")].edge_index = gg_index
    data[("gene", "interacts", "gene")].edge_weight = gg_weight

    data[("cell", "treated_with", "drug")].edge_index = cd_index
    if cd_ic50 is not None:
        data[("cell", "treated_with", "drug")].edge_ln_ic50 = cd_ic50
    if cd_auc is not None:
        data[("cell", "treated_with", "drug")].edge_auc = cd_auc

    # Move to GPU for a quick smoke test (optional)
    try:
        data_cuda = data.to(device, non_blocking=True)
        print("✅ Graph tensors moved to:", device)
        # move back to CPU before saving (PyG saves CPU tensors)
        data = data_cuda.to("cpu")
    except Exception as e:
        print("⚠️ Could not move to CUDA (this is okay for saving).", e)

    # Ensure output dir
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(data, args.out)

    # Stats
    n_genes = data["gene"].num_nodes
    n_drugs = data["drug"].num_nodes
    n_cells = data["cell"].num_nodes
    n_gg = data[("gene","interacts","gene")].edge_index.size(1)
    n_cd = data[("cell","treated_with","drug")].edge_index.size(1)

    print("\n=== Hetero graph summary ===")
    print(f"Nodes: genes={n_genes:,} | drugs={n_drugs:,} | cells={n_cells:,}")
    print(f"Edges: gene-gene (directed count)={n_gg:,} | cell-drug={n_cd:,}")
    print(f"Saved to: {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--string", default="data/processed/string_edges.csv", help="Processed STRING CSV")
    ap.add_argument("--gdsc",   default="data/processed/gdsc_ic50.csv", help="Processed GDSC CSV")
    ap.add_argument("--out",    default="data/processed/hetero_graph.pkl", help="Output PyG graph path")
    ap.add_argument("--min_gene_degree", type=int, default=2, help="Drop genes with degree < this")
    ap.add_argument("--max_genes", type=int, default=0, help="Cap number of top-degree genes (0 = keep all)")
    args = ap.parse_args()
    main(args)
