# NeuroQuantNet: Hybrid Quantum Classical Graph Learning for Drug Sensitivity Prediction

This repository contains **NeuroQuantNet**, a hybrid quantum classical pipeline that integrates **Graph Neural Network style embeddings** with a **parameterized quantum circuit head** for **drug sensitivity prediction**. The system operates on a heterogeneous biomedical graph constructed from **STRING v12** protein protein interactions and **GDSC** drug response measurements, and evaluates prediction of ln(IC50) for cell, drug pairs.

The project provides three trained configurations:
1. **Classical baseline**, embeddings for cells and drugs, followed by a compact MLP regressor.  
2. **Hybrid model**, classical embeddings trained jointly with a quantum head that replaces the classical regressor.  

No future scope section is included, by request.

---

## Highlights

- End to end data processing, from raw STRING and GDSC files to a unified heterogeneous graph.  
- Classical baseline with strong accuracy and simple architecture.  
- Hybrid model implemented with **PennyLane** and trained through **PyTorch**.  
- Drug aware splitting to avoid optimistic evaluation.  
- Reproducible training commands, Windows friendly instructions, and evaluation plots.

---

## Results

All runs use the same split and evaluation protocol.

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Classical baseline | 0.786 | 0.93 | 1.255 |
| Hybrid model | 0.750 | 0.95 | 1.300 |

The hybrid model is close to the classical baseline while using a compact quantum head. Absolute values can vary slightly with seeds and pruning thresholds.

---

## Repository Structure

```
NeuroQuantNet/
├── data/
│   ├── raw/                       # Downloaded inputs
│   │   ├── 9606.protein.links.v12.0.txt.gz
│   │   └── GDSC2_fitted_dose_response_27Oct23.xlsx
│   └── processed/
│       ├── string_edges.csv       # Cleaned STRING edges
│       ├── gdsc_ic50.csv          # Cleaned GDSC response table
│       └── hetero_graph.pkl       # Heterogeneous graph for training
├── scripts/
│   ├── preprocess_string.py       # STRING cleaner
│   ├── preprocess_gdsc.py         # GDSC cleaner
│   ├── build_hetero_graph.py      # Graph assembly and indexing
│   ├── train_baseline.py          # Classical baseline trainer
│   ├── train_hybrid_qgnn.py       # Hybrid model, trained from scratch
│   ├── train_hybrid_finetune.py   # Hybrid model, fine tune on baseline
│   └── eval_compare.py            # Evaluation and plots
├── checkpoints/                   # Saved baseline checkpoints
├── checkpoints_hybrid/            # Saved hybrid checkpoints
├── checkpoints_hybrid_ft/         # Saved hybrid fine tuned checkpoints
├── reports/                       # Metrics tables and plots
└── plots/                         # Standalone comparison plots
```

> Note: if your local script names differ, follow the usage patterns below. Each step can run independently.

---

## Environment

**Python**: 3.10 to 3.12 tested  
**CUDA**: optional, used by PyTorch. Quantum simulation generally runs on CPU for stability.

Install core dependencies:

```bash
pip install -r requirements.txt
```

Minimal `requirements.txt` content that matches the training runs:

```
torch==2.5.1+cu121; platform_system!="Darwin"
torch==2.5.1; platform_system=="Darwin"
torch-geometric==2.6.1
pennylane==0.36.0
numpy>=1.26
pandas>=2.0
scipy>=1.11
matplotlib>=3.8
tqdm>=4.66
openpyxl>=3.1
```

Notes, Windows users may not need to install `torch-scatter`, `torch-sparse`, `torch-cluster`, or `torch-spline-conv` for the parts of PyTorch Geometric used here. If a package requires build tools, either skip it or install Microsoft C++ Build Tools and retry. The core pipeline works with the modules listed above.

---

## Data

Place the following files in `data/raw/`:

- `9606.protein.links.v12.0.txt.gz` from STRING v12, species 9606.  
- `GDSC2_fitted_dose_response_27Oct23.xlsx` from GDSC.

If you only have the uncompressed STRING file, you can recompress it so the scripts accept a `.gz` path.

```python
# one-off helper
import gzip, shutil
src = "data/raw/9606.protein.links.v12.0.txt"
dst = "data/raw/9606.protein.links.v12.0.txt.gz"
with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
    shutil.copyfileobj(fi, fo)
print("wrote", dst)
```

---

## Preprocessing

**1) STRING**

```bash
python scripts/preprocess_string.py \
  --links data/raw/9606.protein.links.v12.0.txt.gz \
  --out   data/processed/string_edges.csv \
  --min_score 400
```

The script filters edges by `combined_score`, writes a CSV, and prints counts for nodes and edges.

**2) GDSC**

```bash
python scripts/preprocess_gdsc.py \
  --in  data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx \
  --out data/processed/gdsc_ic50.csv
```

This extracts drug, cell, ln(IC50) and normalizes identifiers to consistent indices.

---

## Graph Assembly

Build a compact heterogeneous graph that includes, at minimum, cell nodes, drug nodes, and cell treated_with drug edges labeled by ln(IC50). The script also prunes the gene subgraph to control density and keeps index maps for cells and drugs.

```bash
python scripts/build_hetero_graph.py \
  --string_csv data/processed/string_edges.csv \
  --gdsc_csv   data/processed/gdsc_ic50.csv \
  --out        data/processed/hetero_graph.pkl \
  --min_gene_degree 2 \
  --max_genes 8000
```

Typical console output contains counts, for example, genes retained, edges after pruning, and total cell drug pairs.

---

## Training, Classical Baseline

The baseline uses embeddings for cells and drugs, followed by a compact MLP regressor. On some Windows configurations, `torch.compile` can trigger Triton installation errors, so use the `--no_compile` flag if needed.

```bash
python scripts/train_baseline.py \
  --graph data/processed/hetero_graph.pkl \
  --out_dir checkpoints_ft_ready \
  --no_compile
```

Expected final metrics, R² about 0.79, MAE about 0.93, RMSE about 1.255. The best checkpoint is saved in `checkpoints_ft_ready/baseline_best.pt`.

---

## Training, Hybrid Model

This configuration trains a quantum head directly within the model. The quantum layer maps concatenated cell and drug embeddings to a small number of qubits using `AngleEmbedding`, applies `BasicEntanglerLayers`, and returns expectation values to a small post circuit regressor. On Windows, prefer `default.qubit` for stability, which the script selects automatically if `lightning.qubit` is unavailable.

```bash
python scripts/train_hybrid_qgnn.py \
  --graph data/processed/hetero_graph.pkl \
  --out_dir checkpoints_hybrid \
  --epochs 20 \
  --bs 4096 \
  --emb_dim 256 \
  --n_qubits 8 \
  --q_depth 3 \
  --lr 3e-4
```

Observed metrics for the hybrid model, R² about 0.750, MAE about 0.95, RMSE about 1.300. The best checkpoint is saved in `checkpoints_hybrid/hybrid_best.pt`.

---

## Training, Hybrid Model Fine Tuned

This stage initializes a hybrid model from a trained classical baseline checkpoint. By default, only the quantum head and final regressor are trained. You can unfreeze embeddings with `--unfreeze_embeddings`. To handle naming differences between checkpoints, pass `--safe_load` which loads only compatible tensors and initializes the rest.

```bash
python scripts/train_hybrid_finetune.py \
  --graph data/processed/hetero_graph.pkl \
  --baseline_ckpt checkpoints_ft_ready/baseline_best.pt \
  --out_dir checkpoints_hybrid_ft \
  --epochs 20 \
  --bs 4096 \
  --emb_dim 256 \
  --n_qubits 8 \
  --q_depth 3 \
  --lr 3e-4 \
  --safe_load
```

---

## Evaluation and Plots

The evaluation script compares the classical baseline and the hybrid model on the same test set, and writes figures and a CSV summary.

```bash
python scripts/eval_compare.py \
  --graph data/processed/hetero_graph.pkl \
  --baseline_ckpt checkpoints_ft_ready/baseline_best.pt \
  --hybrid_ckpt checkpoints_hybrid/hybrid_best.pt \
  --out_dir reports
```

Artifacts:
- `reports/compare_metrics.csv`  
- `reports/scatter_baseline.png`, `reports/scatter_hybrid.png`  
- `reports/residuals_baseline.png`, `reports/residuals_hybrid.png`  
- `reports/metrics_bar.png`

There is also a minimal plotting snippet that creates simple bar charts and saves them to `plots/`:

```python
import matplotlib.pyplot as plt, os

os.makedirs("plots", exist_ok=True)
models = ['Baseline, Classical', 'Hybrid model']
r2_scores = [0.786, 0.750]
mae_scores = [0.93, 0.95]
rmse_scores = [1.25, 1.30]

for name, vals, ylabel in [
    ("r2_comparison.png", r2_scores, "R²"),
    ("mae_comparison.png", mae_scores, "MAE"),
    ("rmse_comparison.png", rmse_scores, "RMSE"),
]:
    plt.figure(figsize=(8, 5))
    plt.bar(models, vals)
    plt.title(f"{ylabel} Comparison")
    if ylabel == "R²":
        plt.ylim(0, 1)
    plt.ylabel(ylabel)
    plt.grid(True, axis='y')
    plt.savefig(f"plots/{name}")
    plt.close()
```

---

## Reproducibility Notes

- Splitting is drug aware, per drug indices, with a fixed seed.  
- Metrics include MAE, RMSE, R², averaged over all test edges.  
- Quantum backend uses `default.qubit` by default on Windows. If `lightning.qubit` is available and stable, the scripts will report it.  
- Performance variance is expected with qubit count, circuit depth, and batch size. On some machines, a smaller batch improves stability of the quantum layer.

---

## Troubleshooting on Windows

- If `torch.compile` triggers a Triton error, add `--no_compile` on training commands.  
- If `torch-scatter` or similar packages fail to build, proceed without them. The core pipeline in this repo does not require those compiled extensions.  
- PowerShell multiline commands use backticks. To avoid shell issues, prefer single line commands as shown above.

---

## Citation

If you use this code or ideas in academic work, you can cite this repository informally as:

```
NeuroQuantNet, Hybrid Quantum Classical Graph Learning for Drug Sensitivity Prediction, 2025.
```
