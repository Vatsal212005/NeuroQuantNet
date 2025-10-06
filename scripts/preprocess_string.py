#!/usr/bin/env python3
"""
Preprocess STRING human PPI (9606.protein.links.v12.0.txt.gz).

- Unzips if needed (streams .gz; no temp 100s of MB files).
- (Optional) maps ENSP protein IDs -> HGNC gene symbols
  using 9606.protein.info.v12.0.txt.gz if present.
- Filters by combined_score threshold (default 700/1000).
- Deduplicates undirected edges, removes self-loops.
- Writes data/processed/string_edges.csv  (gene_a,gene_b,weight)

Usage:
  python scripts/preprocess_string.py \
    --links data/raw/9606.protein.links.v12.0.txt.gz \
    --info  data/raw/9606.protein.info.v12.0.txt.gz \
    --out   data/processed/string_edges.csv \
    --min_score 700
"""
import argparse
import csv
import gzip
import os
from collections import defaultdict

def load_id_to_gene(info_gz_path: str):
    """
    Returns dict: '9606.ENSPxxxx' -> 'GENE' (symbol).
    If file missing, returns {} and caller will keep protein IDs.
    """
    import gzip
    import os

    mapping = {}
    if not info_gz_path or not os.path.exists(info_gz_path):
        return mapping

    # Auto-detect compression
    open_fn = gzip.open if info_gz_path.endswith('.gz') else open

    with open_fn(info_gz_path, 'rt', encoding='utf-8', newline='') as f:
        header = f.readline().rstrip('\n').split('\t')
        try:
            id_idx = header.index('protein_external_id')
            sym_idx = header.index('preferred_name')
        except ValueError:
            # Fallback to first two columns
            id_idx, sym_idx = 0, 1

        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(id_idx, sym_idx):
                continue
            pid = parts[id_idx]
            sym = parts[sym_idx]
            if pid and sym:
                mapping[pid] = sym.upper()

    return mapping

def preprocess(links_gz_path: str, info_gz_path: str, out_csv: str, min_score: int):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    id2gene = load_id_to_gene(info_gz_path)

    # Use a dictionary to keep the *max* score seen for an undirected pair
    edge2score = defaultdict(int)

    with gzip.open(links_gz_path, 'rt', encoding='utf-8', newline='') as f:
        header = f.readline().rstrip('\n').split(' ')
        # STRING "links" files are either tab- or space-separated depending on mirror.
        # Read robustly by splitting on any whitespace:
        first_line = header
        is_header = any(h.lower().startswith('protein') for h in first_line)
        if not is_header:
            # no header; process this first line as data
            parts = first_line
            if len(parts) == 1:
                # likely tab-delimited file; re-open splitting by tabs
                f.close()
                with gzip.open(links_gz_path, 'rt', encoding='utf-8') as f2:
                    header = f2.readline().rstrip('\n').split('\t')
                    # If still not a header, treat as data
                    data_iter = f2
                    delim = '\t'
                    # Write will happen below
                    pass
            else:
                data_iter = [ ' '.join(first_line) ]  # will be re-parsed below
                delim = None
        else:
            # header line consumed; detect delimiter
            delim = '\t' if '\t' in '\t'.join(first_line) else ' '
            data_iter = f

        def parse_line(line: str):
            parts = line.strip().split(" ")

            if len(parts) < 3:
                return None
            a, b, score_str = parts[0], parts[1], parts[2]
            try:
                score = int(score_str)
            except ValueError:
                # sometimes header row sneaks in
                return None
            if score < min_score:
                return None
            # Map to gene symbols if available; else keep protein IDs
            a_map = id2gene.get(a, a)
            b_map = id2gene.get(b, b)
            if a_map == b_map:
                return None
            # undirected canonical ordering
            u, v = (a_map, b_map) if a_map < b_map else (b_map, a_map)
            return (u, v, score)

        # Iterate and collect max score per undirected edge
        for line in data_iter:
            rec = parse_line(line)
            if rec is None:
                continue
            u, v, score = rec
            key = (u, v)
            if score > edge2score[key]:
                edge2score[key] = score

    # Write CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as w:
        writer = csv.writer(w)
        writer.writerow(['gene_a', 'gene_b', 'weight'])
        for (u, v), score in edge2score.items():
            writer.writerow([u, v, score / 1000.0])

    # Summary
    n_edges = len(edge2score)
    n_nodes = len({x for e in edge2score.keys() for x in e})
    use_mapping = 'YES' if id2gene else 'NO (kept protein IDs)'
    print(f"[STRING] Wrote: {out_csv}")
    print(f"[STRING] Nodes ≈ {n_nodes:,} | Edges = {n_edges:,} | Min score ≥ {min_score}")
    print(f"[STRING] Used ENSP→gene mapping: {use_mapping}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", required=True, help="Path to 9606.protein.links.v12.0.txt.gz")
    ap.add_argument("--info", default="", help="(Optional) 9606.protein.info.v12.0.txt.gz for ENSP->gene mapping")
    ap.add_argument("--out", default="data/processed/string_edges.csv", help="Output CSV")
    ap.add_argument("--min_score", type=int, default=700, help="Keep edges with combined_score >= this")
    args = ap.parse_args()
    preprocess(args.links, args.info, args.out, args.min_score)
