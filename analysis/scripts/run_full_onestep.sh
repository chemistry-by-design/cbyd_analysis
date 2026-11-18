#!/usr/bin/env bash
# Recompute the one-step comparison tables (USPTO-Full ~0.97M, Pistachio ~4.3M)
# over the FULL datasets (no sampling). Streaming + multiprocessing throughout.
# CbyD tables are untouched (already built).
# Run with the cbyd_analysis env active (conda activate cbyd_analysis).
set -e
cd "$(dirname "$0")/../.."
S=analysis/scripts
DA=data/data_analysis
P=${1:-32}          # processes

echo "===== 1. stream full reaction tables ====="
python $S/00_prepare_tables.py --full --only onestep

for d in uspto_full pistachio; do
  echo "===== 2. products ($d) ====="
  python $S/05_extract_products.py \
    --input $DA/reactions/${d}_reactions.jsonl \
    --output $DA/molecules/${d}_products.jsonl --processes $P

  echo "===== 3. reaction features ($d) ====="
  python $S/20_rxn_features.py \
    --input $DA/reactions/${d}_reactions.jsonl \
    --output $DA/reaction_features/${d}_rxn_features.csv --processes $P

  echo "===== 4. reaction class ($d) ====="
  python $S/40_namerxn.py \
    --input $DA/reactions/${d}_reactions.jsonl \
    --output $DA/reaction_class/${d}_namerxn.csv --source onestep
done

echo "===== 5. metal counts (cbyd + full one-step) ====="
python $S/60_metal_counts.py --processes $P

for d in uspto_full pistachio; do
  echo "===== 6. product complexity ($d) ====="
  python $S/10_mol_complexity.py \
    --input  $DA/molecules/${d}_products.jsonl \
    --output $DA/complexity/${d}_products_complexity.csv --processes $P
done
echo "===== DONE (full one-step) ====="
