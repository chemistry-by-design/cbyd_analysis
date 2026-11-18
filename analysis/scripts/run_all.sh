#!/usr/bin/env bash
# Orchestrates the full precompute. Run with the cbyd_analysis env active
# (conda activate cbyd_analysis).
#
# Usage:  bash analysis/scripts/run_all.sh [complexity|features|all]
set -e
cd "$(dirname "$0")/../.."          # repo root
S=analysis/scripts
DA=data/data_analysis
STAGE="${1:-all}"

run_features() {
  echo "### reaction features (uspto_full, pistachio samples)"
  for d in uspto_full pistachio; do
    python $S/20_rxn_features.py \
      --input  $DA/reactions/${d}_reactions.jsonl \
      --output $DA/reaction_features/${d}_rxn_features.csv --processes 32
  done
  echo "### reaction class (namerxn for cbyd; parse for one-step)"
  python $S/40_namerxn.py \
    --input $DA/reactions/cbyd_reactions.jsonl \
    --output $DA/reaction_class/cbyd_namerxn.csv --source cbyd
  for d in uspto_full pistachio; do
    python $S/40_namerxn.py \
      --input  $DA/reactions/${d}_reactions.jsonl \
      --output $DA/reaction_class/${d}_namerxn.csv --source onestep
  done
}

run_complexity() {
  echo "### molecular complexity"
  # molecule tables: id, smiles
  for f in cbyd_products cbyd_targets uspto_full_products pistachio_products \
           uspto190_targets paroutes_targets; do
    [ -f $DA/molecules/${f}.jsonl ] || { echo "skip ${f} (missing)"; continue; }
    python $S/10_mol_complexity.py \
      --input  $DA/molecules/${f}.jsonl \
      --output $DA/complexity/${f}_complexity.csv --processes 32
  done
}

case "$STAGE" in
  features)   run_features ;;
  complexity) run_complexity ;;
  all)        run_features; run_complexity ;;
  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
echo "### DONE ($STAGE)"
