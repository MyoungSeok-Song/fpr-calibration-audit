#!/usr/bin/env bash
# Draw the result figures from results/ (or FIG_SRC=out for regenerated tables). PDFs go to out/ (or OUT_DIR).
set -euo pipefail
cd "$(dirname "$0")/../.."      # repository root
PY=${PYTHON:-python}
for s in fig1_prov_unit fig2_unit_matrix fig3_disc_cal; do
  $PY code/figs/$s.py | grep -E '^(wrote|WARNING|added)' || true
done
