#!/usr/bin/env bash
# Draw every figure and the derived-number sheet from the result tables: results/ by default, FIG_SRC=out (or a path) for
# the regenerated tables. PDFs and derived_numbers.md are written to out/ (or OUT_DIR).
set -euo pipefail
cd "$(dirname "$0")/../.."      # repository root
PY=${PYTHON:-python}
for s in derived_numbers fig1_prov_unit fig2_unit_matrix fig3_disc_cal fig4_openbmat_auroc; do
  $PY code/figs/$s.py | grep -E '^(wrote|WARNING|added)' || true
done
