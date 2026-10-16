#!/usr/bin/env bash
# Recompute every result table from the released window scores, compare with results/, and redraw the figures.
#   B=10000 B2=2000 B3=1000 (default) use the bootstrap sizes of the paper (several hours on one CPU; use --part/--merge to split);
#   B3 is used by the oracle-alpha' comparison only.
#   B=200 B2=200 gives a quick run; point estimates do not depend on B.
# Outputs go to out/ (or OUT_DIR).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
B=${B:-10000}; B2=${B2:-2000}; B3=${B3:-1000}
export OUT_DIR=${OUT_DIR:-$PWD/out}
mkdir -p "$OUT_DIR"
run() { echo "+ $*"; "$@"; }
run $PY code/agg_window.py --B $B
run $PY code/unit_provenance.py --B $B
run $PY code/unit_refswap.py --B $B2
run $PY code/unit_multiref.py --B $B2
run $PY code/unit_strict_refsize.py --mode refsize --draws 500
run $PY code/unit_strict_refsize.py --mode strict --B $B3
run $PY code/unit_alpha_dependence.py
run $PY code/policy_compare.py --B $B2
run $PY code/interpretation_change.py --B $B2
run $PY code/f2_platform_overlap.py --B $B2
run $PY code/openbmat_window_classes.py
run $PY code/openbmat_window_auroc.py --B $B
run $PY code/f2x_exposure_crossover.py --B $B2 --B_obm $B
run $PY code/agg_window_variants.py --models "B0+0;B0+0 (seed 2);B0+SU;B0+AVA;B0+broad" --prefix seednoise --ref "B0+0" --B $B
run $PY code/agg_window_variants.py --models "B0+0;M0" --prefix m0 --ref "B0+0" --parts auroc --B $B
run $PY code/matched_tpr_variants.py --pairs "B0+AVA:openbmat_ai_musicbearing" --prefix seednoise --B $B2
run $PY code/agg_track_tau.py --models "B0;M0" --prefix all --B $B
run $PY code/heads_window.py --B $B --B_aux $B2
run $PY code/heads_report.py
run $PY code/heads_matched_tpr.py --B $B2
run $PY code/table3_matched_baseline.py --B $B2
run env B=$B B2=$B2 PYTHON=$PY bash code/extra_sonics.sh
run $PY code/compare_tables.py --out "$OUT_DIR"
FIG_SRC="$OUT_DIR" PYTHON=$PY bash code/figs/make_all.sh
