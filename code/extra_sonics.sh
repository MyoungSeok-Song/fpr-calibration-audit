#!/usr/bin/env bash
# Pre-specified replication (Section 5.6): SpecTTTra (gamma) next to SpecTTTra-alpha and SpecTTTra-beta, computed with the same
# scripts and settings as the main tables. Intermediate tables go to $OUT_DIR/extra_sonics/, the result to
# $OUT_DIR/extra_sonics_variants.{md,csv} (frozen copy: results/extra_sonics_variants.csv).
#   B=10000 B2=2000 (default) use the bootstrap sizes of the paper; B=200 B2=200 gives a quick run (point estimates do not depend on B).
set -euo pipefail
cd "$(dirname "$0")/.."      # repository root
PY=${PYTHON:-python}
B=${B:-10000}; B2=${B2:-2000}
OUT=${OUT_DIR:-$PWD/out}; X="$OUT/extra_sonics"
mkdir -p "$X"
D="SpecTTTra;SpecTTTra-alpha;SpecTTTra-beta"
run() { echo "+ $*"; "$@"; }
run env OUT_DIR="$X" $PY code/agg_window.py --models "$D" --B $B
run env OUT_DIR="$X" $PY code/unit_provenance.py --dets "$D" --B $B
run env OUT_DIR="$X" $PY code/unit_refswap.py --dets "$D" --B $B2
run env OUT_DIR="$X" $PY code/unit_multiref.py --dets "$D" --B $B2
run env OUT_DIR="$OUT" $PY code/extra_sonics_variants.py --in "$X" --out "$OUT" --B $B
