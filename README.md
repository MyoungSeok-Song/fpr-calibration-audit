# Where You Calibrate Is What You See: Calibration Source and Decision Unit Distort the False-Alarm Rates of AI-Generated Music Detectors

Supplementary material for the anonymous review.

## What this repository reproduces

The repository recomputes the numbers, tables and figures of the paper from released per-window detector scores. It holds the
item lists of the public cohorts the paper evaluates with their cluster and split assignments, the per-window logits of the
detectors trained on public data and of the public detectors (the three of the main analysis and the two SONICS checkpoints of
the pre-specified replication in Section 5.6), the per-window outputs of the original, re-trained and frozen-probe last-layer
heads, the aggregation code (thresholds calibrated on human references at a chosen decision unit, window-level false-positive and
true-positive rates, the two-stage cluster bootstrap that re-estimates the threshold, reference swaps, multi-reference
thresholds, decision-unit mismatch, head contrasts, matched-TPR comparisons) and the frozen result tables behind the paper. No
audio is redistributed, and training code is not included at review time. Detector C1 is not released (see below), so numbers
that involve C1 cannot be recomputed here.

## Layout

```
README.md                 this file
requirements.txt          Python packages
run_all.sh                recompute every table, compare with results/, draw the figures
scan_anon.py              anonymity scan run before release (its pattern list is kept outside the repository)
code/                     aggregation code (one script per analysis, see below)
  config.py               paths, detector names and loaders shared by all scripts
  extra_sonics.sh         pre-specified replication with SpecTTTra-alpha and -beta (Section 5.6)
  figs/                   figure scripts, derived-number sheet, make_all.sh
lists/
  cohorts.csv             cohort registry: item list, subset query, role, bootstrap cluster, size, source, license
  items/<dataset>.csv     item lists (row number = item index); item_id follows the dataset's own layout
  splits/                 Jamendo artist split (A, B, held-out pool), AVA video split (evaluation, reserved)
  openbmat_segments.csv   per-file segment list of AI-OpenBMAT, from the dataset's annotations (window classes)
scores/
  windows/<detector>/<dataset>.csv.gz   per-window logits
  heads/<family>/<dataset>.npz          per-window head outputs (Table 3, probes)
results/                  frozen result tables behind the paper (no C1 rows)
```

**Window scores** (`scores/windows/<detector>/<dataset>.csv.gz`, one row per scored window): `item_id`, `j` (window index within
the item), `n` (windows scored for the item), `start` (window start in s; empty when not recorded), `dur` (item duration in s),
`short` (1 if the item is shorter than one 10.24-s window and was reflect-padded), `lg` (logit, higher = AI), `mp`
(music-presence probability that weights windows in the track-level aggregates of the fine-tuned detectors; empty for the
public detectors). Detector directories: `B0`, `B0prime` (B0′), `B0plus0` (B0+0), `B0plus0_seed2` (B0+0, seed 2), `C0`,
`C0plus0` (C0+0), `M0`, `B0plusSU` (B0+SU), `B0plusAVA` (B0+AVA), `B0plusbroad` (B0+broad), `M0plusSU` (M0+SU: M0's recipe
with the Suno and Udio compilations added), `SpecTTTra`, `ArtifactNet`, `ArtifactNet_native` (ArtifactNet at its native
sampling rate, Table 4), `FakeprintLR` (Fakeprint LR), `SpecTTTra-alpha` and `SpecTTTra-beta` (Section 5.6; the eleven datasets
the replication uses). The probe lists FMA-ov and FMA-test calibrate only the frozen-encoder probes, so no window scores are
released for them (the fine-tuned detectors were trained on their tracks).

**Head outputs** (`scores/heads/<family>/<dataset>.npz`): `item_id`, `j`, `short` and, for every head `h` in `orig` (the
detector's own last layer), `free_split` (UC), `pos_split` (PE), `free_mlp`, `pos_mlp` (unconstrained and non-negative last
layer on a fixed ReLU hidden layer, Section 5.4): `h__z` (logit), `h__ep` (positive evidence E+), `h__en` (negative evidence
E-), `h__z0` (logit of the reference input). Families: `B0`, `C0`, `M0`, `EAT_frozen`, `EAT-AS2M_frozen`, `MERT_frozen`,
`CLAP_frozen`, `FakeprintLR`; outputs on FMA-ov and FMA-test exist for the four frozen-encoder probes only.

**Cohort registry** (`lists/cohorts.csv`): `name`, `dump` (the dataset whose item list and scores the cohort uses), `list`,
`subset` (pandas query on the item list), `role` (`calib`, `calib_probe`, `real`, `null`, `fake`), `cluster` (bootstrap
cluster column), `n`, `n_clusters`, `source`, `license`, `note`. Item identifiers: FMA six-digit track ID (audio at
`fma_large/<first three digits>/<ID>.mp3`); MTG-Jamendo `track_<ID>`; MUSDB18 `<split>/<track>`; GTZAN `<genre>.<number>`;
AI-OpenBMAT `<arm>/<file>`; AVA `<type>/<video ID>__<start>_<end>` (milliseconds after minute 15; seconds in `start_rel`,
`end_rel`, `start_abs`); AudioSet `<video ID>_<start ms>`; ESC-50 and Recent: file name; LibriSpeech utterance ID; SONICS file
stem in `fake_songs/`; MUSIC8K path in the dataset; AIME record ID; Suno compilation clip ID (`shard` column); Udio compilation
`<shard>/<key>` of `data/<shard>.tar`; synthetic nulls: generated file stem.

## Reproducing the tables and figures

Requirements: Python 3.10 with the packages in `requirements.txt`. Run every command from the repository root; outputs go to
`out/` (or `$OUT_DIR`).

All at once, with the bootstrap sizes of the paper (B = 10,000 and 2,000, and 1,000 for the oracle α′ comparison; a few CPU
hours), or quickly (point estimates do not depend on B):

```
bash run_all.sh
B=200 B2=200 bash run_all.sh
```

`run_all.sh` ends with `code/check_paper_numbers.py`, which compares every regenerated table with `results/` and prints
selected numbers quoted in the paper. Step by step:

```
python code/agg_window.py --B 10000
python code/unit_provenance.py --B 10000
python code/unit_refswap.py --B 2000
python code/unit_multiref.py --B 2000
python code/unit_strict_refsize.py --mode refsize --draws 500
python code/unit_strict_refsize.py --mode strict --B 1000
python code/unit_alpha_dependence.py
python code/f2_platform_overlap.py --B 2000
python code/openbmat_window_classes.py
python code/openbmat_window_auroc.py --B 10000
python code/agg_window_variants.py --models "B0+0;B0+0 (seed 2);B0+SU;B0+AVA;B0+broad" --prefix seednoise --ref "B0+0" --B 10000
python code/agg_window_variants.py --models "B0+0;M0" --prefix m0 --ref "B0+0" --parts auroc --B 10000
python code/agg_window_variants.py --models "M0;M0+SU" --prefix m0su_pair --ref "M0" --parts rates --B 10000
python code/matched_tpr_variants.py --pairs "B0+AVA:openbmat_ai_musicbearing" --prefix seednoise --B 2000
python code/agg_track_tau.py --models "B0;M0" --prefix all --B 10000
python code/heads_window.py --B 10000 --B_aux 2000
python code/heads_report.py
python code/heads_matched_tpr.py --B 2000
bash code/extra_sonics.sh                    # Section 5.6; B=10000 B2=2000 by default
python code/check_paper_numbers.py
FIG_SRC=out bash code/figs/make_all.sh      # figures and derived numbers from out/; without FIG_SRC from results/
```

`agg_window.py`, `unit_provenance.py` and `unit_multiref.py` accept `--part` (with `--models`/`--dets`) and `--merge` to split
the work over processes.

| Paper item | Script | Table (`out/`, frozen copy in `results/`) |
|---|---|---|
| Table 1 (cohort sizes) | registry | `lists/cohorts.csv` (`n`, `n_clusters`) |
| Fig. 1, Section 5.1 first paragraph | `unit_provenance.py`, `figs/fig1_prov_unit.py` | `unit_prov.csv` |
| Section 5.1, threshold source and alpha dependence | `agg_window.py`, `unit_alpha_dependence.py` | `public_window_rates.csv`, `unit_alpha_dependence.csv` |
| Section 5.1, frozen-encoder probes (FMA-ov, FMA-test) | `heads_window.py` | `heads_provenance.csv` |
| Section 5.1, reference swap | `unit_refswap.py` | `unit_refswap.csv`, `unit_refswap_summary.csv` |
| Section 5.1, most conservative of five references | `unit_multiref.py` | `unit_multiref.csv`, `unit_multiref_summary.csv`, `unit_multiref_tpr.csv` |
| Section 5.1, reference size (500 artist-level draws of 150 or 322 Jamendo-A tracks) | `unit_strict_refsize.py --mode refsize` | `unit_refsize.csv`, `unit_refsize_summary.csv` |
| Section 5.1, single Jamendo-A threshold at an oracle α′ | `unit_strict_refsize.py --mode strict` | `unit_multiref_vs_strict.csv` |
| Section 5.1, intervention (B0+broad) | `f2_platform_overlap.py`; paired AUROC in `agg_window_variants.py` (seednoise) | `f2_platform_overlap*.csv`, `variants_seednoise_auroc.csv` |
| Fig. 2, Section 5.2 | `unit_provenance.py`, `figs/fig2_unit_matrix.py` | `unit_matrix.csv` |
| Section 5.2, padded inputs | `agg_window.py` | `public_window_rates.csv` (cohorts `...[long]`, `...[short]`) |
| Section 5.2, track-level thresholds | `unit_provenance.py`, `agg_track_tau.py` | `unit_prov.csv` (unit `file`), `track_tau_all_rates.csv` |
| Section 5.3, excess over Jamendo-B | `agg_window.py` | `public_window_g0.csv` |
| Section 5.3, B0+SU, M0+SU, M0; Section 7, seed repeat | `agg_window_variants.py` | `variants_seednoise_{rates,ratios,paired}.csv`, `variants_m0su_pair_{rates,ratios}.csv`, `variants_m0_auroc.csv` |
| Fig. 3, Section 5.3 regimes | `unit_provenance.py`, `figs/fig3_disc_cal.py` | `unit_rank_unseen.csv` |
| Table 3, Section 5.4 (incl. null inputs) | `heads_window.py`, `heads_report.py`, `heads_matched_tpr.py` | `heads_rates.csv`, `heads_contrast_all.csv`, `heads_fa_types.csv`, `heads_null.csv`, `heads_matched_tpr.csv` |
| Section 5.4, B0+AVA at matched TPR | `agg_window_variants.py` (seednoise), `matched_tpr_variants.py` | `matched_tpr_variants.csv` |
| Fig. 4, Section 5.5 | `openbmat_window_classes.py`, `openbmat_window_auroc.py`, `agg_window_variants.py` (m0), `figs/fig4_openbmat_auroc.py` | `openbmat_window_auroc.csv`, `variants_m0_auroc.csv` |
| Table 4 | `unit_provenance.py`; native-rate ArtifactNet in `agg_window.py` | `unit_gen.csv`, `unit_exposure.csv`, `public_window_rates.csv` (cohorts `<dataset>::<generator>`) |
| Section 5.6, pre-specified replication | `extra_sonics.sh` (`agg_window.py`, `unit_provenance.py`, `unit_refswap.py`, `unit_multiref.py`, `extra_sonics_variants.py`) | `extra_sonics_variants.csv` |
| Ranges, means and ratios quoted in the text | `figs/derived_numbers.py` | `derived_numbers.md` |

**What reproduces exactly.** Point estimates (thresholds, FPR, TPR, AUROC, ratios, counts) are deterministic and match
`results/` to floating-point precision. Bootstrap intervals use fixed seeds: with the paper's B they also match, except where
the random stream in the paper also passed through cohorts and analyses that the paper does not report and that are not part of
this repository: the cohort intervals of `agg_window.py` after Jamendo-B (`public_window_rates.csv`, `public_window_g0.csv`) and,
in `unit_provenance.py`, the `unit_matrix.csv` rows after GTZAN, `unit_prov.csv` and `unit_gen.csv`. There the intervals differ
by Monte Carlo error only (for example 2.18 [1.29, 3.84] against 2.18 [1.28, 3.83] in the paper). With B = 200 the intervals
change, the point estimates do not. Holm-adjusted p-values in `heads_contrast_all.csv` are computed over the families present
(eight without C1), while `results/` keeps the paper's nine-family values.

## Pre-specified replication (Section 5.6)

After the main analysis two further official SONICS checkpoints, SpecTTTra-alpha and SpecTTTra-beta (5-s windows), were scored
against four expectations recorded in advance: E1, unit inflation (track-level thresholds raise the single-window Jamendo-B FPR
at least twofold); E2, a small calibration-source effect for detectors not trained on FMA (FMA-dev instead of Jamendo-A
thresholds change the Jamendo-B FPR by at most about twofold); E3, a coverage failure on new generator families (mean TPR at most
20%); E4, no detector-specific in-the-wild excess (AVA-SM, AudioSet-SM and AI-OpenBMAT-H do not meet the failure criterion at
α = 2%). The same scripts as the main tables run for SpecTTTra, SpecTTTra-alpha and SpecTTTra-beta, the SpecTTTra rows are
checked against the main tables, and `code/extra_sonics_variants.py` adds joint-bootstrap intervals for the E1 and E2 ratios:

```
bash code/extra_sonics.sh                    # out/extra_sonics_variants.{md,csv}; frozen copy results/extra_sonics_variants.csv
B=200 B2=200 bash code/extra_sonics.sh       # quick run, same point estimates
```

E1, E2 and E3 held for both checkpoints. E4 failed for SpecTTTra-alpha: it flags 10.1% of AudioSet-SM windows against 2.6% of
Jamendo-B (ratio 3.85 [2.65, 5.79]) and meets the failure criterion at α = 1% and 2% (not at 5%); E4 held for SpecTTTra-beta.

## Detector C1 is not released

C1 continues the C0 lineage through several training stages, and training data used in some of the stages cannot be
redistributed. Its weights, window scores, thresholds and per-item outputs are therefore not part of this repository, and
`results/` contains no C1 rows. Counts and ranges that pool C1 with other detectors are computed here over the nine released
detectors. The multi-reference MAX count, for example, is 7/9 here (C1 exceeds 6% under the most conservative threshold):
C1 not released; with C1's scores the count is 7/10 as reported. Likewise the reference-swap count for AudioSet-MO is 8/9
here against 9/10 in the paper, and the Jamendo-A-threshold FPR range of the fine-tuned detectors on Jamendo-B starts at
2.9% instead of 2.6%.

## Model weights

Model weights of the detectors trained on public data will be released with the camera-ready version, for non-commercial
research, following the licenses of their training data.

## Public datasets

Licenses as given in Table 1 of the paper. Only identifiers, annotations derived from the datasets' own labels and detector
scores are distributed here.

| Dataset | Cohorts | Official URL | License |
|---|---|---|---|
| FMA | FMA-dev, FMA-ov, FMA-test | https://github.com/mdeff/fma | per-track Creative Commons |
| MTG-Jamendo | Jamendo-A, Jamendo-B | https://github.com/MTG/mtg-jamendo-dataset (http://hdl.handle.net/10230/42015) | per-track Creative Commons |
| MUSDB18 | MUSDB18 | https://doi.org/10.5281/zenodo.1117372 | research use |
| GTZAN | GTZAN | https://huggingface.co/datasets/marsyas/gtzan (doi:10.1109/TSA.2002.800560) | none stated |
| AI-OpenBMAT | AI-OpenBMAT-H, AI-OpenBMAT-AI | https://zenodo.org/records/20031232 | CC BY 4.0; upstream music for non-commercial research only |
| AVA-Speech | AVA-SM, AVA-MO, AVA-S | https://research.google.com/ava/download.html#ava_speech_download | CC BY 4.0 labels; audio from YouTube |
| AVASpeech-SMAD | music labels of the AVA regions | https://arxiv.org/abs/2111.01320 | MIT (labels) |
| AudioSet strong labels | AudioSet-SM, AudioSet-MO | https://research.google.com/audioset/download_strong.html | CC BY 4.0 labels; audio from YouTube |
| LibriSpeech | LibriSpeech (test-clean) | https://www.openslr.org/12 | CC BY 4.0 |
| ESC-50 | ESC-50 | https://github.com/karolpiczak/ESC-50 | CC BY-NC 3.0 |
| SONICS | SONICS | https://huggingface.co/datasets/awsaf49/sonics | CC BY-NC 4.0 |
| MUSIC8K | MUSIC8K | https://huggingface.co/datasets/homura23/MUSIC8K | CC BY 4.0 |
| AIME | AIME | https://huggingface.co/datasets/disco-eth/AIME | CC BY 4.0 |
| Lyria 3 | Recent | https://huggingface.co/datasets/sleeping-ai/Lyria3 | MIT (Recent: mixed) |
| Suno v5.5 | Recent | https://huggingface.co/datasets/Kukedlc/suno-ai-music-dataset | CC BY 4.0 (Recent: mixed) |
| Suno compilation (suno-musicality) | Recent | https://huggingface.co/datasets/teamsleeping/suno_musicality | CC BY-NC-ND 4.0 (Recent: mixed) |
| Suno platform compilation | Suno comp. | https://huggingface.co/datasets/nyuuzyou/suno | per dataset card (CC0-1.0) |
| Udio platform compilation | Udio comp. | https://huggingface.co/datasets/blanchon/udio_dataset | per dataset card (CC0-1.0) |
| Synthetic nulls | Synthetic | generated (noise and dithered silence) | n/a |

Several of these datasets permit only non-commercial research; the same restriction applies to the scores derived from them.
