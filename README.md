# Detector Failure or Evaluation Artifact? Diagnosing False Positives in AI-Generated Music Detection

Supplementary material for the anonymous submission.
Code, item lists and per-window detector scores that reproduce the tables and figures of the paper.

## Layout

```
code/      aggregation and figure scripts
lists/     item lists, cluster and split assignments of the public cohorts
scores/    per-window detector scores
results/   result tables
```

## Usage

```
pip install -r requirements.txt
bash run_all.sh            # B=200 B2=200 bash run_all.sh for a quick run
```

## Notes

- Audio is not redistributed. All cohorts come from public datasets; item IDs follow each dataset's own layout.
- One detector of the paper is not released.
- Model weights of the detectors trained on public data will be released with the camera-ready version, for non-commercial research use.
