# Phase 8 Protocol (frozen before any test-set evaluation)

## Model selection criteria (priority order)
1. Image-level defect recall on VAL, plotted against confidence threshold T
2. rolled-in_scale box recall on VAL
3. mAP50 and mAP50-95 on VAL (3-seed mean, 50 epochs)
4. CPU latency and FPS (laptop CPU, batch 1, imgsz 640, warm-up, 100+ images, median latency)

## Rules
- Primary weights: seed 42 (final_v8n_best.pt, final_v8s_best.pt). No seed picking.
- Both models are compared on the same T grid.
- Tie rule: if differences in criteria 1-3 are within the 3-seed spread, the smaller/faster
  model (v8n) is chosen. This is a practical rule, not a statistical test.
- T is tuned on VAL only, using a rule fixed before the analysis.
- The TEST set is evaluated once, with the selected model and fixed T.

## Limitations to report
- All VAL images are defective: false-positive rate / Good accuracy cannot be measured.
- Image-level threshold analysis uses seed-42 weights only; multi-seed variability was
  assessed on validation metrics, not on image-level threshold analysis.
## Threshold rule (frozen before the analysis)
- Strict image-level recall: at least one detection with the same class and IoU >= 0.5 against a ground-truth box.
- Target: recall >= 0.95 (at least 137 of 144 val images); T grid 0.05 to 0.95, step 0.05, same for both models.
- Choose the highest T meeting the target; fallback T = 0.05 if none does.
- Decision-rule recall (any detection with conf >= T) is reported as a sanity check only.

## FINAL DECISION (frozen before any test-set evaluation)
- Selected model: YOLOv8n, weights models/final_v8n_best.pt (seed 42, 50 epochs).
- Confidence threshold: T = 0.15.
- Basis (validation set only):
  - Strict image-level recall at T = 0.15: v8n 137/144, v8s 138/144 (difference of 1 image).
  - rolled-in_scale box recall, 3-seed mean: v8n 0.533, v8s 0.503.
  - mAP50, 3-seed mean: v8n 0.816, v8s 0.802; mAP50-95: 0.440 vs 0.439.
  - CPU latency (laptop, batch 1, imgsz 640): v8n about 129-133 ms, v8s about 298-349 ms.
  - Tie rule applied: v8n is smaller and faster; this is a practical selection rule, not a statistical test.
- Known caveats: v8s had higher inclusion box recall (mean 0.829 vs 0.788, not a selection criterion);
  T = 0.15 meets the 0.95 target for v8n with exactly 137 images; false-alarm behaviour is unmeasured
  because all validation and test images are defective.
- Test set: not used for any decision above. It will be evaluated once with this model and T.
- Weights SHA256: 94A2E409EDC19B4DE670423614F66C70DE41CAC1696D1FBFD0C89A55DF18FEAF
