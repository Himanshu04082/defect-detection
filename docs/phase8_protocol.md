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
