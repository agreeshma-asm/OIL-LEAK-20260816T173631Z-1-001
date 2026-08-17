# JCB Hydraulic Oil Leak Detection Dataset (YOLO11s-seg)

Proof-of-concept synthetic dataset for JCB hydraulic oil leak instance segmentation.

## Modalities
- **RGB**: Visible natural-light inspection images with oily wet regions, drips, and surface trails.
- **UV**: UV-assisted inspection images with simulated fluorescent yellow/green leak signals.

## Class Definition
- `0: hydraulic_oil_leak`
- Negative / no-leak images are annotated with empty label files (`.txt`).

## Split Summary
- **Train**: ~70% of dataset
- **Val**: ~20% of dataset
- **Test**: ~10% of dataset (scene-separated holdout test set)

## Annotation Format
YOLO instance segmentation format:
`0 x1 y1 x2 y2 x3 y3 ... xN yN`
Coordinates are normalized [0, 1].

## Scientific & POC Limitations Disclaimer
1. This dataset uses synthetic JCB images for proof-of-concept pipeline development.
2. The UV images simulate a fluorescent leak indication for inspection workflow modeling; ordinary hydraulic oil does NOT naturally fluoresce under UV without fluorescent tracer dyes added.
3. Model evaluation on synthetic data should be validated with real field test samples.
