# YOLO11s-seg Test Set Evaluation Report

## Overall Metrics Summary

| Modality | Precision | Recall | F1-Score | mAP50 (Seg) | mAP50-95 (Seg) | TP | TN | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Overall** | 0.5000 | 1.0000 | 0.6667 | 0.0000 | 0.0000 | 2 | 0 | 2 | 0 |
| **RGB-Only** | 0.5000 | 1.0000 | 0.6667 | - | - | 1 | 0 | 1 | 0 |
| **UV-Only** | 0.5000 | 1.0000 | 0.6667 | - | - | 1 | 0 | 1 | 0 |

## Detailed Confusion & Error Analysis

### 1. Missed Leaks (False Negatives - FN)
- *None (Perfect detection recall)*

### 2. False Positives (FP)
- `04_cylinder_seal_rgb_no_leak.jpg`
- `04_cylinder_seal_uv_no_leak.jpg`

### 3. No-Leak Images Incorrectly Detected
- `04_cylinder_seal_rgb_no_leak.jpg`
- `04_cylinder_seal_uv_no_leak.jpg`

## Image-by-Image Prediction Details

| Filename | Modality | Ground Truth | Prediction | Result | IoU | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| `04_cylinder_seal_rgb_no_leak.jpg` | RGB | No Leak | Leak | **FP** | 0.0000 | 0.5110 |
| `04_cylinder_seal_uv_no_leak.jpg` | UV | No Leak | Leak | **FP** | 0.0000 | 0.7665 |
| `05_valve_manifold_rgb_leak.jpg` | RGB | Leak | Leak | **TP** | 0.0776 | 0.8437 |
| `05_valve_manifold_uv_leak.jpg` | UV | Leak | Leak | **TP** | 0.0615 | 0.7420 |

Report generated successfully.