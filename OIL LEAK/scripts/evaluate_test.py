import os
import glob
import cv2
import csv
import numpy as np
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMG_DIR = os.path.join(ROOT_DIR, "dataset", "test", "images")
TEST_LBL_DIR = os.path.join(ROOT_DIR, "dataset", "test", "labels")
WEIGHTS_PATH = os.path.join(ROOT_DIR, "runs", "yolo11s_jcb_leak_v6_seg", "weights", "best.pt")
SAVE_DIR = os.path.join(ROOT_DIR, "runs", "test_predictions")

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def main():
    print("=========================================================")
    print("       YOLO11s-seg TEST SET EVALUATION PIPELINE          ")
    print("=========================================================")
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Model weights not found at {WEIGHTS_PATH}")
        print("Please wait for training to complete.")
        return
        
    model = YOLO(WEIGHTS_PATH)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    img_files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.*")))
    
    results_summary = {
        "overall": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "rgb": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "uv": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    }
    
    # Store predictions for detailed reporting
    details = []
    
    # Error classification lists
    missed_leaks = []            # False Negatives (FN)
    false_positives = []         # False Positives (FP)
    no_leak_incorrect = []       # FP on ground-truth no-leak images
    
    for img_path in img_files:
        fname = os.path.basename(img_path)
        bname = os.path.splitext(fname)[0]
        lbl_path = os.path.join(TEST_LBL_DIR, bname + ".txt")
        
        # Load image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # Get ground truth info
        is_leak = False
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            is_leak = True
        is_uv = "uv" in bname.lower()
        modality = "uv" if is_uv else "rgb"
        
        # Get ground truth mask
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        gt_pts_list = []
        if is_leak:
            with open(lbl_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 7:
                    coords = [float(x) for x in parts[1:]]
                    pts = np.array([(coords[i]*w, coords[i+1]*h) for i in range(0, len(coords), 2)], dtype=np.int32)
                    gt_pts_list.append(pts)
                    cv2.fillPoly(gt_mask, [pts], 255)
                    
        # Model Prediction
        pred = model.predict(img, conf=0.25, verbose=False)[0]
        
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        has_pred_leak = False
        max_conf = 0.0
        
        if pred.masks is not None and len(pred.masks) > 0:
            has_pred_leak = True
            for i, mask in enumerate(pred.masks.xy):
                pts = np.array([mask], dtype=np.int32)
                cv2.fillPoly(pred_mask, pts, 255)
            max_conf = float(pred.boxes.conf.max().cpu().numpy())
            
        # Determine classification metrics
        # TP: GT leak, Pred leak
        # FP: GT no leak, Pred leak
        # FN: GT leak, Pred no leak
        # TN: GT no leak, Pred no leak
        metric_type = ""
        if is_leak:
            if has_pred_leak:
                metric_type = "tp"
            else:
                metric_type = "fn"
                missed_leaks.append(fname)
        else:
            if has_pred_leak:
                metric_type = "fp"
                false_positives.append(fname)
                no_leak_incorrect.append(fname)
            else:
                metric_type = "tn"
                
        # Update overall and modality statistics
        results_summary["overall"][metric_type] += 1
        results_summary[modality][metric_type] += 1
        
        # Compute IoU if there is mask overlap
        iou = compute_iou(gt_mask, pred_mask)
        
        # Save visualization (3-Panel: Original Image, Ground Truth, Prediction)
        vis_img = img.copy()
        
        # Overlay GT in Red
        gt_overlay = vis_img.copy()
        for pts in gt_pts_list:
            cv2.fillPoly(gt_overlay, [pts], (0, 0, 255))
        
        # Overlay Prediction in Green
        pred_overlay = vis_img.copy()
        if has_pred_leak:
            pred_xy = pred.masks.xy
            for mask in pred_xy:
                pts = np.int32([mask])
                cv2.fillPoly(pred_overlay, [pts], (0, 255, 0))
                
        cv2.addWeighted(gt_overlay, 0.4, vis_img, 0.6, 0, gt_overlay)
        cv2.addWeighted(pred_overlay, 0.4, vis_img, 0.6, 0, pred_overlay)
        
        # Concatenate 3 panels side-by-side: original, GT overlaid, Pred overlaid
        combined = np.hstack((img, gt_overlay, pred_overlay))
        
        # Draw labels (adjusted scale for small image size)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        # Top titles for panels
        cv2.putText(combined, "1. ORIGINAL", (10, 20), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(combined, "2. GROUND TRUTH", (w + 10, 20), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(combined, f"3. PRED (Conf: {max_conf:.2f})", (2 * w + 10, 20), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Metric info at bottom left
        cv2.putText(combined, f"IoU: {iou:.3f} | {metric_type.upper()}", (10, h - 10), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        
        cv2.imwrite(os.path.join(SAVE_DIR, f"eval_{fname}"), combined)
        
        details.append({
            "filename": fname,
            "modality": modality,
            "gt_leak": is_leak,
            "pred_leak": has_pred_leak,
            "metric": metric_type.upper(),
            "iou": iou,
            "confidence": max_conf
        })

    # Run standard yolo validation to get exact mAP50 / mAP50-95
    print("\nRunning Ultralytics validation on test set...")
    val_results = model.val(data=os.path.join(ROOT_DIR, "dataset", "data.yaml"), split="test", verbose=False)
    
    overall_map50 = val_results.seg.map50 if (hasattr(val_results, "seg") and val_results.seg is not None) else 0.0
    overall_map95 = val_results.seg.map if (hasattr(val_results, "seg") and val_results.seg is not None) else 0.0
    
    # Calculate Precision and Recall
    def get_precision_recall(counts):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1
        
    p_o, r_o, f1_o = get_precision_recall(results_summary["overall"])
    p_rgb, r_rgb, f1_rgb = get_precision_recall(results_summary["rgb"])
    p_uv, r_uv, f1_uv = get_precision_recall(results_summary["uv"])
    
    print("\n" + "="*50)
    print("        EVALUATION TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Overall Metrics:")
    print(f"  - Precision  : {p_o:.4f}")
    print(f"  - Recall     : {r_o:.4f}")
    print(f"  - F1-Score   : {f1_o:.4f}")
    print(f"  - mAP50 (Seg): {overall_map50:.4f}")
    print(f"  - mAP50-95   : {overall_map95:.4f}")
    print(f"  - TP: {results_summary['overall']['tp']} | TN: {results_summary['overall']['tn']} | FP: {results_summary['overall']['fp']} | FN: {results_summary['overall']['fn']}")
    print("-" * 50)
    print(f"RGB-Only Test Performance:")
    print(f"  - Precision  : {p_rgb:.4f}")
    print(f"  - Recall     : {r_rgb:.4f}")
    print(f"  - F1-Score   : {f1_rgb:.4f}")
    print(f"  - TP: {results_summary['rgb']['tp']} | TN: {results_summary['rgb']['tn']} | FP: {results_summary['rgb']['fp']} | FN: {results_summary['rgb']['fn']}")
    print("-" * 50)
    print(f"UV-Only Test Performance:")
    print(f"  - Precision  : {p_uv:.4f}")
    print(f"  - Recall     : {r_uv:.4f}")
    print(f"  - F1-Score   : {f1_uv:.4f}")
    print(f"  - TP: {results_summary['uv']['tp']} | TN: {results_summary['uv']['tn']} | FP: {results_summary['uv']['fp']} | FN: {results_summary['uv']['fn']}")
    print("==================================================")
    
    print("\n==================================================")
    print("        CONFUSION & ERROR ANALYSIS")
    print("==================================================")
    print(f"Missed Leaks (FN) [{len(missed_leaks)}]:")
    for item in missed_leaks:
        print(f"  - {item}")
    print(f"False Positives (FP) [{len(false_positives)}]:")
    for item in false_positives:
        print(f"  - {item}")
    print(f"No-Leak Images Incorrectly Detected [{len(no_leak_incorrect)}]:")
    for item in no_leak_incorrect:
        print(f"  - {item}")
    print("==================================================")
    
    # Save a CSV log of all image predictions
    csv_path = os.path.join(ROOT_DIR, "runs", "test_evaluation_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "modality", "gt_leak", "pred_leak", "metric", "iou", "confidence"])
        writer.writeheader()
        writer.writerows(details)
    print(f"\nSaved evaluation log CSV to {csv_path}")
    print(f"Saved 3-panel test visualizations to {SAVE_DIR}")
    
    # Write a detailed markdown report
    report_path = os.path.join(ROOT_DIR, "runs", "test_evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("# YOLO11s-seg Test Set Evaluation Report\n\n")
        f.write("## Overall Metrics Summary\n\n")
        f.write("| Modality | Precision | Recall | F1-Score | mAP50 (Seg) | mAP50-95 (Seg) | TP | TN | FP | FN |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        f.write(f"| **Overall** | {p_o:.4f} | {r_o:.4f} | {f1_o:.4f} | {overall_map50:.4f} | {overall_map95:.4f} | {results_summary['overall']['tp']} | {results_summary['overall']['tn']} | {results_summary['overall']['fp']} | {results_summary['overall']['fn']} |\n")
        f.write(f"| **RGB-Only** | {p_rgb:.4f} | {r_rgb:.4f} | {f1_rgb:.4f} | - | - | {results_summary['rgb']['tp']} | {results_summary['rgb']['tn']} | {results_summary['rgb']['fp']} | {results_summary['rgb']['fn']} |\n")
        f.write(f"| **UV-Only** | {p_uv:.4f} | {r_uv:.4f} | {f1_uv:.4f} | - | - | {results_summary['uv']['tp']} | {results_summary['uv']['tn']} | {results_summary['uv']['fp']} | {results_summary['uv']['fn']} |\n\n")
        
        f.write("## Detailed Confusion & Error Analysis\n\n")
        f.write("### 1. Missed Leaks (False Negatives - FN)\n")
        if missed_leaks:
            for m in missed_leaks:
                f.write(f"- `{m}`\n")
        else:
            f.write("- *None (Perfect detection recall)*\n")
        f.write("\n")
        
        f.write("### 2. False Positives (FP)\n")
        if false_positives:
            for fp in false_positives:
                f.write(f"- `{fp}`\n")
        else:
            f.write("- *None (Perfect precision)*\n")
        f.write("\n")
        
        f.write("### 3. No-Leak Images Incorrectly Detected\n")
        if no_leak_incorrect:
            for nli in no_leak_incorrect:
                f.write(f"- `{nli}`\n")
        else:
            f.write("- *None (No false triggers on clean background)*\n")
        f.write("\n")
        
        f.write("## Image-by-Image Prediction Details\n\n")
        f.write("| Filename | Modality | Ground Truth | Prediction | Result | IoU | Confidence |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for det in details:
            gt_str = "Leak" if det["gt_leak"] else "No Leak"
            pred_str = "Leak" if det["pred_leak"] else "No Leak"
            f.write(f"| `{det['filename']}` | {det['modality'].upper()} | {gt_str} | {pred_str} | **{det['metric']}** | {det['iou']:.4f} | {det['confidence']:.4f} |\n")
        f.write("\n")
        f.write("Report generated successfully.")
    print(f"Saved evaluation Markdown report to {report_path}")

if __name__ == "__main__":
    main()
