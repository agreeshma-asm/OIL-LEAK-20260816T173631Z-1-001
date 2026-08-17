import os
import argparse
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def evaluate_subset(model, data_path, split="val", name="overall"):
    print(f"\n--- Evaluating [{name}] on split: {split} ---")
    metrics = model.val(data=data_path, split=split)
    
    if hasattr(metrics, "seg"):
        print(f"[{name}] Seg mAP50    : {metrics.seg.map50:.4f}")
        print(f"[{name}] Seg mAP50-95 : {metrics.seg.map:.4f}")
    if hasattr(metrics, "box"):
        print(f"[{name}] Box Precision : {metrics.box.mp:.4f}")
        print(f"[{name}] Box Recall    : {metrics.box.mr:.4f}")
        print(f"[{name}] Box mAP50     : {metrics.box.map50:.4f}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO11s-seg JCB Hydraulic Oil Leak Model (v6)")
    parser.add_argument("--weights", type=str, default=os.path.join(ROOT_DIR, "runs", "yolo11s_jcb_leak_v6_seg", "weights", "best.pt"), help="Path to best.pt weights")
    parser.add_argument("--data", type=str, default=os.path.join(ROOT_DIR, "dataset", "data.yaml"), help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate: test or val")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"Weights file not found: {args.weights}")
        print("Please train the model first using python scripts/train.py")
        return
        
    model = YOLO(args.weights)
    evaluate_subset(model, args.data, split=args.split, name="Overall")
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
