import os
import argparse
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11s-seg for JCB Hydraulic Oil Leak Detection (Clean Dataset v6)")
    parser.add_argument("--data", type=str, default=os.path.join(ROOT_DIR, "dataset", "data.yaml"), help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="yolo11s-seg.pt", help="Initial weights path or model spec")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu", help="CUDA device or cpu")
    parser.add_argument("--project", type=str, default=os.path.join(ROOT_DIR, "runs"), help="Project runs directory")
    parser.add_argument("--name", type=str, default="yolo11s_jcb_leak_v6_seg", help="Experiment name")
    
    args = parser.parse_args()
    
    print(f"Loading YOLO model with weights: {args.weights}...")
    model = YOLO(args.weights)
    
    print(f"Starting training on dataset: {args.data}")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        plots=True
    )
    
    print(f"\nTraining completed! Results saved to {os.path.join(args.project, args.name)}")
    best_weights = os.path.join(args.project, args.name, "weights", "best.pt")
    print(f"Best model weights: {best_weights}")

if __name__ == "__main__":
    main()
