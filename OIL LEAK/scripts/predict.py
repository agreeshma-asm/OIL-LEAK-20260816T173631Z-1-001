import os
import sys
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_image_inference(model, source_path, conf_thresh=0.25, save_dir=os.path.join(ROOT_DIR, "runs", "predict")):
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(source_path):
        print(f"Error: Source image path does not exist: {source_path}")
        return
        
    img = cv2.imread(source_path)
    if img is None:
        print(f"Error: Unable to load image {source_path}")
        return
        
    results = model.predict(source=img, conf=conf_thresh, verbose=False)
    result = results[0]
    
    annotated_img = img.copy()
    
    has_leak = False
    max_conf = 0.0
    
    if result.masks is not None and len(result.masks) > 0:
        has_leak = True
        masks = result.masks.xy
        boxes = result.boxes
        
        overlay = annotated_img.copy()
        
        for i, mask in enumerate(masks):
            conf = float(boxes[i].conf[0].cpu().numpy())
            if conf > max_conf:
                max_conf = conf
                
            pts = np.int32([mask])
            cv2.fillPoly(overlay, pts, (0, 0, 255))
            cv2.polylines(annotated_img, pts, True, (0, 255, 255), 2)
            
            xyxy = boxes[i].xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            
        cv2.addWeighted(overlay, 0.4, annotated_img, 0.6, 0, annotated_img)
        
    print("\n----------------------------------------")
    if has_leak:
        print("HYDRAULIC LEAK DETECTED")
        print(f"confidence: {max_conf:.2f}")
        banner_text = f"HYDRAULIC LEAK DETECTED ({max_conf:.2f})"
        banner_color = (0, 0, 255)
    else:
        print("NO HYDRAULIC LEAK DETECTED")
        banner_text = "NO HYDRAULIC LEAK DETECTED"
        banner_color = (0, 255, 0)
    print("----------------------------------------\n")
    
    cv2.rectangle(annotated_img, (0, 0), (annotated_img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(annotated_img, banner_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, banner_color, 2)
    
    save_path = os.path.join(save_dir, f"result_{os.path.basename(source_path)}")
    cv2.imwrite(save_path, annotated_img)
    print(f"Inference output saved to {save_path}")

def run_webcam_inference(model, conf_thresh=0.25):
    print("Starting webcam inference mode... Press 'q' to exit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access camera / webcam device.")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break
            
        results = model.predict(source=frame, conf=conf_thresh, verbose=False)
        result = results[0]
        
        annotated_frame = frame.copy()
        has_leak = False
        max_conf = 0.0
        
        if result.masks is not None and len(result.masks) > 0:
            has_leak = True
            masks = result.masks.xy
            boxes = result.boxes
            overlay = annotated_frame.copy()
            
            for i, mask in enumerate(masks):
                conf = float(boxes[i].conf[0].cpu().numpy())
                if conf > max_conf:
                    max_conf = conf
                pts = np.int32([mask])
                cv2.fillPoly(overlay, pts, (0, 0, 255))
                cv2.polylines(annotated_frame, pts, True, (0, 255, 255), 2)
                
            cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)
            
        banner_text = f"HYDRAULIC LEAK DETECTED ({max_conf:.2f})" if has_leak else "NO HYDRAULIC LEAK DETECTED"
        banner_color = (0, 0, 255) if has_leak else (0, 255, 0)
        
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(annotated_frame, banner_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, banner_color, 2)
        
        cv2.imshow("JCB Hydraulic Oil Leak Detection POC", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Inference for JCB Hydraulic Oil Leak Detection (YOLO11s-seg)")
    parser.add_argument("--source", type=str, help="Path to input image file")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam inference mode")
    parser.add_argument("--weights", type=str, default=os.path.join(ROOT_DIR, "runs", "yolo11s_jcb_leak_v6_seg", "weights", "best.pt"), help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if not args.source and not args.webcam:
        print("Please specify either --source <image_path> or --webcam")
        parser.print_help()
        return
        
    weights_path = args.weights
    if not os.path.exists(weights_path):
        print(f"Notice: Model weights file not found at {weights_path}")
        print("Using default yolo11s-seg.pt pretrained weights for inference demonstration...")
        weights_path = "yolo11s-seg.pt"
        
    model = YOLO(weights_path)
    
    if args.webcam:
        run_webcam_inference(model, conf_thresh=args.conf)
    elif args.source:
        run_image_inference(model, args.source, conf_thresh=args.conf)

if __name__ == "__main__":
    main()
