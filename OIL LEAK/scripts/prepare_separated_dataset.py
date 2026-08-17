import os
import cv2
import numpy as np
import shutil
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "dataset_jcb_separated", "JCB_Synthetic_Starter_Separated")
TGT_DIR = os.path.join(ROOT_DIR, "dataset")

def segment_uv_leaks():
    print("--- Extracting segmentation masks from UV leak images ---")
    
    lower_green = np.array([30, 80, 80])
    upper_green = np.array([85, 255, 255])
    
    # We define splits based on Category prefix
    # Category 01, 02, 03 -> train
    # Category 04 -> val
    # Category 05 -> test
    category_splits = {
        "01": "train",
        "02": "train",
        "03": "train",
        "04": "val",
        "05": "test"
    }
    
    # Create targets
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(TGT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(TGT_DIR, split, "labels"), exist_ok=True)
        
    # Process images
    rgb_dir = os.path.join(SRC_DIR, "rgb")
    uv_dir = os.path.join(SRC_DIR, "uv")
    
    for filename in os.listdir(rgb_dir):
        if not filename.endswith(".jpg"):
            continue
            
        prefix = filename[:2] # "01", "02", etc.
        split = category_splits.get(prefix, "train")
        
        is_leak = "no_leak" not in filename.lower()
        bname = os.path.splitext(filename)[0]
        
        # Paths
        rgb_img_path = os.path.join(rgb_dir, filename)
        uv_img_name = filename.replace("rgb", "uv")
        uv_img_path = os.path.join(uv_dir, uv_img_name)
        
        # Target paths
        tgt_rgb_img = os.path.join(TGT_DIR, split, "images", filename)
        tgt_rgb_lbl = os.path.join(TGT_DIR, split, "labels", bname + ".txt")
        
        tgt_uv_img = os.path.join(TGT_DIR, split, "images", uv_img_name)
        tgt_uv_lbl = os.path.join(TGT_DIR, split, "labels", os.path.splitext(uv_img_name)[0] + ".txt")
        
        # Copy images
        shutil.copy2(rgb_img_path, tgt_rgb_img)
        shutil.copy2(uv_img_path, tgt_uv_img)
        
        # Generate labels
        if not is_leak:
            # Clean/no-leak images receive empty labels
            with open(tgt_rgb_lbl, "w") as f:
                pass
            with open(tgt_uv_lbl, "w") as f:
                pass
        else:
            # Extract green fluorescent leak contours from UV leak image
            img_uv = cv2.imread(uv_img_path)
            h_uv, w_uv = img_uv.shape[:2]
            hsv = cv2.cvtColor(img_uv, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            lines = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > 4.0: # Filter out tiny noise points
                    # Simplify contour to reduce polygon point count
                    epsilon = 0.005 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, epsilon, True)
                    
                    if len(approx) >= 3:
                        # Convert to normalized coordinates
                        coords = []
                        for pt in approx:
                            px, py = pt[0]
                            coords.append(f"{px / w_uv:.6f} {py / h_uv:.6f}")
                        lines.append(f"0 {' '.join(coords)}\n")
                        
            # Write to label files for both RGB and UV (they share the same normalized coordinates)
            with open(tgt_rgb_lbl, "w") as f:
                f.writelines(lines)
            with open(tgt_uv_lbl, "w") as f:
                f.writelines(lines)
                
    # Create data.yaml
    data_yaml_content = {
        "path": "./dataset",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {
            0: "hydraulic_oil_leak"
        }
    }
    with open(os.path.join(TGT_DIR, "data.yaml"), "w") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)
        
    print(f"Dataset generated at {TGT_DIR} with 20 images.")

if __name__ == "__main__":
    segment_uv_leaks()
