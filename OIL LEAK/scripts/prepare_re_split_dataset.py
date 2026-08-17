import os
import cv2
import numpy as np
import shutil
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "dataset_jcb_separated", "JCB_Synthetic_Starter_Separated")
TGT_DIR = os.path.join(ROOT_DIR, "dataset")

def re_split_dataset():
    print("--- Re-splitting separated dataset into Train/Val/Test (keeping pairs together) ---")
    
    # We clear the existing dataset folder to prevent old files from mixing
    if os.path.exists(TGT_DIR):
        shutil.rmtree(TGT_DIR)
        
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(TGT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(TGT_DIR, split, "labels"), exist_ok=True)
        
    # We define the pair assignment mapping:
    # 10 pairs (represented by their prefix and condition):
    # Train: 01_leak, 01_no_leak, 02_leak, 03_no_leak, 04_leak, 05_no_leak (6 pairs)
    # Val:   02_no_leak, 03_leak (2 pairs)
    # Test:  04_no_leak, 05_leak (2 pairs)
    
    pair_assignments = {
        # Train
        "01_leak": "train",
        "01_no_leak": "train",
        "02_leak": "train",
        "03_no_leak": "train",
        "04_leak": "train",
        "05_no_leak": "train",
        # Val
        "02_no_leak": "val",
        "03_leak": "val",
        # Test
        "04_no_leak": "test",
        "05_leak": "test"
    }
    
    lower_green = np.array([30, 80, 80])
    upper_green = np.array([85, 255, 255])
    
    rgb_dir = os.path.join(SRC_DIR, "rgb")
    
    for filename in os.listdir(rgb_dir):
        if not filename.endswith(".jpg"):
            continue
            
        # Determine pair key (e.g. "01_leak" or "01_no_leak")
        prefix = filename[:2]
        is_leak = "no_leak" not in filename.lower()
        condition_str = "leak" if is_leak else "no_leak"
        pair_key = f"{prefix}_{condition_str}"
        
        split = pair_assignments.get(pair_key, "train")
        
        bname = os.path.splitext(filename)[0]
        rgb_img_path = os.path.join(rgb_dir, filename)
        uv_img_name = filename.replace("rgb", "uv")
        uv_img_path = os.path.join(SRC_DIR, "uv", uv_img_name)
        
        # Target paths
        tgt_rgb_img = os.path.join(TGT_DIR, split, "images", filename)
        tgt_rgb_lbl = os.path.join(TGT_DIR, split, "labels", bname + ".txt")
        
        tgt_uv_img = os.path.join(TGT_DIR, split, "images", uv_img_name)
        tgt_uv_lbl = os.path.join(TGT_DIR, split, "labels", os.path.splitext(uv_img_name)[0] + ".txt")
        
        # Copy images
        shutil.copy2(rgb_img_path, tgt_rgb_img)
        shutil.copy2(uv_img_path, tgt_uv_img)
        
        # Create labels
        if not is_leak:
            with open(tgt_rgb_lbl, "w") as f:
                pass
            with open(tgt_uv_lbl, "w") as f:
                pass
        else:
            # Segment UV green region
            img_uv = cv2.imread(uv_img_path)
            h_uv, w_uv = img_uv.shape[:2]
            hsv = cv2.cvtColor(img_uv, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            lines = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > 4.0:
                    epsilon = 0.005 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, epsilon, True)
                    if len(approx) >= 3:
                        coords = []
                        for pt in approx:
                            px, py = pt[0]
                            coords.append(f"{px / w_uv:.6f} {py / h_uv:.6f}")
                        lines.append(f"0 {' '.join(coords)}\n")
                        
            with open(tgt_rgb_lbl, "w") as f:
                f.writelines(lines)
            with open(tgt_uv_lbl, "w") as f:
                f.writelines(lines)
                
    # data.yaml
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
        
    print("Dataset re-splitting completed successfully.")

if __name__ == "__main__":
    re_split_dataset()
