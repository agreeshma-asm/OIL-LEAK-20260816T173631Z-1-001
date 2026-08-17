import os
import glob
import csv
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATASET_DIR = os.path.join(ROOT_DIR, "dataset_jcb_clean", "dataset")
METADATA_CSV = os.path.join(CLEAN_DATASET_DIR, "metadata.csv")

def calculate_polygon_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0

def validate():
    print("=========================================================")
    print("     JCB CLEAN DATASET v6 AUDIT & VALIDATION REPORT      ")
    print("=========================================================")
    
    splits = ["train", "val", "test"]
    
    stats = {
        "total_images": 0,
        "rgb_count": 0,
        "uv_count": 0,
        "leak_count": 0,
        "no_leak_count": 0,
        "train_count": 0,
        "val_count": 0,
        "test_count": 0,
        "invalid_labels": 0,
        "missing_labels": 0,
        "unique_scenes": set(),
        "scene_split_map": {},
        "scene_pairing": {},
        "missing_labels_list": [],
        "invalid_labels_list": []
    }
    
    metadata = {}
    if os.path.exists(METADATA_CSV):
        with open(METADATA_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row["filename"]] = row
                
    for split in splits:
        img_dir = os.path.join(CLEAN_DATASET_DIR, split, "images")
        lbl_dir = os.path.join(CLEAN_DATASET_DIR, split, "labels")
        
        if not os.path.exists(img_dir):
            continue
            
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        
        for img_path in img_files:
            fname = os.path.basename(img_path)
            bname, ext = os.path.splitext(fname)
            if ext.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            stats["total_images"] += 1
            if split == "train":
                stats["train_count"] += 1
            elif split == "val":
                stats["val_count"] += 1
            elif split == "test":
                stats["test_count"] += 1
                
            # Read metadata
            meta = metadata.get(fname, {})
            scene_id = meta.get("scene_id")
            if not scene_id:
                # Infer scene id from filename, e.g. jcb_rgb_H1.jpg -> H1
                parts = bname.split("_")
                if len(parts) >= 3:
                    scene_id = parts[2]
                else:
                    scene_id = bname
            stats["unique_scenes"].add(scene_id)
            
            modality = meta.get("modality")
            if not modality:
                if "uv" in bname.lower():
                    modality = "uv"
                else:
                    modality = "rgb"
                    
            if modality == "uv":
                stats["uv_count"] += 1
            else:
                stats["rgb_count"] += 1
                
            class_name = meta.get("class_name")
            if not class_name:
                if "no_leak" in bname.lower() or "normal" in bname.lower():
                    class_name = "no_leak"
                else:
                    class_name = "hydraulic_oil_leak"
                    
            if class_name == "hydraulic_oil_leak":
                stats["leak_count"] += 1
            else:
                stats["no_leak_count"] += 1
                
            # Scene pairing validation
            stats["scene_pairing"].setdefault(scene_id, set()).add(modality)
            
            # Scene split validation
            if scene_id in stats["scene_split_map"]:
                if stats["scene_split_map"][scene_id] != split:
                    stats["invalid_labels_list"].append(
                        f"Data leakage! Scene {scene_id} is in both {stats['scene_split_map'][scene_id]} and {split} splits"
                    )
            else:
                stats["scene_split_map"][scene_id] = split
                
            # Check label file
            lbl_path = os.path.join(lbl_dir, bname + ".txt")
            if not os.path.exists(lbl_path):
                stats["missing_labels"] += 1
                stats["missing_labels_list"].append(fname)
            else:
                # Validate annotation
                with open(lbl_path, "r") as f:
                    lines = f.readlines()
                    
                if class_name == "no_leak":
                    if len(lines) > 0:
                        stats["invalid_labels"] += 1
                        stats["invalid_labels_list"].append(f"{fname}: Label file is not empty for no_leak class")
                else:
                    if len(lines) == 0:
                        stats["invalid_labels"] += 1
                        stats["invalid_labels_list"].append(f"{fname}: Label file is empty for leak class")
                    else:
                        for l_idx, line in enumerate(lines):
                            line_str = line.strip()
                            if not line_str:
                                continue
                            parts = line_str.split()
                            try:
                                cls_id = int(parts[0])
                                coords = [float(x) for x in parts[1:]]
                                
                                if cls_id != 0:
                                    stats["invalid_labels"] += 1
                                    stats["invalid_labels_list"].append(f"{fname}: Class ID is {cls_id}, expected 0")
                                    continue
                                    
                                if len(coords) < 6 or len(coords) % 2 != 0:
                                    stats["invalid_labels"] += 1
                                    stats["invalid_labels_list"].append(f"{fname}: Line {l_idx} has invalid coord count {len(coords)}")
                                    continue
                                    
                                oob = [c for c in coords if c < 0.0 or c > 1.0001]
                                if oob:
                                    stats["invalid_labels"] += 1
                                    stats["invalid_labels_list"].append(f"{fname}: Line {l_idx} has coords out of [0, 1] bounds")
                                    continue
                                    
                                pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                                area = calculate_polygon_area(pts)
                                if area <= 0.0:
                                    stats["invalid_labels"] += 1
                                    stats["invalid_labels_list"].append(f"{fname}: Line {l_idx} has zero area polygon")
                            except Exception as e:
                                stats["invalid_labels"] += 1
                                stats["invalid_labels_list"].append(f"{fname}: Parse error {e}")
                                
    # Check RGB/UV scene pairing completeness
    unpaired_scenes = []
    for sid, modalities in stats["scene_pairing"].items():
        if len(modalities) != 2:
            unpaired_scenes.append(f"Scene {sid} has missing modality. Found: {list(modalities)}")
            
    print("\n--- FINAL STATISTICS REPORT ---")
    print(f"total images    : {stats['total_images']}")
    print(f"RGB count       : {stats['rgb_count']}")
    print(f"UV count        : {stats['uv_count']}")
    print(f"leak count      : {stats['leak_count']}")
    print(f"no-leak count   : {stats['no_leak_count']}")
    print(f"train count     : {stats['train_count']}")
    print(f"val count       : {stats['val_count']}")
    print(f"test count      : {stats['test_count']}")
    print(f"invalid labels  : {stats['invalid_labels']}")
    print(f"missing labels  : {stats['missing_labels']}")
    print(f"unique scenes   : {len(stats['unique_scenes'])}")
    print("--------------------------------\n")
    
    if unpaired_scenes:
        print("Scene Pairing Issues:")
        for issue in unpaired_scenes:
            print(f"  - {issue}")
    else:
        print("Verify RGB/UV scene pairing: PASS (All scenes have both RGB and UV images)")
        
    if stats["invalid_labels_list"]:
        print("\nInvalid Label Details:")
        for item in stats["invalid_labels_list"]:
            print(f"  - {item}")
    else:
        print("Verify train/val/test scene-level separation: PASS (No data leakage across splits)")
        
    return stats

if __name__ == "__main__":
    validate()
