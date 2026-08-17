import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
RUNS_DIR = os.path.join(ROOT_DIR, "runs")

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
    print("    JCB RE-SPLIT STARTER DATASET AUDIT & VALIDATION      ")
    print("=========================================================")
    
    splits = ["train", "val", "test"]
    
    stats = {
        "total_images": 0,
        "rgb_count": 0,
        "uv_count": 0,
        "leak_count": 0,
        "no_leak_count": 0,
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "annotations_per_split": {"train": 0, "val": 0, "test": 0},
        "resolutions": {},
        "unique_scenes": set(),
        "invalid_labels": 0,
        "missing_labels": 0,
        "invalid_labels_list": []
    }
    
    samples_for_contact_sheet = []
    
    for split in splits:
        img_dir = os.path.join(DATASET_DIR, split, "images")
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")
        
        if not os.path.exists(img_dir):
            continue
            
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        
        for img_path in img_files:
            fname = os.path.basename(img_path)
            bname, ext = os.path.splitext(fname)
            if ext.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            stats["total_images"] += 1
            stats["split_counts"][split] += 1
            
            # Read exact image resolution
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    stats["resolutions"][fname] = (w, h)
            except Exception as e:
                w, h = 0, 0
                
            is_uv = "uv" in bname.lower()
            if is_uv:
                stats["uv_count"] += 1
            else:
                stats["rgb_count"] += 1
                
            is_leak = "no_leak" not in bname.lower()
            if is_leak:
                stats["leak_count"] += 1
            else:
                stats["no_leak_count"] += 1
                
            # Scene code
            parts = bname.split("_")
            if len(parts) >= 3:
                scene_id = "_".join(parts[:3])
            else:
                scene_id = bname
            stats["unique_scenes"].add(scene_id)
            
            # Check labels
            lbl_path = os.path.join(lbl_dir, bname + ".txt")
            polygon_coords = []
            if not os.path.exists(lbl_path):
                stats["missing_labels"] += 1
                stats["invalid_labels_list"].append(f"{fname}: Label file missing")
            else:
                with open(lbl_path, "r") as f:
                    lines = f.readlines()
                    
                if not is_leak:
                    if len(lines) > 0:
                        stats["invalid_labels"] += 1
                        stats["invalid_labels_list"].append(f"{fname}: No-leak image has labels")
                else:
                    if len(lines) == 0:
                        stats["invalid_labels"] += 1
                        stats["invalid_labels_list"].append(f"{fname}: Leak image has empty labels")
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
                                    stats["invalid_labels_list"].append(f"{fname}: Class ID {cls_id}, expected 0")
                                    continue
                                if len(coords) < 6 or len(coords) % 2 != 0:
                                    stats["invalid_labels"] += 1
                                    stats["invalid_labels_list"].append(f"{fname}: Invalid coord count {len(coords)}")
                                    continue
                                    
                                pts = [(coords[i] * w, coords[i+1] * h) for i in range(0, len(coords), 2)]
                                area = calculate_polygon_area(pts)
                                if area <= 0.0:
                                    stats["invalid_labels"] += 1
                                    stats["invalid_labels_list"].append(f"{fname}: Zero area polygon")
                                else:
                                    polygon_coords.append(pts)
                                    stats["annotations_per_split"][split] += 1
                            except Exception as e:
                                stats["invalid_labels"] += 1
                                stats["invalid_labels_list"].append(f"{fname}: Parse error {e}")
                                
            samples_for_contact_sheet.append({
                "img_path": img_path,
                "bname": bname,
                "modality": "uv" if is_uv else "rgb",
                "is_leak": is_leak,
                "split": split,
                "polygons": polygon_coords
            })

    # Find resolution range
    res_list = list(stats["resolutions"].values())
    w_min = min(r[0] for r in res_list)
    w_max = max(r[0] for r in res_list)
    h_min = min(r[1] for r in res_list)
    h_max = max(r[1] for r in res_list)
    res_range = f"{w_min}x{h_min} to {w_max}x{h_max}"

    # Print exact resolution of every image
    print("\n--- EXACT RESOLUTION OF EVERY IMAGE ---")
    for name, res in sorted(stats["resolutions"].items()):
        print(f"  {name:<32}: {res[0]} x {res[1]}")
    print("----------------------------------------\n")
    
    print("--- FINAL STATISTICS REPORT ---")
    print(f"total images        : {stats['total_images']}")
    print(f"RGB images          : {stats['rgb_count']}")
    print(f"UV images           : {stats['uv_count']}")
    print(f"leak images         : {stats['leak_count']}")
    print(f"no-leak images      : {stats['no_leak_count']}")
    print(f"unique scenes       : {len(stats['unique_scenes'])}")
    print(f"resolution range    : {res_range}")
    print(f"train/val/test counts: Train={stats['split_counts']['train']}, Val={stats['split_counts']['val']}, Test={stats['split_counts']['test']}")
    print(f"annotations per split: Train={stats['annotations_per_split']['train']}, Val={stats['annotations_per_split']['val']}, Test={stats['annotations_per_split']['test']}")
    print("--------------------------------\n")
    
    if stats["invalid_labels_list"]:
        print("Invalid label warnings:")
        for item in stats["invalid_labels_list"]:
            print(f"  - {item}")
    else:
        print("All annotations conform to YOLO11 segmentation format: PASS")
        
    generate_contact_sheet(samples_for_contact_sheet)
    return stats

def generate_contact_sheet(samples):
    os.makedirs(RUNS_DIR, exist_ok=True)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, sample in enumerate(samples[:20]):
        ax = axes[i]
        try:
            img = Image.open(sample["img_path"]).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")
            
            for poly in sample["polygons"]:
                if len(poly) >= 3:
                    pts_tuples = [(int(p[0]), int(p[1])) for p in poly]
                    if sample["modality"] == "uv":
                        fill = (0, 255, 0, 120)
                        outline = (0, 255, 0, 255)
                    else:
                        fill = (255, 0, 0, 120)
                        outline = (255, 255, 0, 255)
                    draw.polygon(pts_tuples, fill=fill, outline=outline)
                    
            ax.imshow(img)
            status = "LEAK" if sample["is_leak"] else "NO-LEAK"
            color = 'red' if sample["is_leak"] else 'green'
            title = f"{sample['modality'].upper()} {status}\n[{sample['split']}] {sample['bname'][:22]}"
            ax.set_title(title, fontsize=9, color=color, fontweight='bold')
            ax.axis("off")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center")
            ax.axis("off")
            
    for j in range(len(samples), len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    contact_sheet_path = os.path.join(RUNS_DIR, "re_split_dataset_validation_sheet.jpg")
    plt.savefig(contact_sheet_path, dpi=150)
    plt.close()
    print(f"\nAnnotated preview contact sheet saved to {contact_sheet_path}")

if __name__ == "__main__":
    validate()
