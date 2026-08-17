import os
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_V1_DIR = os.path.join(ROOT_DIR, "dataset_jcb_v1", "dataset")
METADATA_CSV = os.path.join(DATASET_V1_DIR, "metadata.csv")
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

def audit_and_validate():
    print("=================================================================")
    print("      JCB HYDRAULIC LEAK DATASET (v1) AUDIT & VALIDATION         ")
    print("=================================================================")
    
    splits = ["train", "val", "test"]
    
    stats = {
        "total_images": 0,
        "rgb_images": 0,
        "uv_images": 0,
        "leak_images": 0,
        "no_leak_images": 0,
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "unique_base_scenes": set(),
        "valid_annotations": 0,
        "invalid_annotations": 0,
        "rejected_ambiguous": 0,
        "missing_labels": [],
        "orphaned_labels": [],
        "invalid_polygons_detail": [],
        "out_of_bounds": [],
        "zero_area_polygons": [],
        "resolutions": set(),
        "scene_split_violations": [],
        "duplicates": set()
    }
    
    scene_to_split = {}
    all_filenames = set()
    samples_for_contact_sheet = []
    
    # Load metadata.csv if available
    metadata = {}
    if os.path.exists(METADATA_CSV):
        with open(METADATA_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row["filename"]] = row
                
    for split in splits:
        img_dir = os.path.join(DATASET_V1_DIR, split, "images")
        lbl_dir = os.path.join(DATASET_V1_DIR, split, "labels")
        
        if not os.path.exists(img_dir):
            continue
            
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        lbl_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
        
        img_basenames = set()
        
        for img_path in img_files:
            bname = os.path.splitext(os.path.basename(img_path))[0]
            ext = os.path.splitext(img_path)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                continue
                
            fname = os.path.basename(img_path)
            img_basenames.add(bname)
            
            if fname in all_filenames:
                stats["duplicates"].add(fname)
            all_filenames.add(fname)
            
            stats["total_images"] += 1
            stats["split_counts"][split] += 1
            
            # Resolution check
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    stats["resolutions"].add((w, h))
            except Exception as e:
                w, h = 640, 640
                
            meta_info = metadata.get(fname, {})
            modality = meta_info.get("modality", "")
            if not modality:
                if bname.startswith("uv_"):
                    modality = "uv"
                else:
                    modality = "rgb"
                    
            if modality == "uv":
                stats["uv_images"] += 1
            else:
                stats["rgb_images"] += 1
                
            # Correct leak vs no-leak check
            is_no_leak = "no_leak" in bname.lower() or meta_info.get("class_name") == "no_leak"
            is_leak = not is_no_leak
            
            if is_leak:
                stats["leak_images"] += 1
            else:
                stats["no_leak_images"] += 1
                
            group_code = meta_info.get("group") or meta_info.get("code")
            if not group_code:
                parts = bname.split("_")
                if len(parts) >= 3:
                    group_code = "_".join(parts[:3])
                else:
                    group_code = bname
            stats["unique_base_scenes"].add(group_code)
            
            if group_code in scene_to_split and scene_to_split[group_code] != split:
                stats["scene_split_violations"].append((group_code, scene_to_split[group_code], split))
            else:
                scene_to_split[group_code] = split
                
            # Annotation checks
            lbl_path = os.path.join(lbl_dir, bname + ".txt")
            if not os.path.exists(lbl_path):
                stats["missing_labels"].append(img_path)
            else:
                with open(lbl_path, "r") as f:
                    lines = f.readlines()
                    
                polygon_coords = []
                if is_no_leak:
                    if len(lines) > 0:
                        stats["invalid_annotations"] += 1
                        stats["invalid_polygons_detail"].append((lbl_path, "No-leak image has non-empty label file"))
                    else:
                        stats["valid_annotations"] += 1
                else:
                    if len(lines) == 0:
                        stats["invalid_annotations"] += 1
                        stats["invalid_polygons_detail"].append((lbl_path, "Leak image has empty label file"))
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
                                    stats["invalid_annotations"] += 1
                                    stats["invalid_polygons_detail"].append((lbl_path, f"Class ID is {cls_id}, expected 0"))
                                    continue
                                    
                                if len(coords) < 6 or len(coords) % 2 != 0:
                                    stats["invalid_annotations"] += 1
                                    stats["invalid_polygons_detail"].append((lbl_path, f"Line {l_idx}: Invalid coord count ({len(coords)})"))
                                    continue
                                    
                                oob = [c for c in coords if c < 0.0 or c > 1.0001]
                                if oob:
                                    stats["out_of_bounds"].append((lbl_path, f"Out of bounds coords: {oob[:4]}"))
                                    
                                pts = [(coords[i] * w, coords[i+1] * h) for i in range(0, len(coords), 2)]
                                area = calculate_polygon_area(pts)
                                if area <= 0.5:
                                    stats["zero_area_polygons"].append((lbl_path, f"Zero/tiny area polygon: {area}"))
                                    stats["invalid_annotations"] += 1
                                else:
                                    stats["valid_annotations"] += 1
                                    polygon_coords.append(pts)
                            except Exception as ex:
                                stats["invalid_annotations"] += 1
                                stats["invalid_polygons_detail"].append((lbl_path, f"Parse error: {ex}"))
                                
            if len(samples_for_contact_sheet) < 30:
                samples_for_contact_sheet.append({
                    "img_path": img_path,
                    "bname": bname,
                    "modality": modality,
                    "is_leak": is_leak,
                    "split": split,
                    "polygons": polygon_coords if 'polygon_coords' in locals() else []
                })
                
        for lbl_path in lbl_files:
            lbname = os.path.splitext(os.path.basename(lbl_path))[0]
            if lbname not in img_basenames:
                stats["orphaned_labels"].append(lbl_path)

    print("\n=================================================================")
    print("      EXACT DATASET METRICS (STEP 9 FINAL REPORT)               ")
    print("=================================================================")
    print(f"1. Total image count             : {stats['total_images']}")
    print(f"2. RGB image count               : {stats['rgb_images']}")
    print(f"3. UV image count                : {stats['uv_images']}")
    print(f"4. Leak image count              : {stats['leak_images']}")
    print(f"5. No-leak image count           : {stats['no_leak_images']}")
    print(f"6. Train count                   : {stats['split_counts']['train']}")
    print(f"7. Validation count              : {stats['split_counts']['val']}")
    print(f"8. Test count                    : {stats['split_counts']['test']}")
    print(f"9. Number of unique base scenes  : {len(stats['unique_base_scenes'])}")
    print(f"10. Number of valid annotations  : {stats['valid_annotations']}")
    print(f"11. Number of invalid annotations: {stats['invalid_annotations']}")
    print(f"12. Rejected / ambiguous images  : {stats['rejected_ambiguous']}")
    print("-----------------------------------------------------------------")
    print(f"Resolutions Found          : {list(stats['resolutions'])}")
    print(f"Missing Label Files        : {len(stats['missing_labels'])}")
    print(f"Orphaned Label Files       : {len(stats['orphaned_labels'])}")
    print(f"Out-of-Bounds Coords       : {len(stats['out_of_bounds'])}")
    print(f"Zero-Area Polygons         : {len(stats['zero_area_polygons'])}")
    print(f"Scene-Split Violations     : {len(stats['scene_split_violations'])}")
    print(f"Duplicate Filenames        : {len(stats['duplicates'])}")
    print("=================================================================\n")
    
    generate_annotated_preview_sheet(samples_for_contact_sheet)
    return stats

def generate_annotated_preview_sheet(samples):
    os.makedirs(RUNS_DIR, exist_ok=True)
    out_dir = os.path.join(RUNS_DIR, "dataset_jcb_v1_samples")
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(5, 6, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, sample in enumerate(samples[:30]):
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
                    
            sample_save_path = os.path.join(out_dir, f"{sample['split']}_{sample['bname']}.jpg")
            img.save(sample_save_path)
            
            ax.imshow(img)
            status_str = "LEAK" if sample["is_leak"] else "NO-LEAK"
            color = 'red' if sample["is_leak"] else 'green'
            title = f"{sample['modality'].upper()} {status_str}\n[{sample['split']}] {sample['bname'][:15]}"
            ax.set_title(title, fontsize=8, color=color, fontweight='bold')
            ax.axis("off")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center")
            ax.axis("off")
            
    for j in range(len(samples), len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    contact_sheet_path = os.path.join(RUNS_DIR, "dataset_jcb_v1_contact_sheet.jpg")
    plt.savefig(contact_sheet_path, dpi=150)
    plt.close()
    print(f"Annotated preview contact sheet saved to {contact_sheet_path}")

if __name__ == "__main__":
    audit_and_validate()
