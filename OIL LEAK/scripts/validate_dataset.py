import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
RUNS_DIR = os.path.join(ROOT_DIR, "runs")

def validate_annotations():
    print("==================================================")
    print("        JCB DATASET VALIDATION REPORT             ")
    print("==================================================")
    
    splits = ["train", "val", "test"]
    
    report = {
        "total_images": 0,
        "by_modality": {"rgb": {"leak": 0, "no_leak": 0}, "uv": {"leak": 0, "no_leak": 0}},
        "by_split": {"train": 0, "val": 0, "test": 0},
        "missing_labels": [],
        "orphaned_labels": [],
        "invalid_polygons": [],
        "duplicate_filenames": set(),
        "resolutions": set(),
        "class_counts": {0: 0}
    }
    
    all_filenames = set()
    samples_for_contact_sheet = []
    
    for split in splits:
        img_dir = os.path.join(DATASET_DIR, split, "images")
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")
        
        if not os.path.exists(img_dir):
            print(f"Warning: {img_dir} does not exist.")
            continue
            
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        lbl_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
        
        lbl_basenames = {os.path.splitext(os.path.basename(f))[0] for f in lbl_files}
        
        for img_path in img_files:
            bname = os.path.splitext(os.path.basename(img_path))[0]
            ext = os.path.splitext(img_path)[1].lower()
            
            if ext not in ['.jpg', '.jpeg', '.png']:
                continue
                
            if bname in all_filenames:
                report["duplicate_filenames"].add(bname)
            all_filenames.add(bname)
            
            report["total_images"] += 1
            report["by_split"][split] += 1
            
            # Read image resolution
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    report["resolutions"].add((w, h))
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                w, h = 640, 640
                
            # Classify modality and leak status
            is_uv = "uv_" in bname.lower()
            is_leak = "_leak_" in bname.lower() and "_no_leak_" not in bname.lower()
            modality = "uv" if is_uv else "rgb"
            leak_status = "leak" if is_leak else "no_leak"
            
            report["by_modality"][modality][leak_status] += 1
            
            # Check matching label file
            lbl_path = os.path.join(lbl_dir, bname + ".txt")
            if not os.path.exists(lbl_path):
                report["missing_labels"].append(img_path)
            else:
                # Validate annotation content
                with open(lbl_path, "r") as f:
                    lines = f.readlines()
                    
                if not is_leak and len(lines) > 0:
                    print(f"Notice: No-leak image {bname} has non-empty label lines.")
                    
                polygon_coords = []
                for l_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(parts[0])
                        report["class_counts"][cls_id] = report["class_counts"].get(cls_id, 0) + 1
                        
                        coords = [float(x) for x in parts[1:]]
                        # Polygon requires pairs of (x, y)
                        if len(coords) < 6 or len(coords) % 2 != 0:
                            report["invalid_polygons"].append((lbl_path, f"Line {l_idx}: Invalid coordinate count ({len(coords)})"))
                        else:
                            # Check normalization 0-1
                            out_of_bounds = [c for c in coords if c < 0.0 or c > 1.05]
                            if out_of_bounds:
                                report["invalid_polygons"].append((lbl_path, f"Line {l_idx}: Out of bounds coordinates {out_of_bounds[:3]}"))
                            
                            # Convert to pixel coords for visualization sample
                            pts = [(int(coords[i] * w), int(coords[i+1] * h)) for i in range(0, len(coords), 2)]
                            polygon_coords.append(pts)
                    except Exception as ex:
                        report["invalid_polygons"].append((lbl_path, f"Line {l_idx}: Parse error {ex}"))
                        
            # Keep sample for visual contact sheet
            if len(samples_for_contact_sheet) < 24:
                samples_for_contact_sheet.append({
                    "img_path": img_path,
                    "bname": bname,
                    "modality": modality,
                    "leak_status": leak_status,
                    "split": split,
                    "polygons": polygon_coords if 'polygon_coords' in locals() else []
                })
                
        # Check orphaned labels
        img_basenames = {os.path.splitext(os.path.basename(f))[0] for f in img_files}
        for lbl_path in lbl_files:
            lbname = os.path.splitext(os.path.basename(lbl_path))[0]
            if lbname not in img_basenames:
                report["orphaned_labels"].append(lbl_path)

    # Print summary report
    print(f"Total Images Audited    : {report['total_images']}")
    print(f"Splits Distribution     :")
    print(f"  - Train               : {report['by_split']['train']}")
    print(f"  - Validation          : {report['by_split']['val']}")
    print(f"  - Test                : {report['by_split']['test']}")
    print(f"Modalities & Categories :")
    print(f"  - RGB Leak            : {report['by_modality']['rgb']['leak']}")
    print(f"  - RGB No-Leak         : {report['by_modality']['rgb']['no_leak']}")
    print(f"  - UV Leak             : {report['by_modality']['uv']['leak']}")
    print(f"  - UV No-Leak          : {report['by_modality']['uv']['no_leak']}")
    print(f"Class Index Counts      : {report['class_counts']}")
    print(f"Resolutions Found       : {list(report['resolutions'])}")
    print(f"Missing Label Files     : {len(report['missing_labels'])}")
    print(f"Orphaned Label Files    : {len(report['orphaned_labels'])}")
    print(f"Invalid Polygons        : {len(report['invalid_polygons'])}")
    print(f"Duplicate Filenames     : {len(report['duplicate_filenames'])}")
    
    if report["invalid_polygons"]:
        print("\nInvalid Polygons detail (first 5):")
        for item in report["invalid_polygons"][:5]:
            print(f"  - {item[0]}: {item[1]}")
            
    print("==================================================\n")
    
    # Generate visual contact sheet & sample overlays
    generate_contact_sheet_and_samples(samples_for_contact_sheet)
    
    return report

def generate_contact_sheet_and_samples(samples):
    os.makedirs(RUNS_DIR, exist_ok=True)
    sample_out_dir = os.path.join(RUNS_DIR, "dataset_samples")
    os.makedirs(sample_out_dir, exist_ok=True)
    
    print(f"Generating visual samples in {sample_out_dir}...")
    
    # Create contact sheet grid (4 rows x 6 cols = 24 samples)
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, sample in enumerate(samples[:24]):
        ax = axes[i]
        try:
            img = Image.open(sample["img_path"]).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")
            
            # Draw polygons if leak
            for poly in sample["polygons"]:
                if len(poly) >= 3:
                    # Fill semi-transparent red for RGB, fluorescent green for UV
                    fill_color = (0, 255, 0, 100) if sample["modality"] == "uv" else (255, 0, 0, 120)
                    outline_color = (0, 255, 0, 255) if sample["modality"] == "uv" else (255, 255, 0, 255)
                    draw.polygon(poly, fill=fill_color, outline=outline_color)
                    
            # Save individual sample image
            sample_save_path = os.path.join(sample_out_dir, f"{sample['split']}_{sample['bname']}.jpg")
            img.save(sample_save_path)
            
            ax.imshow(img)
            title = f"{sample['modality'].upper()} {sample['leak_status']}\n[{sample['split']}] {sample['bname'][:15]}"
            ax.set_title(title, fontsize=8, color='green' if sample['leak_status']=='no_leak' else 'red')
            ax.axis("off")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center")
            ax.axis("off")
            
    for j in range(len(samples), len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    contact_sheet_path = os.path.join(RUNS_DIR, "dataset_validation_contact_sheet.jpg")
    plt.savefig(contact_sheet_path, dpi=150)
    plt.close()
    print(f"Contact sheet saved to {contact_sheet_path}")

if __name__ == "__main__":
    validate_annotations()
