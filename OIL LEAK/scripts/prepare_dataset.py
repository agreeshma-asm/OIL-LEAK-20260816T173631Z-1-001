import os
import glob
import shutil
import random
from PIL import Image
import yaml

# Source directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_DATASET_DIR = os.path.join(ROOT_DIR, "dataset_from_zip")
REF_IMAGES_DIR = os.path.join(ROOT_DIR, "reference_images")
TARGET_DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

def crop_reference_collages():
    """
    Extract individual visual scene panels from the reference collage sheets,
    removing headers, captions, numbering tags, and borders.
    """
    print("--- Cropping individual visual scenes from reference collages ---")
    cropped_out_dir = os.path.join(REF_IMAGES_DIR, "cropped_panels")
    os.makedirs(cropped_out_dir, exist_ok=True)
    
    # List reference collage files
    rgb1_path = os.path.join(REF_IMAGES_DIR, "synthetic_jcb_rgb", "a_large_collage_dataset_style_image_with_multipl.png")
    rgb2_path = os.path.join(REF_IMAGES_DIR, "synthetic_jcb_rgb", "a_large_composite_infographic_image_showing_a_data.png")
    uv1_path = os.path.join(REF_IMAGES_DIR, "synthetic_jcb_uv", "a_large_multi_panel_infographic_collage_image_sh.png")
    
    cropped_count = 0
    
    # Process RGB Collage 1 & 2 if they exist
    for path_idx, img_path in enumerate([rgb1_path, rgb2_path, uv1_path]):
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path)
            w, h = img.size
            # The collages are 1536x1024 grids.
            # We define panel grid boundaries to cleanly crop visual contents without text captions/header banners
            # Grid structure: ~5 columns x 6-8 rows or similar layout
            
            rows = 5
            cols = 5
            pad_x = 20
            pad_y = 35  # skip header/caption areas
            
            cell_w = (w - 2 * pad_x) // cols
            cell_h = (h - 2 * pad_y) // rows
            
            is_uv = "uv" in img_path.lower()
            modality = "uv" if is_uv else "rgb"
            
            for r in range(rows):
                for c in range(cols):
                    left = pad_x + c * cell_w + 12
                    top = pad_y + r * cell_h + 30  # trim top number badge & title header
                    right = pad_x + (c + 1) * cell_w - 12
                    bottom = pad_y + (r + 1) * cell_h - 25  # trim bottom caption text
                    
                    if right > left + 50 and bottom > top + 40:
                        cropped = img.crop((left, top, right, bottom))
                        save_name = f"ref_crop_{modality}_{path_idx}_{r}_{c}.jpg"
                        save_path = os.path.join(cropped_out_dir, save_name)
                        cropped.save(save_path, quality=95)
                        cropped_count += 1
        except Exception as e:
            print(f"Error cropping {img_path}: {e}")
            
    print(f"Successfully cropped {cropped_count} individual visual panels from reference sheets into {cropped_out_dir}")

def prepare_yolo_dataset():
    """
    Audit dataset_from_zip, convert class ID from 1 -> 0 (hydraulic_oil_leak),
    create empty labels for no-leak images, and structure into dataset/ with train, val, test.
    """
    print("\n--- Auditing and structuring final YOLO11s-seg dataset ---")
    
    if not os.path.exists(ZIP_DATASET_DIR):
        zip_path = os.path.join(ROOT_DIR, "JCB_Hydraulic_Leak_Dataset_FROM_THESE_IMAGES_v1.zip")
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path} to {ZIP_DATASET_DIR}...")
            import zipfile
            temp_extract_dir = os.path.join(ROOT_DIR, "temp_extract")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if member.startswith("dataset/"):
                        zip_ref.extract(member, temp_extract_dir)
            if os.path.exists(ZIP_DATASET_DIR):
                shutil.rmtree(ZIP_DATASET_DIR)
            shutil.move(os.path.join(temp_extract_dir, "dataset"), ZIP_DATASET_DIR)
            shutil.rmtree(temp_extract_dir)
            print("Extraction complete.")
    
    # Create target directory structure
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(TARGET_DATASET_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DATASET_DIR, split, "labels"), exist_ok=True)
        
    source_splits = ["train", "val", "test"]
    
    total_audited = 0
    stats = {
        "rgb_leak": 0,
        "rgb_no_leak": 0,
        "uv_leak": 0,
        "uv_no_leak": 0,
        "by_split": {"train": 0, "val": 0, "test": 0}
    }
    
    all_samples = []
    
    for split in source_splits:
        split_img_dir = os.path.join(ZIP_DATASET_DIR, split, "images")
        split_lbl_dir = os.path.join(ZIP_DATASET_DIR, split, "labels")
        
        if not os.path.exists(split_img_dir):
            continue
            
        for img_name in os.listdir(split_img_dir):
            if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            base_name = os.path.splitext(img_name)[0]
            lbl_name = base_name + ".txt"
            
            src_img_path = os.path.join(split_img_dir, img_name)
            src_lbl_path = os.path.join(split_lbl_dir, lbl_name)
            
            # Categorize modality and leak status
            is_uv = "uv_" in img_name.lower()
            is_leak = "_leak_" in img_name.lower() and "_no_leak_" not in img_name.lower()
            
            modality = "uv" if is_uv else "rgb"
            leak_status = "leak" if is_leak else "no_leak"
            
            cat_key = f"{modality}_{leak_status}"
            stats[cat_key] += 1
            total_audited += 1
            
            all_samples.append({
                "src_img": src_img_path,
                "src_lbl": src_lbl_path,
                "name": img_name,
                "base_name": base_name,
                "modality": modality,
                "is_leak": is_leak,
                "orig_split": split
            })

    print(f"Audited total {total_audited} images in ZIP dataset:")
    print(f"  RGB Leak: {stats['rgb_leak']}")
    print(f"  RGB No-Leak: {stats['rgb_no_leak']}")
    print(f"  UV Leak: {stats['uv_leak']}")
    print(f"  UV No-Leak: {stats['uv_no_leak']}")
    
    # We maintain or re-balance splits (70% train, 20% val, 10% test) by scene/category
    # Group samples by category to ensure balanced distribution across splits
    random.seed(42)
    
    categories = {}
    for sample in all_samples:
        key = f"{sample['modality']}_{'leak' if sample['is_leak'] else 'no_leak'}"
        categories.setdefault(key, []).append(sample)
        
    final_train, final_val, final_test = [], [], []
    
    for cat_name, items in categories.items():
        random.shuffle(items)
        n = len(items)
        n_test = int(n * 0.10)
        n_val = int(n * 0.20)
        n_train = n - n_val - n_test
        
        final_train.extend(items[:n_train])
        final_val.extend(items[n_train:n_train + n_val])
        final_test.extend(items[n_train + n_val:])
        
    split_map = {
        "train": final_train,
        "val": final_val,
        "test": final_test
    }
    
    for split, samples in split_map.items():
        stats["by_split"][split] = len(samples)
        for s in samples:
            tgt_img_path = os.path.join(TARGET_DATASET_DIR, split, "images", s["name"])
            tgt_lbl_path = os.path.join(TARGET_DATASET_DIR, split, "labels", s["base_name"] + ".txt")
            
            # Copy image
            shutil.copy2(s["src_img"], tgt_img_path)
            
            # Process label file:
            # 1. Convert class 1 -> 0 (hydraulic_oil_leak)
            # 2. For no-leak images: write empty text file
            if s["is_leak"] and os.path.exists(s["src_lbl"]):
                with open(s["src_lbl"], "r") as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 6:  # polygon segmentation format: class x1 y1 x2 y2 ...
                        # Change class index to 0
                        parts[0] = "0"
                        new_lines.append(" ".join(parts) + "\n")
                        
                with open(tgt_lbl_path, "w") as f:
                    f.writelines(new_lines)
            else:
                # Empty file for no-leak
                with open(tgt_lbl_path, "w") as f:
                    pass

    print(f"\nFinal dataset populated in: {TARGET_DATASET_DIR}")
    print(f"  Train: {stats['by_split']['train']} images")
    print(f"  Val:   {stats['by_split']['val']} images")
    print(f"  Test:  {stats['by_split']['test']} images")
    
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
    
    yaml_path = os.path.join(TARGET_DATASET_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)
        
    print(f"Created dataset data.yaml at {yaml_path}")
    
    # Create dataset README.md
    readme_path = os.path.join(TARGET_DATASET_DIR, "README.md")
    readme_content = """# JCB Hydraulic Oil Leak Detection Dataset (YOLO11s-seg)

Proof-of-concept synthetic dataset for JCB hydraulic oil leak instance segmentation.

## Modalities
- **RGB**: Visible natural-light inspection images with oily wet regions, drips, and surface trails.
- **UV**: UV-assisted inspection images with simulated fluorescent yellow/green leak signals.

## Class Definition
- `0: hydraulic_oil_leak`
- Negative / no-leak images are annotated with empty label files (`.txt`).

## Split Summary
- **Train**: ~70% of dataset
- **Val**: ~20% of dataset
- **Test**: ~10% of dataset (scene-separated holdout test set)

## Annotation Format
YOLO instance segmentation format:
`0 x1 y1 x2 y2 x3 y3 ... xN yN`
Coordinates are normalized [0, 1].

## Scientific & POC Limitations Disclaimer
1. This dataset uses synthetic JCB images for proof-of-concept pipeline development.
2. The UV images simulate a fluorescent leak indication for inspection workflow modeling; ordinary hydraulic oil does NOT naturally fluoresce under UV without fluorescent tracer dyes added.
3. Model evaluation on synthetic data should be validated with real field test samples.
"""
    with open(readme_path, "w") as f:
        f.write(readme_content)
        
    print(f"Created dataset README.md at {readme_path}")

if __name__ == "__main__":
    crop_reference_collages()
    prepare_yolo_dataset()
