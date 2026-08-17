# JCB Hydraulic Oil Leak Detection Proof of Concept (YOLO11s-seg)

This project provides a complete **Proof of Concept (POC)** for detecting hydraulic oil leaks on JCB construction machinery using **YOLO11s-seg** instance segmentation. 

### Key Highlights
- **Model**: YOLO11s-seg instance segmentation trained on CPU/GPU for 60 epochs.
- **Dataset**: Custom-separated JCB synthetic dataset (`JCB_Synthetic_Starter_Separated_v1`) containing:
  - 10 RGB Modality images (5 leak, 5 no-leak)
  - 10 UV Modality images (5 leak, 5 no-leak)
  - Clean train (12 images), validation (4 images), and test (4 images) splits.
- **Modality Support**: Supports both standard **Visible (RGB)** light inspections and **UV fluorescence** light inspections.
- **Separated Architecture**:
  - **Backend**: Flask API server (`app.py`) on port 5000 that loads the trained weights checkpoint, runs instance segmentation, renders modality-specific overlays (Red for RGB, Fluorescent Green for UV), and outputs detection metadata.
  - **Frontend**: Vite + React + TypeScript web application (`frontend/`) on port 3000 that provides a high-fidelity industrial dashboard UI, secure context webcam capture, scanning laser animations, real-time threshold adjustments (translating to LEAK, REVIEW, or NO LEAK states), and image-upload fallback capabilities.

---

## Dataset Architecture

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
└── README.md
```

### Classes
- `0: hydraulic_oil_leak`
- Negative / clean / no-leak images use **empty** annotation files (`.txt`).

---

## Reference Sheet Visual Panel Cropping
The reference sheets (`reference_images/synthetic_jcb_rgb/...` and `reference_images/synthetic_jcb_uv/...`) are contact-sheet collages containing multiple scenes, captions, numbers, and borders.
`scripts/prepare_dataset.py` extracts individual visual scenes into clean standalone panel images (`reference_images/cropped_panels/`), stripping away infographic text, legends, headers, and borders.

---

## Setup & Usage Instructions

### 1. Create Virtual Environment
> **Windows Note:** Due to Windows MAX_PATH (260 character) limits and corporate Group Policy restrictions, create the virtual environment at a **short path** (e.g., `C:\Users\<USER>\venv_oil`) rather than inside the project directory.

```bash
# Create venv at a short path (recommended on Windows)
python -m venv C:\Users\%USERNAME%\venv_oil

# Activate the virtual environment
C:\Users\%USERNAME%\venv_oil\Scripts\activate
```

### 2. Install Dependencies
With the virtual environment activated:
```bash
pip install -r requirements.txt
```
This installs PyTorch (CPU-only), Ultralytics YOLO, OpenCV, NumPy, Matplotlib, and all other dependencies with pinned compatible versions.

### 3. Prepare & Audit Dataset
```bash
python scripts/prepare_dataset.py
```
This script:
- Extracts clean visual panel scenes from reference collages.
- Audits all 400 images from the synthetic dataset pool (100 RGB leak, 100 RGB no-leak, 100 UV leak, 100 UV no-leak).
- Formats labels to single class `0` (`hydraulic_oil_leak`) and creates empty label files for negative examples.
- Splits data cleanly into `train`, `val`, and `test` sets by scene to prevent data leakage.
- Generates `dataset/data.yaml` and `dataset/README.md`.

### 4. Validate Dataset & Generate Quality Report
```bash
python scripts/validate_dataset.py
```
Outputs:
- Detailed statistics report in console (split counts, modality breakdown, polygon coordinate validity, image resolutions).
- Visual contact sheet saved to `runs/dataset_validation_contact_sheet.jpg`.
- Overlay sample images saved to `runs/dataset_samples/`.

### 5. Train Model (YOLO11s-seg)
```bash
python scripts/train.py --epochs 50 --batch 16 --imgsz 640
```
Trains YOLO11s-seg using Ultralytics framework and saves best weights to `runs/yolo11s_jcb_leak_seg/weights/best.pt`.

### 6. Evaluate Model
```bash
python scripts/evaluate.py --split test
```
Reports precision, recall, mAP50, and mAP50-95 for segmentation masks and bounding boxes.

### 7. Run Inference
#### Single Image Prediction
```bash
python scripts/predict.py --source path/to/image.jpg
```
Output text:
```
HYDRAULIC LEAK DETECTED
confidence: 0.94
```
or
```
NO HYDRAULIC LEAK DETECTED
```

#### Live Webcam Mode
```bash
python scripts/predict.py --webcam
```

### 8. Run Inspection Web Application (POC UI) in VS Code

The project features a separated frontend-backend web application. We have pre-configured a VS Code task and set up a portable Node.js v22.2.0 runtime to allow you to run the application easily without needing administrator access.

#### Option A: Running via VS Code Tasks (Recommended & Easiest)
1. Open the project folder in VS Code (`File` -> `Open Folder...` -> Select `OIL LEAK`).
2. Open the Command Palette by pressing **`Ctrl` + `Shift` + `P`**.
3. Select **`Tasks: Run Task`** (Note: do not select *Run Build Task*).
4. Click on **`Start JCB Leak Dashboard (Both)`**.

This will automatically open two terminal tabs in VS Code and start both the backend API and the frontend dashboard.

#### Option B: Running manually in VS Code Terminal
1. Open the integrated terminal in VS Code by pressing **`Ctrl` + `\`** (Control + Backtick) or selecting `Terminal` -> `New Terminal`.
2. **Start the Backend API (Terminal 1)**:
   ```cmd
   C:\Users\A30165\venv_oil\Scripts\python.exe app.py
   ```
3. **Start the Frontend Dashboard (Terminal 2)**:
   * Open a second terminal window (click the **`+`** icon in the terminal panel).
   * Depending on whether your terminal is **PowerShell** or **Command Prompt (CMD)**, copy and paste the correct commands:

     **For PowerShell (if prompt starts with `PS`):**
     ```powershell
     $env:PATH = "C:\Users\A30165\Downloads\OIL LEAK-20260816T173631Z-1-001\OIL LEAK\node_portable\node-v22.2.0-win-x64;" + $env:PATH
     cd frontend
     npm run dev
     ```
 
     **For Command Prompt (CMD):**
     ```cmd
    set PATH=C:\Users\A30165\Downloads\OIL LEAK-20260816T173631Z-1-001\OIL LEAK\node_portable\node-v22.2.0-win-x64;%PATH%
     cd oil leak
     cd frontend
     npm run dev
     ```

#### Step 3: Access the Interface
Once running, open your browser and visit: **[http://127.0.0.1:3000/](http://127.0.0.1:3000/)**

---

## Scientific & POC Limitations Disclaimer
1. **Synthetic Data**: Reference images and dataset images are synthetic representations created for POC architecture and pipeline development.
2. **UV Signal Simulation**: The UV images simulate a fluorescent visual signal under UV lighting. Ordinary hydraulic oil does **not** naturally fluoresce under UV unless a dedicated fluorescent dye additive is mixed into the hydraulic fluid.
3. **Next Steps for Field Deployment**:
   - Collect real-world RGB photos of JCB equipment operating in construction field environments.
   - Collect real UV-inspection photos using UV illumination lamps with fluorescent dye-doped hydraulic fluid.
   - Annotate real field dataset with instance segmentation masks and train a production-grade YOLO11-seg model.
