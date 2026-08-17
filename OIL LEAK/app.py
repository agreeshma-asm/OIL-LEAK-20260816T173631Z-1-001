import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
# Enable CORS for frontend communication (running on port 3000/5173)
CORS(app)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(ROOT_DIR, "runs", "yolo11s_jcb_leak_v6_seg", "weights", "best.pt")

# Initialize Model variables
model = None
model_error = None
device_name = "cpu"

# Attempt to load ONLY the trained best.pt weights
if not os.path.exists(WEIGHTS_PATH):
    model_error = f"Model weights file NOT found at: {WEIGHTS_PATH}. Please ensure training completed successfully."
    print(f"CRITICAL ERROR: {model_error}")
else:
    try:
        print(f"Loading YOLO11s-seg model from {WEIGHTS_PATH}...")
        model = YOLO(WEIGHTS_PATH)
        print("Model loaded successfully.")
        device_name = str(model.device)
    except Exception as e:
        model_error = f"Failed to load YOLO model: {str(e)}"
        print(f"CRITICAL ERROR: {model_error}")

@app.route('/', methods=['GET'])
def home():
    return '''
    <html>
        <head>
            <title>JCB Leak API Server</title>
            <style>
                body { font-family: sans-serif; background-color: #080a0e; color: #f1f2f6; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 90vh; margin: 0; }
                a { color: #FFC80A; text-decoration: none; font-weight: bold; }
                a:hover { text-decoration: underline; }
                .box { border: 1px solid rgba(255, 200, 10, 0.2); padding: 30px; border-radius: 12px; background: rgba(18, 22, 32, 0.9); max-width: 500px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
            </style>
        </head>
        <body>
            <div class="box">
                <h2>JCB Leak Inspection API Service</h2>
                <p style="margin: 15px 0; color: #a4b0be; font-size: 14px;">This port (5000) hosts the backend machine learning API.</p>
                <p style="font-size: 16px;">To access the user interface dashboard, visit:<br/><br/>
                   <a href="http://127.0.0.1:3000/">http://127.0.0.1:3000/</a>
                </p>
            </div>
        </body>
    </html>
    '''

@app.route('/api/status', methods=['GET'])
def get_status():
    if model_error or model is None:
        return jsonify({
            "status": "error",
            "model_loaded": False,
            "backend_ready": False,
            "model_path": WEIGHTS_PATH,
            "device": "unknown",
            "message": model_error or "Model is null."
        }), 500
        
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "backend_ready": True,
        "model_path": WEIGHTS_PATH,
        "device": device_name
    })

@app.route('/api/scan', methods=['POST'])
def scan():
    if model_error or model is None:
        return jsonify({
            "status": "error",
            "message": model_error or "Model not loaded."
        }), 500

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing image data in request."
            }), 400

        mode = data.get('mode', 'rgb').upper()  # 'RGB' or 'UV'
        source_param = data.get('source', 'unknown').upper()

        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            header, image_data = image_data.split(',', 1)
            
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "status": "error",
                "message": "Failed to decode captured image."
            }), 400

        h, w = img.shape[:2]

        # Log request source and image size
        print("\n" + "="*45, flush=True)
        print("    INCOMING API REQUEST DETAILS", flush=True)
        print("="*45, flush=True)
        print(f"REQUEST SOURCE: {source_param}", flush=True)
        print(f"IMAGE WIDTH: {w}", flush=True)
        print(f"IMAGE HEIGHT: {h}", flush=True)
        
        # Save temporary debug copy
        debug_path = os.path.join(ROOT_DIR, "debug_received.jpg")
        cv2.imwrite(debug_path, img)
        print(f"Saved debug received image to: {debug_path}", flush=True)
        print("="*45 + "\n", flush=True)

        # 1. Run low-threshold inference to capture raw candidates for debug logging
        results_raw = model.predict(source=img, conf=0.001, classes=[0], verbose=False)[0]
        raw_count = len(results_raw.boxes) if results_raw.boxes is not None else 0

        # 2. Count candidates that pass our confidence filter (>= 0.50) before NMS/max_det
        conf_filtered_count = 0
        if results_raw.boxes is not None:
            conf_filtered_count = sum(1 for box in results_raw.boxes if float(box.conf[0].cpu().numpy()) >= 0.50)

        # 3. Run final strict inference (conf=0.50, iou=0.50, max_det=1)
        results_final = model.predict(source=img, conf=0.50, iou=0.50, max_det=1, classes=[0], verbose=False)
        result = results_final[0]

        # Debug console logging
        print("\n" + "="*45, flush=True)
        print("    YOLO INFERENCE PIPELINE DEBUG LOG", flush=True)
        print("="*45, flush=True)
        print(f"RAW MODEL RESULTS:\n{raw_count}", flush=True)
        print(f"AFTER CONFIDENCE FILTER:\n{conf_filtered_count}", flush=True)

        # Initialize outputs
        detected = False
        confidence = None
        class_id = None
        class_name = None
        detections = 0
        
        annotated_img = img.copy()

        # Check if any detection passed NMS
        if result.boxes is not None and len(result.boxes) > 0:
            detected = True
            detections = len(result.boxes)
            box = result.boxes[0]
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = "hydraulic_oil_leak"
            
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            print(f"AFTER NMS:\n{detections}", flush=True)
            print(f"FINAL:\n{class_name} {confidence:.2f}", flush=True)
            print(f"BOUNDING BOX:\n[{xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}]", flush=True)

            # Overlay mask only if present
            if result.masks is not None and len(result.masks) > 0:
                mask = result.masks.xy[0]
                
                # Colors BGR: Red (0,0,255) for Visible/RGB, Fluorescent Green (0,255,0) for UV
                if mode == 'UV':
                    fill_color = (0, 255, 0)       # Fluorescent Green
                    outline_color = (0, 255, 255)   # Neon Yellow
                    box_color = (255, 255, 0)       # Neon Cyan
                else:
                    fill_color = (0, 0, 255)       # Red
                    outline_color = (0, 255, 255)   # Yellow
                    box_color = (0, 255, 0)         # Green

                # 1. Create overlay layer for the mask fill only
                mask_overlay = annotated_img.copy()
                pts = np.int32([mask])
                cv2.fillPoly(mask_overlay, pts, fill_color)

                # 2. Blend the filled mask transparently (alpha=0.25)
                cv2.addWeighted(mask_overlay, 0.25, annotated_img, 0.75, 0, annotated_img)

                # 3. Draw border lines, bounding box, and labels at 100% opacity on top
                cv2.polylines(annotated_img, pts, True, outline_color, 2)
                cv2.rectangle(annotated_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)

                cv2.putText(
                    annotated_img, 
                    f"Leak: {confidence:.2f}", 
                    (xyxy[0], xyxy[1] - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    box_color, 
                    1, 
                    cv2.LINE_AA
                )
        else:
            print("AFTER NMS:\n0", flush=True)
            print("FINAL:\nNo detections", flush=True)
        print("="*45 + "\n", flush=True)

        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        result_img_url = f"data:image/jpeg;base64,{encoded_image}"

        # Return JSON schema
        return jsonify({
            "detected": detected,
            "confidence": confidence,
            "detections": detections,
            "class_id": class_id,
            "class_name": class_name,
            "mode": mode,
            "overlay_image": result_img_url
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Inference pipeline failure: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Run Flask server locally
    app.run(host='127.0.0.1', port=5000, debug=False)
