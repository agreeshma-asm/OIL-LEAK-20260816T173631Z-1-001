// ==========================================================================
// JCB LEAK INSPECTION CONTROLLER
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const video = document.getElementById('webcam');
    const feedPlaceholder = document.getElementById('feed-placeholder');
    const cameraIndicator = document.getElementById('camera-indicator');
    const scanline = document.getElementById('scanline');
    const viewportWrapper = document.querySelector('.viewport-wrapper');

    const modeRgb = document.getElementById('mode-rgb');
    const modeUv = document.getElementById('mode-uv');
    const btnScan = document.getElementById('btn-scan');
    const btnClear = document.getElementById('btn-clear');

    const fileInput = document.getElementById('file-input');

    const statusReady = document.getElementById('status-card-ready');
    const statusLeak = document.getElementById('status-card-leak');
    const statusNoLeak = document.getElementById('status-card-noleak');
    const statusReview = document.getElementById('status-card-review');

    const resConfidence = document.getElementById('res-confidence');
    const resMode = document.getElementById('res-mode');
    const resDetections = document.getElementById('res-detections');
    const resThreshold = document.getElementById('res-threshold');

    const thresholdSlider = document.getElementById('threshold');
    const thresholdVal = document.getElementById('threshold-val');

    const resultImage = document.getElementById('result-image');
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultLoading = document.getElementById('result-loading');

    const modelStatusDot = document.getElementById('model-status-dot');
    const modelStatusText = document.getElementById('model-status-text');

    // App State
    let currentMode = 'rgb';
    let localStream = null;
    let uploadedImageBase64 = null; // Stores uploaded image if fallback used
    let lastScanResult = null; // Stores last successful scan response

    // Create an image preview element dynamically for uploaded files
    const uploadPreview = document.createElement('img');
    uploadPreview.id = 'upload-preview';
    uploadPreview.className = 'feed-element';
    uploadPreview.style.display = 'none';
    uploadPreview.style.transform = 'none'; // Overrides mirrored layout of webcam
    video.parentNode.insertBefore(uploadPreview, video.nextSibling);

    // Initialize UI Defaults
    thresholdVal.textContent = parseFloat(thresholdSlider.value).toFixed(2);
    resThreshold.textContent = Math.round(parseFloat(thresholdSlider.value) * 100) + '%';

    // Verify Model Status on Backend Startup
    async function checkModelStatus() {
        try {
            const res = await fetch('/status');
            const data = await res.json();
            if (data.status === 'ok') {
                modelStatusDot.className = 'status-dot pulse green';
                modelStatusText.textContent = 'SYSTEM READY';
                console.log("Backend model initialized successfully.");
            } else {
                showModelError(data.message);
            }
        } catch (err) {
            showModelError("BACKEND OFFLINE");
            console.error("Error contacting backend:", err);
        }
    }

    function showModelError(msg) {
        modelStatusDot.className = 'status-dot pulse red';
        modelStatusText.textContent = 'MODEL ERROR';
        alert(`CRITICAL ERROR: ${msg}\nInference will be unavailable.`);
    }

    checkModelStatus();

    // 1. Initialize Webcam
    async function initWebcam() {
        try {
            console.log("Requesting camera access...");
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });
            
            localStream = stream;
            video.srcObject = stream;
            
            // Webcam OK state
            feedPlaceholder.classList.remove('active');
            video.style.display = 'block';
            uploadPreview.style.display = 'none';
            
            cameraIndicator.textContent = '🟢 CAMERA ACTIVE';
            cameraIndicator.style.color = 'var(--color-green)';
            btnScan.disabled = false;
            
            console.log("Webcam stream started successfully.");
        } catch (err) {
            console.warn("Camera access denied or unavailable. Fallback to image upload.", err);
            cameraIndicator.textContent = '🔴 CAM OFFLINE';
            cameraIndicator.style.color = 'var(--color-red)';
            
            // Keep preview elements in placeholder mode
            video.style.display = 'none';
            feedPlaceholder.classList.add('active');
            
            // Do not enable scan button unless a file is uploaded
            if (!uploadedImageBase64) {
                btnScan.disabled = true;
            }
        }
    }

    // Attempt to start webcam automatically
    initWebcam();

    // 2. Modality Switchers
    modeRgb.addEventListener('click', () => {
        currentMode = 'rgb';
        modeRgb.classList.add('active');
        modeUv.classList.remove('active');
        if (lastScanResult) {
            // Trigger dynamic re-display if there is previous data (mode is UI display)
            lastScanResult.mode = 'rgb';
            updateResultUI(lastScanResult, parseFloat(thresholdSlider.value));
        }
    });

    modeUv.addEventListener('click', () => {
        currentMode = 'uv';
        modeUv.classList.add('active');
        modeRgb.classList.remove('active');
        if (lastScanResult) {
            lastScanResult.mode = 'uv';
            updateResultUI(lastScanResult, parseFloat(thresholdSlider.value));
        }
    });

    // 3. Fallback File Upload
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            uploadedImageBase64 = event.target.result;

            // Stop live video stream if running
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }

            // Show preview image instead of live video
            video.style.display = 'none';
            feedPlaceholder.classList.remove('active');
            
            uploadPreview.src = uploadedImageBase64;
            uploadPreview.style.display = 'block';
            
            cameraIndicator.textContent = '📂 FILE LOADED';
            cameraIndicator.style.color = 'var(--jcb-yellow)';
            
            btnScan.disabled = false;
            console.log("File loaded for fallback inspection.");
        };
        reader.readAsDataURL(file);
    });

    // 4. Threshold Slider Interactions
    thresholdSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        thresholdVal.textContent = val.toFixed(2);
        resThreshold.textContent = Math.round(val * 100) + '%';
        
        // Dynamically update results if a scan was already conducted
        if (lastScanResult) {
            updateResultUI(lastScanResult, val);
        }
    });

    // 5. SCAN Trigger
    btnScan.addEventListener('click', async () => {
        let payloadImage = null;

        // If a file is uploaded, use it. Otherwise, capture from live video.
        if (uploadedImageBase64) {
            payloadImage = uploadedImageBase64;
        } else if (localStream && video.readyState === video.HAVE_ENOUGH_DATA) {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            // Draw video frame to canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            payloadImage = canvas.toDataURL('image/jpeg');
        }

        if (!payloadImage) {
            alert("Error: No image source available to scan.");
            return;
        }

        // Trigger SCAN animations
        viewportWrapper.classList.add('scanning');
        resultLoading.classList.add('active');
        btnScan.disabled = true;

        try {
            const response = await fetch('/scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: payloadImage,
                    mode: currentMode,
                    threshold: parseFloat(thresholdSlider.value)
                })
            });

            const data = await response.json();
            
            if (data.status === 'ok') {
                lastScanResult = data;
                updateResultUI(data, parseFloat(thresholdSlider.value));
            } else {
                alert(`Inference Failure: ${data.message}`);
                resetResultUI();
            }
        } catch (err) {
            console.error("Scan request failed:", err);
            alert("Scan failed: Unable to connect to backend server.");
            resetResultUI();
        } finally {
            // Stop scanning animations
            viewportWrapper.classList.remove('scanning');
            resultLoading.classList.remove('active');
            btnScan.disabled = false;
        }
    });

    // Update Result UI Cards & Metadata
    function updateResultUI(data, threshold) {
        // Hide all cards
        statusReady.classList.remove('active');
        statusLeak.classList.remove('active');
        statusNoLeak.classList.remove('active');
        statusReview.classList.remove('active');

        // Apply threshold logic dynamically
        let finalState = data.result_state;
        if (data.detection_count > 0) {
            if (data.max_confidence >= threshold) {
                finalState = "LEAK_DETECTED";
            } else {
                finalState = "REVIEW_REQUIRED";
            }
        } else {
            finalState = "NO_LEAK";
        }

        // Activate matching card
        if (finalState === 'LEAK_DETECTED') {
            statusLeak.classList.add('active');
        } else if (finalState === 'NO_LEAK') {
            statusNoLeak.classList.add('active');
        } else {
            statusReview.classList.add('active');
        }

        // Update metadata
        resConfidence.textContent = data.detection_count > 0 ? Math.round(data.max_confidence * 100) + '%' : '0%';
        resMode.textContent = data.mode.toUpperCase();
        resDetections.textContent = data.detection_count;

        // Display overlay image
        resultImage.src = data.annotated_image;
        resultImage.classList.add('active');
        resultPlaceholder.classList.remove('active');
    }

    // Reset Result UI back to ready
    function resetResultUI() {
        statusReady.classList.add('active');
        statusLeak.classList.remove('active');
        statusNoLeak.classList.remove('active');
        statusReview.classList.remove('active');

        resConfidence.textContent = '-- %';
        resMode.textContent = '--';
        resDetections.textContent = '--';

        resultImage.classList.remove('active');
        resultImage.src = '';
        resultPlaceholder.classList.add('active');
        lastScanResult = null;
    }

    // 6. CLEAR Handler
    btnClear.addEventListener('click', () => {
        // Clear file input
        fileInput.value = '';
        uploadedImageBase64 = null;
        
        // Reset results
        resetResultUI();

        // Restart video stream if it was stopped
        if (!localStream) {
            initWebcam();
        } else {
            video.style.display = 'block';
            uploadPreview.style.display = 'none';
            uploadPreview.src = '';
            feedPlaceholder.classList.remove('active');
            cameraIndicator.textContent = '🟢 CAMERA ACTIVE';
            cameraIndicator.style.color = 'var(--color-green)';
            btnScan.disabled = false;
        }
        
        console.log("Inspector UI cleared.");
    });
});
