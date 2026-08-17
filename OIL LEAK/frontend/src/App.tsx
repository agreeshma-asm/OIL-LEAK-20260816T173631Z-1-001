import React, { useState, useEffect, useRef } from 'react';
import { 
  Camera, 
  Upload, 
  Trash2, 
  Sliders, 
  Cpu, 
  AlertTriangle, 
  CheckCircle2, 
  Play, 
  Loader2, 
  RefreshCw,
  Gauge,
  Lightbulb,
  Zap,
  Info,
  Power,
  PowerOff,
  Video
} from 'lucide-react';
import './App.css';

const API_BASE = 'http://127.0.0.1:5000';

interface ScanResult {
  detected: boolean;
  confidence: number;
  class_name: string;
  mode: string;
  detections: number;
  overlay_image: string;
}

interface ModelStatus {
  status: string;
  model_loaded: boolean;
  backend_ready: boolean;
  model_path: string;
  device: string;
  message?: string;
}

type CameraState = 'CONNECTED' | 'NOT_AVAILABLE' | 'PERMISSION_DENIED' | 'STARTING' | 'DISCONNECTED';

export default function App() {
  // App States
  const [mode, setMode] = useState<'RGB' | 'UV'>('RGB');
  const [videoActive, setVideoActive] = useState<boolean>(false);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [scanning, setScanning] = useState<boolean>(false);
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [threshold, setThreshold] = useState<number>(0.50);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);

  // Advanced Webcam States
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [cameraState, setCameraState] = useState<CameraState>('DISCONNECTED');

  // References
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // 1. Check Backend & Model Status
  const checkStatus = async () => {
    try {
      setBackendError(null);
      const res = await fetch(`${API_BASE}/api/status`);
      const data: ModelStatus = await res.json();
      if (data.status === 'ok') {
        setModelStatus(data);
      } else {
        setBackendError(data.message || 'Model loading error.');
      }
    } catch (err) {
      setBackendError('Backend service offline. Please start Flask app.py.');
      console.error('Failed to query backend status:', err);
    }
  };

  // 2. Enumerate Cameras
  const enumerateCameras = async () => {
    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = allDevices.filter(d => d.kind === 'videoinput');
      setDevices(videoDevices);
      if (videoDevices.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevices[0].deviceId);
      }
    } catch (err) {
      console.error('Error enumerating cameras:', err);
    }
  };

  // 3. Initialize Webcam
  const initWebcam = async (deviceId?: string) => {
    setCameraState('STARTING');
    setWebcamError(null);
    setUploadedImage(null);

    // Stop current stream if exists
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    try {
      const idToUse = deviceId || selectedDeviceId;
      const constraints: MediaStreamConstraints = {
        video: idToUse ? { deviceId: { exact: idToUse } } : { facingMode: 'environment' },
        audio: false
      };
      
      console.log('Accessing webcam with constraints:', constraints);
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setVideoActive(true);
      }
      setCameraState('CONNECTED');
      
      // Re-enumerate to get labels after permission is granted
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = allDevices.filter(d => d.kind === 'videoinput');
      setDevices(videoDevices);
      if (videoDevices.length > 0) {
        const activeDevice = videoDevices.find(d => d.deviceId === idToUse) || videoDevices[0];
        setSelectedDeviceId(activeDevice.deviceId);
      }
    } catch (err: any) {
      console.warn('Webcam permission denied or camera not found.', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setCameraState('PERMISSION_DENIED');
        setWebcamError('Camera permission denied.');
      } else {
        setCameraState('NOT_AVAILABLE');
        setWebcamError(err.message || 'Camera not available or blocked.');
      }
      setVideoActive(false);
    }
  };

  // 4. Stop Webcam Stream
  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setVideoActive(false);
    setCameraState('DISCONNECTED');
  };

  // Initial mount: check backend and try starting webcam
  useEffect(() => {
    checkStatus();
    initWebcam();
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // 5. Modality Change
  const handleModeChange = (newMode: 'RGB' | 'UV') => {
    setMode(newMode);
    if (scanResult) {
      setScanResult({
        ...scanResult,
        mode: newMode
      });
    }
  };

  // 6. File Uploader Fallback
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Immediately reset previous scan results so we do not show stale info
    setScanResult(null);

    // Step 1: Log file metadata
    console.log('UPLOAD FILE:');
    console.log('name:', file.name);
    console.log('size:', file.size);
    console.log('type:', file.type);

    const reader = new FileReader();
    reader.onload = (event) => {
      const base64Data = event.target?.result as string;
      setUploadedImage(base64Data);
      
      // Stop webcam when a file is loaded
      stopWebcam();
      setWebcamError(null);
    };
    reader.readAsDataURL(file);
  };

  // 7. SCAN Trigger (Canvas capture or uploaded image)
  const handleScan = async () => {
    // Immediately clear previous scan result to avoid showing stale data while scanning
    setScanResult(null);

    let imagePayload = null;

    if (uploadedImage) {
      imagePayload = uploadedImage;
      
      // Step 3: Log scan source and details
      console.log('SCAN SOURCE: UPLOAD');
      const file = fileInputRef.current?.files?.[0];
      console.log('FILE NAME:', file ? file.name : 'unknown');
    } else if (videoActive && videoRef.current) {
      console.log('SCAN SOURCE: WEBCAM');
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth || 640;
      canvas.height = videoRef.current.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        // Draw mirrored webcam frame
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        imagePayload = canvas.toDataURL('image/jpeg');
      }
    }

    if (!imagePayload) {
      alert('Error: No image source available to scan. Turn on camera or upload a file.');
      return;
    }

    // Step 3: Log image dimensions
    const imgObj = new Image();
    imgObj.onload = () => {
      console.log('IMAGE WIDTH:', imgObj.width);
      console.log('IMAGE HEIGHT:', imgObj.height);
    };
    imgObj.src = imagePayload;

    setScanning(true);
    try {
      const res = await fetch(`${API_BASE}/api/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: imagePayload,
          mode: mode,
          source: uploadedImage ? 'upload' : 'webcam'
        })
      });

      const data = await res.json();
      if (data.overlay_image) {
        setScanResult(data);
      } else {
        alert(`Inference failed: ${data.message || 'Unknown error'}`);
        setScanResult(null);
      }
    } catch (err) {
      console.error('Scan request failed:', err);
      alert('Network Error: Failed to contact Flask backend API.');
      setScanResult(null);
    } finally {
      setScanning(false);
    }
  };

  // 8. CLEAR / Reset Trigger
  const handleClear = () => {
    setUploadedImage(null);
    setScanResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    // Re-initialize webcam if was disconnected or upload was shown
    if (cameraState === 'DISCONNECTED') {
      initWebcam();
    }
  };

  // Camera Status Badges & Styles
  const getCameraStatusMarkup = () => {
    switch (cameraState) {
      case 'CONNECTED':
        return <span className="status-label connected">● CAMERA CONNECTED</span>;
      case 'STARTING':
        return <span className="status-label starting pulse">● STARTING CAMERA...</span>;
      case 'PERMISSION_DENIED':
        return <span className="status-label denied">● CAMERA PERMISSION DENIED</span>;
      case 'NOT_AVAILABLE':
        return <span className="status-label error">● CAMERA NOT AVAILABLE</span>;
      case 'DISCONNECTED':
      default:
        return <span className="status-label offline">● CAMERA OFFLINE</span>;
    }
  };

  // Calculate Result Badge details on client side
  const getInspectionState = () => {
    if (!scanResult) {
      return {
        label: 'AWAITING DISPATCH',
        color: 'ready',
        description: 'Position component and trigger scan.',
        icon: <Zap className="card-icon-svg" />
      };
    }

    if (!scanResult.detected) {
      return {
        label: 'NO LEAK DETECTED',
        color: 'noleak',
        description: 'Component is clean and sealed.',
        icon: <CheckCircle2 className="card-icon-svg" />
      };
    }

    if (scanResult.confidence >= threshold) {
      return {
        label: 'HYDRAULIC LEAK DETECTED',
        color: 'leak',
        description: 'Critical fluid loss detected! Maintenance required.',
        icon: <AlertTriangle className="card-icon-svg" />
      };
    }

    return {
      label: 'REVIEW REQUIRED',
      color: 'review',
      description: 'Potential leak detected below threshold.',
      icon: <Info className="card-icon-svg" />
    };
  };

  const currentResult = getInspectionState();

  return (
    <div className="dashboard-container">
      {/* Header Banner */}
      <header className="dashboard-header">
        <div className="header-brand">
          <div className="logo-kobelco">KOBELCO</div>
          <div className="divider" />
          <div className="system-title">
            <h1>HYDRAULIC LEAK INSPECTION</h1>
            <p>YOLO11s-seg Instance Segmentation POC Console</p>
          </div>
        </div>

        {/* AI Engine Status badge */}
        <div className={`engine-badge ${backendError ? 'offline' : 'online'}`}>
          <div className="badge-core">
            <Cpu className="badge-icon" />
            <div className="badge-text">
              <span className="status-indicator">
                {backendError ? '● OFFLINE' : '● AI ENGINE READY'}
              </span>
              <span className="engine-meta">
                {modelStatus ? `${modelStatus.device.toUpperCase()} | ${modelStatus.model_path.split('\\').pop()}` : 'No Model Loaded'}
              </span>
            </div>
          </div>
          {backendError && (
            <button className="btn-retry" onClick={checkStatus} title="Retry backend connection">
              <RefreshCw className="retry-icon" />
            </button>
          )}
        </div>
      </header>

      {/* Main Grid Layout */}
      <main className="dashboard-grid">
        
        {/* LEFT COLUMN: Webcam & Captures */}
        <section className="dashboard-card feed-card">
          <div className="card-header">
            <div className="header-meta">
              <div className="card-number">01</div>
              <h2>LIVE VIEWPORT FEED</h2>
            </div>
            {getCameraStatusMarkup()}
          </div>

          {/* Camera Selection Dropdown */}
          {devices.length > 1 && (
            <div className="device-selector-row">
              <div className="select-wrapper">
                <Video className="select-icon" />
                <select 
                  value={selectedDeviceId} 
                  onChange={(e) => {
                    setSelectedDeviceId(e.target.value);
                    initWebcam(e.target.value);
                  }}
                  className="camera-dropdown"
                >
                  {devices.map((device, index) => (
                    <option key={device.deviceId} value={device.deviceId}>
                      {device.label || `Camera Source ${index + 1}`}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}

          <div className="feed-viewport-container">
            {/* Live video */}
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              className="feed-element" 
              style={{ display: videoActive ? 'block' : 'none' }}
            />

            {/* Fallback image */}
            {uploadedImage && (
              <img 
                src={uploadedImage} 
                className="feed-element upload-preview" 
                alt="Uploaded source" 
              />
            )}

            {/* Placeholder / error state */}
            {!videoActive && !uploadedImage && (
              <div className="feed-error-placeholder">
                <Camera className="placeholder-icon" />
                <h3>CAMERA FEED OFFLINE</h3>
                <p className="placeholder-error">{webcamError || 'Webcam stream is stopped.'}</p>
                <p className="placeholder-help">Start the webcam or upload a file below to run analysis.</p>
              </div>
            )}

            {/* Scanning Overlay line */}
            <div className={`laser-scanner ${scanning ? 'animating' : ''}`} />
          </div>

          {/* Controls */}
          <div className="card-controls">
            
            {/* Start / Stop Webcam buttons */}
            <div className="camera-power-buttons">
              <button 
                className="power-btn btn-start" 
                onClick={() => initWebcam(selectedDeviceId)}
                disabled={cameraState === 'CONNECTED' || cameraState === 'STARTING'}
              >
                <Power className="power-icon" /> START CAMERA
              </button>
              <button 
                className="power-btn btn-stop" 
                onClick={stopWebcam}
                disabled={cameraState === 'DISCONNECTED'}
              >
                <PowerOff className="power-icon" /> STOP CAMERA
              </button>
            </div>

            <div className="modality-setting">
              <span className="control-label">INSPECTION LIGHT MODALITY</span>
              <div className="toggle-group">
                <button 
                  className={`toggle-btn ${mode === 'RGB' ? 'active' : ''}`}
                  onClick={() => handleModeChange('RGB')}
                >
                  <Lightbulb className="toggle-icon" /> VISIBLE / RGB
                </button>
                <button 
                  className={`toggle-btn ${mode === 'UV' ? 'active' : ''}`}
                  onClick={() => handleModeChange('UV')}
                >
                  <Zap className="toggle-icon" /> UV MODALITY
                </button>
              </div>
            </div>

            <div className="scan-actions">
              <button 
                className="btn-scan" 
                onClick={handleScan}
                disabled={scanning || (!videoActive && !uploadedImage)}
              >
                {scanning ? (
                  <>
                    <Loader2 className="btn-spinner" /> ANALYZING...
                  </>
                ) : (
                  <>
                    <Play className="btn-icon" /> TRIGGER SCAN
                  </>
                )}
              </button>
              <button className="btn-clear" onClick={handleClear}>
                <Trash2 className="btn-icon" /> RESET
              </button>
            </div>

            {/* Fallback File Inspector */}
            <div className="upload-section">
              <span className="control-label">IMAGE FILE FALLBACK DISPATCHER</span>
              <div className="file-dropzone" onClick={() => fileInputRef.current?.click()}>
                <Upload className="upload-icon" />
                <div className="upload-text">
                  <span className="upload-highlight">Click to upload image</span> (JPG or PNG)
                </div>
                <input 
                  type="file" 
                  ref={fileInputRef}
                  accept="image/*" 
                  className="file-input-hidden" 
                  onChange={handleFileUpload} 
                />
              </div>
            </div>

          </div>
        </section>

        {/* RIGHT COLUMN: Results Details */}
        <section className="dashboard-card analysis-card">
          <div className="card-header">
            <div className="header-meta">
              <div className="card-number">02</div>
              <h2>ANALYSIS REPORT</h2>
            </div>
          </div>

          {/* Result Banner Card */}
          <div className={`inspection-badge ${currentResult.color}`}>
            <div className="badge-visual">
              {currentResult.icon}
            </div>
            <div className="badge-details">
              <h3>{currentResult.label}</h3>
              <p>{currentResult.description}</p>
            </div>
          </div>

          {/* Analytics Details */}
          <div className="analytics-grid">
            <div className="analytic-tile">
              <span className="tile-label">LEAK CONFIDENCE</span>
              <span className="tile-value value-mono">
                {scanResult && scanResult.detected ? `${Math.round(scanResult.confidence * 100)}%` : '0%'}
              </span>
            </div>

            <div className="analytic-tile">
              <span className="tile-label">DETECTIONS</span>
              <span className="tile-value value-mono">
                {scanResult ? scanResult.detections : '0'}
              </span>
            </div>

            <div className="analytic-tile">
              <span className="tile-label">MODALITY</span>
              <span className="tile-value">
                {scanResult ? scanResult.mode : mode}
              </span>
            </div>

            <div className="analytic-tile">
              <span className="tile-label">CLASS LABEL</span>
              <span className="tile-value truncate-text">
                {scanResult && scanResult.detected ? scanResult.class_name : 'none'}
              </span>
            </div>
          </div>

          {/* Threshold Slider control */}
          <div className="threshold-panel">
            <div className="threshold-header">
              <span className="threshold-title">
                <Sliders className="title-icon" /> CONFIDENCE DISPATCH THRESHOLD
              </span>
              <span className="threshold-percentage">
                {Math.round(threshold * 100)}%
              </span>
            </div>
            <input 
              type="range" 
              min="0.25" 
              max="0.95" 
              step="0.05" 
              value={threshold} 
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="custom-range"
            />
            <div className="slider-limits">
              <span>0.25</span>
              <span>0.50 (Default)</span>
              <span>0.95</span>
            </div>
          </div>

          {/* Overlay Visualization Display */}
          <div className="overlay-viewport-container">
            {/* Loader indicator when scanning */}
            {scanning && (
              <div className="loader-overlay">
                <Loader2 className="spinning-loader" />
                <p>RUNNING INSTANCE SEGMENTATION...</p>
              </div>
            )}

            {/* Prediction overlay image output */}
            {scanResult && !scanning ? (
              <img 
                src={scanResult.overlay_image} 
                className="result-overlay-image" 
                alt="Segmentation result" 
              />
            ) : (
              !scanning && (
                <div className="result-placeholder">
                  <div className="scanner-target">
                    <div className="bracket top-left" />
                    <div className="bracket top-right" />
                    <div className="bracket bottom-left" />
                    <div className="bracket bottom-right" />
                    <Gauge className="placeholder-visual" />
                  </div>
                  <p>Inference visualization overlay will be rendered here.</p>
                </div>
              )
            )}
          </div>
        </section>

      </main>
    </div>
  );
}
