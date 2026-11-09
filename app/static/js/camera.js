/**
 * SignBridge Automatic Hand & Finger Detection Engine
 * Uses MediaPipe Hands for real-time 21-point hand tracking, dynamic bounding box cropping,
 * stability-based auto-capture countdown, and automatic video start/stop on hand entry/departure.
 * Features built-in client-side WASM tracking with transparent server-side fallback.
 */

// Global State
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let capturedBlob = null;
let capturedMetadata = {
    handedness: null,
    confidence: null,
    bbox: null,
    landmarks_count: 0
};

let currentMediaType = 'IMAGE'; // 'IMAGE' or 'VIDEO'
let captureState = 'IDLE';       // 'IDLE', 'WAITING_FOR_HAND', 'HAND_DETECTED', 'STABILIZING', 'COUNTDOWN', 'RECORDING', 'REVIEW'

// Settings
let showLandmarks = true;
let autoCaptureEnabled = true;
let handSelection = 'Auto';      // 'Auto', 'Right', 'Left'
let stabilityThresholdMs = 1000; // ms required steady before countdown
let disappearanceDebounceMs = 800; // ms hand missing before stopping video

// Hand Tracking State
let handsDetector = null;
let isDetecting = false;
let framesProcessedCount = 0;
let lastFpsCalculationTime = performance.now();
let fpsFrameCount = 0;
let lastDetectedHands = [];
let primaryHand = null;
let serverFallbackActive = false;
let lastServerRequestTime = 0;

// Stability & Countdown State
let lastHandCenter = null;
let handSteadySince = null;
let countdownTimer = null;
let countdownRemaining = 3;
let disappearanceTimer = null;
let recordingStartTime = 0;
let recordingTimerInterval = null;
let recordedDurationSeconds = 0;

// DOM Elements
let liveVideo, trackingCanvas, snapshotCanvas, snapshotReviewContainer, recordedVideoReview;
let cameraErrorOverlay, cameraStatusBadge, handStateBadge, handStatsBadge, handStatsText, cameraSelect;
let recordingBadge, recTimer, captureCountdownOverlay, countdownNumber, countdownLabel, reviewHandDetails;
let liveControls, reviewControls, btnCaptureImage, btnStartRecord, btnStopRecord, btnSaveSample;
let targetClassSelect, targetParticipantSelect, metaReviewSign, metaReviewParticipant, metaReviewType;

// Diagnostics Panel Elements
let diagMpRuntime, diagHandLandmarker, diagCameraState, diagFramesProcessed, diagLastDetection;
let diagHandsCount, diagLandmarksCount, diagConfidence, diagBbox, diagCaptureState, diagLastError, diagFpsBadge;

// Standard MediaPipe 21 Hand Landmark Connection Pairs
const HAND_CONNECTIONS = [
    // Thumb
    [0, 1], [1, 2], [2, 3], [3, 4],
    // Index
    [0, 5], [5, 6], [6, 7], [7, 8],
    // Middle
    [5, 9], [9, 10], [10, 11], [11, 12],
    // Ring
    [9, 13], [13, 14], [14, 15], [15, 16],
    // Pinky
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];

// Initialize on DOM Ready or immediately if DOM is already parsed
function initializeCaptureStudio() {
    liveVideo = document.getElementById('liveVideo');
    trackingCanvas = document.getElementById('trackingCanvas');
    snapshotCanvas = document.getElementById('snapshotCanvas');
    snapshotReviewContainer = document.getElementById('snapshotReviewContainer');
    recordedVideoReview = document.getElementById('recordedVideoReview');
    cameraErrorOverlay = document.getElementById('cameraErrorOverlay');
    cameraStatusBadge = document.getElementById('cameraStatusBadge');
    handStateBadge = document.getElementById('handStateBadge');
    handStatsBadge = document.getElementById('handStatsBadge');
    handStatsText = document.getElementById('handStatsText');
    cameraSelect = document.getElementById('cameraSelect');
    recordingBadge = document.getElementById('recordingBadge');
    recTimer = document.getElementById('recTimer');
    captureCountdownOverlay = document.getElementById('captureCountdownOverlay');
    countdownNumber = document.getElementById('countdownNumber');
    countdownLabel = document.getElementById('countdownLabel');
    reviewHandDetails = document.getElementById('reviewHandDetails');

    liveControls = document.getElementById('liveControls');
    reviewControls = document.getElementById('reviewControls');
    btnCaptureImage = document.getElementById('btnCaptureImage');
    btnStartRecord = document.getElementById('btnStartRecord');
    btnStopRecord = document.getElementById('btnStopRecord');
    btnSaveSample = document.getElementById('btnSaveSample');

    targetClassSelect = document.getElementById('targetClassSelect');
    targetParticipantSelect = document.getElementById('targetParticipantSelect');
    metaReviewSign = document.getElementById('metaReviewSign');
    metaReviewParticipant = document.getElementById('metaReviewParticipant');
    metaReviewType = document.getElementById('metaReviewType');

    diagMpRuntime = document.getElementById('diagMpRuntime');
    diagHandLandmarker = document.getElementById('diagHandLandmarker');
    diagCameraState = document.getElementById('diagCameraState');
    diagFramesProcessed = document.getElementById('diagFramesProcessed');
    diagLastDetection = document.getElementById('diagLastDetection');
    diagHandsCount = document.getElementById('diagHandsCount');
    diagLandmarksCount = document.getElementById('diagLandmarksCount');
    diagConfidence = document.getElementById('diagConfidence');
    diagBbox = document.getElementById('diagBbox');
    diagCaptureState = document.getElementById('diagCaptureState');
    diagLastError = document.getElementById('diagLastError');
    diagFpsBadge = document.getElementById('diagFpsBadge');

    if (targetClassSelect) {
        targetClassSelect.addEventListener('change', onSignClassChange);
    }
    if (targetParticipantSelect) {
        targetParticipantSelect.addEventListener('change', updateReviewMetadata);
    }

    const stabilityRange = document.getElementById('stabilityRange');
    if (stabilityRange) {
        stabilityRange.addEventListener('input', (e) => {
            stabilityThresholdMs = parseInt(e.target.value) || 1000;
        });
    }

    updateReviewMetadata();
    setCaptureState('WAITING_FOR_HAND');
    initMediaPipeHands();
    startCamera();
    onModeChange('IMAGE');
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeCaptureStudio);
} else {
    initializeCaptureStudio();
}

// Update Diagnostics State Helper
function setCaptureState(newState) {
    captureState = newState;
    if (diagCaptureState) {
        diagCaptureState.textContent = newState;
        diagCaptureState.className = newState === 'RECORDING' ? 'text-danger fw-bold' :
                                    (newState === 'HAND_DETECTED' || newState === 'STABILIZING' ? 'text-success fw-bold' :
                                    (newState === 'COUNTDOWN' ? 'text-warning fw-bold' : 'text-light'));
    }
}

function updateHandStateBadge(text, bgClass) {
    if (handStateBadge) {
        handStateBadge.textContent = text;
        handStateBadge.className = `badge ${bgClass} border border-opacity-25 px-2 py-1`;
    }
}

function updateReviewMetadata() {
    if (targetClassSelect && targetClassSelect.selectedOptions.length > 0) {
        const opt = targetClassSelect.selectedOptions[0];
        if (metaReviewSign) {
            metaReviewSign.textContent = opt.dataset.name || opt.textContent.split('•')[0].trim();
        }
    }
    if (targetParticipantSelect && targetParticipantSelect.selectedOptions.length > 0) {
        if (metaReviewParticipant) {
            metaReviewParticipant.textContent = targetParticipantSelect.value;
        }
    }
    if (metaReviewType) {
        metaReviewType.textContent = currentMediaType;
    }
}

function onSignClassChange() {
    updateReviewMetadata();
    if (!targetClassSelect) return;
    const opt = targetClassSelect.selectedOptions[0];
    if (opt && opt.dataset.type === 'DYNAMIC') {
        const modeVid = document.getElementById('modeVideo');
        if (modeVid) {
            modeVid.checked = true;
            onModeChange('VIDEO');
        }
    }
}

function onToggleLandmarks(checked) {
    showLandmarks = checked;
}

function onAutoCaptureToggle(checked) {
    autoCaptureEnabled = checked;
    if (!checked) {
        cancelCountdown();
        updateHandStateBadge('MANUAL CAPTURE', 'bg-secondary');
    }
}

function onHandSelectionChange(val) {
    handSelection = val;
}

function onModeChange(mode) {
    currentMediaType = mode;
    updateReviewMetadata();
    cancelCountdown();
    if (btnCaptureImage && btnStartRecord && btnStopRecord) {
        if (mode === 'IMAGE') {
            btnCaptureImage.classList.remove('d-none');
            btnStartRecord.classList.add('d-none');
            btnStopRecord.classList.add('d-none');
        } else {
            btnCaptureImage.classList.add('d-none');
            btnStartRecord.classList.remove('d-none');
            btnStopRecord.classList.add('d-none');
        }
    }
    retakeMedia();
}

// ---------------- MEDIAPIPE HANDS INITIALIZATION ----------------

function initMediaPipeHands() {
    if (diagMpRuntime) {
        diagMpRuntime.textContent = 'CHECKING...';
        diagMpRuntime.className = 'text-warning';
    }

    if (typeof Hands === 'undefined') {
        console.warn('[SignBridge] MediaPipe Hands global object undefined. Retrying in 400ms...');
        setTimeout(initMediaPipeHands, 400);
        return;
    }

    try {
        if (diagMpRuntime) {
            diagMpRuntime.textContent = 'LOADED';
            diagMpRuntime.className = 'text-success';
        }
        if (diagHandLandmarker) {
            diagHandLandmarker.textContent = 'INITIALIZING...';
            diagHandLandmarker.className = 'text-warning';
        }

        // Initialize MediaPipe Hands
        handsDetector = new Hands({
            locateFile: (file) => {
                // Try local vendor path with fallback
                return `/static/vendor/mediapipe/${file}`;
            }
        });

        handsDetector.setOptions({
            maxNumHands: 2,
            modelComplexity: 0,
            minDetectionConfidence: 0.35,
            minTrackingConfidence: 0.35
        });

        handsDetector.onResults(onHandResults);

        if (diagHandLandmarker) {
            diagHandLandmarker.textContent = 'READY (WASM)';
            diagHandLandmarker.className = 'text-success';
        }
        if (diagLastError) {
            diagLastError.textContent = 'NONE';
            diagLastError.className = 'text-success';
        }
        console.log('[SignBridge] MediaPipe Hands initialized successfully.');
    } catch (e) {
        console.error('[SignBridge] MediaPipe Hands init error:', e);
        serverFallbackActive = true;
        if (diagHandLandmarker) {
            diagHandLandmarker.textContent = 'FALLBACK (SERVER)';
            diagHandLandmarker.className = 'text-info';
        }
        if (diagLastError) {
            diagLastError.textContent = 'WASM unavailable, using server pipeline';
            diagLastError.className = 'text-info';
        }
    }
}

// ---------------- CAMERA INITIALIZATION & FRAME LOOP ----------------

async function startCamera(deviceId = null) {
    if (diagCameraState) {
        diagCameraState.textContent = 'REQUESTING...';
        diagCameraState.className = 'text-warning';
    }
    if (cameraStatusBadge) {
        cameraStatusBadge.className = 'badge bg-warning bg-opacity-25 text-warning border border-warning border-opacity-25';
        cameraStatusBadge.textContent = 'Requesting access...';
    }
    if (cameraErrorOverlay) {
        cameraErrorOverlay.classList.add('d-none');
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    const constraints = {
        video: deviceId ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
                        : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false
    };

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        if (liveVideo) {
            liveVideo.srcObject = mediaStream;
            await liveVideo.play();

            const vw = liveVideo.videoWidth || 1280;
            const vh = liveVideo.videoHeight || 720;

            if (diagCameraState) {
                diagCameraState.textContent = `ACTIVE (${vw}x${vh})`;
                diagCameraState.className = 'text-success';
            }

            if (cameraStatusBadge) {
                cameraStatusBadge.className = 'badge bg-success bg-opacity-25 text-success border border-success border-opacity-25';
                cameraStatusBadge.textContent = 'Live Feed (Active)';
            }
        }
        
        await enumerateCameras();
        startDetectionLoop();
    } catch (err) {
        console.error('[SignBridge] Camera access error:', err);
        if (diagCameraState) {
            diagCameraState.textContent = 'FAILED';
            diagCameraState.className = 'text-danger';
        }
        if (diagLastError) {
            diagLastError.textContent = err.message || 'Camera permission denied';
            diagLastError.className = 'text-danger';
        }
        if (cameraStatusBadge) {
            cameraStatusBadge.className = 'badge bg-danger bg-opacity-25 text-danger border border-danger border-opacity-25';
            cameraStatusBadge.textContent = 'Camera Unavailable';
        }
        if (cameraErrorOverlay) {
            cameraErrorOverlay.classList.remove('d-none');
            const errTitle = document.getElementById('cameraErrorTitle');
            const errMsg = document.getElementById('cameraErrorMessage');
            if (errTitle) errTitle.textContent = 'Camera Access Error';
            if (errMsg) {
                errMsg.textContent = err.name === 'NotAllowedError' 
                    ? 'Camera permission was denied. Please allow camera permissions in your browser address bar.'
                    : (err.message || 'No compatible video input device found.');
            }
        }
    }
}

async function enumerateCameras() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');
        
        if (cameraSelect && cameraSelect.options.length <= 1) {
            cameraSelect.innerHTML = '';
            videoDevices.forEach((device, index) => {
                const opt = document.createElement('option');
                opt.value = device.deviceId;
                opt.textContent = device.label || `Camera ${index + 1}`;
                cameraSelect.appendChild(opt);
            });
        }
    } catch (e) {
        console.warn('Could not enumerate camera devices:', e);
    }
}

function switchCamera(deviceId) {
    if (deviceId) {
        startCamera(deviceId);
    }
}

// Continuous Frame Loop
function startDetectionLoop() {
    let lastFrameTime = performance.now();

    async function processFrame(now) {
        if (!mediaStream || !liveVideo || liveVideo.paused || liveVideo.ended || captureState === 'REVIEW') {
            requestAnimationFrame(processFrame);
            return;
        }

        // 1. Client-Side WASM MediaPipe Detection
        if (handsDetector && !serverFallbackActive && liveVideo.readyState >= 2 && !isDetecting) {
            isDetecting = true;
            try {
                await handsDetector.send({ image: liveVideo });
                framesProcessedCount++;
                if (diagFramesProcessed) {
                    diagFramesProcessed.textContent = framesProcessedCount;
                }

                // Calculate Live FPS
                fpsFrameCount++;
                const elapsed = now - lastFpsCalculationTime;
                if (elapsed >= 1000) {
                    const currentFps = Math.round((fpsFrameCount * 1000) / elapsed);
                    if (diagFpsBadge) {
                        diagFpsBadge.textContent = `${currentFps} FPS`;
                    }
                    fpsFrameCount = 0;
                    lastFpsCalculationTime = now;
                }
            } catch (e) {
                console.warn('[SignBridge] Client MediaPipe send error, falling back to server detection:', e);
                serverFallbackActive = true;
                if (diagHandLandmarker) {
                    diagHandLandmarker.textContent = 'SERVER FALLBACK';
                    diagHandLandmarker.className = 'text-info';
                }
            } finally {
                isDetecting = false;
            }
        } 
        // 2. Server-Side MediaPipe Fallback Frame Sampling (every 160ms)
        else if ((serverFallbackActive || !handsDetector) && liveVideo.readyState >= 2 && !isDetecting) {
            if (now - lastServerRequestTime >= 160) {
                lastServerRequestTime = now;
                isDetecting = true;
                sendFrameToServerForDetection().finally(() => {
                    isDetecting = false;
                });
            }
        }

        requestAnimationFrame(processFrame);
    }

    requestAnimationFrame(processFrame);
}

// Server Fallback: sends a lightweight snapshot frame for server MediaPipe processing
async function sendFrameToServerForDetection() {
    if (!liveVideo || liveVideo.readyState < 2) return;
    try {
        const offCanvas = document.createElement('canvas');
        offCanvas.width = 320;
        offCanvas.height = 240;
        const ctx = offCanvas.getContext('2d');
        ctx.drawImage(liveVideo, 0, 0, 320, 240);
        const base64Data = offCanvas.toDataURL('image/jpeg', 0.80);

        const formData = new FormData();
        formData.append('frame_base64', base64Data);

        const resp = await fetch(`/api/models/${typeof MODEL_ID !== 'undefined' ? MODEL_ID : ''}/detect-hand`, {
            method: 'POST',
            body: formData
        });
        const data = await resp.json();

        framesProcessedCount++;
        if (diagFramesProcessed) diagFramesProcessed.textContent = framesProcessedCount;

        if (data.success && data.hand_present && data.hands && data.hands.length > 0) {
            // Convert server detection into MediaPipe format
            const mpFormat = {
                multiHandLandmarks: data.hands.map(h => h.landmarks || []),
                multiHandedness: data.hands.map(h => ({
                    label: h.handedness || 'Right',
                    score: h.confidence || 0.95
                }))
            };
            onHandResults(mpFormat);
        } else {
            onHandResults({ multiHandLandmarks: [], multiHandedness: [] });
        }
    } catch (err) {
        console.warn('Server fallback frame error:', err);
    }
}

// ---------------- MEDIAPIPE HANDS RESULTS HANDLER ----------------

function onHandResults(results) {
    if (captureState === 'REVIEW') return;

    const canvas = trackingCanvas;
    if (!canvas || !liveVideo) return;
    const ctx = canvas.getContext('2d');

    // Synchronize canvas size with displayed video stream
    const vw = liveVideo.videoWidth || 640;
    const vh = liveVideo.videoHeight || 480;
    if (canvas.width !== vw || canvas.height !== vh) {
        canvas.width = vw;
        canvas.height = vh;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const rawHands = results.multiHandLandmarks || [];
    const rawHandedness = results.multiHandedness || [];

    // Case 1: No Hands Detected in this frame
    if (rawHands.length === 0) {
        handleNoHandsDetected(ctx, canvas);
        return;
    }

    // Case 2: Hand(s) Detected
    if (diagLastDetection) {
        diagLastDetection.textContent = 'DETECTED';
        diagLastDetection.className = 'text-success fw-bold';
    }
    if (diagHandsCount) {
        diagHandsCount.textContent = rawHands.length;
        diagHandsCount.className = 'text-success fw-bold';
    }
    if (diagLandmarksCount) {
        diagLandmarksCount.textContent = rawHands.length * 21;
        diagLandmarksCount.className = 'text-success fw-bold';
    }

    const detectedList = [];
    for (let i = 0; i < rawHands.length; i++) {
        const landmarks = rawHands[i];
        let handednessLabel = 'Right';
        let conf = 0.90;
        if (rawHandedness[i]) {
            handednessLabel = rawHandedness[i].label || 'Right';
            conf = rawHandedness[i].score || 0.90;
        }

        // Calculate Dynamic Bounding Box from all 21 landmarks
        let minX = 1.0, maxX = 0.0, minY = 1.0, maxY = 0.0;
        for (const lm of landmarks) {
            if (lm.x < minX) minX = lm.x;
            if (lm.x > maxX) maxX = lm.x;
            if (lm.y < minY) minY = lm.y;
            if (lm.y > maxY) maxY = lm.y;
        }

        // Padding around hand (22%) + Force Square Aspect Ratio (1:1)
        const boxW = maxX - minX;
        const boxH = maxY - minY;
        const padW = boxW * 0.22;
        const padH = boxH * 0.22;
        const cx = (minX + maxX) / 2.0;
        const cy = (minY + maxY) / 2.0;
        const side = Math.max(boxW + 2 * padW, boxH + 2 * padH);

        const sqYmin = Math.max(0.0, cy - side / 2.0);
        const sqYmax = Math.min(1.0, cy + side / 2.0);
        const sqXmin = Math.max(0.0, cx - side / 2.0);
        const sqXmax = Math.min(1.0, cx + side / 2.0);

        const area = (sqYmax - sqYmin) * (sqXmax - sqXmin);

        detectedList.push({
            index: i,
            landmarks: landmarks,
            handedness: handednessLabel,
            confidence: conf,
            bbox: { ymin: sqYmin, xmin: sqXmin, ymax: sqYmax, xmax: sqXmax },
            pixelBbox: {
                y1: Math.round(sqYmin * canvas.height),
                x1: Math.round(sqXmin * canvas.width),
                y2: Math.round(sqYmax * canvas.height),
                x2: Math.round(sqXmax * canvas.width)
            },
            center: { x: cx, y: cy },
            area: area
        });
    }

    lastDetectedHands = detectedList;

    // Determine Primary Hand based on configuration
    if (handSelection === 'Right') {
        const rHands = detectedList.filter(h => h.handedness === 'Right');
        primaryHand = rHands.length > 0 ? rHands[0] : detectedList.reduce((max, h) => h.area > max.area ? h : max, detectedList[0]);
    } else if (handSelection === 'Left') {
        const lHands = detectedList.filter(h => h.handedness === 'Left');
        primaryHand = lHands.length > 0 ? lHands[0] : detectedList.reduce((max, h) => h.area > max.area ? h : max, detectedList[0]);
    } else {
        // Auto: select largest / most prominent hand
        primaryHand = detectedList.reduce((max, h) => h.area > max.area ? h : max, detectedList[0]);
    }

    // Diagnostics update
    if (diagConfidence) {
        diagConfidence.textContent = `${(primaryHand.confidence * 100).toFixed(1)}%`;
        diagConfidence.className = 'text-success fw-bold';
    }
    if (diagBbox) {
        const pb = primaryHand.pixelBbox;
        diagBbox.textContent = `[x:${pb.x1}, y:${pb.y1}, w:${pb.x2 - pb.x1}, h:${pb.y2 - pb.y1}]`;
    }

    // Cancel disappearance debounce if hand returned
    if (disappearanceTimer) {
        clearTimeout(disappearanceTimer);
        disappearanceTimer = null;
    }

    // Draw Visualization Overlays
    drawHandTracking(ctx, canvas, detectedList, primaryHand);

    // Update UI Stats Badge
    if (handStatsBadge && handStatsText) {
        handStatsBadge.classList.remove('d-none');
        handStatsText.textContent = `${primaryHand.handedness} Hand (${Math.round(primaryHand.confidence * 100)}%) • 21 Landmarks`;
    }

    // Process State Automation
    if (currentMediaType === 'IMAGE') {
        handleImageStabilityAndAutoCapture(primaryHand);
    } else if (currentMediaType === 'VIDEO') {
        handleVideoAutoRecording(primaryHand);
    }
}

function handleNoHandsDetected(ctx, canvas) {
    primaryHand = null;
    lastDetectedHands = [];
    lastHandCenter = null;
    handSteadySince = null;

    cancelCountdown();

    if (diagLastDetection) {
        diagLastDetection.textContent = 'NO HAND';
        diagLastDetection.className = 'text-secondary';
    }
    if (diagHandsCount) {
        diagHandsCount.textContent = '0';
        diagHandsCount.className = 'text-secondary';
    }
    if (diagLandmarksCount) {
        diagLandmarksCount.textContent = '0';
        diagLandmarksCount.className = 'text-secondary';
    }
    if (diagConfidence) {
        diagConfidence.textContent = '--%';
        diagConfidence.className = 'text-secondary';
    }
    if (diagBbox) {
        diagBbox.textContent = 'N/A';
    }

    if (captureState !== 'RECORDING') {
        setCaptureState('WAITING_FOR_HAND');
        updateHandStateBadge('WAITING FOR HAND', 'bg-secondary bg-opacity-25 text-secondary');
        if (handStatsBadge) handStatsBadge.classList.add('d-none');
    } else {
        // Hand Disappeared During Recording: Trigger 800ms debounce
        updateHandStateBadge('HAND LOST (STOPPING...)', 'bg-warning bg-opacity-25 text-warning');
        if (!disappearanceTimer) {
            disappearanceTimer = setTimeout(() => {
                if (captureState === 'RECORDING') {
                    console.log('[SignBridge] Hand disappearance debounce expired; stopping recording.');
                    stopRecording();
                }
            }, disappearanceDebounceMs);
        }
    }
}

// ---------------- VISUALIZATION: SKELETON & DYNAMIC BOUNDING BOX ----------------

function drawHandTracking(ctx, canvas, allHands, targetHand) {
    allHands.forEach(hand => {
        const isPrimary = (hand === targetHand);
        const pb = hand.pixelBbox;
        const bw = pb.x2 - pb.x1;
        const bh = pb.y2 - pb.y1;

        // 1. Draw Dynamic Bounding Box around hand
        ctx.save();
        ctx.strokeStyle = isPrimary ? '#00d2ff' : 'rgba(255, 255, 255, 0.4)';
        ctx.lineWidth = isPrimary ? 2.5 : 1.5;
        ctx.setLineDash(isPrimary ? [8, 4] : [4, 4]);
        ctx.strokeRect(pb.x1, pb.y1, bw, bh);
        ctx.restore();

        // 2. Dynamic Label Tag above hand
        ctx.save();
        const tagText = `✋ ${hand.handedness} Hand (${Math.round(hand.confidence * 100)}%)`;
        ctx.font = 'bold 12px Inter, sans-serif';
        const tagWidth = ctx.measureText(tagText).width + 16;
        const tagHeight = 22;
        const tagY = Math.max(0, pb.y1 - tagHeight - 2);

        ctx.fillStyle = isPrimary ? '#00d2ff' : 'rgba(100, 116, 139, 0.9)';
        ctx.fillRect(pb.x1, tagY, tagWidth, tagHeight);

        ctx.fillStyle = isPrimary ? '#000000' : '#ffffff';
        ctx.fillText(tagText, pb.x1 + 8, tagY + 15);
        ctx.restore();

        // 3. Draw 21 Hand Landmarks & Skeleton if enabled
        if (showLandmarks && hand.landmarks) {
            // Draw Bone Connections
            ctx.save();
            ctx.strokeStyle = isPrimary ? 'rgba(0, 255, 136, 0.85)' : 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 2.0;

            HAND_CONNECTIONS.forEach(([i, j]) => {
                const p1 = hand.landmarks[i];
                const p2 = hand.landmarks[j];
                if (p1 && p2) {
                    ctx.beginPath();
                    ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
                    ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
                    ctx.stroke();
                }
            });
            ctx.restore();

            // Draw Landmark Joint Circles
            hand.landmarks.forEach((lm, idx) => {
                const lx = lm.x * canvas.width;
                const ly = lm.y * canvas.height;

                ctx.beginPath();
                ctx.arc(lx, ly, idx === 0 ? 5.5 : 3.5, 0, 2 * Math.PI);
                // Wrist: Coral Red, Tips: Cyan, Joints: Neon Green
                if (idx === 0) {
                    ctx.fillStyle = '#ff0055';
                } else if (idx % 4 === 0) {
                    ctx.fillStyle = '#00ffff';
                } else {
                    ctx.fillStyle = '#00ff88';
                }
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1;
                ctx.stroke();
            });
        }
    });
}

// ---------------- IMAGE MODE: STABILITY & AUTO-CAPTURE COUNTDOWN ----------------

function handleImageStabilityAndAutoCapture(hand) {
    if (!autoCaptureEnabled || captureState === 'REVIEW') {
        setCaptureState('HAND_DETECTED');
        updateHandStateBadge('HAND DETECTED', 'bg-success bg-opacity-25 text-success');
        return;
    }

    const currentCenter = hand.center;
    const now = performance.now();

    if (!lastHandCenter) {
        lastHandCenter = currentCenter;
        handSteadySince = now;
        setCaptureState('HAND_DETECTED');
        updateHandStateBadge('HAND DETECTED', 'bg-success bg-opacity-25 text-success');
        return;
    }

    // Compute centroid displacement delta
    const dx = currentCenter.x - lastHandCenter.x;
    const dy = currentCenter.y - lastHandCenter.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    // Movement threshold for stability (3.5% of frame)
    const movementThreshold = 0.035;

    if (distance > movementThreshold) {
        // Hand moved significantly; reset steady timer
        lastHandCenter = currentCenter;
        handSteadySince = now;

        if (captureState === 'COUNTDOWN') {
            cancelCountdown();
            if (countdownLabel) countdownLabel.textContent = 'Keep hand steady...';
        }
        setCaptureState('HAND_DETECTED');
        updateHandStateBadge('STABILIZING HAND...', 'bg-info bg-opacity-25 text-info');
    } else {
        // Hand is steady
        const steadyDuration = now - handSteadySince;
        if (steadyDuration >= stabilityThresholdMs && captureState !== 'COUNTDOWN') {
            startAutoCaptureCountdown(hand);
        } else if (captureState !== 'COUNTDOWN') {
            setCaptureState('STABILIZING');
            updateHandStateBadge('HOLD STEADY...', 'bg-primary bg-opacity-25 text-primary');
        }
    }
}

function startAutoCaptureCountdown(hand) {
    setCaptureState('COUNTDOWN');
    updateHandStateBadge('CAPTURING...', 'bg-warning bg-opacity-25 text-warning');
    if (captureCountdownOverlay) captureCountdownOverlay.classList.remove('d-none');
    countdownRemaining = 3;
    if (countdownNumber) countdownNumber.textContent = countdownRemaining;
    if (countdownLabel) countdownLabel.textContent = 'Hold Hand Steady';

    if (countdownTimer) clearInterval(countdownTimer);

    countdownTimer = setInterval(() => {
        countdownRemaining--;
        if (countdownRemaining > 0) {
            if (countdownNumber) countdownNumber.textContent = countdownRemaining;
        } else {
            clearInterval(countdownTimer);
            countdownTimer = null;
            if (captureCountdownOverlay) captureCountdownOverlay.classList.add('d-none');
            // Execute automatic crop capture
            executeHandCropCapture(primaryHand || hand);
        }
    }, 800);
}

function cancelCountdown() {
    if (countdownTimer) {
        clearInterval(countdownTimer);
        countdownTimer = null;
    }
    if (captureCountdownOverlay) {
        captureCountdownOverlay.classList.add('d-none');
    }
}

function manualCaptureSnapshot() {
    if (!primaryHand && lastDetectedHands.length === 0) {
        alert('No hand detected. Please position your hand in view before capturing.');
        return;
    }
    cancelCountdown();
    executeHandCropCapture(primaryHand || lastDetectedHands[0]);
}

// ---------------- EXACT HAND CROP & SNAPSHOT EXTRACTION ----------------

function executeHandCropCapture(hand) {
    if (!liveVideo || liveVideo.readyState < 2) return;
    setCaptureState('REVIEW');
    updateHandStateBadge('REVIEW SAMPLE', 'bg-info');

    const pb = hand ? hand.pixelBbox : {
        x1: 0, y1: 0,
        x2: liveVideo.videoWidth, y2: liveVideo.videoHeight
    };

    const cropW = Math.max(10, pb.x2 - pb.x1);
    const cropH = Math.max(10, pb.y2 - pb.y1);

    // Render exact 224x224 hand crop onto snapshotCanvas
    snapshotCanvas.width = 224;
    snapshotCanvas.height = 224;
    const sCtx = snapshotCanvas.getContext('2d');
    sCtx.imageSmoothingQuality = 'high';

    // Draw ONLY the hand bounding box from live video feed
    sCtx.drawImage(
        liveVideo,
        pb.x1, pb.y1, cropW, cropH,
        0, 0, 224, 224
    );

    // Save metadata
    capturedMetadata = {
        handedness: hand ? hand.handedness : 'Auto',
        confidence: hand ? hand.confidence : 0.95,
        bbox: hand ? [hand.bbox.ymin, hand.bbox.xmin, hand.bbox.ymax, hand.bbox.xmax] : [0, 0, 1, 1],
        landmarks_count: hand && hand.landmarks ? hand.landmarks.length : 21,
        preprocessing_version: 'hand_landmark_v1'
    };

    // Convert crop canvas to Blob
    snapshotCanvas.toBlob((blob) => {
        capturedBlob = blob;
    }, 'image/jpeg', 0.95);

    // Show Review UI
    if (snapshotReviewContainer) snapshotReviewContainer.classList.remove('d-none');
    if (liveControls) liveControls.classList.add('d-none');
    if (reviewControls) reviewControls.classList.remove('d-none');

    if (reviewHandDetails) {
        reviewHandDetails.textContent = `Hand: ${capturedMetadata.handedness} (${Math.round(capturedMetadata.confidence * 100)}%) • 21 Landmarks • 224×224 Hand Crop`;
    }
}

// ---------------- VIDEO MODE: AUTOMATIC RECORDING & DEBOUNCE ----------------

function handleVideoAutoRecording(hand) {
    if (!autoCaptureEnabled || captureState === 'REVIEW') {
        if (captureState !== 'RECORDING') {
            setCaptureState('HAND_DETECTED');
            updateHandStateBadge('HAND DETECTED', 'bg-success bg-opacity-25 text-success');
        }
        return;
    }

    if (captureState !== 'RECORDING') {
        // Hand appeared: Start recording automatically
        console.log('[SignBridge] Hand confirmed in video mode; auto-starting recording.');
        startRecording();
    }
}

function startRecording() {
    if (!mediaStream) return;
    recordedChunks = [];
    capturedBlob = null;

    try {
        const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' :
                         (MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : 'video/mp4');
        
        mediaRecorder = new MediaRecorder(mediaStream, { mimeType: mimeType });

        mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
                recordedChunks.push(e.data);
            }
        };

        mediaRecorder.onstop = () => {
            capturedBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
            if (recordedVideoReview) {
                recordedVideoReview.src = URL.createObjectURL(capturedBlob);
                recordedVideoReview.classList.remove('d-none');
            }
            if (liveControls) liveControls.classList.add('d-none');
            if (reviewControls) reviewControls.classList.remove('d-none');
            setCaptureState('REVIEW');
            updateHandStateBadge('RECORDING COMPLETE', 'bg-success');
        };

        mediaRecorder.start(200);
        setCaptureState('RECORDING');
        updateHandStateBadge('RECORDING...', 'bg-danger animate-pulse');
        if (recordingBadge) recordingBadge.classList.remove('d-none');
        if (btnStartRecord) btnStartRecord.classList.add('d-none');
        if (btnStopRecord) btnStopRecord.classList.remove('d-none');

        recordingStartTime = Date.now();
        if (recTimer) recTimer.textContent = '00:00';
        if (recordingTimerInterval) clearInterval(recordingTimerInterval);
        recordingTimerInterval = setInterval(updateRecTimer, 1000);
    } catch (err) {
        console.error('MediaRecorder start error:', err);
        alert('Could not start video recording: ' + err.message);
    }
}

function updateRecTimer() {
    const elapsedSeconds = Math.floor((Date.now() - recordingStartTime) / 1000);
    recordedDurationSeconds = elapsedSeconds;
    const mins = String(Math.floor(elapsedSeconds / 60)).padStart(2, '0');
    const secs = String(elapsedSeconds % 60).padStart(2, '0');
    if (recTimer) recTimer.textContent = `${mins}:${secs}`;
}

function stopRecording() {
    if (recordingTimerInterval) {
        clearInterval(recordingTimerInterval);
        recordingTimerInterval = null;
    }
    if (disappearanceTimer) {
        clearTimeout(disappearanceTimer);
        disappearanceTimer = null;
    }
    if (recordingBadge) recordingBadge.classList.add('d-none');
    if (btnStopRecord) btnStopRecord.classList.add('d-none');
    if (btnStartRecord) btnStartRecord.classList.remove('d-none');

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
}

// ---------------- RETAKE & SAVE CAPTURED MEDIA ----------------

function retakeMedia() {
    cancelCountdown();
    if (disappearanceTimer) {
        clearTimeout(disappearanceTimer);
        disappearanceTimer = null;
    }
    capturedBlob = null;

    if (snapshotReviewContainer) snapshotReviewContainer.classList.add('d-none');
    if (recordedVideoReview) {
        recordedVideoReview.classList.add('d-none');
        recordedVideoReview.src = '';
    }

    if (reviewControls) reviewControls.classList.add('d-none');
    if (liveControls) liveControls.classList.remove('d-none');

    setCaptureState('WAITING_FOR_HAND');
    updateHandStateBadge('WAITING FOR HAND', 'bg-secondary bg-opacity-25 text-secondary');
    if (handStatsBadge) handStatsBadge.classList.add('d-none');

    lastHandCenter = null;
    handSteadySince = null;
}

async function saveCapturedSample() {
    if (!capturedBlob) {
        alert('No captured image or video available to save.');
        return;
    }

    if (!targetClassSelect || !targetClassSelect.value) {
        alert('Please select a Sign Class before saving.');
        return;
    }
    if (!targetParticipantSelect || !targetParticipantSelect.value) {
        alert('Please select a Participant before saving.');
        return;
    }

    const classId = targetClassSelect.value;
    const participantId = targetParticipantSelect.value;
    const notesInput = document.getElementById('sampleNotes');
    const notes = notesInput ? notesInput.value : '';

    if (btnSaveSample) {
        btnSaveSample.disabled = true;
        btnSaveSample.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Saving to Database...`;
    }

    try {
        const formData = new FormData();
        const ext = currentMediaType === 'IMAGE' ? 'jpg' : 'webm';
        formData.append('file', capturedBlob, `capture_${Date.now()}.${ext}`);
        formData.append('class_id', classId);
        formData.append('participant_id', participantId);
        formData.append('notes', notes);

        if (capturedMetadata) {
            if (capturedMetadata.handedness) formData.append('handedness', capturedMetadata.handedness);
            if (capturedMetadata.confidence) formData.append('hand_detection_confidence', capturedMetadata.confidence.toString());
            if (capturedMetadata.bbox) formData.append('hand_bbox', JSON.stringify(capturedMetadata.bbox));
        }

        const endpoint = currentMediaType === 'IMAGE' 
            ? `/api/models/${typeof MODEL_ID !== 'undefined' ? MODEL_ID : ''}/samples/capture-image`
            : `/api/models/${typeof MODEL_ID !== 'undefined' ? MODEL_ID : ''}/samples/capture-video`;

        const resp = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.detail || 'Server error while saving sample');
        }

        // Success Alert
        showToastNotification(`Hand sample saved successfully! [${data.label || 'Sign'}] added to dataset.`);
        retakeMedia();
    } catch (err) {
        console.error('Save sample error:', err);
        alert('Failed to save sample: ' + err.message);
    } finally {
        if (btnSaveSample) {
            btnSaveSample.disabled = false;
            btnSaveSample.innerHTML = `<i class="bi bi-cloud-arrow-up-fill me-2"></i><span class="fw-semibold">Save Hand Sample</span>`;
        }
    }
}

async function handleMediaUpload(event) {
    event.preventDefault();
    const fileInput = document.getElementById('uploadFileInput');
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        alert('Please select a file to upload.');
        return;
    }

    const file = fileInput.files[0];
    const isVid = file.type.startsWith('video') || file.name.endsWith('.mp4') || file.name.endsWith('.webm');
    const classId = targetClassSelect ? targetClassSelect.value : '';
    const participantId = targetParticipantSelect ? targetParticipantSelect.value : '';
    const notesInput = document.getElementById('sampleNotes');
    const notes = notesInput ? notesInput.value : '';

    if (!classId) {
        alert('Please select a Sign Class.');
        return;
    }
    if (!participantId) {
        alert('Please select a Participant.');
        return;
    }

    const btnUpload = document.getElementById('btnUploadMedia');
    if (btnUpload) {
        btnUpload.disabled = true;
        btnUpload.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Uploading...`;
    }

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('class_id', classId);
        formData.append('participant_id', participantId);
        formData.append('notes', notes);

        const endpoint = isVid 
            ? `/api/models/${typeof MODEL_ID !== 'undefined' ? MODEL_ID : ''}/samples/capture-video`
            : `/api/models/${typeof MODEL_ID !== 'undefined' ? MODEL_ID : ''}/samples/capture-image`;

        const resp = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.detail || 'Upload failed');
        }

        showToastNotification(`File uploaded successfully! [${data.label}] saved.`);
        fileInput.value = '';
    } catch (e) {
        alert('Error uploading media: ' + e.message);
    } finally {
        if (btnUpload) {
            btnUpload.disabled = false;
            btnUpload.innerHTML = `<i class="bi bi-cloud-arrow-up me-1"></i>Upload & Save Sample`;
        }
    }
}

function showToastNotification(msg) {
    const toastDiv = document.createElement('div');
    toastDiv.className = 'position-fixed bottom-0 end-0 p-3';
    toastDiv.style.zIndex = '1050';
    toastDiv.innerHTML = `
        <div class="toast align-items-center text-white bg-success border-0 show shadow" role="alert">
            <div class="d-flex">
                <div class="toast-body"><i class="bi bi-check-circle-fill me-2"></i>${msg}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" onclick="this.parentElement.parentElement.remove()"></button>
            </div>
        </div>
    `;
    document.body.appendChild(toastDiv);
    setTimeout(() => {
        toastDiv.remove();
    }, 4000);
}
