/**
 * AERO-TACTIC Ground Control Station Client Application
 * Handles Telemetry WebSocket, Leaflet Map, Lawnmower Grid Drawing, 
 * Dual Side-by-Side Viewports (Original Raw + Terrain Segmented), Opacity Slider,
 * Multi-Spectrum Adaptive Water Classification, and Camera Mount Control.
 */

document.addEventListener('DOMContentLoaded', () => {

    let currentCamAngle = -45.0; // Default 45 deg oblique
    let activePerceptionMode = "TERRAIN"; // "TERRAIN", "OBJECTS", "COMBINED", "THERMAL", "NDVI"
    let isDualView = false;

    // --- Telemetry WebSocket ---
    let ws = null;
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            document.getElementById('val-link').innerText = 'CONNECTED';
            document.getElementById('val-link').classList.remove('disconnected');
            document.getElementById('val-link').style.color = '#00ff88';
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                updateTelemetryUI(data.telemetry, data.mission_status);
            } catch (err) {
                console.error("Error parsing telemetry:", err);
            }
        };

        ws.onclose = () => {
            document.getElementById('val-link').innerText = 'DISCONNECTED';
            document.getElementById('val-link').classList.add('disconnected');
            setTimeout(connectWebSocket, 2000);
        };
    }
    connectWebSocket();

    function updateTelemetryUI(telem, mission) {
        if (!telem) return;

        document.getElementById('val-mode').innerText = telem.mode || 'GUIDED';
        document.getElementById('val-mission-state').innerText = mission?.state || 'IDLE';
        document.getElementById('val-alt').innerHTML = `${(telem.alt_relative_m || 0).toFixed(1)} <small>m</small>`;
        document.getElementById('val-spd').innerHTML = `${(telem.ground_speed_mps || 0).toFixed(1)} <small>m/s</small>`;
        
        // 4S LiPo Battery Display
        const batVolts = (telem.battery_voltage || 16.4).toFixed(1);
        const batPct = telem.battery_remaining_pct || 96;
        document.getElementById('val-bat').innerText = `${batVolts}V (${batPct}%)`;

        const lat = telem.lat || defaultLat;
        const lon = telem.lon || defaultLon;
        uavMarker.setLatLng([lat, lon]);
        flightPath.addLatLng([lat, lon]);

        const horizon = document.getElementById('horizon-line');
        if (horizon) {
            const pitch = telem.pitch_deg || 0;
            const roll = telem.roll_deg || 0;
            horizon.style.transform = `rotate(${-roll}deg) translateY(${pitch * 1.5}px)`;
        }

        // Active Filter Tag
        const queryTag = document.getElementById('query-tag');
        if (mission?.active_filter && (mission.active_filter.color || mission.active_filter.target_class)) {
            const f = mission.active_filter;
            queryTag.innerText = `FILTER: ${f.color || ''} ${f.target_class || ''}`.toUpperCase().trim();
            queryTag.style.background = 'rgba(0, 255, 136, 0.25)';
        } else {
            queryTag.innerText = activePerceptionMode === 'TERRAIN' ? 'ALL TERRAIN TYPES ACTIVE' : 'NO FILTER (ALL TARGETS)';
            queryTag.style.background = 'rgba(0, 240, 255, 0.15)';
        }

        // Camera Mount Angle Badge
        if (mission?.camera_mount_angle_deg !== undefined) {
            currentCamAngle = mission.camera_mount_angle_deg;
            const isOblique = Math.abs(currentCamAngle - (-45.0)) < 1.0;
            document.getElementById('val-cam-angle').innerText = isOblique ? '45Â° OBLIQUE' : '90Â° NADIR';
        }

        // --- Live Terrain Classification Percentages (Multi-Spectrum Water & Land) ---
        if (mission?.terrain_stats) {
            const ts = mission.terrain_stats;
            const veg = ts.vegetation_pct ?? ts.dense_forest_pct ?? 0;
            const water = ts.water_pct ?? ts.ocean_water_pct ?? 0;
            const bldg = ts.building_pct ?? 0;
            const road = ts.road_pct ?? ts.roads_pavement_pct ?? 0;
            const bare = ts.bare_ground_pct ?? ts.beach_sand_pct ?? 0;
            const sky = ts.sky_pct ?? ts.snow_clouds_pct ?? 0;
            const other = ts.other_pct ?? ts.mountain_rock_pct ?? 0;

            if (document.getElementById('pct-forest')) document.getElementById('pct-forest').innerText = `${veg}%`;
            if (document.getElementById('pct-grassland')) document.getElementById('pct-grassland').innerText = `${bldg}%`;
            if (document.getElementById('pct-water')) document.getElementById('pct-water').innerText = `${water}%`;
            if (document.getElementById('pct-sand')) document.getElementById('pct-sand').innerText = `${bare}%`;
            if (document.getElementById('pct-rock')) document.getElementById('pct-rock').innerText = `${other}%`;
            if (document.getElementById('pct-roads')) document.getElementById('pct-roads').innerText = `${road}%`;
            if (document.getElementById('pct-clouds')) document.getElementById('pct-clouds').innerText = `${sky}%`;
        }

        // Terrain Scan Progress
        const scanPct = mission?.scan_progress_pct || 0;
        const scannedArea = Math.round(mission?.scanned_area_m2 || 0);
        document.getElementById('scan-bar-fill').style.width = `${scanPct}%`;
        document.getElementById('terrain-scan-progress').innerText = `SCAN: ${scanPct}% (${scannedArea} mÂ²)`;

        const surveyChip = document.getElementById('survey-chip');
        if (mission?.state === 'SURVEY_GRID') {
            surveyChip.innerText = `SCANNING GRID (${mission.current_waypoint_idx + 1}/${mission.active_waypoints_count})`;
            surveyChip.style.borderColor = '#00ff88';
            surveyChip.style.color = '#00ff88';
        } else {
            surveyChip.innerText = 'GRID STANDBY';
            surveyChip.style.borderColor = '#00f0ff';
            surveyChip.style.color = '#00f0ff';
        }

        // Render Lawnmower Survey Grid on Map
        if (mission?.survey_waypoints && mission.survey_waypoints.length > 0) {
            renderSurveyGrid(mission.survey_waypoints, mission.current_waypoint_idx);
        }

        if (mission?.latest_detections) {
            renderDetections(mission.latest_detections);
        }
    }

    function renderSurveyGrid(waypoints, currentIdx) {
        const latlngs = waypoints.map(wp => [wp.lat, wp.lon]);
        surveyGridPath.setLatLngs(latlngs);

        if (surveyWaypointMarkers.length !== waypoints.length) {
            surveyWaypointMarkers.forEach(m => map.removeLayer(m));
            surveyWaypointMarkers = [];

            waypoints.forEach((wp, idx) => {
                const marker = L.circleMarker([wp.lat, wp.lon], {
                    radius: 4,
                    color: '#00ff88',
                    fillColor: '#00ff88',
                    fillOpacity: 0.6
                }).addTo(map);
                marker.bindPopup(`<b>Survey Waypoint #${idx + 1}</b><br>Alt: ${wp.alt_m}m`);
                surveyWaypointMarkers.push(marker);
            });
        }
    }

    function renderDetections(dets) {
        const list = document.getElementById('detections-list');
        document.getElementById('det-count').innerText = `${dets.length} DETECTED`;

        if (dets.length === 0) {
            list.innerHTML = '<div class="empty-hint">Scanning for localized objects (people, vehicles, boats, hotspots)...</div>';
            return;
        }

        list.innerHTML = '';
        dets.forEach((d, idx) => {
            const item = document.createElement('div');
            const isMatch = d.is_matched !== false;
            item.className = `detection-item ${d.source || 'rgb'} ${isMatch ? 'matched-item' : ''}`;
            const gpsStr = d.gps ? `[${d.gps.lat.toFixed(5)}, ${d.gps.lon.toFixed(5)}]` : 'Acquiring GPS...';
            const label = d.display_label || `${d.class.toUpperCase()} (${Math.round(d.confidence * 100)}%)`;

            item.innerHTML = `
                <div>
                    <strong>${label}</strong>
                    <div style="font-size: 0.65rem; color: #94a3b8;">${gpsStr}</div>
                </div>
                <div><span class="badge ${isMatch ? 'matched' : ''}">${isMatch ? 'MATCH' : d.source.toUpperCase()}</span></div>
            `;
            list.appendChild(item);

            if (d.gps && d.gps.lat && d.gps.lon) {
                const markerKey = `${d.class}_${idx}`;
                if (!detectionMarkers[markerKey]) {
                    const color = isMatch ? '#00ff88' : (d.source === 'thermal' ? '#ffb800' : '#00f0ff');
                    const marker = L.circleMarker([d.gps.lat, d.gps.lon], {
                        radius: isMatch ? 8 : 5,
                        color: color,
                        fillColor: color,
                        fillOpacity: 0.85
                    }).addTo(map);
                    marker.bindPopup(`<b>${label}</b><br>Lat: ${d.gps.lat}<br>Lon: ${d.gps.lon}`);
                    detectionMarkers[markerKey] = marker;
                }
            }
        });
    }

    // --- Dual Side-by-Side View Toggle ---
    const btnDualView = document.getElementById('btn-dual-view');
    const videoContainer = document.getElementById('video-container');
    const viewportRaw = document.getElementById('viewport-raw');

    function toggleDualView(enable) {
        isDualView = (enable !== undefined) ? enable : !isDualView;
        if (isDualView) {
            videoContainer.classList.add('dual-mode-grid');
            viewportRaw.style.display = 'flex';
            btnDualView.classList.add('active');
            btnDualView.innerHTML = '<i class="fa-solid fa-square"></i> SINGLE VIEW';
        } else {
            videoContainer.classList.remove('dual-mode-grid');
            viewportRaw.style.display = 'none';
            btnDualView.classList.remove('active');
            btnDualView.innerHTML = '<i class="fa-solid fa-table-columns"></i> DUAL VIEW: RAW + TERRAIN';
        }
    }
    btnDualView.addEventListener('click', () => toggleDualView());

    // --- Mode Ribbon Switcher (Terrain Mode vs Object Mode vs Dual) ---
    const feedImg = document.getElementById('active-video-feed');
    const feedTag = document.getElementById('feed-tag');
    const terrainPanel = document.getElementById('terrain-legend-panel');
    const detectionsPanel = document.getElementById('detections-panel');
    const opacityBar = document.getElementById('opacity-bar');

    function switchPerceptionMode(mode, stream) {
        activePerceptionMode = mode;
        document.querySelectorAll('.mode-ribbon-btn').forEach(b => b.classList.remove('active'));
        const btn = document.querySelector(`.mode-ribbon-btn[data-mode="${mode}"]`);
        if (btn) btn.classList.add('active');

        feedImg.src = `/stream/${stream}`;

        if (mode === 'TERRAIN') {
            feedTag.innerText = 'TERRAIN SEGMENTATION (ADAPTIVE WATER & LAND)';
            terrainPanel.style.display = 'flex';
            opacityBar.style.display = 'flex';
            detectionsPanel.style.display = 'none';
        } else if (mode === 'OBJECTS') {
            feedTag.innerText = 'OBJECT DETECTION & ATTRIBUTE TRACKING';
            terrainPanel.style.display = 'none';
            opacityBar.style.display = 'none';
            detectionsPanel.style.display = 'flex';
        } else if (mode === 'COMBINED') {
            feedTag.innerText = 'COMBINED DUAL HUD (TERRAIN + OBJECTS)';
            terrainPanel.style.display = 'flex';
            opacityBar.style.display = 'flex';
            detectionsPanel.style.display = 'flex';
        } else if (mode === 'THERMAL') {
            feedTag.innerText = 'THERMAL RADIOMETRIC (IRONBOW)';
            terrainPanel.style.display = 'none';
            opacityBar.style.display = 'none';
            detectionsPanel.style.display = 'flex';
        } else if (mode === 'NDVI') {
            feedTag.innerText = 'MULTISPECTRAL NDVI VEGETATION INDEX';
            terrainPanel.style.display = 'flex';
            opacityBar.style.display = 'none';
            detectionsPanel.style.display = 'none';
        }
    }

    document.querySelectorAll('.mode-ribbon-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.getAttribute('data-mode');
            const stream = btn.getAttribute('data-stream');
            switchPerceptionMode(mode, stream);
        });
    });

    // --- Opacity Slider Control ---
    const opacitySlider = document.getElementById('opacity-slider');
    const valOpacity = document.getElementById('val-opacity');
    opacitySlider.addEventListener('input', async (e) => {
        const pct = e.target.value;
        valOpacity.innerText = `${pct}%`;
        const alpha = pct / 100.0;
        try {
            await fetch('/api/set_terrain_alpha', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ alpha: alpha })
            });
        } catch (err) {
            console.error("Failed to set opacity:", err);
        }
    });

    // --- Camera Mount Angle Toggle ---
    const btnCamAngle = document.getElementById('btn-camera-angle');
    btnCamAngle.addEventListener('click', async () => {
        const nextAngle = (Math.abs(currentCamAngle - (-45.0)) < 1.0) ? -90.0 : -45.0;
        try {
            const resp = await fetch('/api/set_camera_angle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ angle_deg: nextAngle })
            });
            const data = await resp.json();
            currentCamAngle = nextAngle;
            document.getElementById('val-cam-angle').innerText = data.mode;
            speakFeedback(`Camera mount set to ${data.mode}`);
        } catch (err) {
            console.error("Failed to toggle camera angle:", err);
        }
    });

    // --- Preset Feeds & Custom Scenery Video / Image Upload ---
    const presetSelect = document.getElementById('preset-feed-select');
    const uploadInput = document.getElementById('video-upload-input');

    async function loadPresetFeeds() {
        if (!presetSelect) return;
        try {
            const resp = await fetch('/api/list_feeds');
            const data = await resp.json();
            if (data.status === 'OK' && Array.isArray(data.feeds)) {
                presetSelect.innerHTML = '<option value="">📂 SELECT VIDEO FEED...</option>';
                data.feeds.forEach(f => {
                    const opt = document.createElement('option');
                    opt.value = f;
                    opt.innerText = f.length > 28 ? f.substring(0, 25) + '...' : f;
                    opt.title = f;
                    presetSelect.appendChild(opt);
                });
            }
        } catch (err) {
            console.error("Failed to load preset feeds list:", err);
        }
    }
    loadPresetFeeds();

    if (presetSelect) {
        presetSelect.addEventListener('change', async (e) => {
            const filename = e.target.value;
            if (!filename) return;

            document.getElementById('feed-tag').innerText = `SWITCHING FEED: ${filename}...`;
            try {
                const resp = await fetch('/api/select_feed', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename: filename })
                });
                const data = await resp.json();
                if (data.status === 'SUCCESS') {
                    document.getElementById('feed-tag').innerText = `FEED ACTIVE: ${filename}`;
                    speakFeedback(`Switched feed to ${filename}`);

                    setTimeout(() => {
                        const currentSrc = feedImg.src.split('?')[0];
                        feedImg.src = `${currentSrc}?t=${Date.now()}`;
                        if (viewportRaw && isDualView) {
                            const rawImg = document.getElementById('raw-video-feed');
                            if (rawImg) rawImg.src = `/stream/raw?t=${Date.now()}`;
                        }
                    }, 300);
                } else {
                    alert(`Failed to activate feed: ${data.message}`);
                }
            } catch (err) {
                console.error("Feed selection error:", err);
            }
        });
    }

    uploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);
        document.getElementById('feed-tag').innerText = `UPLOADING: ${file.name}...`;

        try {
            const resp = await fetch('/api/upload_feed', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();
            if (data.status === 'SUCCESS') {
                document.getElementById('feed-tag').innerText = `INGESTING FEED: ${file.name}`;
                speakFeedback(`Loaded scenery feed ${file.name}`);
                loadPresetFeeds(); // Refresh dropdown list

                // Force browser image reconnection to new video feed stream
                setTimeout(() => {
                    const currentSrc = feedImg.src.split('?')[0];
                    feedImg.src = `${currentSrc}?t=${Date.now()}`;
                    if (viewportRaw && isDualView) {
                        const rawImg = document.getElementById('raw-video-feed');
                        if (rawImg) rawImg.src = `/stream/raw?t=${Date.now()}`;
                    }
                }, 300);
            } else {
                alert(`Upload failed: ${data.message}`);
            }
        } catch (err) {
            console.error("Upload error:", err);
            alert(`Error uploading file: ${err}`);
        } finally {
            uploadInput.value = ''; // Reset file input so user can re-upload easily
        }
    });

    document.getElementById('btn-recenter').addEventListener('click', () => {
        map.setView(uavMarker.getLatLng(), 18);
    });

    // --- Speech Recognition ---
    const voiceBtn = document.getElementById('btn-voice');
    const voiceStatus = document.getElementById('voice-status');
    const voiceFeedback = document.getElementById('voice-feedback');
    let recognition = null;
    let isListening = false;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            isListening = true;
            voiceBtn.classList.add('recording');
            voiceStatus.innerText = 'LISTENING... (Speak command now)';
            voiceStatus.style.color = '#ff3366';
        };

        recognition.onresult = (event) => {
            const spokenText = event.results[0][0].transcript;
            voiceStatus.innerText = `HEARD: "${spokenText}"`;
            handleSpokenOrTypedCommand(spokenText);
        };

        recognition.onerror = (event) => {
            console.warn("Speech recognition error:", event.error);
            resetVoiceUI();
        };

        recognition.onend = () => {
            resetVoiceUI();
        };

        voiceBtn.addEventListener('click', () => {
            if (isListening) {
                recognition.stop();
            } else {
                try {
                    recognition.start();
                } catch (e) {
                    console.error("Speech recognition start failed:", e);
                }
            }
        });
    }

    function resetVoiceUI() {
        isListening = false;
        voiceBtn.classList.remove('recording');
        voiceStatus.innerText = 'VOICE COMMAND STANDBY (Click Mic or Type)';
        voiceStatus.style.color = '#00f0ff';
    }

    function handleSpokenOrTypedCommand(cmdText) {
        const lower = cmdText.toLowerCase().trim();
        if (lower.includes('terrain mode') || lower.includes('segmentation mode')) {
            switchPerceptionMode('TERRAIN', 'terrain_seg');
            speakFeedback("Switched to Terrain Segmentation Mode");
            return;
        } else if (lower.includes('object mode') || lower.includes('detection mode')) {
            switchPerceptionMode('OBJECTS', 'rgb');
            speakFeedback("Switched to Object Detection Mode");
            return;
        } else if (lower.includes('combined mode') || lower.includes('dual mode') || lower.includes('dual view')) {
            toggleDualView(true);
            speakFeedback("Engaged Dual View: Raw Video plus Terrain Map");
            return;
        }
        sendCommand(cmdText);
    }

    async function sendCommand(cmdText) {
        if (!cmdText || !cmdText.trim()) return;
        voiceFeedback.innerText = `Dispatching: "${cmdText}"...`;

        try {
            const resp = await fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: cmdText })
            });
            const data = await resp.json();

            if (data.status === 'OK') {
                const msg = data.execution?.message || data.parsed?.feedback_msg || "Command executed.";
                voiceFeedback.innerText = `[SUCCESS] ${msg}`;
                speakFeedback(msg);
            } else {
                const errMsg = data.execution?.message || "Command not recognized.";
                voiceFeedback.innerText = `[WARNING] ${errMsg}`;
                speakFeedback(errMsg);
            }
        } catch (err) {
            voiceFeedback.innerText = `[ERROR] Failed: ${err}`;
        }
    }

    function speakFeedback(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.05;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }
    }

    const cmdInput = document.getElementById('cmd-input');
    const sendBtn = document.getElementById('btn-send-cmd');

    sendBtn.addEventListener('click', () => {
        handleSpokenOrTypedCommand(cmdInput.value);
        cmdInput.value = '';
    });

    cmdInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            handleSpokenOrTypedCommand(cmdInput.value);
            cmdInput.value = '';
        }
    });

    document.querySelectorAll('.chip-btn, .quick-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const cmd = btn.getAttribute('data-cmd');
            handleSpokenOrTypedCommand(cmd);
        });
    });
});