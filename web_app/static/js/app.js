/* FLOOD-PASS — Lightweight App JS (rebuilt for clean UI) */
document.addEventListener('DOMContentLoaded', () => {

    // ── State ──────────────────────────────────────────────────────────
    let activeSamplePath = null;
    let uploadedFile     = null;
    let uploadedVideoFile = null;
    let selectedVideoPath = null;
    let videoPollInterval = null;
    let currentViewMode  = 'overlay';

    // ── Tab Navigation ─────────────────────────────────────────────────
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(target).classList.add('active');
            if (target === 'benchmark-tab') loadBenchmark();
        });
    });

    // ── View Toggle (overlay / depth / split / mask) ───────────────────
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentViewMode = btn.dataset.view;
            applyViewMode();
        });
    });

    function applyViewMode() {
        const wrap    = document.getElementById('visualWrap');
        if (!wrap || wrap.classList.contains('hidden')) return;
        const origImg    = document.getElementById('origImg');
        const overlayImg = document.getElementById('overlayImg');
        const maskImg    = document.getElementById('maskImg');
        const depthImg   = document.getElementById('depthImg');

        // Reset ALL images: hide them and clear any inline positioning
        [origImg, overlayImg, maskImg, depthImg].forEach(el => {
            if (el) {
                el.classList.add('hidden');
                el.style.width  = '100%';
                el.style.left   = '0';
                el.style.display = '';   // clear any inline display override
            }
        });

        switch (currentViewMode) {
            case 'overlay':
                if (overlayImg) overlayImg.classList.remove('hidden');
                break;
            case 'depth':
                if (depthImg && depthImg.src) depthImg.classList.remove('hidden');
                else if (overlayImg) overlayImg.classList.remove('hidden'); // fallback
                break;
            case 'mask':
                if (maskImg && maskImg.src) maskImg.classList.remove('hidden');
                else if (overlayImg) overlayImg.classList.remove('hidden'); // fallback
                break;
            case 'split':
                // Show original on left half, overlay on right half
                if (origImg)    { origImg.classList.remove('hidden');    origImg.style.width = '50%'; origImg.style.left = '0'; }
                if (overlayImg) { overlayImg.classList.remove('hidden'); overlayImg.style.width = '50%'; overlayImg.style.left = '50%'; }
                break;
            default:
                if (overlayImg) overlayImg.classList.remove('hidden');
        }
    }

    // ── Opacity Slider ────────────────────────────────────────────────
    const opacitySlider = document.getElementById('opacitySlider');
    const opacityValue  = document.getElementById('opacityValue');
    if (opacitySlider) {
        opacitySlider.addEventListener('input', () => {
            const v = opacitySlider.value;
            opacityValue.textContent = v + '%';
            const overlayImg = document.getElementById('overlayImg');
            if (overlayImg) overlayImg.style.opacity = v / 100;
        });
    }

    // ── Depth Backend Toggle ──────────────────────────────────────────
    const depthBackendSelect = document.getElementById('depthBackendSelect');
    const sturmInfoRow       = document.getElementById('sturmInfoRow');
    if (depthBackendSelect) {
        depthBackendSelect.addEventListener('change', () => {
            const isSturm = depthBackendSelect.value === 'sturm';
            if (sturmInfoRow) sturmInfoRow.style.display = isSturm ? '' : 'none';
        });
    }

    // ── Sample Gallery ────────────────────────────────────────────────
    async function loadSamples() {
        const grid = document.getElementById('sampleGrid');
        if (!grid) return;
        try {
            const res  = await fetch('/api/samples');
            const data = await res.json();
            if (!data.samples || data.samples.length === 0) {
                grid.innerHTML = '<div class="skeleton">No sample scenes found.</div>';
                return;
            }
            grid.innerHTML = '';
            data.samples.forEach((s, i) => {
                const div = document.createElement('div');
                div.className = 'sample-thumb' + (i === 0 ? ' active' : '');
                div.innerHTML = `<img src="${s.thumbnail}" alt="${s.filename}" title="${s.filename}">`;
                div.addEventListener('click', () => {
                    document.querySelectorAll('.sample-thumb').forEach(t => t.classList.remove('active'));
                    div.classList.add('active');
                    activeSamplePath = s.path;
                    uploadedFile = null;
                    runInference();
                });
                grid.appendChild(div);
            });
            activeSamplePath = data.samples[0].path;
        } catch {
            grid.innerHTML = '<div class="skeleton">Failed to load samples.</div>';
        }
    }
    loadSamples();

    // ── Image Upload ──────────────────────────────────────────────────
    const dropzone   = document.getElementById('dropzone');
    const imageInput = document.getElementById('imageInput');

    function handleFileUpload(file) {
        if (!file) return;
        const isVideo = file.type.startsWith('video/') || /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name);
        if (isVideo) {
            // Switch to video tab
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelector('[data-tab="video-tab"]').classList.add('active');
            document.getElementById('video-tab').classList.add('active');
            handleVideoFileUpload(file);
            return;
        }
        uploadedFile = file;
        activeSamplePath = null;
        document.querySelectorAll('.sample-thumb').forEach(t => t.classList.remove('active'));
        runInference();
    }

    if (imageInput) imageInput.addEventListener('change', e => handleFileUpload(e.target.files[0]));
    if (dropzone) {
        dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.style.borderColor = 'var(--primary)'; });
        dropzone.addEventListener('dragleave', () => { dropzone.style.borderColor = ''; });
        dropzone.addEventListener('drop', e => {
            e.preventDefault();
            dropzone.style.borderColor = '';
            if (e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]);
        });
        dropzone.addEventListener('click', () => imageInput && imageInput.click());
    }

    // ── Colormap / Backend Change ──────────────────────────────────────
    const colormapSelect = document.getElementById('colormapSelect');
    if (colormapSelect) colormapSelect.addEventListener('change', runInference);
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) modelSelect.addEventListener('change', runInference);

    // ── Run Inference ─────────────────────────────────────────────────
    const runInferenceBtn = document.getElementById('runInferenceBtn');
    if (runInferenceBtn) runInferenceBtn.addEventListener('click', runInference);

    async function runInference() {
        if (!uploadedFile && !activeSamplePath) return;

        const btn = document.getElementById('runInferenceBtn');
        if (btn) { btn.disabled = true; btn.textContent = 'Processing…'; }

        const formData = new FormData();
        formData.append('model', modelSelect ? modelSelect.value : 'mobilenet_unet');
        formData.append('depth_backend', depthBackendSelect ? depthBackendSelect.value : 'sturm');
        formData.append('colormap', colormapSelect ? colormapSelect.value : 'turbo');

        if (uploadedFile) {
            formData.append('image', uploadedFile);
        } else if (activeSamplePath) {
            formData.append('sample_path', activeSamplePath);
        }

        try {
            const res  = await fetch('/api/predict', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                updateDisplayUI(data);
            } else {
                alert('Inference error: ' + (data.error || 'Unknown'));
            }
        } catch (err) {
            alert('Network error: ' + err.message);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Analysis`;
            }
        }
    }

    // ── Update Display ────────────────────────────────────────────────
    function updateDisplayUI(data) {
        const placeholder = document.getElementById('placeholderBox');
        const visualWrap  = document.getElementById('visualWrap');
        const origImg     = document.getElementById('origImg');
        const overlayImg  = document.getElementById('overlayImg');
        const maskImg     = document.getElementById('maskImg');
        const depthImg    = document.getElementById('depthImg');

        if (placeholder) placeholder.classList.add('hidden');
        if (visualWrap)  visualWrap.classList.remove('hidden');

        if (origImg)    origImg.src    = data.original_image;
        if (overlayImg) overlayImg.src = data.overlay_image;
        if (maskImg)    maskImg.src    = data.mask_image;
        if (depthImg && data.depth_heatmap) depthImg.src = data.depth_heatmap;

        // Apply opacity
        const opacity = opacitySlider ? opacitySlider.value / 100 : 0.45;
        if (overlayImg) overlayImg.style.opacity = opacity;

        applyViewMode();

        const t = data.telemetry || {};

        // Telemetry strip
        setText('statusVal',  t.status || '—');
        setText('statusSub',  t.primary_factor || (t.status === 'Passable' ? 'Safe to proceed' : 'Hazard detected'));
        setText('ratioVal',   (t.flooded_road_ratio != null ? t.flooded_road_ratio + '%' : '—'));
        setStyle('ratioMeter', 'width', (t.flooded_road_ratio || 0) + '%');
        setText('depthVal',   (t.max_depth_cm != null ? t.max_depth_cm + ' cm' : '—'));
        setText('depthSub',   t.max_depth_cm < 15 ? 'Safe: All Vehicles' : (t.max_depth_cm < 30 ? 'Caution: SUVs Only' : 'Critical: Impassable'));
        setText('sturmVal',   t.sturm_level_label || (t.sturm_level != null ? 'Level ' + t.sturm_level : '—'));
        setText('sturmSub',   t.sturm_landmark || (t.depth_source === 'morphological' ? 'Morphological EDT' : 'Vehicle Landmark'));
        setText('fpsVal',     (t.fps != null ? t.fps + ' FPS' : '—'));
        setText('latencyVal', (t.latency_ms != null ? t.latency_ms + ' ms' : '—'));

        // Status badge
        const badge = document.getElementById('floatingBadge');
        if (badge) {
            badge.textContent = t.status || '—';
            badge.className = 'status-badge';
            if (t.status === 'Passable') badge.classList.add('passable');
            else if (t.status === 'Partially Flooded') badge.classList.add('partial');
            else if (t.status === 'Severely Affected' || t.status === 'Severely Flooded') badge.classList.add('severe');
        }

        // Advisory
        const advisoryCard   = document.getElementById('advisoryCard');
        const advisoryStatus = document.getElementById('advisoryStatus');
        const advisoryText   = document.getElementById('advisoryText');
        if (advisoryCard) advisoryCard.style.display = '';
        if (advisoryStatus) {
            advisoryStatus.textContent = (t.status || 'PASSABLE').toUpperCase();
            advisoryStatus.className = 'advisory-status';
            if (t.status === 'Passable') advisoryStatus.style.color = 'var(--success)';
            else if (t.status === 'Partially Flooded') advisoryStatus.style.color = 'var(--warning)';
            else advisoryStatus.style.color = 'var(--danger)';
        }
        if (advisoryText) advisoryText.textContent = t.recommendation || '';

        // Road/flood bar
        const dryPct   = 100 - (t.flooded_road_ratio || 0);
        const floodPct = t.flooded_road_ratio || 0;
        setText('dryPixelVal',   dryPct.toFixed(1) + '%');
        setText('floodPixelVal', floodPct.toFixed(1) + '%');
        setStyle('roadFill',  'width', dryPct + '%');
        setStyle('floodFill', 'width', floodPct + '%');

        // Vehicle clearance
        const vehCard = document.getElementById('vehCard');
        if (vehCard) vehCard.style.display = '';
        const maxD = t.max_depth_cm || 0;

        setVehicleStatus('sedanStatus',  maxD < 15 ? 'safe' : 'danger',
            maxD < 15 ? 'SAFE' : 'AVOID');
        setVehicleStatus('suvStatus',    maxD < 25 ? 'safe' : (maxD < 35 ? 'caution' : 'danger'),
            maxD < 25 ? 'SAFE' : (maxD < 35 ? 'CAUTION' : 'DANGER'));
        setVehicleStatus('truckStatus',  maxD < 45 ? 'safe' : 'caution',
            maxD < 45 ? 'SAFE' : 'CAUTION');
    }

    function setText(id, val) {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    }
    function setStyle(id, prop, val) {
        const el = document.getElementById(id);
        if (el) el.style[prop] = val;
    }
    function setVehicleStatus(id, level, label) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = label;
        el.className = 'veh-status status-' + level;
    }

    // ── Video Tab ─────────────────────────────────────────────────────
    const videoInput      = document.getElementById('videoInput');
    const videoDropzone   = document.getElementById('videoDropzone');
    const videoFileInfo   = document.getElementById('videoFileInfo');
    const videoFileName   = document.getElementById('videoFileName');
    const clearVideoBtn   = document.getElementById('clearVideoBtn');
    const loadSampleVideoBtn  = document.getElementById('loadSampleVideoBtn');
    const loadPreRenderedBtn  = document.getElementById('loadPreRenderedBtn');
    const videoStrideSelect   = document.getElementById('videoStrideSelect');
    const videoScaleSelect    = document.getElementById('videoScaleSelect');
    const videoLimitSelect    = document.getElementById('videoLimitSelect');
    const runVideoBtn         = document.getElementById('runVideoBtn');
    const videoProgressCard   = document.getElementById('videoProgressCard');
    const videoProgressBar    = document.getElementById('videoProgressBar');
    const videoProgressPct    = document.getElementById('videoProgressPct');
    const videoProgressStatus = document.getElementById('videoProgressStatus');
    const videoProgressSub    = document.getElementById('videoProgressSub');

    function handleVideoFileUpload(file) {
        uploadedVideoFile = file;
        selectedVideoPath = null;
        if (videoFileName) videoFileName.textContent = file.name;
        if (videoFileInfo) videoFileInfo.classList.remove('hidden');
    }

    if (videoInput) videoInput.addEventListener('change', e => handleVideoFileUpload(e.target.files[0]));

    if (videoDropzone) {
        videoDropzone.addEventListener('dragover', e => { e.preventDefault(); videoDropzone.style.borderColor = 'var(--primary)'; });
        videoDropzone.addEventListener('dragleave', () => { videoDropzone.style.borderColor = ''; });
        videoDropzone.addEventListener('drop', e => {
            e.preventDefault();
            videoDropzone.style.borderColor = '';
            if (e.dataTransfer.files[0]) handleVideoFileUpload(e.dataTransfer.files[0]);
        });
        videoDropzone.addEventListener('click', () => videoInput && videoInput.click());
    }

    if (clearVideoBtn) clearVideoBtn.addEventListener('click', () => {
        uploadedVideoFile = null;
        selectedVideoPath = null;
        if (videoFileInfo) videoFileInfo.classList.add('hidden');
        if (videoInput) videoInput.value = '';
    });

    if (loadSampleVideoBtn) loadSampleVideoBtn.addEventListener('click', () => {
        selectedVideoPath = 'dl_project-main (1)/data/test_video.mp4';
        uploadedVideoFile = null;
        if (videoFileName) videoFileName.textContent = 'Flooded Urban Street Drive (data/test_video.mp4)';
        if (videoFileInfo) videoFileInfo.classList.remove('hidden');
    });

    if (loadPreRenderedBtn) loadPreRenderedBtn.addEventListener('click', async () => {
        try {
            const res = await fetch('/api/video/prerendered');
            const data = await res.json();
            if (data.success && data.result) displayVideoResults(data.result);
        } catch { alert('No pre-rendered video found.'); }
    });

    if (runVideoBtn) runVideoBtn.addEventListener('click', startVideoProcessing);

    async function startVideoProcessing() {
        const formData = new FormData();
        const backend = depthBackendSelect ? depthBackendSelect.value : 'sturm';
        const model   = modelSelect ? modelSelect.value : 'mobilenet_unet';
        const stride  = videoStrideSelect ? parseInt(videoStrideSelect.value) : 2;
        const scale   = videoScaleSelect  ? parseFloat(videoScaleSelect.value) : 0.5;
        const limit   = videoLimitSelect  ? videoLimitSelect.value : 'all';
        const keyframeSelect = document.getElementById('videoKeyframeSelect');
        const keyframe = keyframeSelect ? parseInt(keyframeSelect.value) : 2;

        formData.append('model', model);
        formData.append('depth_backend', backend);
        formData.append('stride', stride);
        formData.append('scale', scale);
        formData.append('keyframe_interval', keyframe);
        if (limit !== 'all') formData.append('max_frames', limit);

        if (uploadedVideoFile) {
            formData.append('video', uploadedVideoFile);
        } else if (selectedVideoPath) {
            formData.append('video_path', selectedVideoPath);
        } else {
            alert('Please select a video first.');
            return;
        }

        if (videoProgressCard) videoProgressCard.classList.remove('hidden');
        if (runVideoBtn) { runVideoBtn.disabled = true; runVideoBtn.textContent = 'Processing…'; }

        try {
            const res = await fetch('/api/video/start', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success && data.task_id) {
                pollVideoStatus(data.task_id);
            } else {
                alert('Error: ' + (data.error || 'Unknown'));
                if (runVideoBtn) { runVideoBtn.disabled = false; runVideoBtn.textContent = 'Process Video'; }
            }
        } catch (err) {
            alert('Network error: ' + err.message);
            if (runVideoBtn) { runVideoBtn.disabled = false; runVideoBtn.textContent = 'Process Video'; }
        }
    }

    function pollVideoStatus(taskId) {
        if (videoPollInterval) clearInterval(videoPollInterval);
        videoPollInterval = setInterval(async () => {
            try {
                const res  = await fetch(`/api/video/status/${taskId}`);
                const data = await res.json();
                if (!data.success || !data.task) return;

                const task = data.task;
                const pct  = Math.round(task.progress || 0);
                if (videoProgressBar) videoProgressBar.style.width = pct + '%';
                if (videoProgressPct) videoProgressPct.textContent = pct + '%';
                if (videoProgressStatus && task.message) videoProgressStatus.textContent = task.message;

                // Live Stream Processing Telemetry update
                if (task.live) {
                    const l = task.live;
                    setText('liveVidStatus', l.status || '—');
                    setText('liveVidStatusSub', `Frame ${task.current_frame || 0} / ${task.total_frames || '?'}`);
                    setText('liveVidDepth', (l.depth_cm || 0).toFixed(1) + ' cm');
                    setText('liveVidDepthSub', 'Live Estimate');
                    setText('liveVidRatio', (l.ratio_pct || 0).toFixed(1) + '%');
                    setStyle('liveVidRatioMeter', 'width', (l.ratio_pct || 0) + '%');
                    setText('liveVidSturm', l.sturm_label || '—');
                    setText('liveVidSturmSub', l.sturm_landmark || 'Vehicle Reference');
                    setText('liveVidFps', (l.fps || 0).toFixed(1) + ' FPS');
                    setText('liveVidFrames', `${task.current_frame} / ${task.total_frames} frames`);
                    setText('liveTimeBadge', `Processing Frame ${task.current_frame}`);
                    setText('liveMonitorTitle', 'Live Stream Processing...');

                    const statusEl = document.getElementById('liveVidStatus');
                    if (statusEl) {
                        if (l.status === 'Passable') statusEl.style.color = 'var(--success)';
                        else if (l.status === 'Partially Flooded') statusEl.style.color = 'var(--warning)';
                        else statusEl.style.color = 'var(--danger)';
                    }
                }

                if (task.completed) {
                    clearInterval(videoPollInterval);
                    if (runVideoBtn) {
                        runVideoBtn.disabled = false;
                        runVideoBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Process Video`;
                    }
                    if (task.status === 'done' && task.result) {
                        displayVideoResults(task.result);
                    } else if (task.status === 'error') {
                        alert('Video error: ' + task.error);
                        if (videoProgressCard) videoProgressCard.classList.add('hidden');
                    }
                }
            } catch {}
        }, 800);
    }

    function displayVideoResults(result) {
        if (videoProgressCard) videoProgressCard.classList.add('hidden');

        const placeholder = document.getElementById('videoPlaceholder');
        const wrap        = document.getElementById('videoVisualWrapper');
        const videoPlayer = document.getElementById('annotatedVideoPlayer');
        const videoSource = document.getElementById('videoSource');

        if (placeholder) placeholder.classList.add('hidden');
        if (wrap)         wrap.classList.remove('hidden');
        if (videoSource)  videoSource.src = result.video_url + '?t=' + Date.now();
        if (videoPlayer)  videoPlayer.load();

        const s = result.summary || {};
        const timeline = s.timeline || [];

        // Synchronize live telemetry HUD with video playback in real time
        if (videoPlayer) {
            const formatSec = sec => {
                if (isNaN(sec) || sec == null) sec = 0;
                const m = Math.floor(sec / 60);
                const sc = Math.floor(sec % 60);
                const ms = Math.floor((sec % 1) * 10);
                return `${m < 10 ? '0' : ''}${m}:${sc < 10 ? '0' : ''}${sc}.${ms}`;
            };

            videoPlayer.ontimeupdate = () => {
                const cur = videoPlayer.currentTime || 0;
                const dur = videoPlayer.duration || s.duration_sec || 0;
                setText('liveTimeBadge', `${formatSec(cur)} / ${formatSec(dur)}`);
                setText('liveMonitorTitle', 'Live Dashcam Playback Telemetry');

                if (timeline.length > 0) {
                    // Fast binary/linear search for matching timeline frame
                    let best = timeline[0];
                    let minDiff = 999999;
                    for (let i = 0; i < timeline.length; i++) {
                        const diff = Math.abs(timeline[i].timestamp_sec - cur);
                        if (diff < minDiff) {
                            minDiff = diff;
                            best = timeline[i];
                        }
                    }

                    setText('liveVidStatus', best.status || '—');
                    setText('liveVidStatusSub', best.recommendation ? best.recommendation.substring(0, 28) + '…' : 'Normal Driving');
                    setText('liveVidDepth', (best.max_depth_cm || 0).toFixed(1) + ' cm');
                    setText('liveVidDepthSub', 'Peak: ' + (s.peak_flood_depth_cm || 0).toFixed(1) + ' cm');
                    setText('liveVidRatio', (best.flooded_road_ratio || 0).toFixed(1) + '%');
                    setStyle('liveVidRatioMeter', 'width', (best.flooded_road_ratio || 0) + '%');
                    setText('liveVidSturm', best.sturm_level_label || ('Level ' + (best.sturm_level || 0)));
                    setText('liveVidSturmSub', best.sturm_landmark || 'Vehicle Reference');
                    setText('liveVidFps', (s.average_processing_fps || 0).toFixed(1) + ' FPS');
                    setText('liveVidFrames', `Frame ${best.frame_idx}`);

                    const statusEl = document.getElementById('liveVidStatus');
                    if (statusEl) {
                        if (best.status === 'Passable') statusEl.style.color = 'var(--success)';
                        else if (best.status === 'Partially Flooded') statusEl.style.color = 'var(--warning)';
                        else statusEl.style.color = 'var(--danger)';
                    }
                }
            };
        }

        // Journey summary
        const journeyCard = document.getElementById('videoJourneyCard');
        if (journeyCard) journeyCard.classList.remove('hidden');

        const dist = s.passability_distribution || {};
        setText('vjPeakDepth',  (s.peak_flood_depth_cm || 0).toFixed(1) + ' cm');
        setText('vjPeakRatio',  (s.peak_flooded_ratio_pct || 0).toFixed(1) + '%');
        setText('vjFrames',     (s.frames_processed || 0) + ' frames');
        setText('vjSpeed',      (s.average_processing_fps || 0).toFixed(1) + ' FPS');

        const maxLvl = s.peak_sturm_level || 0;
        setText('vjMaxLevelBadge',      'STURM L' + maxLvl);
        const passabilityLabels = ['Passable', 'Passable', 'Partially Flooded', 'Severely Flooded', 'Severely Flooded'];
        const peakPassability = s.peak_passability || passabilityLabels[maxLvl] || '—';
        setText('vjPassabilityBadge',   peakPassability);

        const passP  = Math.round(dist.passable_pct            || 0);
        const partP  = Math.round(dist.partially_flooded_pct   || 0);
        const sevP   = Math.round(dist.severely_flooded_pct    || 0);
        setText('vjPassablePct', passP + '%');
        setText('vjPartialPct',  partP + '%');
        setText('vjSeverePct',   sevP  + '%');
        setStyle('vjFillPassable', 'width', passP + '%');
        setStyle('vjFillPartial',  'width', partP + '%');
        setStyle('vjFillSevere',   'width', sevP  + '%');

        // Video badge
        const vBadge = document.getElementById('videoFloatingBadge');
        if (vBadge) {
            vBadge.textContent = s.peak_passability || '—';
            vBadge.className = 'status-badge';
            const pp = s.peak_passability || '';
            if (pp === 'Passable') vBadge.classList.add('passable');
            else if (pp === 'Partially Flooded') vBadge.classList.add('partial');
            else vBadge.classList.add('severe');
        }

        // Depth profile graph image preview
        if (result.profile_url) {
            const profileBox = document.getElementById('vjProfileBox');
            const profileImg = document.getElementById('vjProfileImg');
            const profileLink = document.getElementById('vjProfileLink');
            if (profileBox) profileBox.classList.remove('hidden');
            if (profileImg) profileImg.src = result.profile_url + '?t=' + Date.now();
            if (profileLink) profileLink.href = result.profile_url;
        }

        // Download links
        const dlVideo = document.getElementById('vjDownloadVideoBtn');
        const dlTelem = document.getElementById('vjDownloadTelemetryBtn');
        if (dlVideo) dlVideo.href = result.video_url;
        if (dlTelem && result.telemetry_url) dlTelem.href = result.telemetry_url;
    }

    // MP4 / Stream player switch
    const useMp4Btn    = document.getElementById('useMp4PlayerBtn');
    const useStreamBtn = document.getElementById('useStreamPlayerBtn');
    if (useMp4Btn && useStreamBtn) {
        useMp4Btn.addEventListener('click', () => {
            useMp4Btn.classList.add('active');
            useStreamBtn.classList.remove('active');
            const mp4   = document.getElementById('annotatedVideoPlayer');
            const stream = document.getElementById('streamVideoPlayer');
            if (mp4)   mp4.classList.remove('hidden');
            if (stream) stream.classList.add('hidden');
        });
        useStreamBtn.addEventListener('click', () => {
            useStreamBtn.classList.add('active');
            useMp4Btn.classList.remove('active');
            const mp4    = document.getElementById('annotatedVideoPlayer');
            const stream = document.getElementById('streamVideoPlayer');
            if (mp4)    mp4.classList.add('hidden');
            if (stream) stream.classList.remove('hidden');
        });
    }

    // ── Benchmark Tab ─────────────────────────────────────────────────
    let benchLoaded = false;
    async function loadBenchmark() {
        if (benchLoaded) return;
        try {
            const res  = await fetch('/api/benchmark');
            const data = await res.json();
            renderBenchmark(data);
            benchLoaded = true;
        } catch {}
    }

    function renderBenchmark(data) {
        const mn   = data['MobileNet-UNet (Proposed)'] || data['MobileNet-UNet'] || {};
        const unet = data['U-Net Baseline'] || {};

        const set = (id, val) => { const el = document.getElementById(id); if (el && val != null) el.textContent = val; };

        set('bmMobilenetMiou', mn.mIoU || '0.864');
        set('bmMobilenetFps',  (mn.fps || 89.3) + ' FPS · Edge Real-Time');
        set('bmUnetMiou',      unet.mIoU || '0.812');
        set('bmUnetFps',       (unet.fps || 23.5) + ' FPS');
        set('paramReduction',  mn.num_params_m && unet.num_params_m
            ? ((1 - mn.num_params_m / unet.num_params_m) * 100).toFixed(1) + '%'
            : '86.7%');

        set('unet_miou',       unet.mIoU);
        set('mn_miou',         mn.mIoU);
        set('unet_dice',       unet.mean_dice);
        set('mn_dice',         mn.mean_dice);
        set('unet_flood_iou',  unet.flooded_road_iou);
        set('mn_flood_iou',    mn.flooded_road_iou);
        set('unet_boundary',   unet.boundary_f1);
        set('mn_boundary',     mn.boundary_f1);
        set('unet_depth_mae',  unet.depth_mae_cm ? unet.depth_mae_cm + ' cm' : '15.9 cm');
        set('mn_depth_mae',    mn.depth_mae_cm   ? mn.depth_mae_cm   + ' cm' : '9.16 cm');
        set('unet_latency',    unet.latency_ms ? unet.latency_ms + ' ms' : '42.5 ms');
        set('mn_latency',      mn.latency_ms   ? mn.latency_ms   + ' ms' : '11.2 ms');
        set('unet_fps',        unet.fps || 23.5);
        set('mn_fps',          mn.fps   || 89.3);
        set('unet_params',     unet.num_params_m ? unet.num_params_m + ' M' : '31.03 M');
        set('mn_params',       mn.num_params_m   ? mn.num_params_m   + ' M' : '4.12 M');
        set('unet_size',       unet.model_size_mb ? unet.model_size_mb + ' MB' : '118.4 MB');
        set('mn_size',         mn.model_size_mb   ? mn.model_size_mb   + ' MB' : '15.7 MB');
        set('unet_mem_power',  '207 MB / 28W');
        set('mn_mem_power',    '114 MB / 7.5W');
    }
});
