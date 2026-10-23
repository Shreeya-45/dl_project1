document.addEventListener('DOMContentLoaded', () => {
    // State management
    let activeSamplePath = null;
    let uploadedFile = null;
    let currentPredictionData = null;
    let accuracyChartInstance = null;
    let speedChartInstance = null;

    // Element References
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPages = document.querySelectorAll('.tab-page');
    const modelSelect = document.getElementById('modelSelect');
    const imageInput = document.getElementById('imageInput');
    const dropzone = document.getElementById('dropzone');
    const sampleGrid = document.getElementById('sampleGrid');
    const opacitySlider = document.getElementById('opacitySlider');
    const opacityValue = document.getElementById('opacityValue');
    const runInferenceBtn = document.getElementById('runInferenceBtn');

    const viewerContainer = document.getElementById('viewerContainer');
    const placeholderBox = document.getElementById('placeholderBox');
    const visualWrapper = document.getElementById('visualWrapper');
    const origImg = document.getElementById('origImg');
    const overlayImg = document.getElementById('overlayImg');
    const maskImg = document.getElementById('maskImg');
    const floatingBadge = document.getElementById('floatingBadge');
    const badgeText = document.getElementById('badgeText');

    const statusVal = document.getElementById('statusVal');
    const ratioVal = document.getElementById('ratioVal');
    const latencyVal = document.getElementById('latencyVal');
    const fpsVal = document.getElementById('fpsVal');
    const advisoryText = document.getElementById('advisoryText');
    const dryPixelVal = document.getElementById('dryPixelVal');
    const floodPixelVal = document.getElementById('floodPixelVal');
    const roadFill = document.getElementById('roadFill');
    const floodFill = document.getElementById('floodFill');

    // 1. Tab Switcher
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            navButtons.forEach(b => b.classList.remove('active'));
            tabPages.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetTab = btn.getAttribute('data-tab');
            document.getElementById(targetTab).classList.add('active');

            if (targetTab === 'benchmark-tab') {
                fetchBenchmarks();
            }
        });
    });

    // 2. View Toggle Mode (HUD Overlay, Side-by-Side, Pure Mask)
    const toggleBtns = document.querySelectorAll('.toggle-btn');
    toggleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            toggleBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const viewMode = btn.getAttribute('data-view');
            if (viewMode === 'overlay') {
                overlayImg.classList.remove('hidden');
                maskImg.classList.add('hidden');
                overlayImg.style.opacity = (opacitySlider.value / 100).toString();
            } else if (viewMode === 'mask') {
                overlayImg.classList.add('hidden');
                maskImg.classList.remove('hidden');
            } else if (viewMode === 'split') {
                overlayImg.classList.remove('hidden');
                maskImg.classList.add('hidden');
                overlayImg.style.opacity = "1.0";
            }
        });
    });

    // 3. Opacity Slider Adjustment
    opacitySlider.addEventListener('input', (e) => {
        const val = e.target.value;
        opacityValue.textContent = `${val}%`;
        if (overlayImg) {
            overlayImg.style.opacity = (val / 100).toString();
        }
    });

    // 4. Fetch Sample Scenes from Server
    async function loadSamples() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            
            if (data.samples && data.samples.length > 0) {
                sampleGrid.innerHTML = '';
                data.samples.forEach((sample, idx) => {
                    const card = document.createElement('div');
                    card.className = `sample-card ${idx === 0 ? 'active' : ''}`;
                    card.innerHTML = `<img src="${sample.thumbnail}" alt="Sample ${idx + 1}">`;
                    card.addEventListener('click', () => {
                        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
                        card.classList.add('active');
                        activeSamplePath = sample.path;
                        uploadedFile = null;
                        runInference();
                    });
                    sampleGrid.appendChild(card);
                });
                activeSamplePath = data.samples[0].path;
                runInference();
            } else {
                sampleGrid.innerHTML = '<div class="sample-skeleton">No sample scenes found in data directory.</div>';
            }
        } catch (err) {
            console.error('Error loading sample scenes:', err);
        }
    }

    // 5. Drag & Drop / File Input Handlers
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--primary-color)';
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });

    function handleFileUpload(file) {
        uploadedFile = file;
        activeSamplePath = null;
        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
        runInference();
    }

    runInferenceBtn.addEventListener('click', runInference);
    modelSelect.addEventListener('change', runInference);

    // 6. Run Segmentation Inference API
    async function runInference() {
        const formData = new FormData();
        formData.append('model', modelSelect.value);

        if (uploadedFile) {
            formData.append('image', uploadedFile);
        } else if (activeSamplePath) {
            formData.append('sample_path', activeSamplePath);
        } else {
            return;
        }

        runInferenceBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Segmentation...';
        runInferenceBtn.disabled = true;

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            if (data.success) {
                currentPredictionData = data;
                updateDisplayUI(data);
            } else {
                alert(`Inference Error: ${data.error}`);
            }
        } catch (err) {
            console.error('Error running inference:', err);
        } finally {
            runInferenceBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run Segmentation & Passability Analysis';
            runInferenceBtn.disabled = false;
        }
    }

    // 7. Update UI with Prediction Results & Telemetry
    function updateDisplayUI(data) {
        placeholderBox.classList.add('hidden');
        visualWrapper.classList.remove('hidden');

        origImg.src = data.original_image;
        overlayImg.src = data.overlay_image;
        maskImg.src = data.mask_image;

        overlayImg.style.opacity = (opacitySlider.value / 100).toString();

        const t = data.telemetry;
        statusVal.textContent = t.status;
        statusVal.style.color = t.color_hex;

        badgeText.textContent = t.status.toUpperCase();
        floatingBadge.style.borderColor = t.color_hex;
        const badgeDot = floatingBadge.querySelector('.badge-dot');
        if (badgeDot) badgeDot.style.backgroundColor = t.color_hex;

        ratioVal.textContent = `${t.flooded_road_ratio}%`;
        latencyVal.textContent = `${t.latency_ms} ms`;
        fpsVal.textContent = `${t.fps} FPS`;
        advisoryText.textContent = t.recommendation;

        // Calculate dry road vs flood percentage for breakdown progress bar
        const total = t.total_road_pixels || 1;
        const dryPct = Math.round((t.dry_road_pixels / total) * 100);
        const floodPct = Math.round((t.flooded_pixels / total) * 100);

        dryPixelVal.textContent = `${dryPct}%`;
        floodPixelVal.textContent = `${floodPct}%`;

        roadFill.style.width = `${dryPct}%`;
        floodFill.style.width = `${floodPct}%`;
    }

    // 8. Fetch & Render Benchmarks & Chart.js Visualizations
    async function fetchBenchmarks() {
        try {
            const res = await fetch('/api/benchmark');
            const data = await res.json();
            renderBenchmarkUI(data);
        } catch (err) {
            console.error('Error fetching benchmarks:', err);
        }
    }

    function renderBenchmarkUI(data) {
        const unet = data['U-Net Baseline'] || {};
        const mn = data['MobileNet-UNet (Proposed)'] || data['MobileNet-UNet'] || {};

        document.getElementById('bmMobilenetMiou').textContent = (mn.mIoU || 0).toFixed(3);
        document.getElementById('bmMobilenetFps').textContent = (mn.fps || 0).toFixed(1);

        document.getElementById('bmUnetMiou').textContent = (unet.mIoU || 0).toFixed(3);
        document.getElementById('bmUnetFps').textContent = (unet.fps || 0).toFixed(1);

        // Update table cells
        document.getElementById('unet_miou').textContent = (unet.mIoU || 0).toFixed(3);
        document.getElementById('mn_miou').textContent = (mn.mIoU || 0).toFixed(3);

        document.getElementById('unet_dice').textContent = (unet.mean_dice || 0).toFixed(3);
        document.getElementById('mn_dice').textContent = (mn.mean_dice || 0).toFixed(3);

        document.getElementById('unet_flood_iou').textContent = (unet.flooded_road_iou || 0).toFixed(3);
        document.getElementById('mn_flood_iou').textContent = (mn.flooded_road_iou || 0).toFixed(3);

        document.getElementById('unet_boundary').textContent = (unet.boundary_f1 || 0).toFixed(3);
        document.getElementById('mn_boundary').textContent = (mn.boundary_f1 || 0).toFixed(3);

        document.getElementById('unet_params').textContent = `${(unet.num_params_m || 0).toFixed(2)} M`;
        document.getElementById('mn_params').textContent = `${(mn.num_params_m || 0).toFixed(2)} M`;

        document.getElementById('unet_size').textContent = `${(unet.model_size_mb || 0).toFixed(1)} MB`;
        document.getElementById('mn_size').textContent = `${(mn.model_size_mb || 0).toFixed(1)} MB`;

        document.getElementById('unet_latency').textContent = `${(unet.latency_ms || 0).toFixed(1)} ms`;
        document.getElementById('mn_latency').textContent = `${(mn.latency_ms || 0).toFixed(1)} ms`;

        document.getElementById('unet_fps').textContent = `${(unet.fps || 0).toFixed(1)} FPS`;
        document.getElementById('mn_fps').textContent = `${(mn.fps || 0).toFixed(1)} FPS`;

        // Render Chart.js
        renderCharts(unet, mn);
    }

    function renderCharts(unet, mn) {
        const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
        const speedCtx = document.getElementById('speedChart').getContext('2d');

        if (accuracyChartInstance) accuracyChartInstance.destroy();
        if (speedChartInstance) speedChartInstance.destroy();

        // 1. Accuracy Chart (mIoU, Dice, Flooded Road IoU, Boundary F1)
        accuracyChartInstance = new Chart(accuracyCtx, {
            type: 'bar',
            data: {
                labels: ['Mean IoU (mIoU)', 'Dice Score', 'Flooded Road IoU', 'Boundary F1'],
                datasets: [
                    {
                        label: 'U-Net Baseline',
                        data: [unet.mIoU, unet.mean_dice, unet.flooded_road_iou, unet.boundary_f1],
                        backgroundColor: 'rgba(148, 163, 184, 0.6)',
                        borderColor: '#94a3b8',
                        borderWidth: 1
                    },
                    {
                        label: 'MobileNet-UNet (Proposed)',
                        data: [mn.mIoU, mn.mean_dice, mn.flooded_road_iou, mn.boundary_f1],
                        backgroundColor: 'rgba(59, 130, 246, 0.8)',
                        borderColor: '#3b82f6',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 1.0, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                    x: { grid: { display: false } }
                },
                plugins: { legend: { labels: { color: '#f8fafc' } } }
            }
        });

        // 2. Speed Chart (FPS & Latency ms)
        speedChartInstance = new Chart(speedCtx, {
            type: 'bar',
            data: {
                labels: ['Frames Per Sec (FPS)', 'Latency (ms)'],
                datasets: [
                    {
                        label: 'U-Net Baseline',
                        data: [unet.fps, unet.latency_ms],
                        backgroundColor: 'rgba(239, 68, 68, 0.6)',
                        borderColor: '#ef4444',
                        borderWidth: 1
                    },
                    {
                        label: 'MobileNet-UNet (Proposed)',
                        data: [mn.fps, mn.latency_ms],
                        backgroundColor: 'rgba(16, 185, 129, 0.8)',
                        borderColor: '#10b981',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                    x: { grid: { display: false } }
                },
                plugins: { legend: { labels: { color: '#f8fafc' } } }
            }
        });
    }

    // Initial load
    loadSamples();
});
