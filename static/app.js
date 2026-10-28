/**
 * Frontend Controller for Continuous Smartphone Batch-Scanning & Subject Router.
 * Handles Continuous Camera Snapping ("Snap & Queue Next"), Background Upload Worker,
 * Subject Lock Overrides, Live Subject Tallies, and Real-Time Ledger Table.
 */

const state = {
  mode: "camera", // 'camera' | 'multi_upload' | 'sample'
  forcedCourseCode: "", // '' (Auto-Detect) | '24MCPC522' | '24MCPC517' | '24MCPC521'
  mediaStream: null,
  facingMode: "environment",
  snapCounter: 0,
  uploadQueue: [],
  isProcessingQueue: false,
  allRecords: [],
  subjectFilter: "ALL",
  samples: [],
  sampleIndex: 0,
};

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  await Promise.all([
    fetchSystemStatus(),
    fetchSubjectSummary(),
    fetchSamplesList(),
  ]);
});

function bindEvents() {
  // 1. Subject Lock Pills
  document.querySelectorAll(".lock-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".lock-pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      state.forcedCourseCode = pill.dataset.code || "";
      console.log("Subject Lock Override:", state.forcedCourseCode || "AUTO-DETECT");
    });
  });

  // 2. Mode Tabs
  document.querySelectorAll(".mode-tab").forEach((tab) => {
    tab.addEventListener("click", () => switchMode(tab.dataset.mode));
  });

  // 3. Camera Controls
  document.getElementById("btnStartCam")?.addEventListener("click", startCamera);
  document.getElementById("btnStopCam")?.addEventListener("click", stopCamera);
  document.getElementById("btnFlipCam")?.addEventListener("click", flipCamera);
  document.getElementById("btnSnapAndQueue")?.addEventListener("click", snapAndQueueNext);
  document.getElementById("btnDockShutter")?.addEventListener("click", snapAndQueueNext);

  // 4. Multi-Upload File Input
  document.getElementById("batchFileInput")?.addEventListener("change", handleMultiFileInput);

  // 5. Benchmark Sample Batch Controls
  document.getElementById("btnQueueSampleBatch5")?.addEventListener("click", () => queueSampleBatch(5));
  document.getElementById("btnQueueSampleBatch10")?.addEventListener("click", () => queueSampleBatch(10));
  document.getElementById("btnQueueSingleSample")?.addEventListener("click", queueSingleSampleFromSelect);

  // 6. Queue Clear Button
  document.getElementById("btnClearCompletedQueue")?.addEventListener("click", clearCompletedQueueCards);

  // 7. Ledger Filter Tabs & Search
  document.querySelectorAll(".filter-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.subjectFilter = btn.dataset.filter || "ALL";
      renderLedgerTable();
    });
  });

  document.getElementById("ledgerSearch")?.addEventListener("input", renderLedgerTable);

  // 8. Lightbox Modal Close
  document.getElementById("imageModal")?.addEventListener("click", (e) => {
    if (e.target.id === "imageModal" || e.target.id === "btnCloseModal") {
      closeImageModal();
    }
  });
}

// ---------------------------------------------------------------------------
// Mode Switching
// ---------------------------------------------------------------------------
function switchMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode-tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.mode === mode);
  });

  document.getElementById("paneCamera").style.display = mode === "camera" ? "block" : "none";
  document.getElementById("paneMultiUpload").style.display = mode === "multi_upload" ? "block" : "none";
  document.getElementById("paneSample").style.display = mode === "sample" ? "block" : "none";

  const dock = document.getElementById("mobileShutterDock");
  if (dock) dock.style.display = mode === "camera" ? "block" : "none";
}

// ---------------------------------------------------------------------------
// HTML5 WebRTC Live Camera Viewfinder
// ---------------------------------------------------------------------------
async function startCamera() {
  const video = document.getElementById("cameraVideo");
  const placeholder = document.getElementById("cameraPlaceholder");
  const overlay = document.getElementById("cameraGuideOverlay");
  const btnSnap = document.getElementById("btnSnapAndQueue");
  const btnDock = document.getElementById("btnDockShutter");
  const btnStop = document.getElementById("btnStopCam");

  try {
    if (state.mediaStream) {
      state.mediaStream.getTracks().forEach((t) => t.stop());
    }

    const constraints = {
      video: {
        facingMode: state.facingMode,
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    };

    state.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = state.mediaStream;
    await video.play();

    placeholder.style.display = "none";
    if (overlay) overlay.style.display = "flex";
    btnSnap.disabled = false;
    if (btnDock) btnDock.disabled = false;
    if (btnStop) btnStop.style.display = "inline-flex";
  } catch (err) {
    console.error("Camera access error:", err);
    alert("Camera permission denied or camera unavailable. Please use file upload.");
  }
}

function stopCamera() {
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((t) => t.stop());
    state.mediaStream = null;
  }
  const placeholder = document.getElementById("cameraPlaceholder");
  const overlay = document.getElementById("cameraGuideOverlay");
  const btnSnap = document.getElementById("btnSnapAndQueue");
  const btnDock = document.getElementById("btnDockShutter");
  const btnStop = document.getElementById("btnStopCam");

  if (placeholder) placeholder.style.display = "flex";
  if (overlay) overlay.style.display = "none";
  if (btnSnap) btnSnap.disabled = true;
  if (btnDock) btnDock.disabled = true;
  if (btnStop) btnStop.style.display = "none";
}

function flipCamera() {
  state.facingMode = state.facingMode === "environment" ? "user" : "environment";
  if (state.mediaStream) {
    startCamera();
  }
}

// ---------------------------------------------------------------------------
// Continuous Camera "Snap & Queue Next"
// ---------------------------------------------------------------------------
function snapAndQueueNext() {
  const video = document.getElementById("cameraVideo");
  if (!video || !video.videoWidth) return;

  state.snapCounter += 1;
  const snapIndex = state.snapCounter;

  // Update button indices
  const nextNum = snapIndex + 1;
  document.getElementById("nextBookletNum").textContent = nextNum;
  const dockNum = document.getElementById("dockBookletNum");
  if (dockNum) dockNum.textContent = nextNum;
  document.getElementById("sessionSnapCount").textContent = snapIndex;

  // Capture video frame to Canvas
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Subtle flash animation on viewfinder
  video.style.opacity = "0.3";
  setTimeout(() => { video.style.opacity = "1"; }, 100);

  canvas.toBlob((blob) => {
    if (!blob) return;

    const queueId = `queue_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
    const previewUrl = URL.createObjectURL(blob);

    const queueItem = {
      id: queueId,
      seq: snapIndex,
      type: "blob",
      blob: blob,
      previewUrl: previewUrl,
      label: `Booklet Scan #${snapIndex}`,
      forcedCourseCode: state.forcedCourseCode,
      status: "queued", // 'queued' | 'uploading' | 'extracting' | 'saved' | 'failed'
      result: null,
      error: null,
      timestamp: new Date().toLocaleTimeString(),
    };

    state.uploadQueue.push(queueItem);
    renderQueueCard(queueItem);
    updateQueueChips();

    // Trigger background queue worker
    processUploadQueue();
  }, "image/jpeg", 0.92);
}

// ---------------------------------------------------------------------------
// Multi-Photo File Picker Ingestion
// ---------------------------------------------------------------------------
function handleMultiFileInput(event) {
  const files = Array.from(event.target.files || []);
  if (files.length === 0) return;

  files.forEach((file) => {
    state.snapCounter += 1;
    const snapIndex = state.snapCounter;
    const queueId = `queue_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
    const previewUrl = URL.createObjectURL(file);

    const queueItem = {
      id: queueId,
      seq: snapIndex,
      type: "file",
      file: file,
      previewUrl: previewUrl,
      label: file.name,
      forcedCourseCode: state.forcedCourseCode,
      status: "queued",
      result: null,
      error: null,
      timestamp: new Date().toLocaleTimeString(),
    };

    state.uploadQueue.push(queueItem);
    renderQueueCard(queueItem);
  });

  document.getElementById("sessionSnapCount").textContent = state.snapCounter;
  document.getElementById("nextBookletNum").textContent = state.snapCounter + 1;
  const dockNum = document.getElementById("dockBookletNum");
  if (dockNum) dockNum.textContent = state.snapCounter + 1;

  updateQueueChips();
  processUploadQueue();

  // Reset file input so user can choose again
  event.target.value = "";
}

// ---------------------------------------------------------------------------
// Benchmark Sample Batch Ingestion
// ---------------------------------------------------------------------------
function queueSampleBatch(count) {
  if (state.samples.length === 0) {
    alert("Sample booklets list not loaded yet.");
    return;
  }

  for (let i = 0; i < count; i++) {
    const s = state.samples[state.sampleIndex % state.samples.length];
    state.sampleIndex += 1;
    state.snapCounter += 1;

    const queueId = `queue_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
    const queueItem = {
      id: queueId,
      seq: state.snapCounter,
      type: "sample",
      sampleId: s.booklet_id,
      previewUrl: s.p1_url,
      label: `${s.booklet_id.toUpperCase()} (Benchmark)`,
      forcedCourseCode: state.forcedCourseCode,
      status: "queued",
      result: null,
      error: null,
      timestamp: new Date().toLocaleTimeString(),
    };

    state.uploadQueue.push(queueItem);
    renderQueueCard(queueItem);
  }

  document.getElementById("sessionSnapCount").textContent = state.snapCounter;
  document.getElementById("nextBookletNum").textContent = state.snapCounter + 1;
  const dockNum = document.getElementById("dockBookletNum");
  if (dockNum) dockNum.textContent = state.snapCounter + 1;

  updateQueueChips();
  processUploadQueue();
}

function queueSingleSampleFromSelect() {
  const sel = document.getElementById("singleSampleSelect");
  const sId = sel.value;
  if (!sId) {
    alert("Please select a benchmark booklet.");
    return;
  }

  const opt = sel.selectedOptions[0];
  state.snapCounter += 1;

  const queueId = `queue_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
  const queueItem = {
    id: queueId,
    seq: state.snapCounter,
    type: "sample",
    sampleId: sId,
    previewUrl: opt.dataset.p1 || "",
    label: `${sId.toUpperCase()} (Benchmark)`,
    forcedCourseCode: state.forcedCourseCode,
    status: "queued",
    result: null,
    error: null,
    timestamp: new Date().toLocaleTimeString(),
  };

  state.uploadQueue.push(queueItem);
  renderQueueCard(queueItem);
  updateQueueChips();
  processUploadQueue();
}

// ---------------------------------------------------------------------------
// Asynchronous Background Queue Worker
// ---------------------------------------------------------------------------
async function processUploadQueue() {
  if (state.isProcessingQueue) return;
  state.isProcessingQueue = true;

  while (true) {
    // Find next pending queued item
    const item = state.uploadQueue.find((q) => q.status === "queued");
    if (!item) break;

    // 1. Transition: Uploading
    item.status = "uploading";
    updateQueueCard(item, "☁️ Uploading...", "Uploading scan to cloud storage...");
    updateQueueChips();

    try {
      const formData = new FormData();
      if (item.forcedCourseCode) {
        formData.append("forced_course_code", item.forcedCourseCode);
      }

      if (item.type === "blob") {
        formData.append("files", item.blob, `snap_${item.seq}.jpg`);
      } else if (item.type === "file") {
        formData.append("files", item.file);
      } else if (item.type === "sample") {
        formData.append("sample_ids", item.sampleId);
      }

      // 2. Transition: Extracting & Solving
      const extractingTimer = setTimeout(() => {
        if (item.status === "uploading") {
          item.status = "extracting";
          updateQueueCard(item, "🧠 Extracting & Solving...", "Running TrOCR & Z3 Presburger Solver...");
          updateQueueChips();
        }
      }, 700);

      const res = await fetch("/api/booklets/batch-upload", {
        method: "POST",
        body: formData,
      });

      clearTimeout(extractingTimer);
      const data = await res.json();

      if (!res.ok || !data.success || !data.results || data.results.length === 0) {
        throw new Error(data.detail || "Digitization failed");
      }

      const result = data.results[0];
      item.status = "saved";
      item.result = result;

      // 3. Transition: Saved
      const statusText = `✅ Saved to [${result.course_code}] (USN: ${result.usn} | Total: ${result.total_cie_50}/50)`;
      updateQueueCard(item, statusText, `Validated via Z3 MaxSMT • ${result.status}`);
      updateQueueChips();

      // Refresh Subject Cards & Ledger
      await fetchSubjectSummary();
    } catch (err) {
      console.error("Queue item error:", err);
      item.status = "failed";
      item.error = err.message;
      updateQueueCard(item, "❌ Processing Error", err.message);
      updateQueueChips();
    }
  }

  state.isProcessingQueue = false;
}

// ---------------------------------------------------------------------------
// Queue Cards UI Rendering
// ---------------------------------------------------------------------------
function renderQueueCard(item) {
  const notice = document.getElementById("queueEmptyNotice");
  if (notice) notice.style.display = "none";

  const container = document.getElementById("queueCardsContainer");
  if (!container) return;

  const card = document.createElement("div");
  card.className = `queue-card ${item.status}`;
  card.id = `card_${item.id}`;

  const thumbSrc = item.previewUrl || "/static/placeholder.jpg";

  card.innerHTML = `
    <img src="${thumbSrc}" class="queue-thumb" alt="Thumbnail" onclick="openImageModal('${thumbSrc}', '${escapeHtml(item.label)}')" />
    <div class="queue-info">
      <div class="queue-header-row">
        <span class="queue-label">${escapeHtml(item.label)}</span>
        <span class="queue-badge status-queued" id="badge_${item.id}">⏳ Queued</span>
      </div>
      <div class="queue-desc" id="desc_${item.id}">Waiting in smartphone stream queue...</div>
    </div>
  `;

  // Prepend so newly snapped card is at the top
  container.insertBefore(card, container.firstChild);
}

function updateQueueCard(item, badgeText, descText) {
  const card = document.getElementById(`card_${item.id}`);
  const badge = document.getElementById(`badge_${item.id}`);
  const desc = document.getElementById(`desc_${item.id}`);
  if (!card) return;

  card.className = `queue-card ${item.status}`;

  if (badge) {
    badge.className = `queue-badge status-${item.status}`;
    badge.textContent = badgeText;
  }

  if (desc) {
    desc.innerHTML = descText;
  }

  // If saved, update thumbnail with final cloud scan URL if available
  if (item.status === "saved" && item.result && item.result.image_url) {
    const thumbImg = card.querySelector(".queue-thumb");
    if (thumbImg) {
      thumbImg.src = item.result.image_url;
      thumbImg.onclick = () => openImageModal(item.result.image_url, `USN: ${item.result.usn} (${item.result.course_code})`);
    }
  }
}

function updateQueueChips() {
  const queuedCount = state.uploadQueue.filter((q) => q.status === "queued").length;
  const procCount = state.uploadQueue.filter((q) => q.status === "uploading" || q.status === "extracting").length;
  const doneCount = state.uploadQueue.filter((q) => q.status === "saved").length;

  document.getElementById("badgeQueued").textContent = `⏳ Queued: ${queuedCount}`;
  document.getElementById("badgeProcessing").textContent = `⚡ Processing: ${procCount}`;
  document.getElementById("badgeCompleted").textContent = `✅ Completed: ${doneCount}`;
}

function clearCompletedQueueCards() {
  const completed = state.uploadQueue.filter((q) => q.status === "saved" || q.status === "failed");
  completed.forEach((item) => {
    const el = document.getElementById(`card_${item.id}`);
    if (el) el.remove();
  });

  state.uploadQueue = state.uploadQueue.filter((q) => q.status === "queued" || q.status === "uploading" || q.status === "extracting");
  updateQueueChips();

  if (state.uploadQueue.length === 0) {
    const notice = document.getElementById("queueEmptyNotice");
    if (notice) notice.style.display = "block";
  }
}

// ---------------------------------------------------------------------------
// Subject Summary & Metrics Fetcher
// ---------------------------------------------------------------------------
async function fetchSubjectSummary() {
  try {
    const res = await fetch("/api/subjects/summary");
    const data = await res.json();
    if (!data.success) return;

    // Update Top Pill
    document.getElementById("countPill").textContent = `${data.total_booklets} Booklets Logged`;

    // Update 3 Subject Cards
    const subjects = data.subjects || {};
    ["24MCPC522", "24MCPC517", "24MCPC521"].forEach((code) => {
      const s = subjects[code] || {};
      const cntEl = document.getElementById(`cnt-${code}`);
      const avgEl = document.getElementById(`avg-${code}`);
      const corEl = document.getElementById(`cor-${code}`);

      if (cntEl) cntEl.textContent = s.student_count ?? 0;
      if (avgEl) avgEl.innerHTML = `${(s.average_cie ?? 0.0).toFixed(1)}<small style="font-size:0.75rem;color:#94a3b8;">/50</small>`;
      if (corEl) corEl.textContent = s.auto_corrected_count ?? 0;
    });

    // Flatten all records for the Ledger Table
    const combined = [];
    Object.keys(subjects).forEach((code) => {
      (subjects[code].records || []).forEach((r) => combined.push(r));
    });

    state.allRecords = combined;
    renderLedgerTable();
  } catch (err) {
    console.error("Failed to fetch subject summary:", err);
  }
}

// ---------------------------------------------------------------------------
// Ledger Table Rendering
// ---------------------------------------------------------------------------
function renderLedgerTable() {
  const tbody = document.getElementById("ledgerTableBody");
  if (!tbody) return;

  const query = (document.getElementById("ledgerSearch")?.value || "").toLowerCase().trim();
  const filter = state.subjectFilter;

  let records = state.allRecords || [];

  // Filter by subject tab
  if (filter !== "ALL") {
    records = records.filter((r) => (r.Course_Code || "").toUpperCase() === filter);
  }

  // Filter by search query
  if (query) {
    records = records.filter((r) => {
      return (
        (r.USN || "").toLowerCase().includes(query) ||
        (r.Course_Code || "").toLowerCase().includes(query) ||
        (r.Status || "").toLowerCase().includes(query) ||
        (r.Booklet_ID || "").toLowerCase().includes(query)
      );
    });
  }

  // Sort ascending by USN strictly
  records.sort((a, b) => (a.USN || "").localeCompare(b.USN || ""));

  if (records.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="14" style="text-align:center;padding:28px;color:#94a3b8;">
          No matching student records found.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = records
    .map((r) => {
      const thumbSrc = r.Image_Url || "";
      const thumbHtml = thumbSrc
        ? `<img src="${escapeHtml(thumbSrc)}" class="ledger-thumb" alt="Scan" onclick="openImageModal('${escapeHtml(thumbSrc)}', 'USN: ${escapeHtml(r.USN)} • ${escapeHtml(r.Course_Code)}')" />`
        : `<span style="color:#64748b;font-size:0.75rem;">N/A</span>`;

      const statusBadge = `<span class="badge-status status-${escapeHtml(r.Status || 'VALIDATED')}">${escapeHtml(r.Status || 'VALIDATED')}</span>`;

      return `
        <tr>
          <td>${thumbHtml}</td>
          <td><strong style="color:#60a5fa;font-family:'JetBrains Mono',monospace;">${escapeHtml(r.USN)}</strong></td>
          <td><span class="subject-code-tag" style="margin:0;font-size:0.74rem;">${escapeHtml(r.Course_Code)}</span></td>
          <td>${r.Test1_15}</td>
          <td>${r.Test2_15}</td>
          <td>${r.ABA_10}</td>
          <td>${r.Theory_Total_40}</td>
          <td>${r.Viva_5}</td>
          <td>${r.Lab_5}</td>
          <td>${r.Practical_Total_10}</td>
          <td><strong style="color:#34d399;font-size:1.05rem;font-family:'JetBrains Mono',monospace;">${r.Total_CIE_50}</strong><span style="color:#94a3b8;font-size:0.72rem;">/50</span></td>
          <td>${statusBadge}</td>
          <td style="max-width:240px;overflow:hidden;text-overflow:ellipsis;font-size:0.75rem;color:#94a3b8;" title="${escapeHtml(r.Solver_Overrides || '')}">${escapeHtml(r.Solver_Overrides || '✓ None (Exact)')}</td>
          <td style="font-size:0.75rem;color:#94a3b8;">${escapeHtml(r.Scanned_At || 'Seed Batch')}</td>
        </tr>
      `;
    })
    .join("");
}

// ---------------------------------------------------------------------------
// System Diagnostics & Samples Fetchers
// ---------------------------------------------------------------------------
async function fetchSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    document.getElementById("devicePill").textContent = `Device: ${data.device}`;
    document.getElementById("cloudPill").textContent = `Vault: ${(data.cloud?.provider || "LOCAL").toUpperCase()}`;
  } catch (err) {
    console.error("Error fetching system status:", err);
  }
}

async function fetchSamplesList() {
  try {
    const res = await fetch("/api/booklets/samples");
    const data = await res.json();
    state.samples = data.samples || [];

    const sel = document.getElementById("singleSampleSelect");
    if (!sel) return;

    sel.innerHTML = `<option value="">-- Choose verified benchmark booklet (1 to ${data.count}) --</option>`;
    state.samples.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.booklet_id;
      opt.dataset.p1 = s.p1_url;
      opt.textContent = `${s.booklet_id.toUpperCase()} (Page 1 Front Scan)`;
      sel.appendChild(opt);
    });
  } catch (err) {
    console.error("Failed to load sample booklets list:", err);
  }
}

// ---------------------------------------------------------------------------
// Lightbox Modal
// ---------------------------------------------------------------------------
function openImageModal(imgSrc, captionText) {
  const modal = document.getElementById("imageModal");
  const modalImg = document.getElementById("modalImg");
  const modalCaption = document.getElementById("modalCaption");

  if (!modal || !modalImg) return;
  modalImg.src = imgSrc;
  if (modalCaption) modalCaption.textContent = captionText || "Booklet Scan Preview";
  modal.classList.add("active");
}

function closeImageModal() {
  const modal = document.getElementById("imageModal");
  if (modal) modal.classList.remove("active");
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
