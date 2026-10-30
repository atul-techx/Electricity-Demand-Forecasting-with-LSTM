/**
 * script.js
 * ---------
 * Dynamic interactive dashboard controller for EDF-LSTM.
 * Integrates with FastAPI endpoints /api/metrics and /api/forecast,
 * and powers the Chart.js visual display.
 */

// Global State
let chartInstance = null;
let currentForecastHours = 168;
let currentHistoryHours = 168;
let cachedForecastData = null;

// DOM Elements
const systemStatusPill = document.getElementById("system-status-pill");
const statusText = document.getElementById("status-text");
const btnRefresh = document.getElementById("btn-refresh");
const btnSimulate = document.getElementById("btn-simulate");
const horizonPills = document.querySelectorAll("#horizon-pills .pill-btn");
const hoursSlider = document.getElementById("hours-slider");
const sliderValueDisplay = document.getElementById("slider-value-display");
const historySelect = document.getElementById("history-select");
const chartLoader = document.getElementById("chart-loader");
const latencyVal = document.getElementById("latency-val");

// KPI Elements
const kpiModelStatus = document.getElementById("kpi-model-status");
const kpiLstmMae = document.getElementById("kpi-lstm-mae");
const kpiMaeImprovement = document.getElementById("kpi-mae-improvement");
const kpiLstmRmse = document.getElementById("kpi-lstm-rmse");
const kpiRmseImprovement = document.getElementById("kpi-rmse-improvement");
const kpiBaselineMae = document.getElementById("kpi-baseline-mae");
const kpiBaselineRmse = document.getElementById("kpi-baseline-rmse");
const kpiTotalPoints = document.getElementById("kpi-total-points");
const kpiSplitMeta = document.getElementById("kpi-split-meta");
const kpiForecastPeak = document.getElementById("kpi-forecast-peak");
const kpiForecastRange = document.getElementById("kpi-forecast-range");
const kpiForecastSpan = document.getElementById("kpi-forecast-span");

// Chart Footer Stats
const statBoundary = document.getElementById("stat-boundary");
const statStart = document.getElementById("stat-start");
const statEnd = document.getElementById("stat-end");
const statMean = document.getElementById("stat-mean");
const statPeak = document.getElementById("stat-peak");
const breakdownTbody = document.getElementById("breakdown-tbody");

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  initChart();
  loadAllData();
});

function initEventListeners() {
  // Preset horizon pills
  horizonPills.forEach((btn) => {
    btn.addEventListener("click", () => {
      horizonPills.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const hours = parseInt(btn.getAttribute("data-hours"), 10);
      currentForecastHours = hours;
      hoursSlider.value = hours;
      sliderValueDisplay.textContent = `${hours} Hours`;
      loadForecastData(currentForecastHours, currentHistoryHours);
    });
  });

  // Slider fine adjustment
  hoursSlider.addEventListener("input", (e) => {
    const hours = parseInt(e.target.value, 10);
    sliderValueDisplay.textContent = `${hours} Hours`;
    // Sync pill button active state
    horizonPills.forEach((p) => {
      const pHours = parseInt(p.getAttribute("data-hours"), 10);
      p.classList.toggle("active", pHours === hours);
    });
  });

  hoursSlider.addEventListener("change", (e) => {
    currentForecastHours = parseInt(e.target.value, 10);
    loadForecastData(currentForecastHours, currentHistoryHours);
  });

  // History select dropdown
  historySelect.addEventListener("change", (e) => {
    currentHistoryHours = parseInt(e.target.value, 10);
    loadForecastData(currentForecastHours, currentHistoryHours);
  });

  // Refresh & Simulate buttons
  btnRefresh.addEventListener("click", () => {
    loadAllData();
  });

  btnSimulate.addEventListener("click", () => {
    loadForecastData(currentForecastHours, currentHistoryHours, true);
  });

  // Series Toggle Chips
  const toggles = [
    { id: "toggle-actual", datasetIndex: 0 },
    { id: "toggle-lstm", datasetIndex: 1 },
    { id: "toggle-baseline", datasetIndex: 2 },
    { id: "toggle-future", datasetIndex: 3 },
  ];

  toggles.forEach(({ id, datasetIndex }) => {
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.addEventListener("click", () => {
      if (!chartInstance) return;
      const isVisible = chartInstance.isDatasetVisible(datasetIndex);
      chartInstance.setDatasetVisibility(datasetIndex, !isVisible);
      chartInstance.update();
      btn.classList.toggle("active", !isVisible);
    });
  });
}

/**
 * Initialize Chart.js with dark-themed styling and multi-dataset configuration
 */
function initChart() {
  const ctx = document.getElementById("demandChart").getContext("2d");

  // Create Gradients
  const gradActual = ctx.createLinearGradient(0, 0, 0, 400);
  gradActual.addColorStop(0, "rgba(56, 189, 248, 0.22)");
  gradActual.addColorStop(1, "rgba(56, 189, 248, 0.0)");

  const gradFuture = ctx.createLinearGradient(0, 0, 0, 400);
  gradFuture.addColorStop(0, "rgba(250, 204, 21, 0.25)");
  gradFuture.addColorStop(1, "rgba(250, 204, 21, 0.0)");

  const isMobile = window.innerWidth < 640;
  const isSmallMobile = window.innerWidth < 480;

  chartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Historical Actual Demand",
          data: [],
          borderColor: "#38bdf8",
          backgroundColor: gradActual,
          borderWidth: isMobile ? 1.8 : 2,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: "#38bdf8",
          pointHoverBorderColor: "#ffffff",
          pointHoverBorderWidth: 2,
          fill: true,
          tension: 0.25,
        },
        {
          label: "LSTM Test Prediction",
          data: [],
          borderColor: "#a855f7",
          backgroundColor: "transparent",
          borderWidth: isMobile ? 1.6 : 1.8,
          borderDash: [4, 4],
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: "#a855f7",
          fill: false,
          tension: 0.25,
        },
        {
          label: "Naive Baseline Benchmark",
          data: [],
          borderColor: "#fb923c",
          backgroundColor: "transparent",
          borderWidth: 1.2,
          borderDash: [2, 2],
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "#fb923c",
          fill: false,
          tension: 0.2,
        },
        {
          label: "Future Out-of-Sample Forecast",
          data: [],
          borderColor: "#facc15",
          backgroundColor: gradFuture,
          borderWidth: isMobile ? 2.2 : 2.5,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: "#facc15",
          pointHoverBorderColor: "#ffffff",
          pointHoverBorderWidth: 2,
          fill: true,
          tension: 0.28,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          display: false, // controlled via custom UI chips
        },
        tooltip: {
          backgroundColor: "rgba(15, 20, 32, 0.94)",
          titleColor: "#ffffff",
          titleFont: { family: "Outfit", size: isMobile ? 11 : 13, weight: "bold" },
          bodyColor: "#94a3b8",
          bodyFont: { family: "JetBrains Mono", size: isMobile ? 10 : 12 },
          borderColor: "rgba(255, 255, 255, 0.12)",
          borderWidth: 1,
          padding: isMobile ? 8 : 12,
          cornerRadius: 8,
          boxPadding: 4,
          callbacks: {
            label: function (context) {
              const val = context.parsed.y;
              if (val === null || val === undefined) return null;
              return ` ${context.dataset.label}: ${val.toLocaleString()} MW`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: {
            color: "rgba(255, 255, 255, 0.05)",
            drawBorder: false,
          },
          ticks: {
            color: "#64748b",
            font: { family: "JetBrains Mono", size: isMobile ? 9 : 10 },
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: isSmallMobile ? 4 : (isMobile ? 6 : 10),
            callback: function (val, index) {
              const raw = this.getLabelForValue(val);
              if (!raw || typeof raw !== "string") return raw;
              // Clean mobile format: "04-28 16h" instead of "2024-04-28 16:00"
              if (window.innerWidth < 640 && raw.length >= 16) {
                return raw.slice(5, 10) + " " + raw.slice(11, 13) + "h";
              }
              return raw;
            },
          },
        },
        y: {
          grid: {
            color: "rgba(255, 255, 255, 0.05)",
            drawBorder: false,
          },
          ticks: {
            color: "#64748b",
            font: { family: "JetBrains Mono", size: isMobile ? 9 : 11 },
            callback: function (value) {
              return (value / 1000).toFixed(0) + "k MW";
            },
          },
        },
      },
    },
  });

  // Responsive resize handler for Chart.js
  let resizeTimeout;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      if (!chartInstance) return;
      const w = window.innerWidth;
      const small = w < 480;
      const mobile = w < 640;
      chartInstance.options.scales.x.ticks.maxTicksLimit = small ? 4 : (mobile ? 6 : 10);
      chartInstance.options.scales.x.ticks.font.size = mobile ? 9 : 10;
      chartInstance.options.scales.y.ticks.font.size = mobile ? 9 : 11;
      chartInstance.options.plugins.tooltip.titleFont.size = mobile ? 11 : 13;
      chartInstance.options.plugins.tooltip.bodyFont.size = mobile ? 10 : 12;
      chartInstance.options.plugins.tooltip.padding = mobile ? 8 : 12;
      chartInstance.update("none");
    }, 150);
  });
}

/**
 * Fetch and display all metrics and default forecast with automatic retry
 */
async function loadAllData(retries = 15) {
  const success = await fetchMetrics();
  if (success) {
    await loadForecastData(currentForecastHours, currentHistoryHours);
  } else if (retries > 0) {
    statusText.textContent = "Pipeline Initializing (Background warm-up)...";
    setTimeout(() => loadAllData(retries - 1), 2000);
  }
}

/**
 * Load model metrics from /api/metrics
 */
async function fetchMetrics() {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Model Status
    statusText.textContent = "Pipeline Active";
    systemStatusPill.classList.remove("status-offline");
    kpiModelStatus.textContent = "Active";

    // LSTM Metrics
    kpiLstmMae.textContent = `${data.lstm.mae.toLocaleString(undefined, {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    })} MW`;
    kpiMaeImprovement.textContent = `▼ ${data.improvements.mae_percentage}% vs Baseline`;

    kpiLstmRmse.textContent = `${data.lstm.rmse.toLocaleString(undefined, {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    })} MW`;
    kpiRmseImprovement.textContent = `▼ ${data.improvements.rmse_percentage}% vs Baseline`;

    // Baseline Metrics
    kpiBaselineMae.textContent = `${data.baseline.mae.toLocaleString(undefined, {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    })} MW`;
    kpiBaselineRmse.textContent = `RMSE: ${data.baseline.rmse.toLocaleString(
      undefined,
      { minimumFractionDigits: 1, maximumFractionDigits: 1 }
    )} MW`;

    // Dataset records
    kpiTotalPoints.textContent = data.total_records.toLocaleString();
    kpiSplitMeta.textContent = `${data.train_records.toLocaleString()} Train / ${data.test_records.toLocaleString()} Test`;
    return true;
  } catch (err) {
    console.warn("Metrics still initializing:", err.message);
    statusText.textContent = "Model Initializing...";
    kpiModelStatus.textContent = "Standby";
    return false;
  }
}

/**
 * Fetch forecast series from /api/forecast
 */
async function loadForecastData(hours = 168, historyHours = 168, isSim = false) {
  chartLoader.classList.add("visible");
  const t0 = performance.now();

  try {
    const res = await fetch(`/api/forecast?hours=${hours}&history_hours=${historyHours}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    cachedForecastData = data;

    const latency = Math.round(performance.now() - t0);
    latencyVal.textContent = `~${latency}ms`;

    // Update Chart Data
    const combined = data.combined;
    chartInstance.data.labels = combined.labels;
    chartInstance.data.datasets[0].data = combined.actual;
    chartInstance.data.datasets[1].data = combined.lstm;
    chartInstance.data.datasets[2].data = combined.baseline;
    chartInstance.data.datasets[3].data = combined.future_forecast;
    chartInstance.update();

    // Update Meta Cards
    const meta = data.metadata;
    kpiForecastPeak.textContent = `${meta.forecast_max_mw.toLocaleString()} MW`;
    kpiForecastRange.textContent = `Min: ${meta.forecast_min_mw.toLocaleString()} MW`;
    kpiForecastSpan.textContent = `Next ${meta.requested_forecast_hours}h Horizon`;

    // Update Footer Stats
    statBoundary.textContent = meta.last_known_timestamp;
    statStart.textContent = meta.forecast_start;
    statEnd.textContent = meta.forecast_end;
    statMean.textContent = `${meta.forecast_avg_mw.toLocaleString()} MW`;
    statPeak.textContent = `${meta.forecast_max_mw.toLocaleString()} MW`;

    // Populate Daily Breakdown Table
    populateBreakdownTable(data.future.timestamps, data.future.forecast);
  } catch (err) {
    console.error("Failed to load forecast data:", err);
  } finally {
    setTimeout(() => {
      chartLoader.classList.remove("visible");
    }, 200);
  }
}

/**
 * Populate 24h daily buckets in the breakdown table
 */
function populateBreakdownTable(timestamps, values) {
  if (!timestamps || !timestamps.length) return;
  breakdownTbody.innerHTML = "";

  const daysCount = Math.ceil(timestamps.length / 24);

  for (let d = 0; d < daysCount; d++) {
    const startIdx = d * 24;
    const endIdx = Math.min((d + 1) * 24, timestamps.length);
    const dayVals = values.slice(startIdx, endIdx);
    const dayDates = timestamps.slice(startIdx, endIdx);

    const peak = Math.max(...dayVals);
    const valley = Math.min(...dayVals);
    const avg = dayVals.reduce((a, b) => a + b, 0) / dayVals.length;
    const swing = peak - valley;
    const dayDateStr = dayDates[0].split(" ")[0];

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>Day ${d + 1}</strong></td>
      <td>${dayDateStr}</td>
      <td style="color: var(--gold-accent); font-weight: 600;">${Math.round(peak).toLocaleString()} MW</td>
      <td style="color: var(--cyan-accent);">${Math.round(valley).toLocaleString()} MW</td>
      <td>${Math.round(avg).toLocaleString()} MW</td>
      <td><span class="badge-tag">± ${Math.round(swing / 2).toLocaleString()} MW</span></td>
    `;
    breakdownTbody.appendChild(tr);
  }
}
