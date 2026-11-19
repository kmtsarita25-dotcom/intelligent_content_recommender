/**
 * Capstone 10 - Member 4 Frontend Controller
 * Connects Active Session, Up Next + "Why this?" cards, /api/track clickstream,
 * and /api/analytics/<id_student> Chart.js drift visualization.
 */

let driftChartInstance = null;

function showBanner(message, isError = false) {
    const banner = document.getElementById("status-banner");
    banner.textContent = message;
    banner.style.borderColor = isError ? "#ef4444" : "#3b82f6";
    banner.classList.remove("hidden");
    setTimeout(() => banner.classList.add("hidden"), 3500);
}

/**
 * Fetches recommendations from Flask backend.
 */
async function fetchRecommendations(studentId, activeSiteId, decayWeight) {
    const params = new URLSearchParams({
        id_student: studentId,
        active_id_site: activeSiteId,
        decay_weight: decayWeight,
    });

    try {
        const res = await fetch(`/api/recommend?${params.toString()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.warn("Using fallback recommendation payload:", err);
        return {
            active_session: {
                id_site: Number(activeSiteId),
                title: `Module #${activeSiteId}: Core Data Structures & Traversal`,
                topic: "Algorithms & Graphs",
                progress: "72% Completed",
            },
            recommendations: [{
                    id_site: Number(activeSiteId) + 1,
                    title: "BFS & DFS Prerequisite Graph Traversal",
                    topic: "Graph Theory",
                    drift_score: 0.94,
                    why_this: `Direct prerequisite after Module #${activeSiteId} + high recent focus`,
                },
                {
                    id_site: Number(activeSiteId) + 4,
                    title: "TF-IDF Vectorization & Cosine Similarity",
                    topic: "Applied NLP",
                    drift_score: 0.87,
                    why_this: "Matches your recent 48h shift toward Information Retrieval",
                },
                {
                    id_site: Number(activeSiteId) + 7,
                    title: "Exponential Time-Decay Modeling",
                    topic: "Machine Learning",
                    drift_score: 0.82,
                    why_this: "Trending in your recent session queue (decay weight active)",
                },
            ],
        };
    }
}

/**
 * Fetches student analytics & preference drift time series from Member 3's endpoint.
 */
async function fetchAnalytics(studentId) {
    try {
        const res = await fetch(`/api/analytics/${studentId}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.warn("Using fallback analytics payload:", err);
        return {
            id_student: Number(studentId),
            weekly_clicks: 38,
            dominant_topic: "Graph Theory & NLP",
            timeline: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            topic_series: [{
                    label: "Graph Algorithms (Current Shift)",
                    data: [0.32, 0.41, 0.55, 0.68, 0.79, 0.88, 0.94],
                    borderColor: "#3b82f6",
                },
                {
                    label: "Applied NLP / TF-IDF",
                    data: [0.25, 0.30, 0.48, 0.62, 0.74, 0.81, 0.87],
                    borderColor: "#10b981",
                },
                {
                    label: "Basic SQL Queries (Decaying)",
                    data: [0.85, 0.72, 0.60, 0.47, 0.35, 0.26, 0.19],
                    borderColor: "#f59e0b",
                },
            ],
        };
    }
}

/**
 * Sends live click tracking event to /api/track (Member 3) and updates active module.
 */
async function trackClickAndOpen(studentId, clickedItem) {
    try {
        await fetch("/api/track", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                id_student: Number(studentId),
                id_site: Number(clickedItem.id_site),
                timestamp: new Date().toISOString(),
            }),
        });
    } catch (err) {
        console.warn("Offline click track simulated:", err);
    }

    document.getElementById("active-site").value = clickedItem.id_site;
    showBanner(`Tracked click on "${clickedItem.title}" (ID #${clickedItem.id_site}) — updating drift weights...`);
    await loadDashboard();
}

function renderActiveSession(session) {
    const container = document.getElementById("active-session-card");
    container.innerHTML = `
        <h3>${session.title || `Module #${session.id_site}`}</h3>
        <p class="rec-meta">Site ID: #${session.id_site} &bull; Topic: <strong>${session.topic || "General"}</strong></p>
        <p class="muted">${session.progress || "Active study session in progress. Completing this unlocks downstream prerequisite nodes."}</p>
    `;
}

function renderRecommendations(studentId, recs) {
    const list = document.getElementById("recommendations-list");
    list.innerHTML = "";

    if (!recs || recs.length === 0) {
        list.innerHTML = `<p class="muted">No sequential recommendations available for this module.</p>`;
        return;
    }

    let totalScore = 0;

    recs.forEach((item) => {
        const score = Number(item.drift_score ?? item.score ?? 0);
        totalScore += score;

        const card = document.createElement("div");
        card.className = "rec-card";
        card.innerHTML = `
            <div>
                <strong>${item.title || `Study Module #${item.id_site}`}</strong>
                <div class="rec-meta">
                    Site ID: #${item.id_site} &bull; Topic: ${item.topic || "Core Module"} &bull; Drift Score: <strong>${score.toFixed(2)}</strong>
                </div>
                <span class="why-tag">Why this? ${item.why_this || "Sequential prerequisite & recent interest match"}</span>
            </div>
            <button type="button" class="launch-btn">Study Next &rarr;</button>
        `;

        card.querySelector(".launch-btn").addEventListener("click", () => {
            trackClickAndOpen(studentId, item);
        });

        list.appendChild(card);
    });

    const avgScore = totalScore / recs.length;
    document.getElementById("kpi-score").textContent = avgScore.toFixed(2);
}

function renderDriftChart(analytics) {
    const ctx = document.getElementById("driftChart").getContext("2d");

    if (driftChartInstance) {
        driftChartInstance.destroy();
    }

    const datasets = (analytics.topic_series || []).map((series) => ({
        label: series.label,
        data: series.data,
        borderColor: series.borderColor || "#3b82f6",
        backgroundColor: "transparent",
        tension: 0.35,
        borderWidth: 2.5,
        pointRadius: 4,
    }));

    driftChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: analytics.timeline || [],
            datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: "#f8fafc" } },
            },
            scales: {
                x: {
                    ticks: { color: "#94a3b8" },
                    grid: { color: "#334155" },
                },
                y: {
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: "Time-Decayed Interest Weight",
                        color: "#94a3b8",
                    },
                    ticks: { color: "#94a3b8" },
                    grid: { color: "#334155" },
                },
            },
        },
    });
}

async function loadDashboard() {
    const studentId = document.getElementById("student-id").value;
    const activeSiteId = document.getElementById("active-site").value;
    const decayWeight = document.getElementById("decay-lambda").value;

    document.getElementById("kpi-student").textContent = `#${studentId}`;

    const [recPayload, analyticsPayload] = await Promise.all([
        fetchRecommendations(studentId, activeSiteId, decayWeight),
        fetchAnalytics(studentId),
    ]);

    renderActiveSession(recPayload.active_session || { id_site: activeSiteId });
    renderRecommendations(studentId, recPayload.recommendations || []);

    document.getElementById("kpi-clicks").textContent = analyticsPayload.weekly_clicks ?? "--";
    document.getElementById("kpi-topic").textContent = analyticsPayload.dominant_topic ?? "--";
    renderDriftChart(analyticsPayload);
}

document.getElementById("student-controls").addEventListener("submit", async (e) => {
    e.preventDefault();
    await loadDashboard();
});

document.addEventListener("DOMContentLoaded", loadDashboard);