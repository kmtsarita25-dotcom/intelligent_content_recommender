let driftChartInstance = null;

function showBanner(message, isError = false) {
    const banner = document.getElementById("status-banner");
    banner.textContent = message;
    banner.style.borderColor = isError ? "#ef4444" : "#3b82f6";
    banner.classList.remove("hidden");
    setTimeout(() => banner.classList.add("hidden"), 4000);
}

async function populateSampleStudents() {
    try {
        const res = await fetch("/api/sample-students");
        if (!res.ok) return;
        const students = await res.json();
        const select = document.getElementById("sample-student-select");
        students.forEach((s) => {
            const opt = document.createElement("option");
            opt.value = s.id_student;
            opt.textContent = `#${s.id_student} (${s.code_module} • ${s.final_result})`;
            select.appendChild(opt);
        });

        select.addEventListener("change", () => {
            if (select.value) {
                document.getElementById("student-id").value = select.value;
                document.getElementById("active-site").value = "";
                loadDashboard();
            }
        });
    } catch (err) {
        console.error("Failed to load sample student IDs:", err);
    }
}

async function fetchRecommendations(studentId, activeSiteId, decayWeight) {
    const params = new URLSearchParams({
        id_student: studentId,
        decay_weight: decayWeight,
    });
    if (activeSiteId) {
        params.set("active_id_site", activeSiteId);
    }

    const res = await fetch(`/api/recommend?${params.toString()}`);
    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
}

async function fetchAnalytics(studentId, decayWeight) {
    const res = await fetch(`/api/analytics/${studentId}?decay_weight=${decayWeight}`);
    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
}

async function trackClickAndOpen(studentId, clickedItem) {
    try {
        const res = await fetch("/api/track", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                id_student: Number(studentId),
                id_site: Number(clickedItem.id_site),
            }),
        });
        if (!res.ok) throw new Error("Failed to write click to MySQL");
    } catch (err) {
        showBanner(`Database error logging click: ${err.message}`, true);
        return;
    }

    document.getElementById("active-site").value = clickedItem.id_site;
    showBanner(`Logged real click on OULAD Site #${clickedItem.id_site} (${clickedItem.activity_type}) — recalculating drift...`);
    await loadDashboard();
}

function renderActiveSession(session, profile) {
    const container = document.getElementById("active-session-card");
    const modInfo = profile
        ? `Module ${profile.code_module} (${profile.code_presentation}) &bull; Result: ${profile.final_result}`
        : `Module ${session.code_module} (${session.code_presentation})`;

    document.getElementById("kpi-profile").innerHTML = modInfo;

    container.innerHTML = `
        <h3>OULAD Site #${session.id_site} &mdash; ${session.activity_type.toUpperCase()}</h3>
        <p class="rec-meta">Module: <strong>${session.code_module}</strong> &bull; Scheduled: <strong>Week ${session.week_from} to Week ${session.week_to}</strong></p>
        <p class="muted">Last recorded click on course day <strong>${session.latest_day}</strong> (${session.visited_count} distinct materials visited in recent window).</p>
    `;
}

function renderRecommendations(studentId, recs) {
    const list = document.getElementById("recommendations-list");
    list.innerHTML = "";

    if (!recs || recs.length === 0) {
        list.innerHTML = `<p class="muted">No remaining unvisited OULAD materials found after this week.</p>`;
        document.getElementById("kpi-score").textContent = "--";
        return;
    }

    let totalScore = 0;

    recs.forEach((item) => {
        const score = Number(item.drift_score || 0);
        totalScore += score;

        const card = document.createElement("div");
        card.className = "rec-card";
        card.innerHTML = `
            <div>
                <strong>OULAD Site #${item.id_site} (${item.activity_type.toUpperCase()})</strong>
                <div class="rec-meta">
                    Module: ${item.code_module} &bull; Week ${item.week_from} &bull; Engine: ${item.source_rule} &bull; Score: <strong>${score.toFixed(2)}</strong>
                </div>
                <span class="why-tag">Why this? ${item.why_this}</span>
            </div>
            <button type="button" class="launch-btn">Study Next &rarr;</button>
        `;

        card.querySelector(".launch-btn").addEventListener("click", () => {
            trackClickAndOpen(studentId, item);
        });

        list.appendChild(card);
    });

    document.getElementById("kpi-score").textContent = (totalScore / recs.length).toFixed(2);
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
        pointRadius: 3.5,
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
                    grid: { color: "#1e293b" },
                },
                y: {
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: "Normalized Time-Decayed Click Weight",
                        color: "#94a3b8",
                    },
                    ticks: { color: "#94a3b8" },
                    grid: { color: "#1e293b" },
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

    try {
        const [recPayload, analyticsPayload] = await Promise.all([
            fetchRecommendations(studentId, activeSiteId, decayWeight),
            fetchAnalytics(studentId, decayWeight),
        ]);

        renderActiveSession(recPayload.active_session, recPayload.student_profile);
        renderRecommendations(studentId, recPayload.recommendations);

        document.getElementById("kpi-clicks").textContent = (analyticsPayload.total_clicks ?? 0).toLocaleString();
        document.getElementById("kpi-weeks").textContent = `Across ${analyticsPayload.active_weeks ?? 0} active study weeks`;
        document.getElementById("kpi-topic").textContent = (analyticsPayload.dominant_activity || "--").toUpperCase();

        renderDriftChart(analyticsPayload);
    } catch (err) {
        showBanner(err.message, true);
        document.getElementById("active-session-card").innerHTML = `<p class="muted">${err.message}</p>`;
        document.getElementById("recommendations-list").innerHTML = "";
        document.getElementById("kpi-clicks").textContent = "--";
        document.getElementById("kpi-topic").textContent = "--";
        document.getElementById("kpi-score").textContent = "--";
    }
}

document.getElementById("student-controls").addEventListener("submit", async (e) => {
    e.preventDefault();
    await loadDashboard();
});

document.addEventListener("DOMContentLoaded", async () => {
    await populateSampleStudents();
    await loadDashboard();
});