let driftChartInstance = null;

const FRIENDLY_TYPES = {
    oucontent: "Core Course Reading",
    quiz: "Practice Quiz",
    forumng: "Peer Discussion Forum",
    resource: "Study Resource / PDF",
    url: "External Reference Link",
    subpage: "Topic Subpage",
    homepage: "Course Overview Hub",
    oucollaborate: "Live Group Session",
    glossary: "Key Terms Glossary",
};

function friendlyLabel(type) {
    if (!type) return "Study Resource";
    return FRIENDLY_TYPES[type] || type.toUpperCase();
}

function showBanner(message, isError = false) {
    const banner = document.getElementById("status-banner");
    banner.textContent = message;
    banner.style.borderColor = isError ? "#ef4444" : "#3b82f6";
    banner.classList.remove("hidden");
    setTimeout(() => banner.classList.add("hidden"), 3500);
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
            opt.textContent = `Student #${s.id_student} (${s.code_module})`;
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
        console.error("Sample load error:", err);
    }
}

async function trackClickAndOpen(studentId, clickedItem, triggerBtn = null) {
    if (triggerBtn) {
        triggerBtn.disabled = true;
        triggerBtn.textContent = "Loading...";
    }

    try {
        const res = await fetch("/api/track", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                id_student: Number(studentId),
                id_site: Number(clickedItem.id_site),
            }),
        });
        if (!res.ok) throw new Error("Could not save activity");
    } catch (err) {
        showBanner(err.message, true);
        if (triggerBtn) {
            triggerBtn.disabled = false;
            triggerBtn.innerHTML = "Continue Studying &rarr;";
        }
        return;
    }

    document.getElementById("active-site").value = clickedItem.id_site;
    showBanner(`Opened ${friendlyLabel(clickedItem.activity_type)} (Site #${clickedItem.id_site}) — updating next steps...`);
    await loadDashboard();
}

function renderActiveSession(session, profile) {
    const container = document.getElementById("active-session-card");
    const studentId = document.getElementById("student-id").value;
    const modText = profile
        ? `Module ${profile.code_module} (${profile.code_presentation})`
        : `Module ${session.code_module}`;

    document.getElementById("kpi-profile").textContent = modText;

    const weekDisplay = session.week_from !== null && session.week_from !== undefined
        ? `Week ${session.week_from}`
        : "General Schedule";

    container.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;">
            <div>
                <h3>${friendlyLabel(session.activity_type)} <span class="muted">(Site #${session.id_site})</span></h3>
                <p class="muted">${weekDisplay} &bull; ${session.visited_count} materials explored so far</p>
            </div>
            <button type="button" id="continue-active-btn" class="launch-btn">Continue Studying &rarr;</button>
        </div>
    `;

    const continueBtn = document.getElementById("continue-active-btn");
    continueBtn.addEventListener("click", () => {
        trackClickAndOpen(studentId, session, continueBtn);
    });
}

function renderRecommendations(studentId, recs) {
    const list = document.getElementById("recommendations-list");
    list.innerHTML = "";

    if (!recs || recs.length === 0) {
        list.innerHTML = `<p class="muted">You're all caught up on this module!</p>`;
        document.getElementById("kpi-score").textContent = "--";
        return;
    }

    let totalScore = 0;

    recs.forEach((item) => {
        const score = Number(item.drift_score || item.recommendation_score || 0);
        totalScore += score;

        const card = document.createElement("div");
        card.className = "rec-card";
        card.innerHTML = `
            <div>
                <div class="rec-title">${friendlyLabel(item.activity_type)} <span class="muted">&bull; Site #${item.id_site}</span></div>
                <div class="rec-meta">Module ${item.code_module} &bull; Week ${item.week_from ?? "Any"} &bull; ${Math.round(score * 100)}% Match</div>
                <span class="why-pill">${item.why_this}</span>
            </div>
            <button type="button" class="launch-btn">Start Studying &rarr;</button>
        `;

        const btn = card.querySelector(".launch-btn");
        btn.addEventListener("click", () => {
            trackClickAndOpen(studentId, item, btn);
        });

        list.appendChild(card);
    });

    document.getElementById("kpi-score").textContent = `${Math.round((totalScore / recs.length) * 100)}%`;
}

function renderDriftChart(analytics) {
    const ctx = document.getElementById("driftChart").getContext("2d");
    if (driftChartInstance) driftChartInstance.destroy();

    const datasets = (analytics.topic_series || []).map((series) => ({
        label: series.label.replace("OULAD: ", ""),
        data: series.data,
        borderColor: series.borderColor || "#3b82f6",
        backgroundColor: "transparent",
        tension: 0.35,
        borderWidth: 2,
        pointRadius: 3,
    }));

    driftChartInstance = new Chart(ctx, {
        type: "line",
        data: { labels: analytics.timeline || [], datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: "#e2e8f0" } } },
            scales: {
                x: { ticks: { color: "#94a3b8" }, grid: { color: "#334155" } },
                y: {
                    min: 0,
                    max: 1,
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

    const params = new URLSearchParams({ id_student: studentId, decay_weight: decayWeight });
    if (activeSiteId) params.set("active_id_site", activeSiteId);

    try {
        const [recRes, anaRes] = await Promise.all([
            fetch(`/api/recommend?${params.toString()}`),
            fetch(`/api/analytics/${studentId}?decay_weight=${decayWeight}`),
        ]);

        const recPayload = await recRes.json();
        const analyticsPayload = await anaRes.json();

        if (!recRes.ok) throw new Error(recPayload.error || "Unable to load student path");

        renderActiveSession(recPayload.active_session, recPayload.student_profile);
        renderRecommendations(studentId, recPayload.recommendations);

        document.getElementById("kpi-clicks").textContent = (analyticsPayload.total_clicks ?? 0).toLocaleString();
        document.getElementById("kpi-topic").textContent = friendlyLabel(analyticsPayload.dominant_activity || "oucontent");

        renderDriftChart(analyticsPayload);
    } catch (err) {
        showBanner(err.message, true);
    }
}

// UI Toggles
document.getElementById("toggle-advanced").addEventListener("click", () => {
    document.getElementById("advanced-drawer").classList.toggle("hidden");
});

document.getElementById("decay-lambda").addEventListener("input", (e) => {
    document.getElementById("lambda-display").textContent = Number(e.target.value).toFixed(2);
});

document.getElementById("toggle-chart").addEventListener("click", () => {
    const wrapper = document.getElementById("chart-wrapper");
    const arrow = document.getElementById("chart-arrow");
    wrapper.classList.toggle("hidden");
    arrow.innerHTML = wrapper.classList.contains("hidden") ? "&#9662;" : "&#9652;";
});

document.getElementById("student-controls").addEventListener("submit", async (e) => {
    e.preventDefault();
    document.getElementById("active-site").value = "";
    await loadDashboard();
});

document.addEventListener("DOMContentLoaded", async () => {
    await populateSampleStudents();
    await loadDashboard();
});