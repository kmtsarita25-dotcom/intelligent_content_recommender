import os
import math
from functools import wraps
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from mysql.connector import pooling

load_dotenv()

app = Flask(__name__)

db_pool = pooling.MySQLConnectionPool(
    pool_name="pathwise_pool",
    pool_size=5,
    pool_reset_session=True,
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME", "pathwise"),
)

VLE_CACHE = {}
PREREQ_GRAPH = {}
MODULE_SITES = {}

SERIES_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]


def load_static_caches():
    """Loads OULAD vle_materials and module_prerequisites into RAM once at startup."""
    conn = db_pool.get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT id_site, code_module, code_presentation, activity_type, week_from, week_to FROM vle_materials"
    )
    for row in cursor.fetchall():
        VLE_CACHE[row["id_site"]] = row
        MODULE_SITES.setdefault(row["code_module"], []).append(row)

    for mod in MODULE_SITES:
        MODULE_SITES[mod].sort(key=lambda x: (x["week_from"], x["id_site"]))

    cursor.execute("SELECT source_site_id, next_site_id, relation_reason FROM module_prerequisites")
    for edge in cursor.fetchall():
        PREREQ_GRAPH.setdefault(edge["source_site_id"], []).append(edge)

    cursor.close()
    conn.close()


def get_default_student_id():
    """Fetches a real student ID from student_vle_logs for initial page load."""
    conn = db_pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id_student FROM student_vle_logs LIMIT 1")
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row else 11391


def build_recommendations(cursor, id_student, active_site_override=None, decay_lambda=0.03, top_k=3):
    # Fetch real student demographic/course info from student_info
    cursor.execute(
        """
        SELECT id_student, code_module, code_presentation, highest_education, age_band, final_result
        FROM student_info
        WHERE id_student = %s
        LIMIT 1
        """,
        (id_student,),
    )
    student_profile = cursor.fetchone()

    # Fetch recent chronological clickstream from student_vle_logs
    cursor.execute(
        """
        SELECT id_site, date_offset, sum_click
        FROM student_vle_logs
        WHERE id_student = %s
        ORDER BY date_offset DESC, log_id DESC
        LIMIT 50
        """,
        (id_student,),
    )
    recent_rows = cursor.fetchall()

    if not recent_rows:
        return {"error": f"No OULAD interaction records found for Student ID {id_student}."}

    latest_day = recent_rows[0]["date_offset"]
    if active_site_override and active_site_override in VLE_CACHE:
        latest_site = active_site_override
    else:
        latest_site = recent_rows[0]["id_site"]

    active_meta = VLE_CACHE.get(latest_site)
    if not active_meta:
        return {"error": f"Site ID #{latest_site} does not exist in vle_materials."}

    visited_sites = {r["id_site"] for r in recent_rows}

    # Compute live exponential time-decay weights per OULAD activity_type
    type_scores = {}
    for r in recent_rows:
        meta = VLE_CACHE.get(r["id_site"])
        if meta:
            delta_t = max(0, latest_day - r["date_offset"])
            w = float(r["sum_click"]) * math.exp(-decay_lambda * delta_t)
            atype = meta["activity_type"]
            type_scores[atype] = type_scores.get(atype, 0.0) + w

    max_type_score = max(type_scores.values()) if type_scores else 1.0
    preferred_type = max(type_scores, key=type_scores.get) if type_scores else active_meta["activity_type"]

    recommendations = []
    seen = set()

    # 1. Sequential Prerequisite Graph Edges (from module_prerequisites)
    for edge in PREREQ_GRAPH.get(latest_site, []):
        nxt_id = edge["next_site_id"]
        if nxt_id not in visited_sites and nxt_id not in seen and nxt_id in VLE_CACHE:
            seen.add(nxt_id)
            meta = VLE_CACHE[nxt_id]
            type_boost = (type_scores.get(meta["activity_type"], 0.0) / max_type_score) * 0.15
            drift_score = min(0.99, round(0.82 + type_boost, 2))
            recommendations.append({
                "id_site": nxt_id,
                "code_module": meta["code_module"],
                "activity_type": meta["activity_type"],
                "week_from": meta["week_from"],
                "drift_score": drift_score,
                "source_rule": "Sequential Prerequisite Graph",
                "why_this": edge["relation_reason"],
            })

    # 2. Time-Decayed Preference Shift Candidates (from same OULAD module)
    mod = active_meta["code_module"]
    curr_week = active_meta["week_from"]
    candidates = [
        m for m in MODULE_SITES.get(mod, [])
        if m["week_from"] >= curr_week and m["id_site"] not in visited_sites and m["id_site"] not in seen
    ]
    candidates.sort(key=lambda m: (m["activity_type"] != preferred_type, m["week_from"], m["id_site"]))

    for idx, meta in enumerate(candidates[: max(0, top_k - len(recommendations))]):
        seen.add(meta["id_site"])
        drift_score = round(max(0.60, 0.89 - (idx * 0.05)), 2)
        recommendations.append({
            "id_site": meta["id_site"],
            "code_module": meta["code_module"],
            "activity_type": meta["activity_type"],
            "week_from": meta["week_from"],
            "drift_score": drift_score,
            "source_rule": "Time-Decay Preference Shift",
            "why_this": (
                f"Matches your highest time-decayed OULAD activity '{preferred_type}' "
                f"(λ={decay_lambda}) scheduled for Week {meta['week_from']}"
            ),
        })

    return {
        "student_profile": student_profile,
        "active_session": {
            "id_site": latest_site,
            "code_module": active_meta["code_module"],
            "code_presentation": active_meta["code_presentation"],
            "activity_type": active_meta["activity_type"],
            "week_from": active_meta["week_from"],
            "week_to": active_meta["week_to"],
            "latest_day": latest_day,
            "visited_count": len(visited_sites),
        },
        "recommendations": recommendations[:top_k],
    }


def track_interaction(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        data = request.get_json(silent=True) or {}
        id_student = data.get("id_student")
        id_site = data.get("id_site")

        if id_student and id_site:
            conn = db_pool.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE student_vle_logs
                SET decay_weight = ROUND(decay_weight * 0.92, 4)
                WHERE id_student = %s
                ORDER BY date_offset DESC, log_id DESC
                LIMIT 30
                """,
                (id_student,),
            )
            cursor.execute(
                """
                INSERT INTO student_vle_logs (id_student, id_site, date_offset, sum_click, decay_weight)
                SELECT %s, %s, COALESCE(MAX(date_offset), 0) + 1, 1, 1.0
                FROM student_vle_logs WHERE id_student = %s
                """,
                (id_student, id_site, id_student),
            )
            conn.commit()
            cursor.close()
            conn.close()

        return func(*args, **kwargs)
    return wrapper


@app.route("/")
def index():
    default_id = get_default_student_id()
    return render_template("dashboard.html", default_student_id=default_id)


@app.route("/api/sample-students", methods=["GET"])
def api_sample_students():
    """Returns 6 real OULAD student IDs from pathwise for quick demo switching."""
    conn = db_pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT s.id_student, s.code_module, s.final_result
        FROM student_info s
        WHERE EXISTS (SELECT 1 FROM student_vle_logs l WHERE l.id_student = s.id_student)
        LIMIT 6
        """
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(rows)


@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    id_student = request.args.get("id_student", type=int)
    active_site = request.args.get("active_id_site", type=int)
    decay_lambda = request.args.get("decay_weight", default=0.03, type=float)

    if not id_student:
        return jsonify({"error": "Please enter a valid OULAD Student ID."}), 400

    conn = db_pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    payload = build_recommendations(cursor, id_student, active_site, decay_lambda)
    cursor.close()
    conn.close()

    if "error" in payload:
        return jsonify(payload), 404
    return jsonify(payload)


@app.route("/api/track", methods=["POST"])
@track_interaction
def api_track():
    return jsonify({"status": "logged"})


@app.route("/api/analytics/<int:id_student>", methods=["GET"])
def api_analytics(id_student):
    decay_lambda = request.args.get("decay_weight", default=0.03, type=float)

    conn = db_pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT FLOOR(date_offset / 7) AS study_week,
               id_site,
               SUM(sum_click) AS raw_clicks,
               MAX(date_offset) AS latest_week_day
        FROM student_vle_logs
        WHERE id_student = %s
        GROUP BY study_week, id_site
        ORDER BY study_week ASC
        """,
        (id_student,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        return jsonify({
            "id_student": id_student,
            "total_clicks": 0,
            "active_weeks": 0,
            "dominant_activity": "None",
            "timeline": [],
            "topic_series": [],
        })

    max_day = max(r["latest_week_day"] for r in rows)
    weeks_set = sorted({int(r["study_week"]) for r in rows})
    week_labels = [f"Wk {w}" for w in weeks_set]

    type_week_scores = {}
    type_totals = {}
    total_clicks = 0

    for r in rows:
        total_clicks += int(r["raw_clicks"])
        atype = VLE_CACHE.get(r["id_site"], {}).get("activity_type", "resource")
        wk = int(r["study_week"])
        delta_t = max(0, max_day - r["latest_week_day"])
        decayed = float(r["raw_clicks"]) * math.exp(-decay_lambda * delta_t)

        type_week_scores.setdefault(atype, {})
        type_week_scores[atype][wk] = type_week_scores[atype].get(wk, 0.0) + decayed
        type_totals[atype] = type_totals.get(atype, 0.0) + decayed

    top_types = sorted(type_totals, key=type_totals.get, reverse=True)[:4]
    dominant_activity = top_types[0] if top_types else "None"

    global_peak = max(
        (val for t in top_types for val in type_week_scores[t].values()),
        default=1.0,
    ) or 1.0

    topic_series = []
    for idx, atype in enumerate(top_types):
        series_data = [
            round(type_week_scores[atype].get(w, 0.0) / global_peak, 3)
            for w in weeks_set
        ]
        topic_series.append({
            "label": f"OULAD: {atype}",
            "data": series_data,
            "borderColor": SERIES_COLORS[idx % len(SERIES_COLORS)],
        })

    return jsonify({
        "id_student": id_student,
        "total_clicks": total_clicks,
        "active_weeks": len(weeks_set),
        "dominant_activity": dominant_activity,
        "timeline": week_labels,
        "topic_series": topic_series,
    })


load_static_caches()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)