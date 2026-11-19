import os
from collections import deque
from functools import wraps
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from mysql.connector import pooling

load_dotenv()

app = Flask(__name__)

# 1. Persistent MySQL Connection Pool (reuses open connections instead of reconnecting per request)
db_pool = pooling.MySQLConnectionPool(
    pool_name="pathwise_pool",
    pool_size=5,
    pool_reset_session=True,
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME", "pathwise"),
)

# 2. Lightweight In-Memory Cache for static curriculum metadata & prerequisite edges
VLE_CACHE = {}
PREREQ_GRAPH = {}
MODULE_SITES = {}


def load_static_caches():
    """Loads vle_materials and module_prerequisites into RAM once at startup (~5ms)."""
    conn = db_pool.get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT id_site, code_module, code_presentation, activity_type, week_from FROM vle_materials")
    for row in cursor.fetchall():
        VLE_CACHE[row["id_site"]] = row
        MODULE_SITES.setdefault(row["code_module"], []).append(row)

    # Sort each module's resources once by week_from so recommendation filtering is instant
    for mod in MODULE_SITES:
        MODULE_SITES[mod].sort(key=lambda x: (x["week_from"], x["id_site"]))

    cursor.execute("SELECT source_site_id, next_site_id, relation_reason FROM module_prerequisites")
    for edge in cursor.fetchall():
        PREREQ_GRAPH.setdefault(edge["source_site_id"], []).append(edge)

    cursor.close()
    conn.close()


def fetch_recommendations_fast(cursor, id_student, top_k=3):
    """Combines indexed student log lookup with O(1) in-memory graph & metadata lookups."""
    # Single indexed query for recent 20 interactions
    cursor.execute(
        """
        SELECT id_site, date_offset, decay_weight
        FROM student_vle_logs
        WHERE id_student = %s
        ORDER BY date_offset DESC, log_id DESC
        LIMIT 20
        """,
        (id_student,),
    )
    recent_rows = cursor.fetchall()
    if not recent_rows:
        return {"active_item": None, "shifted_focus": None, "recommendations": []}

    session_queue = deque(recent_rows, maxlen=10)
    latest_site = session_queue[0]["id_site"]
    active_meta = VLE_CACHE.get(latest_site, {})
    visited_sites = {r["id_site"] for r in recent_rows}

    # Compute current shifted activity_type preference in Python from recent decayed weights
    type_scores = {}
    for r in recent_rows:
        meta = VLE_CACHE.get(r["id_site"])
        if meta:
            atype = meta["activity_type"]
            type_scores[atype] = type_scores.get(atype, 0.0) + r["decay_weight"]

    preferred_type = max(type_scores, key=type_scores.get) if type_scores else active_meta.get("activity_type", "oucontent")

    recommendations = []
    seen = set()

    # 1. Immediate next-step edges from in-memory prerequisite graph
    for edge in PREREQ_GRAPH.get(latest_site, []):
        nxt_id = edge["next_site_id"]
        if nxt_id not in visited_sites and nxt_id not in seen and nxt_id in VLE_CACHE:
            seen.add(nxt_id)
            meta = VLE_CACHE[nxt_id]
            recommendations.append({
                "id_site": nxt_id,
                "activity_type": meta["activity_type"],
                "week_from": meta["week_from"],
                "reason_tag": "Next-Step Prerequisite",
                "explanation": edge["relation_reason"],
            })

    # 2. Preference-drift matches from pre-sorted module cache
    mod = active_meta.get("code_module")
    curr_week = active_meta.get("week_from", 0)
    candidates = [
        m for m in MODULE_SITES.get(mod, [])
        if m["week_from"] >= curr_week and m["id_site"] not in visited_sites and m["id_site"] not in seen
    ]
    # Prioritize shifted preference type first, then chronological week
    candidates.sort(key=lambda m: (m["activity_type"] != preferred_type, m["week_from"]))

    for meta in candidates[: max(0, top_k - len(recommendations))]:
        seen.add(meta["id_site"])
        recommendations.append({
            "id_site": meta["id_site"],
            "activity_type": meta["activity_type"],
            "week_from": meta["week_from"],
            "reason_tag": "Preference Shift Match",
            "explanation": f"Matches your recent shift toward '{preferred_type}' in Week {meta['week_from']}",
        })

    return {
        "active_item": {**active_meta, "date_offset": session_queue[0]["date_offset"]},
        "shifted_focus": preferred_type,
        "recommendations": recommendations[:top_k],
    }


def track_interaction(func):
    """Logs click and decays only the student's recent active rows (fast indexed update)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        data = request.get_json(silent=True) or {}
        id_student = data.get("id_student")
        id_site = data.get("id_site")

        if id_student and id_site:
            conn = db_pool.get_connection()
            cursor = conn.cursor()
            # Only decay the 30 most recent interactions instead of scanning full history
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
    return render_template("dashboard.html")


@app.route("/api/recommendations/<int:id_student>", methods=["GET"])
def api_recommendations(id_student):
    conn = db_pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    result = fetch_recommendations_fast(cursor, id_student)
    cursor.close()
    conn.close()
    return jsonify(result)


@app.route("/api/track", methods=["POST"])
@track_interaction
def api_track():
    data = request.get_json()
    id_student = int(data["id_student"])
    conn = db_pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    refreshed = fetch_recommendations_fast(cursor, id_student)
    cursor.close()
    conn.close()
    return jsonify({"status": "logged", "updated": refreshed})


@app.route("/api/analytics/<int:id_student>", methods=["GET"])
def api_analytics(id_student):
    conn = db_pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    # Query only student_vle_logs (no SQL JOIN needed since VLE_CACHE is in RAM)
    cursor.execute(
        """
        SELECT FLOOR(date_offset / 7) AS study_week,
               id_site,
               SUM(sum_click) AS raw_clicks,
               SUM(decay_weight) AS decayed_score
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

    # Map id_site -> activity_type in Python memory
    aggregated = {}
    for r in rows:
        atype = VLE_CACHE.get(r["id_site"], {}).get("activity_type", "unknown")
        key = (int(r["study_week"]), atype)
        if key not in aggregated:
            aggregated[key] = {"study_week": key[0], "activity_type": atype, "raw_clicks": 0, "decayed_score": 0.0}
        aggregated[key]["raw_clicks"] += int(r["raw_clicks"])
        aggregated[key]["decayed_score"] = round(aggregated[key]["decayed_score"] + float(r["decayed_score"]), 2)

    timeline = sorted(aggregated.values(), key=lambda x: (x["study_week"], x["activity_type"]))
    return jsonify({"id_student": id_student, "timeline": timeline})


# Warm up caches before serving requests
load_static_caches()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)