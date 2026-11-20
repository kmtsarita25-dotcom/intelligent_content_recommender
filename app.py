import math
import os
from functools import wraps

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from mysql.connector import pooling

from ml.recommender import generate_recommendations

load_dotenv()

app = Flask(__name__)

# 1. Persistent MySQL Connection Pool
db_pool = pooling.MySQLConnectionPool(
    pool_name="pathwise_pool",
    pool_size=5,
    pool_reset_session=True,
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME", "pathwise"),
)

# 2. In-Memory Caches for Fast ML Execution
VLE_CACHE = {}
VLE_DF = pd.DataFrame()
POPULARITY_DF = pd.DataFrame()
PREREQ_GRAPH = {}

SERIES_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]


def load_static_caches():
    """Load OULAD vle_materials, prerequisite edges, and baseline popularity into RAM at startup."""
    global VLE_DF, POPULARITY_DF

    conn = db_pool.get_connection()
    cursor = conn.cursor(dictionary=True)

    # Load all columns from vle_materials so ML feature preprocessing never misses expected fields
    cursor.execute("SELECT * FROM vle_materials")
    vle_rows = cursor.fetchall()
    for row in vle_rows:
        VLE_CACHE[row["id_site"]] = row

    VLE_DF = pd.DataFrame(vle_rows)
    if not VLE_DF.empty:
        # Ensure 'is_evergreen' exists for ml/text_processor.py
        if "is_evergreen" not in VLE_DF.columns:
            VLE_DF["is_evergreen"] = (
                VLE_DF["week_from"].isna() & VLE_DF["week_to"].isna()
            ).astype(int)
        else:
            VLE_DF["is_evergreen"] = pd.to_numeric(
                VLE_DF["is_evergreen"], errors="coerce"
            ).fillna(0).astype(int)

    # Pre-aggregate course-level popularity from MySQL
    cursor.execute(
        """
        SELECT l.id_site, v.code_module, v.code_presentation,
               MAX(l.date_offset) AS date, SUM(l.sum_click) AS sum_click
        FROM student_vle_logs l
        JOIN vle_materials v ON l.id_site = v.id_site
        GROUP BY l.id_site, v.code_module, v.code_presentation
        """
    )
    pop_rows = cursor.fetchall()
    POPULARITY_DF = pd.DataFrame(pop_rows)
    if not POPULARITY_DF.empty:
        POPULARITY_DF["id_student"] = -1  # Synthetic aggregate row for course popularity

    # Load sequential prerequisite graph for explainability tags
    try:
        cursor.execute(
            "SELECT source_site_id, next_site_id, relation_reason FROM module_prerequisites"
        )
        for edge in cursor.fetchall():
            PREREQ_GRAPH.setdefault(edge["source_site_id"], []).append(edge)
    except Exception:
        # Keep running even if module_prerequisites table is empty or omitted
        pass

    cursor.close()
    conn.close()


def get_default_student_id():
    conn = db_pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id_student FROM student_vle_logs LIMIT 1")
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row else 24734


def build_recommendations(cursor, id_student, active_site_override=None, decay_lambda=0.1, top_k=5):
    # 1. Fetch student profile
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

    # 2. Fetch student clickstream from MySQL
    cursor.execute(
        """
        SELECT l.id_student, l.id_site, l.date_offset AS date, l.sum_click,
               v.code_module, v.code_presentation
        FROM student_vle_logs l
        JOIN vle_materials v ON l.id_site = v.id_site
        WHERE l.id_student = %s
        ORDER BY l.date_offset DESC, l.log_id DESC
        """,
        (id_student,),
    )
    student_logs = cursor.fetchall()

    if not student_logs:
        return {"error": f"No interaction records found for Student ID {id_student}."}

    # Fallback profile if student exists in logs but not in student_info
    if not student_profile:
        student_profile = {
            "id_student": id_student,
            "code_module": student_logs[0]["code_module"],
            "code_presentation": student_logs[0]["code_presentation"],
            "highest_education": "N/A",
            "age_band": "N/A",
            "final_result": "In Progress",
        }

    latest_day = student_logs[0]["date"]
    if active_site_override and active_site_override in VLE_CACHE:
        latest_site = active_site_override
    else:
        latest_site = student_logs[0]["id_site"]

    active_meta = VLE_CACHE.get(latest_site)
    if not active_meta:
        return {"error": f"Site ID #{latest_site} does not exist in vle_materials."}

    # 3. Combine live student logs + course popularity for the ML engine
    student_df = pd.DataFrame(student_logs)
    if not POPULARITY_DF.empty:
        mod_pop_df = POPULARITY_DF[
            (POPULARITY_DF["code_module"] == active_meta["code_module"])
            & (POPULARITY_DF["code_presentation"] == active_meta["code_presentation"])
        ]
        combined_student_vle = pd.concat([student_df, mod_pop_df], ignore_index=True)
    else:
        combined_student_vle = student_df

    ml_datasets = {
        "vle": VLE_DF,
        "student_vle": combined_student_vle,
    }

    # 4. Run ML Recommender
    raw_ml_recs = generate_recommendations(
        student_id=id_student,
        datasets=ml_datasets,
        top_n=top_k,
        decay_rate=decay_lambda,
        interest_weight=0.65,
    )

    direct_next_map = {
        edge["next_site_id"]: edge["relation_reason"]
        for edge in PREREQ_GRAPH.get(latest_site, [])
    }

    # 5. Format ML results for frontend display
    enriched_recs = []
    for item in raw_ml_recs:
        sid = item["id_site"]
        meta = VLE_CACHE.get(sid, {})
        atype = item["activity_type"]
        wk = meta.get("week_from") if meta.get("week_from") is not None else "Any"
        mod = meta.get("code_module", active_meta["code_module"])
        score = round(float(item["recommendation_score"]), 3)

        if sid in direct_next_map:
            rule = "ML Hybrid + Sequential Graph"
            why = direct_next_map[sid]
        else:
            rule = "ML TF-IDF + Time-Decay + MMR"
            why = (
                f"Matched your recent {active_meta['activity_type']} interest with time-decay "
                f"(λ={decay_lambda}) and MMR diversity (Week {wk})."
            )

        enriched_recs.append({
            "id_site": sid,
            "code_module": mod,
            "activity_type": atype,
            "week_from": wk,
            "recommendation_score": score,
            "drift_score": round(min(0.99, max(0.50, score)), 2),
            "source_rule": rule,
            "why_this": why,
        })

    visited_sites = {r["id_site"] for r in student_logs}

    return {
        "student_profile": student_profile,
        "active_session": {
            "id_site": latest_site,
            "code_module": active_meta["code_module"],
            "code_presentation": active_meta["code_presentation"],
            "activity_type": active_meta["activity_type"],
            "week_from": active_meta.get("week_from"),
            "week_to": active_meta.get("week_to"),
            "latest_day": latest_day,
            "visited_count": len(visited_sites),
        },
        "recommendations": enriched_recs,
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
                SELECT %s, %s, COALESCE(MAX(date_offset), 0) + 1, 5, 1.0
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
    """Return active students available in student_vle_logs for quick selection/search."""
    limit = request.args.get("limit", default=100, type=int)
    conn = db_pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT DISTINCT l.id_student,
               COALESCE(s.code_module, 'OULAD') AS code_module,
               COALESCE(s.final_result, 'Active') AS final_result
        FROM student_vle_logs l
        LEFT JOIN student_info s ON l.id_student = s.id_student
        ORDER BY l.id_student ASC
        LIMIT %s
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(rows)


@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    id_student = request.args.get("id_student", type=int)
    active_site = request.args.get("active_id_site", type=int)
    decay_lambda = request.args.get("decay_weight", default=0.1, type=float)

    if not id_student:
        return jsonify({"error": "Please enter a valid numeric OULAD Student ID."}), 400

    conn = db_pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        payload = build_recommendations(cursor, id_student, active_site, decay_lambda)
    except ValueError as e:
        payload = {"error": str(e)}
    except Exception as e:
        payload = {"error": f"Recommendation engine error: {str(e)}"}
    finally:
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
    decay_lambda = request.args.get("decay_weight", default=0.1, type=float)

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
        age_in_weeks = max(0.0, (max_day - r["latest_week_day"]) / 7.0)
        decayed = float(r["raw_clicks"]) * math.exp(-decay_lambda * age_in_weeks)

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