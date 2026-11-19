import os
import numpy as np
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

# Resolve project root folder (one level up from database/seed_db.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

CLEAN_DIR = os.path.join(BASE_DIR, "cleaned")
TARGET_MODULES = ["AAA", "BBB"]  # Fast & rich subset (~650k rows); set to None for all 7 modules
LAMBDA_DECAY = 0.03

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME", "pathwise"),
    "allow_local_infile": True,
}


def seed_all():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Temporarily disable index/FK overhead for fast bulk insertion
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
    cursor.execute("SET UNIQUE_CHECKS = 0;")
    cursor.execute("SET autocommit = 0;")

    print("1/4 Loading courses & VLE materials...")
    courses = pd.read_csv(os.path.join(CLEAN_DIR, "courses_cleaned.csv"))
    vle = pd.read_csv(os.path.join(CLEAN_DIR, "vle_cleaned.csv"))
    students = pd.read_csv(os.path.join(CLEAN_DIR, "studentInfo_cleaned.csv"))

    if TARGET_MODULES:
        courses = courses[courses["code_module"].isin(TARGET_MODULES)]
        vle = vle[vle["code_module"].isin(TARGET_MODULES)]
        students = students[students["code_module"].isin(TARGET_MODULES)]

    if "is_evergreen" not in vle.columns:
        vle["is_evergreen"] = 0

    cursor.executemany(
        "INSERT IGNORE INTO courses VALUES (%s, %s, %s)",
        list(
            courses[["code_module", "code_presentation", "module_presentation_length"]].itertuples(
                index=False, name=None
            )
        ),
    )

    cursor.executemany(
        """
        INSERT IGNORE INTO vle_materials
        (id_site, code_module, code_presentation, activity_type, week_from, week_to, is_evergreen)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        list(
            vle[
                ["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to", "is_evergreen"]
            ].itertuples(index=False, name=None)
        ),
    )

    print("2/4 Loading student profiles...")
    cursor.executemany(
        """
        INSERT IGNORE INTO student_info
        (id_student, code_module, code_presentation, gender, region, highest_education, imd_band, age_band, final_result)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        list(
            students[
                [
                    "id_student",
                    "code_module",
                    "code_presentation",
                    "gender",
                    "region",
                    "highest_education",
                    "imd_band",
                    "age_band",
                    "final_result",
                ]
            ].itertuples(index=False, name=None)
        ),
    )

    print("3/4 Building prerequisite graph edges...")
    cursor.execute("TRUNCATE TABLE module_prerequisites;")
    edges = []
    for (mod, pres), group in vle.groupby(["code_module", "code_presentation"]):
        ordered = group.sort_values(["week_from", "id_site"]).reset_index(drop=True)
        for i in range(len(ordered) - 1):
            curr_row = ordered.iloc[i]
            next_row = ordered.iloc[i + 1]
            reason = (
                f"Next sequential {next_row['activity_type']} in {mod} "
                f"(Week {curr_row['week_from']} -> Week {next_row['week_from']})"
            )
            edges.append((int(curr_row["id_site"]), int(next_row["id_site"]), reason))

    cursor.executemany(
        "INSERT INTO module_prerequisites (source_site_id, next_site_id, relation_reason) VALUES (%s, %s, %s)",
        edges,
    )
    conn.commit()

    print("4/4 Loading studentVle interaction logs...")
    cursor.execute("TRUNCATE TABLE student_vle_logs;")
    valid_sites = set(vle["id_site"])
    total_inserted = 0

    for chunk in pd.read_csv(os.path.join(CLEAN_DIR, "studentVle_cleaned.csv"), chunksize=100_000):
        if TARGET_MODULES and "code_module" in chunk.columns:
            chunk = chunk[chunk["code_module"].isin(TARGET_MODULES)]
        chunk = chunk[chunk["id_site"].isin(valid_sites)].copy()
        if chunk.empty:
            continue

        if "decay_weight" not in chunk.columns:
            max_day = chunk.groupby("id_student")["date"].transform("max")
            delta_t = np.maximum(0, (max_day - chunk["date"]).to_numpy(dtype=np.float32))
            clicks = chunk["sum_click"].to_numpy(dtype=np.float32)
            chunk["decay_weight"] = np.round(clicks * np.exp(-LAMBDA_DECAY * delta_t), 4)

        rows = list(
            chunk[["id_student", "id_site", "date", "sum_click", "decay_weight"]].itertuples(index=False, name=None)
        )
        cursor.executemany(
            "INSERT INTO student_vle_logs (id_student, id_site, date_offset, sum_click, decay_weight) VALUES (%s, %s, %s, %s, %s)",
            rows,
        )
        conn.commit()
        total_inserted += len(rows)
        print(f"   -> Inserted {total_inserted:,} rows so far...")

    # Re-enable safety checks
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
    cursor.execute("SET UNIQUE_CHECKS = 1;")
    cursor.execute("SET autocommit = 1;")
    conn.commit()

    cursor.close()
    conn.close()
    print(f"Done! Pathwise database seeded with {total_inserted:,} interaction logs.")


if __name__ == "__main__":
    seed_all()