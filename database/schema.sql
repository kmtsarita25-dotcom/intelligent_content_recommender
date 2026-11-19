CREATE DATABASE IF NOT EXISTS chronofair_edu;
USE chronofair_edu;

CREATE TABLE IF NOT EXISTS courses (
    code_module VARCHAR(10),
    code_presentation VARCHAR(10),
    module_presentation_length INT,
    PRIMARY KEY (code_module, code_presentation)
);

CREATE TABLE IF NOT EXISTS vle_materials (
    id_site INT PRIMARY KEY,
    code_module VARCHAR(10),
    code_presentation VARCHAR(10),
    activity_type VARCHAR(50),
    week_from INT,
    week_to INT,
    is_evergreen TINYINT(1) DEFAULT 0,
    INDEX idx_module_week (code_module, code_presentation, week_from)
);

CREATE TABLE IF NOT EXISTS student_info (
    id_student INT,
    code_module VARCHAR(10),
    code_presentation VARCHAR(10),
    gender VARCHAR(10),
    region VARCHAR(50),
    highest_education VARCHAR(100),
    imd_band VARCHAR(20),
    age_band VARCHAR(20),
    final_result VARCHAR(30),
    PRIMARY KEY (id_student, code_module, code_presentation)
);

CREATE TABLE IF NOT EXISTS student_vle_logs (
    log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    id_student INT NOT NULL,
    id_site INT NOT NULL,
    date_offset INT NOT NULL,
    sum_click INT NOT NULL,
    decay_weight FLOAT NOT NULL,
    INDEX idx_student_timeline (id_student, date_offset),
    FOREIGN KEY (id_site) REFERENCES vle_materials(id_site)
);

CREATE TABLE IF NOT EXISTS module_prerequisites (
    edge_id INT AUTO_INCREMENT PRIMARY KEY,
    source_site_id INT NOT NULL,
    next_site_id INT NOT NULL,
    relation_reason VARCHAR(255) NOT NULL,
    INDEX idx_source_site (source_site_id),
    FOREIGN KEY (source_site_id) REFERENCES vle_materials(id_site),
    FOREIGN KEY (next_site_id) REFERENCES vle_materials(id_site)
);