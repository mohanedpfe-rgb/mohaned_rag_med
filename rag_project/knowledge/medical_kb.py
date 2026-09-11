"""Local structured knowledge store schema for MedEvidence Pro."""
from __future__ import annotations
import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS drugs (drug_name TEXT PRIMARY KEY, standard_dose TEXT, dose_range_min REAL, dose_range_max REAL, frequency TEXT, route TEXT, indication TEXT, contraindications TEXT, side_effects TEXT, source TEXT, source_date TEXT);
CREATE TABLE IF NOT EXISTS interactions (drug_a TEXT NOT NULL, drug_b TEXT NOT NULL, interaction_type TEXT, severity TEXT, mechanism TEXT, management TEXT, source TEXT, PRIMARY KEY(drug_a, drug_b));
CREATE TABLE IF NOT EXISTS guidelines (id INTEGER PRIMARY KEY AUTOINCREMENT, condition TEXT NOT NULL, recommendation TEXT NOT NULL, strength TEXT, evidence_level TEXT, citation TEXT, updated_date TEXT);
CREATE TABLE IF NOT EXISTS contraindications (drug_name TEXT PRIMARY KEY, absolute_list TEXT, relative_list TEXT, conditional_json TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS disease_graph (source_node TEXT NOT NULL, relation TEXT NOT NULL, target_node TEXT NOT NULL, source TEXT, PRIMARY KEY(source_node, relation, target_node));
CREATE TABLE IF NOT EXISTS knowledge_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_interactions_a ON interactions(drug_a);
CREATE INDEX IF NOT EXISTS idx_interactions_b ON interactions(drug_b);
CREATE INDEX IF NOT EXISTS idx_guidelines_condition ON guidelines(condition);
"""

def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path); path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path); db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA busy_timeout=10000"); db.executescript(SCHEMA_SQL); return db

def initialize(db_path: str | Path) -> dict[str, int]:
    db = connect(db_path)
    try:
        db.execute("INSERT OR REPLACE INTO knowledge_meta(key,value) VALUES('schema_version','1')")
        db.commit()
    finally:
        db.close()
    return counts(db_path)

def counts(db_path: str | Path) -> dict[str, int]:
    tables = ('drugs','interactions','guidelines','contraindications','disease_graph')
    db = connect(db_path)
    try:
        return {t: int(db.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]) for t in tables}
    finally:
        db.close()
