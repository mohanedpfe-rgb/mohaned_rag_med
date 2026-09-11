"""Validate the MedEvidence Pro structured KB before human testing."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rag_project.knowledge.medical_kb import counts, connect

MINIMUMS = {"drugs": 10_000, "interactions": 50_000, "guidelines": 100_000, "contraindications": 5_000, "disease_graph": 5_000}
REQUIRED_COLUMNS = {
    "drugs": {"drug_name", "standard_dose", "frequency", "route", "indication", "contraindications", "side_effects", "source_date"},
    "interactions": {"drug_a", "drug_b", "severity", "mechanism", "management", "source"},
    "guidelines": {"condition", "recommendation", "strength", "evidence_level", "citation", "updated_date"},
    "contraindications": {"drug_name", "absolute_list", "relative_list", "conditional_json", "source"},
    "disease_graph": {"source_node", "relation", "target_node", "source"},
}

def validate(db_path: str | Path, strict: bool = True) -> dict:
    path = Path(db_path)
    if not path.exists():
        return {"ready": False, "error": "database_missing", "db": str(path)}
    failures = []
    with connect(path) as db:
        for table, expected in REQUIRED_COLUMNS.items():
            actual = {row[1] for row in db.execute(f'PRAGMA table_info("{table}")').fetchall()}
            missing = sorted(expected - actual)
            if missing:
                failures.append({"table": table, "missing_columns": missing})
        counts_value = counts(path)
        for table, minimum in MINIMUMS.items():
            if strict and counts_value.get(table, 0) < minimum:
                failures.append({"table": table, "count": counts_value.get(table, 0), "minimum": minimum})
    return {"ready": not failures, "db": str(path), "counts": counts_value, "failures": failures}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/medical_knowledge.sqlite3")
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()
    report = validate(args.db, strict=not args.schema_only)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
