"""Import public/local CSV or JSON exports into the MedEvidence Pro KB.

The importer is schema-driven and does not bundle third-party datasets.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from rag_project.knowledge.medical_kb import connect

KIND_COLUMNS = {
    "drugs": ["drug_name","standard_dose","dose_range_min","dose_range_max","frequency","route","indication","contraindications","side_effects","source","source_date"],
    "interactions": ["drug_a","drug_b","interaction_type","severity","mechanism","management","source"],
    "guidelines": ["condition","recommendation","strength","evidence_level","citation","updated_date"],
    "contraindications": ["drug_name","absolute_list","relative_list","conditional_json","source"],
    "disease_graph": ["source_node","relation","target_node","source"],
}

def load_rows(path: Path):
    if path.suffix.lower() == ".csv":
        with path.open("r",encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
    value=json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value,list) else value.get("rows",[])

def main():
    p=argparse.ArgumentParser(); p.add_argument("--db",default="data/medical_knowledge.sqlite3"); p.add_argument("--kind",choices=sorted(KIND_COLUMNS)); p.add_argument("--input",required=True)
    args=p.parse_args(); db=Path(args.db); db.parent.mkdir(parents=True,exist_ok=True)
    rows=load_rows(Path(args.input)); columns=KIND_COLUMNS[args.kind]
    placeholders=','.join('?' for _ in columns); sql=f'INSERT OR REPLACE INTO {args.kind} ({",".join(columns)}) VALUES ({placeholders})'
    with connect(db) as conn:
        conn.executemany(sql, [tuple(row.get(c,'') for c in columns) for row in rows]); conn.commit()
    print({"kind":args.kind,"rows":len(rows),"db":str(db)})

if __name__ == "__main__": main()
