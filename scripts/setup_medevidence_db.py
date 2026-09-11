"""Initialize the local MedEvidence Pro structured knowledge store."""
from __future__ import annotations
import argparse
from pathlib import Path
from rag_project.knowledge.medical_kb import initialize

p = argparse.ArgumentParser()
p.add_argument("--db", default="data/medical_knowledge.sqlite3")
args = p.parse_args()
path = Path(args.db)
print({"db": str(path), "counts": initialize(path)})
