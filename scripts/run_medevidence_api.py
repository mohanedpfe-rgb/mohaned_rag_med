"""Launch the optional FastAPI adapter around the canonical MedEvidence engine."""
from __future__ import annotations
from pathlib import Path
from rag_project.application import create_rag_system
from rag_project.intelligence.production_ops import OperationsStore
from rag_project.api.med_evidence_api import create_app

system = create_rag_system()
store = OperationsStore(Path(getattr(system.settings, "project_root", Path.cwd())) / "data" / "med_evidence_ops.sqlite3")
app = create_app(system.med_evidence_engine if hasattr(system, "med_evidence_engine") else system, store=store, project_root=getattr(system.settings, "project_root", Path.cwd()))
