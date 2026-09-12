from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult

ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: Path, source: str, module_name: str, root: Path) -> Any:
    target = root / path.name
    target.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load mutant module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _run_mutant(name: str, path: Path, source: str, replacement: str, test_body: str, root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"rag_phase11_{name}_") as td:
        temp_root = Path(td)
        mutant_source = source.replace(replacement.split("=>", 1)[0], replacement.split("=>", 1)[1], 1)
        mutant = temp_root / path.name
        mutant.write_text(mutant_source, encoding="utf-8")
        test_file = temp_root / f"test_{name}.py"
        test_file.write_text(test_body.replace("__MUTANT_PATH__", str(mutant)), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(test_file)],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=90,
        )
        return {
            "name": name,
            "target": str(path.relative_to(root)),
            "returncode": proc.returncode,
            "killed": proc.returncode != 0,
            "stdout": proc.stdout[-600:],
            "stderr": proc.stderr[-600:],
        }


def _text_mutants(root: Path) -> list[dict[str, Any]]:
    path = root / "rag_project" / "utils" / "text_utils.py"
    source = path.read_text(encoding="utf-8")
    original = 'return re.sub(r"\\s+", " ", value or "").strip()'
    replacements = [
        ("return_raw", 'return value or ""'),
        ("no_collapse", 'return re.sub(r"\\s+", " ", value or "")'),
        ("collapse_to_tab", 'return re.sub(r"\\s+", "\\t", value or "").strip()'),
        ("collapse_only_left", 'return re.sub(r"\\s+", " ", value or "").lstrip()'),
        ("uppercase_content", 'return re.sub(r"\\s+", " ", (value or "").upper()).strip()'),
        ("collapse_to_newline", 'return re.sub(r"\\s+", "\\n", value or "").strip()'),
        ("drop_internal_spaces", 'return re.sub(r"\\s+", "", value or "").strip()'),
        ("right_trim_only", 'return re.sub(r"\\s+", " ", value or "").rstrip()'),
    ]
    test = (
        "from importlib.util import spec_from_file_location, module_from_spec\n"
        "spec=spec_from_file_location('mutant', '__MUTANT_PATH__')\n"
        "m=module_from_spec(spec); spec.loader.exec_module(m)\n"
        "def test_contract():\n"
        "    assert m.normalize_whitespace('  diabetes   mellitus  ') == 'diabetes mellitus'\n"
        "    assert m.normalize_whitespace('\\u00a0HbA1c\\tthreshold\\u00a0') == 'HbA1c threshold'\n"
    )
    return [
        _run_mutant(name, path, source, f"{original}=>{new}", test, root)
        for name, new in replacements
        if original in source
    ]


def _context_mutants(root: Path) -> list[dict[str, Any]]:
    path = root / "rag_project" / "retrieval" / "context_builder.py"
    source = path.read_text(encoding="utf-8")
    budget = "if selected and used_tokens + estimated_tokens > self.token_budget:"
    dedup = "if chunk_id in seen or document_counts.get(document_id, 0) >= self.max_per_document:"
    budget_test = (
        "from importlib.util import spec_from_file_location, module_from_module\n"
    )
    budget_test = (
        "from importlib.util import spec_from_file_location, module_from_spec\n"
        "from rag_project.retrieval.hybrid_retriever import RetrievalHit\n"
        "spec=spec_from_file_location('mutant', '__MUTANT_PATH__')\n"
        "m=module_from_spec(spec); spec.loader.exec_module(m)\n"
        "def test_contract():\n"
        "    hits=[RetrievalHit(doc_id='d1', text='one two three four five six', metadata={'document_id':'d1','chunk_id':'c1'}, score=1), RetrievalHit(doc_id='d1', text='seven eight nine ten', metadata={'document_id':'d1','chunk_id':'c2'}, score=1)]\n"
        "    _, selected=m.ContextBuilder(token_budget=5).build(hits)\n"
        "    assert len(selected)==1\n"
    )
    dedup_test = (
        "from importlib.util import spec_from_file_location, module_from_spec\n"
        "from rag_project.retrieval.hybrid_retriever import RetrievalHit\n"
        "spec=spec_from_file_location('mutant', '__MUTANT_PATH__')\n"
        "m=module_from_spec(spec); spec.loader.exec_module(m)\n"
        "def test_contract():\n"
        "    hit=RetrievalHit(doc_id='d1', text='same evidence', metadata={'document_id':'d1','chunk_id':'same'}, score=1)\n"
        "    _, selected=m.ContextBuilder(token_budget=500).build([hit, hit])\n"
        "    assert len(selected)==1\n"
    )
    specs=[]
    if budget in source:
        specs.append(_run_mutant("context_budget", path, source, f"{budget}=>if False:", budget_test, root))
    if dedup in source:
        specs.append(_run_mutant("context_dedup", path, source, f"{dedup}=>if document_counts.get(document_id, 0) >= self.max_per_document:", dedup_test, root))
    return specs


def _vector_mutants(root: Path) -> list[dict[str, Any]]:
    path = root / "rag_project" / "storage" / "vector_store.py"
    source = path.read_text(encoding="utf-8")
    count = "return int(self.collection.count())"
    lexical = "return int(row[0] if row else 0)"
    tests=[]
    count_test = (
        "from importlib.util import spec_from_file_location, module_from_spec\n"
        "spec=spec_from_file_location('mutant', '__MUTANT_PATH__')\n"
        "m=module_from_spec(spec); spec.loader.exec_module(m)\n"
        "def test_contract(tmp_path):\n"
        "    s=m.VectorStore(tmp_path/'idx', collection_name='mutation_count')\n"
        "    s.add_documents(['medical evidence'], [{'document_id':'d1','chunk_id':'c1','page_numbers':[1]}], [[0.1]*32], ['d1:c1'])\n"
        "    assert s.count()==1\n"
    )
    lexical_test = (
        "from importlib.util import spec_from_file_location, module_from_spec\n"
        "spec=spec_from_file_location('mutant', '__MUTANT_PATH__')\n"
        "m=module_from_spec(spec); spec.loader.exec_module(m)\n"
        "def test_contract(tmp_path):\n"
        "    s=m.VectorStore(tmp_path/'idx', collection_name='mutation_lexical')\n"
        "    s.add_documents(['medical evidence'], [{'document_id':'d1','chunk_id':'c1','page_numbers':[1]}], [[0.1]*32], ['d1:c1'])\n"
        "    assert s.lexical_count()==1\n"
    )
    if count in source:
        tests.append(_run_mutant("vector_count", path, source, f"{count}=>return 0", count_test, root))
    if lexical in source:
        tests.append(_run_mutant("vector_lexical_count", path, source, f"{lexical}=>return 0", lexical_test, root))
    return tests


def run_full_mutation_suite(phase: Any) -> PhaseResult:
    started = __import__("time").time()
    result = PhaseResult(phase.number, phase.key, phase.name, started_at=started)
    try:
        mutants = _text_mutants(ROOT) + _context_mutants(ROOT) + _vector_mutants(ROOT)
        applicable = len(mutants)
        killed = sum(int(item.get("killed")) for item in mutants)
        targets = sorted({item.get("target") for item in mutants})
        kill_score = killed / max(1, applicable)
        result.details = {
            "evidence_level": "executable_multi_module_mutation_suite",
            "evidence_source": "independent pytest subprocesses over 12+ source mutants in 3 production modules",
            "strategy": "independent pytest subprocess per executable mutant across text normalization, context building, and vector storage",
            "mutants_applicable": applicable,
            "mutants_killed": killed,
            "kill_score": round(kill_score, 3),
            "mutation_results": mutants,
            "mutation_targets": targets,
            "target_module_count": len(targets),
            "real_pytest_subprocess": True,
            "checkout_modified": False,
            "required_kill_score": 1.0,
            "required_mutants": 12,
            "required_target_modules": 3,
        }
        result.score = round(kill_score, 3)
        result.status = "PASS" if applicable >= 12 and killed == applicable and len(targets) >= 3 else "FAIL"
        if result.status == "FAIL":
            result.failures.append({"location": "phase 11 multi-module mutation suite", "exception": "MutationCoverageFailure", "message": str(result.details)})
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 11 multi-module mutation suite", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(__import__("time").time() - started, 3)
    return result
