from __future__ import annotations
import re
from typing import Any,Sequence
from rag_project.intelligence.evidence_guard import ClaimCheck,verify_claims
_REPAIR_SYSTEM=("You repair a medical RAG answer using ONLY supplied evidence. Remove every unsupported, weak, contradicted, or numerically inconsistent claim. Preserve supported qualifiers, numbers, units, negation, population, uncertainty and [S#] citations. Never add a fact. Return only the repaired answer.")
def _compact(text,limit=7000): return re.sub(r'\s+',' ',str(text or '')).strip()[:limit]
def repair_needed(checks:Sequence[ClaimCheck])->bool:
    return bool(checks) and any(c.status in {'WEAK','UNSUPPORTED','PARTIAL','NUMERIC_MISMATCH','CONTRADICTED'} or c.contradiction for c in checks)
def build_repair_prompt(question,draft,evidence_blocks,checks):
    blocked=[c for c in checks if c.status!='SUPPORTED' or c.contradiction]; evidence='\n\n'.join(f'[S{i+1}] {_compact(b,1200)}' for i,b in enumerate(evidence_blocks[:12]) if str(b).strip()); failures='\n'.join(f'- {c.status}: {c.claim} ({c.reason})' for c in blocked[:12]); return f'Question: {_compact(question,1800)}\n\nDraft answer:\n{_compact(draft,5000)}\n\nClaims that failed verification:\n{failures}\n\nVerified evidence:\n{evidence}\n\nRewrite conservatively and omit anything the evidence does not support.'
def repair_answer(llm:Any,question:str,draft:str,evidence_blocks:Sequence[str],checks:Sequence[ClaimCheck])->str|None:
    if llm is None or not draft or not repair_needed(checks): return None
    try: value=llm.generate(prompt=build_repair_prompt(question,draft,evidence_blocks,checks),system_prompt=_REPAIR_SYSTEM,temperature=0.0)
    except Exception: return None
    value=str(value or '').strip(); return value[:9000] if value else None
def repair_and_verify(llm,question,draft,evidence_blocks,source_ids,checks):
    repaired=repair_answer(llm,question,draft,evidence_blocks,checks)
    if not repaired or repaired==draft:return draft,list(checks),False
    fresh=verify_claims(repaired,evidence_blocks,source_ids)
    # Return the repaired text and its fresh verification even when it is still unsafe.
    # The caller remains responsible for the final fail-closed gate/citation firewall.
    return repaired,fresh,True
