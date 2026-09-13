from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from rag_project.utils.text_utils import keyword_overlap_score, meaningful_tokens


@dataclass(frozen=True)
class ClaimCheck:
    claim: str
    support: float
    status: str
    sources: tuple[str, ...]
    numeric_mismatch: bool = False
    contradiction: bool = False
    reason: str = ""

    def __post_init__(self):
        if isinstance(self.support, str) and isinstance(self.status, (int, float)) and isinstance(self.sources, str):
            legacy_source = self.support
            legacy_support = float(self.status)
            legacy_status = self.sources
            object.__setattr__(self, "support", legacy_support)
            object.__setattr__(self, "status", legacy_status)
            object.__setattr__(self, "sources", (legacy_source,))
        elif isinstance(self.sources, str):
            object.__setattr__(self, "sources", (self.sources,))

    def to_dict(self):
        return asdict(self)


class NumericConsistencyResult(dict):
    """Structured numeric result that remains truthy only when no mismatch exists."""

    def __bool__(self) -> bool:
        return not bool(self.get("mismatch", False))


SENT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
MEASURE = re.compile(
    r"(?P<value>[-+]?\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?)\s*"
    r"(?P<unit>mg|g|kg|mcg|µg|ug|ml|l|mmhg|cmh2o|mmol/l|mol/l|iu|units?|%|bpm|°c|c|mm|cm|m|hz|khz|m/s|h|min|s|day|days|week|weeks|month|months|year|years)"
    r"(?=\s|$|[^\w])",
    re.I,
)
NEG = re.compile(
    r"\b(no|not|without|never|none|cannot|does not|doesn't|non|aucun|sans|jamais|ne pas|ممنوع|منع|لا|ليس|دون|absent|absence|inexistent|absente?|غير موجود|غياب)\b",
    re.I | re.UNICODE,
)
OPPOSITES = (
    (r"\bcontraindicated\b", r"\bindicated\b"),
    (r"\bshould not\b", r"\bshould\b"),
    (r"\bavoid\b", r"\brecommended\b"),
    (r"\bno\b", r"\bhas\b|\bwith\b"),
    (r"\bwithout\b", r"\bwith\b"),
    (r"\babsent\b|\babsence\b", r"\bpresent\b|\bdetected\b"),
    (r"\bnegative\b", r"\bpositive\b"),
)
SCALE = {
    "ug": ("mass", 1e-6),
    "mcg": ("mass", 1e-6),
    "mg": ("mass", 1e-3),
    "g": ("mass", 1),
    "kg": ("mass", 1000),
    "ml": ("volume", 1),
    "l": ("volume", 1000),
    "mmhg": ("pressure", 1),
    "cmh2o": ("pressure", 0.735559),
    "mmol/l": ("amount_concentration", 1),
    "mol/l": ("amount_concentration", 1000),
    "iu": ("activity", 1),
    "%": ("percent", 1),
    "bpm": ("rate", 1),
    "c": ("temperature", 1),
    "°c": ("temperature", 1),
    "mm": ("length", 1),
    "cm": ("length", 10),
    "m": ("length", 1000),
    "hz": ("frequency", 1),
    "khz": ("frequency", 1000),
    "m/s": ("velocity", 1),
    "s": ("time", 1),
    "min": ("time", 60),
    "h": ("time", 3600),
    "day": ("time", 86400),
    "days": ("time", 86400),
    "week": ("time", 604800),
    "weeks": ("time", 604800),
    "month": ("time", 2592000),
    "months": ("time", 2592000),
    "year": ("time", 31536000),
    "years": ("time", 31536000),
}
_TINY = {"yes", "no", "ok", "okay", "thanks", "thank", "maybe", "sure"}
_METADATA_BLOCK = re.compile(r"\[(?:section|source|file|page|document|metadata|citation|reference)\s*:\s*[^\]]*\]\s*", re.I)
_METADATA_LABEL = re.compile(r"^\s*(?:sources?|citations?|references?)\s*:", re.I)
_CONCEPT_SYNONYMS = (
    (r"\bhyperglyc(?:emia|émie)\b|\bhyperglycemia\b", "hyperglycemia"),
    (r"\bcétose\b|\bketosis\b", "ketosis"),
    (r"\bacidose métabolique\b|\bmetabolic acidosis\b", "metabolic acidosis"),
    (r"\bhypoglyc(?:emia|émie)\b|\bhypoglycemia\b", "hypoglycemia"),
    (r"\bhypokali(?:emia|émie)\b|\bhypokalemia\b", "hypokalemia"),
    (r"\bacidocétose diabétique\b|\bdiabetic ketoacidosis\b", "diabetic ketoacidosis"),
    (r"\bnéphropathie diabétique\b|\bnephropathie diabetique\b|\bdiabetic nephropathy\b", "diabetic nephropathy"),
    (r"\bdiabète\b|\bdiabete\b|\bdiabetes mellitus\b", "diabetes"),
    (r"\bcomplication microvasculaire\b|\bmicrovascular complication\b", "microvascular complication"),
    (r"\bchronique\b|\bchronic\b", "chronic"),
    (r"\bdu diabète\b|\bof diabetes\b", "of diabetes"),
)


def _normalize_semantic_text(text: str) -> str:
    value = str(text or "").casefold()
    for pattern, replacement in _CONCEPT_SYNONYMS:
        value = re.sub(pattern, replacement, value, flags=re.I | re.UNICODE)
    try:
        from rag_project.intelligence.semantic_reasoning import ALIASES
        for canonical, aliases in sorted(ALIASES.items(), key=lambda item: max(map(len, item[1])), reverse=True):
            for alias in sorted(aliases, key=len, reverse=True):
                escaped = re.escape(str(alias).casefold().strip())
                if escaped:
                    value = re.sub(rf"(?<!\w){escaped}(?!\w)", canonical.casefold(), value, flags=re.I | re.UNICODE)
    except Exception:
        pass
    return re.sub(r"\s+", " ", value).strip()


def split_claims(answer: str) -> list[str]:
    raw = str(answer or "").strip()
    if not raw:
        return []
    raw = re.sub(r"(?<=[.!?。！？])\s+(?=\[S\d+\])", " ", raw)
    raw = _METADATA_BLOCK.sub("", raw)
    out = []
    for sentence in SENT.split(raw):
        sentence = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", sentence.strip())
        if _METADATA_LABEL.match(sentence):
            continue
        if re.fullmatch(r"(?:\[S\d+\]\s*)+", sentence, re.I):
            if out:
                out[-1] = f"{out[-1]} {sentence}".strip()
            continue
        toks = [t.casefold() for t in meaningful_tokens(sentence)]
        if not toks or (len(toks) <= 2 and set(toks).issubset(_TINY)):
            continue
        out.append(sentence)
        if len(out) >= 40:
            break
    return out


def _norm_unit(u: str) -> str:
    return {"µg": "ug", "mcg": "ug", "°c": "c", "mmol/l": "mmol/l", "mol/l": "mol/l"}.get(u.casefold(), u.casefold())


def _num(v: str) -> float | None:
    try:
        return float(v.replace(",", ".").replace(" ", ""))
    except Exception:
        return None


def extract_measurements(text: str) -> list[tuple[str, str]]:
    out = []
    for match in MEASURE.finditer(text or ""):
        item = (match.group("value").replace(",", ".").replace(" ", ""), _norm_unit(match.group("unit")))
        if item not in out:
            out.append(item)
    return out


def _compatible(a, b):
    av, au = a
    bv, bu = b
    if "-" in av or "-" in bv:
        return av == bv and au == bu
    af, bf = _num(av), _num(bv)
    if af is None or bf is None:
        return au == bu and av == bv
    sa, sb = SCALE.get(au), SCALE.get(bu)
    if not sa or not sb or sa[0] != sb[0]:
        return au == bu and math.isclose(af, bf, abs_tol=1e-9)
    return math.isclose(af * sa[1] / sb[1], bf, rel_tol=0, abs_tol=1e-6)


def _measurement_compatible(a, b):
    return _compatible(a, b)


def numeric_consistency_details(claim, evidence):
    cv, ev = extract_measurements(claim), extract_measurements(evidence)
    bad = [x for x in cv if not any(_compatible(x, y) for y in ev)] if cv else []
    return {
        "checked": bool(cv),
        "mismatch": bool(bad),
        "claim_values": [f"{v} {u}" for v, u in cv],
        "evidence_values": [f"{v} {u}" for v, u in ev],
        "unsupported_numeric": [f"{v} {u}" for v, u in bad],
    }


def numeric_consistency(claim, evidence):
    return NumericConsistencyResult(numeric_consistency_details(claim, evidence))


def _polarity(t):
    return -1 if NEG.search(t or "") else 1


def _score_text(t):
    return re.sub(r"\[S\d+\]", "", t or "").strip()


def _remove_measurements(text: str) -> str:
    return MEASURE.sub(" ", text or "")


def semantic_support(claim, evidence):
    claim = _score_text(claim)
    evidence = str(evidence or "").strip()
    if not claim or not evidence:
        return 0.0
    nclaim = re.sub(r"\s+", " ", _normalize_semantic_text(claim)).strip()
    nevidence = re.sub(r"\s+", " ", _normalize_semantic_text(evidence)).strip()
    if nclaim == nevidence:
        return 1.0
    ct = set(meaningful_tokens(nclaim))
    et = set(meaningful_tokens(nevidence))
    if not ct or not et:
        return 0.0
    framing = {"the", "a", "an", "main", "findings", "finding", "include", "includes", "included", "reported", "reports", "observed", "shows", "show", "identified", "described", "key", "primary", "principales", "conséquences", "biologiques", "sont", "les", "des"}
    ct = {t for t in ct if t not in framing} or ct
    shared = ct & et
    if len(shared) <= 2 and (ct - shared) and (et - shared):
        return 0.0
    coverage = len(shared) / len(ct)
    if coverage < 0.50:
        return 0.0
    char = keyword_overlap_score(nclaim, nevidence)
    jac = len(shared) / max(1, len(ct | et))
    polarity_penalty = 0.35 if _polarity(nclaim) != _polarity(nevidence) else 0
    return max(0.0, min(1.0, 0.50 * coverage + 0.25 * jac + 0.25 * char - polarity_penalty))


def detect_contradiction(claim, evidence_blocks):
    cl = _score_text(claim).casefold()
    blocks = [evidence_blocks] if isinstance(evidence_blocks, str) else list(evidence_blocks or ())
    cl_tokens = set(meaningful_tokens(cl))
    generic = {"patient", "the", "is", "has", "with", "present", "presence", "absent", "absence", "not", "no"}
    for ev in blocks:
        el = str(ev or "").casefold()
        shared = (cl_tokens & set(meaningful_tokens(el))) - generic
        explicit = any(re.search(a, cl, re.I) and re.search(b, el, re.I) for a, b in OPPOSITES)
        polarity = bool(NEG.search(cl)) != bool(NEG.search(el)) and bool(shared)
        if (explicit or polarity) and (semantic_support(cl, el) >= 0.08 or len(shared) >= 1):
            return True
    return False


def _best_support(claim, blocks, ids):
    rows = sorted(((semantic_support(claim, b), ids[i] if i < len(ids) else f"S{i + 1}") for i, b in enumerate(blocks)), reverse=True)
    rows = [r for r in rows if r[0] > 0.05]
    return (rows[0][0], tuple(x[1] for x in rows[:3])) if rows else (0.0, ())


def _cited_evidence(claim: str, evidence_blocks: Sequence[str], source_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """Restrict verification to sources explicitly cited by the generated claim when markers exist."""
    markers = [f"S{number}" for number in re.findall(r"\[S(\d+)\]", str(claim or ""), flags=re.I)]
    if not markers:
        return list(evidence_blocks), list(source_ids)
    index_by_id = {str(source_id).casefold(): index for index, source_id in enumerate(source_ids)}
    marker_keys = [marker.casefold() for marker in markers]
    if any(marker not in index_by_id for marker in marker_keys):
        return [], []
    selected_indices = [index_by_id[marker] for marker in marker_keys]
    unique_indices = list(dict.fromkeys(selected_indices))
    return [str(evidence_blocks[index]) for index in unique_indices if index < len(evidence_blocks)], [str(source_ids[index]) for index in unique_indices if index < len(source_ids)]


def verify_claims(answer, evidence_blocks: Sequence[str], source_ids: Sequence[str]) -> list[ClaimCheck]:
    checks = []
    for claim in split_claims(answer):
        cited_blocks, cited_ids = _cited_evidence(claim, evidence_blocks, source_ids)
        support_blocks = cited_blocks if cited_blocks else []
        support_ids = cited_ids if cited_ids else []
        best, sources = _best_support(claim, support_blocks, support_ids) if support_blocks else (0.0, ())
        cited_joined = "\n".join(cited_blocks)
        num = numeric_consistency_details(claim, cited_joined) if cited_blocks else {"checked": bool(extract_measurements(claim)), "mismatch": bool(extract_measurements(claim)), "claim_values": [], "evidence_values": [], "unsupported_numeric": [f"{v} {u}" for v, u in extract_measurements(claim)]}
        contra = detect_contradiction(claim, evidence_blocks)
        numeric_bridge = max((semantic_support(_remove_measurements(claim), _remove_measurements(block)) for block in cited_blocks), default=0.0) if num["checked"] and not num["mismatch"] else 0.0
        if contra:
            status, reason = "CONTRADICTED", "A source conflicts with the claim polarity or safety meaning."
        elif num["mismatch"]:
            status, reason = "NUMERIC_MISMATCH", "The stated measurement is not supported by a compatible value in the cited evidence."
        elif best >= 0.62 or numeric_bridge >= 0.35:
            status, reason = "SUPPORTED", "Strong evidence support."
        elif best >= 0.38:
            status, reason = "PARTIAL", "Partial evidence support."
        elif best > 0.05:
            status, reason = "WEAK", "Weak evidence overlap."
        elif not support_blocks and re.search(r"\[S\d+\]", claim, flags=re.I):
            status, reason = "UNSUPPORTED", "The cited evidence source does not exist in the supplied evidence set."
        else:
            status, reason = "UNSUPPORTED", "No meaningful evidence support."
        checks.append(ClaimCheck(claim, round(max(best, numeric_bridge), 4), status, sources, bool(num["mismatch"]), contra, reason))
    return checks


def grounding_decision(claims: Sequence[ClaimCheck], min_supported_ratio: float = 0.70) -> dict[str, Any]:
    """Return the fail-closed grounding decision used by answer generation paths."""
    rows = list(claims or ())
    blocked = [claim for claim in rows if claim.status in {"UNSUPPORTED", "WEAK", "NUMERIC_MISMATCH", "CONTRADICTED"} or claim.numeric_mismatch or claim.contradiction]
    supported = sum(1 for claim in rows if claim.status in {"SUPPORTED", "PARTIAL"} and not claim.numeric_mismatch and not claim.contradiction)
    ratio = supported / max(1, len(rows))
    threshold = max(0.0, min(1.0, float(min_supported_ratio)))
    allow = bool(rows) and not blocked and ratio >= threshold
    return {
        "allow": allow,
        "checked": bool(rows),
        "claim_count": len(rows),
        "blocked_claims": len(blocked),
        "supported_claims": supported,
        "supported_ratio": round(ratio, 4),
        "threshold": threshold,
        "reason": "grounded" if allow else "insufficient_or_unsafe_support",
    }


def contradiction_report(claims: Sequence[ClaimCheck]) -> dict[str, Any]:
    """Summarize claim-level contradictions without changing the underlying checks."""
    rows = list(claims or ())
    conflicts = [claim.to_dict() for claim in rows if claim.contradiction or claim.status == "CONTRADICTED"]
    agreement = sum(1 for claim in rows if not claim.contradiction and claim.status in {"SUPPORTED", "PARTIAL"}) / max(1, len(rows))
    return {
        "has_contradiction": bool(conflicts),
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "agreement": round(agreement, 4),
    }


def evidence_confidence(*, retrieval: float, rerank: float, entailment: float, quality: float, contradiction: float = 0.0) -> float:
    """Combine retrieval, verification and evidence quality into a bounded confidence score."""
    score = (0.30 * float(retrieval) + 0.25 * float(rerank) + 0.25 * float(entailment) + 0.20 * float(quality)) - 0.40 * float(contradiction)
    return round(max(0.0, min(1.0, score)), 4)


def citation_firewall(answer: str, claims: Sequence[ClaimCheck]) -> tuple[str, bool]:
    """Withhold claims that fail evidence verification while preserving safe claims."""
    rows = list(claims or ())
    unsafe = [claim for claim in rows if claim.status not in {"SUPPORTED", "PARTIAL"} or claim.numeric_mismatch or claim.contradiction]
    if not unsafe:
        return str(answer or ""), False
    safe_claims = [claim.claim.strip() for claim in rows if claim not in unsafe and claim.claim.strip()]
    if safe_claims:
        return "\n".join(f"- {claim}" for claim in safe_claims) + "\n\n[Some claims were withheld because they could not be verified against the evidence.]", True
    return "The requested claims were withheld because they could not be verified against the indexed evidence.", True
