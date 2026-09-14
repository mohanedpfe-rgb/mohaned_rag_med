"""Functionality correction for numeric range grounding."""
from __future__ import annotations

import re
from typing import Any

_MEASURE_RE = re.compile(r"(?P<low>[-+]?\d+(?:[.,]\d+)?)(?:\s*[-–]\s*(?P<high>[-+]?\d+(?:[.,]\d+)?))?\s*(?P<unit>mg|g|kg|mcg|µg|ug|ml|l|mmhg|cmh2o|mmol/l|mol/l|iu|units?|%|bpm|°c|c|mm|cm|m|hz|khz|m/s|h|min|s|day|days|week|weeks|month|months|year|years)\b",re.I)
_SCALE={"ug":("mass",1e-6),"mcg":("mass",1e-6),"mg":("mass",1e-3),"g":("mass",1.0),"kg":("mass",1000.0),"ml":("volume",1.0),"l":("volume",1000.0),"mmhg":("pressure",1.0),"cmh2o":("pressure",0.735559),"mmol/l":("amount_concentration",1.0),"mol/l":("amount_concentration",1000.0),"iu":("activity",1.0),"unit":("activity",1.0),"units":("activity",1.0),"%":("percent",1.0),"bpm":("rate",1.0),"c":("temperature",1.0),"°c":("temperature",1.0),"mm":("length",1.0),"cm":("length",10.0),"m":("length",1000.0),"hz":("frequency",1.0),"khz":("frequency",1000.0),"m/s":("velocity",1.0),"s":("time",1.0),"min":("time",60.0),"h":("time",3600.0),"day":("time",86400.0),"days":("time",86400.0),"week":("time",604800.0),"weeks":("time",604800.0),"month":("time",2592000.0),"months":("time",2592000.0),"year":("time",31536000.0),"years":("time",31536000.0)}


def _parse_measurement(text: str):
    match=_MEASURE_RE.fullmatch(str(text or "").strip())
    if not match: return None
    unit=match.group("unit").casefold().replace(" ",""); info=_SCALE.get(unit)
    if info is None: return None
    try:
        low=float(match.group("low").replace(",","."))*info[1]; raw_high=match.group("high"); high=low if raw_high is None else float(raw_high.replace(",","."))*info[1]
    except (TypeError,ValueError): return None
    if high<low: low,high=high,low
    return info[0],low,high


def _range_compatible(answer_value: str,evidence_value: str)->bool:
    answer=_parse_measurement(answer_value); evidence=_parse_measurement(evidence_value)
    if answer is None or evidence is None or answer[0]!=evidence[0]: return False
    _,al,ah=answer; _,el,eh=evidence; answer_point=al==ah; evidence_point=el==eh
    if answer_point and evidence_point: return abs(al-el)<=1e-9*max(1.0,abs(al),abs(el))
    if answer_point: return el<=al<=eh
    if evidence_point: return False
    return el<=al and ah<=eh


def _wrap_measurement_compatibility(original: Any):
    def wrapped(left: Any,right: Any)->bool:
        try:
            if _range_compatible(f"{left[0]} {left[1]}",f"{right[0]} {right[1]}"): return True
        except (IndexError,KeyError,TypeError): pass
        return bool(original(left,right))
    wrapped._functionality_numeric_range_guard=True
    return wrapped


def _wrap_numeric_verifier(original: Any):
    def wrapped(self: Any,answer: str,hits: Any,route: Any,compiled: dict[str,Any]):
        result=dict(original(self,answer,hits,route,compiled) or {})
        if not answer or not hits or not bool(getattr(route,"numeric_sensitivity",False)): return result
        answer_values=[m.group(0) for m in _MEASURE_RE.finditer(str(answer))]; evidence_values=[m.group(0) for hit in hits for m in _MEASURE_RE.finditer(str(getattr(hit,"text","") or ""))]
        if not answer_values or not evidence_values: return result
        supported=all(any(_range_compatible(a,e) for e in evidence_values) for a in answer_values)
        if not supported: result["numeric_mismatch"]=True; result["allow"]=False; return result
        grounding=result.get("grounding") if isinstance(result.get("grounding"),dict) else {}; final=result.get("final_answer") if isinstance(result.get("final_answer"),dict) else {}
        result["numeric_mismatch"]=False; result["allow"]=bool(grounding.get("allow")) and bool(final.get("allow",True)); return result
    wrapped._functionality_numeric_range_guard=True
    return wrapped


def _install_direct_evidence_guard() -> None:
    from rag_project.intelligence import evidence_guard
    original=getattr(evidence_guard,"numeric_consistency_details",None)
    if not callable(original) or getattr(original,"_functionality_numeric_range_guard",False): return
    def numeric_consistency_details(claim,evidence):
        result=dict(original(claim,evidence) or {})
        claim_values=[m.group(0) for m in _MEASURE_RE.finditer(str(claim or ""))]
        evidence_values=[m.group(0) for m in _MEASURE_RE.finditer(str(evidence or ""))]
        if claim_values and evidence_values and all(any(_range_compatible(c,e) for e in evidence_values) for c in claim_values):
            result["mismatch"]=False; result["unsupported_numeric"]=[]
        return result
    numeric_consistency_details._functionality_numeric_range_guard=True
    evidence_guard.numeric_consistency_details=numeric_consistency_details


_install_direct_evidence_guard()


def install() -> None:
    from rag_project.intelligence import evidence_guard, med_evidence_pro
    if not getattr(evidence_guard._compatible,"_functionality_numeric_range_guard",False): evidence_guard._compatible=_wrap_measurement_compatibility(evidence_guard._compatible)
    if not getattr(evidence_guard._measurement_compatible,"_functionality_numeric_range_guard",False): evidence_guard._measurement_compatible=evidence_guard._compatible
    original_verify=med_evidence_pro.ActiveVerifier.verify
    if not getattr(original_verify,"_functionality_numeric_range_guard",False): med_evidence_pro.ActiveVerifier.verify=_wrap_numeric_verifier(original_verify)


__all__=["_range_compatible","install"]
