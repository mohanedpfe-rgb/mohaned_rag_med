"""Portable local load-test harness for the MedEvidence Pro answer callable."""
from __future__ import annotations
import argparse, json
from rag_project.intelligence.production_ops import benchmark_callable

def main():
    p=argparse.ArgumentParser(); p.add_argument("--count",type=int,default=100); p.add_argument("--workers",type=int,default=10); args=p.parse_args()
    # Replace this callable with a bound engine.answer in a local benchmark session.
    result=benchmark_callable(lambda q:q, ["benchmark query"]*max(1,args.count), workers=max(1,args.workers))
    print(json.dumps(result.__dict__,indent=2))
if __name__=="__main__": main()
