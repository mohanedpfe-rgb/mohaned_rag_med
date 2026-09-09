from __future__ import annotations

import hashlib
import json
import math
import time
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import requests

from rag_project.security import sanitize_model_text, validate_ollama_url


class EmbeddingProfile:
    """Immutable identity for a specific embedding configuration and index profile."""
    def __init__(self, provider: str, model: str | None = None, dimension: int | None = None, *, model_name: str | None = None, model_version: str | None = None, normalization: str = "none", metric: str = "cosine", implementation_version: str = "embedding-v2", configuration: dict[str, Any] | None = None, creation_timestamp: str | None = None):
        if model is None:
            model = model_name
        if model is None or dimension is None:
            raise ValueError("Embedding profile requires a model name and dimension.")
        self.provider = provider; self.model = str(model); self.model_name = self.model; self.model_version = model_version
        self.dimension = int(dimension); self.normalization = normalization; self.metric = metric
        self.implementation_version = implementation_version; self.configuration = dict(configuration or {})
        self.creation_timestamp = creation_timestamp or datetime.now(timezone.utc).isoformat(); self._configuration_fingerprint = self._compute_fingerprint()
    @property
    def normalized_model_identifier(self) -> str:
        return f"{self.provider}:{self.model}:{self.model_version or 'unknown'}"
    @property
    def configuration_fingerprint(self) -> str: return self._configuration_fingerprint
    @property
    def fingerprint(self) -> str: return self._configuration_fingerprint
    @property
    def embedding_id(self) -> str:
        return ":".join((self.provider,self.model,self.model_version or "unknown",str(self.dimension),self.normalization,self.metric,self.implementation_version))
    def _compute_fingerprint(self) -> str:
        payload = {"provider":self.provider,"model":self.model,"model_version":self.model_version,"dimension":self.dimension,"normalization":self.normalization,"metric":self.metric,"implementation_version":self.implementation_version,**self.configuration}
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
    def to_dict(self) -> dict[str, Any]:
        return {"provider":self.provider,"model":self.model,"model_name":self.model,"model_version":self.model_version,"dimension":self.dimension,"normalization":self.normalization,"metric":self.metric,"implementation_version":self.implementation_version,"normalized_model_identifier":self.normalized_model_identifier,"configuration_fingerprint":self.configuration_fingerprint,"fingerprint":self.configuration_fingerprint,"creation_timestamp":self.creation_timestamp,"embedding_id":self.embedding_id}


EmbeddingIdentity = EmbeddingProfile


class EmbeddingService:
    """Ollama-first embedding service with bounded caches, requests, and concurrency."""
    def __init__(self, base_url: str, model: str, *, batch_size: int = 4, retries: int = 4, timeout_seconds: float = 180.0, test_mode: bool = False, cache_size: int = 128, cache_ttl_seconds: float = 900.0, max_concurrency: int = 1, prefer_local_transformers: bool = False):
        self.base_url = validate_ollama_url(base_url) if not test_mode else base_url.rstrip("/") if base_url else ""
        self.model = sanitize_model_text(model, limit=200).strip()
        if not self.model: raise ValueError("Embedding model identifier cannot be empty.")
        self.batch_size = max(1, min(int(batch_size), 64)); self.retries = max(1, min(int(retries), 6)); self.timeout_seconds = max(5.0, min(float(timeout_seconds), 180.0))
        self.test_mode=test_mode; self.dimension=None; self.provider="deterministic-test" if test_mode else "ollama"; self.prefer_local_transformers=bool(prefer_local_transformers)
        self.cache_size=max(0,int(cache_size)); self.cache_ttl_seconds=max(0.0,float(cache_ttl_seconds)); self._profile_fingerprint=None
        self._query_cache=OrderedDict(); self._embedding_cache=OrderedDict(); self._active_batch_size=self.batch_size; self._consecutive_timeouts=0
        self._inference_semaphore=threading.BoundedSemaphore(max(1,int(max_concurrency))); self._sentence_transformer=None; self._ollama_available=None; self._ollama_last_check=0.0; self.last_error=None
    def _get_sentence_transformer(self):
        if self._sentence_transformer is None:
            from sentence_transformers import SentenceTransformer
            self._sentence_transformer=SentenceTransformer(self.model)
        return self._sentence_transformer
    def _check_ollama_available(self, force: bool=False) -> bool:
        if self.test_mode: return False
        now=time.monotonic()
        if not force and self._ollama_available is not None and now-self._ollama_last_check<30: return self._ollama_available
        try:
            response=requests.get(f"{self.base_url}/api/tags",timeout=(1.5,3.0),allow_redirects=False)
            if 300 <= response.status_code < 400: raise requests.RequestException("redirect rejected")
            response.raise_for_status(); self._ollama_available=True
        except requests.RequestException:
            self._ollama_available=False; self.last_error="Ollama health probe failed"
        self._ollama_last_check=now; return self._ollama_available
    @property
    def identity(self) -> EmbeddingProfile | None:
        if self.dimension is None: return None
        implementation="deterministic-test-v1" if self.test_mode else ("sentence-transformers-v1" if self.provider=="sentence-transformers" else "ollama-api-v1")
        profile=EmbeddingProfile(provider=self.provider,model=self.model,dimension=self.dimension,model_version="latest",normalization="none",metric="cosine",implementation_version=implementation)
        if self._profile_fingerprint is None: self._profile_fingerprint=profile.fingerprint
        return profile
    def discover_dimension(self) -> int:
        self.embed_query("__rag_dimension_probe__")
        if self.dimension is None: raise RuntimeError("Embedding dimension discovery produced no dimension.")
        return self.dimension
    def embed_texts(self,texts:list[str])->list[list[float]]:
        if not texts:return []
        texts=[sanitize_model_text(t,limit=12000) for t in texts]
        if self.test_mode:return [self._test_embedding(t) for t in texts]
        ordered=[None]*len(texts); missing=[]; missing_indices=[]; prefix=self._profile_fingerprint or "no_profile"
        for i,text in enumerate(texts):
            key=f"{prefix}::{text}"; cached=self._embedding_cache.get(key)
            if cached is not None: ordered[i]=cached
            else: missing_indices.append(i); missing.append(text)
        if missing:
            with self._inference_semaphore: new=self._embed_batch(missing)
            if len(new)!=len(missing): raise RuntimeError("Embedding backend returned an unexpected result count.")
            for offset,(i,text) in enumerate(zip(missing_indices,missing,strict=True)):
                vector=new[offset]; key=f"{prefix}::{text}"; self._embedding_cache[key]=vector
                if self.cache_size and len(self._embedding_cache)>self.cache_size:self._embedding_cache.popitem(last=False)
                ordered[i]=vector
        if any(v is None for v in ordered):raise RuntimeError("Embedding backend returned incomplete results.")
        return [v for v in ordered if v is not None]
    def _ollama_embed_batch(self,texts:list[str])->list[list[float]]:
        attempt_texts=list(texts); last_error=None
        for attempt in range(self.retries):
            if self._consecutive_timeouts>=2:self._active_batch_size=1
            if len(attempt_texts)>self._active_batch_size:
                half=max(1,len(attempt_texts)//2); return self._ollama_embed_batch(attempt_texts[:half])+self._ollama_embed_batch(attempt_texts[half:])
            try:
                response=requests.post(f"{self.base_url}/api/embed",json={"model":self.model,"input":attempt_texts},timeout=(5,self.timeout_seconds),allow_redirects=False)
                if 300 <= response.status_code < 400: raise RuntimeError("Ollama redirect rejected")
                response.raise_for_status(); payload=response.json()
                if not isinstance(payload,dict): raise ValueError("Embedding response was not an object.")
                if "embeddings" in payload: result=payload["embeddings"]
                elif isinstance(payload.get("embedding"),list): result=[payload["embedding"]]
                elif isinstance(payload.get("data"),list): result=[item["embedding"] for item in payload["data"]]
                else: raise ValueError("Embedding response did not contain embeddings.")
                self._validate(result,len(attempt_texts)); self.provider="ollama"; self.last_error=None; self._consecutive_timeouts=0; self._ollama_available=True
                self._active_batch_size=min(self.batch_size,self._active_batch_size+1); return result
            except requests.exceptions.Timeout as exc:
                last_error=exc; self.last_error="Embedding request timed out"; self._consecutive_timeouts+=1; self._active_batch_size=max(1,self._active_batch_size//2)
                if attempt+1<self.retries: time.sleep(min(4,1.5*(2**attempt)))
            except (requests.RequestException,ConnectionError,ValueError,KeyError,TypeError,RuntimeError) as exc:
                last_error=exc; self.last_error=type(exc).__name__; self._ollama_available=False
                if attempt+1<self.retries: time.sleep(min(2,0.5*(2**attempt)))
        raise RuntimeError(f"Ollama embedding service failed after {self.retries} attempts for model {self.model!r}.") from last_error
    def _transformers_embed_batch(self,texts:list[str])->list[list[float]]:
        try:
            vectors=self._get_sentence_transformer().encode(texts,batch_size=max(1,min(self.batch_size,len(texts))),show_progress_bar=False,normalize_embeddings=False,convert_to_numpy=True); self.provider="sentence-transformers"; self.last_error=None
        except Exception as exc:
            self.last_error=type(exc).__name__; self._sentence_transformer=None; raise RuntimeError("SentenceTransformers fallback failed.") from exc
        if isinstance(vectors,np.ndarray):vectors=vectors.tolist()
        if isinstance(vectors,list) and vectors and not isinstance(vectors[0],list):vectors=[list(vectors)]
        self._validate(vectors,len(texts)); return [list(map(float,v)) for v in vectors]
    def _embed_batch(self,texts:list[str])->list[list[float]]:
        if not texts:return []
        ollama_up=self._check_ollama_available()
        if self.prefer_local_transformers:
            try:return self._transformers_embed_batch(texts)
            except Exception as exc:
                if ollama_up:return self._ollama_embed_batch(texts)
                raise exc
        if ollama_up:
            try:return self._ollama_embed_batch(texts)
            except Exception as exc:
                if ":" not in self.model:
                    try:return self._transformers_embed_batch(texts)
                    except Exception:pass
                raise exc
        if self.base_url:
            try:return self._ollama_embed_batch(texts)
            except Exception:pass
        if ":" not in self.model:return self._transformers_embed_batch(texts)
        raise RuntimeError(f"No embedding backend available at {self.base_url!r} for model {self.model!r}.")
    def _validate(self,vectors:list[list[float]],expected_count:int)->None:
        if len(vectors)!=expected_count or not vectors:raise ValueError("Embedding service returned an unexpected number of vectors.")
        dimension=len(vectors[0])
        if dimension==0 or any(len(v)!=dimension or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in v) for v in vectors):raise ValueError("Embedding vectors have inconsistent dimensions.")
        if any(math.sqrt(sum(float(x)*float(x) for x in v))<=1e-12 for v in vectors):raise ValueError("Embedding vectors must have a non-zero norm.")
        if self.dimension is None:self.dimension=dimension
        elif self.dimension!=dimension:raise ValueError(f"Embedding dimension changed from {self.dimension} to {dimension}.")
    def embed_query(self,query:str)->list[float]:
        query=sanitize_model_text(query,limit=4000); key=query.strip(); now=time.monotonic(); cached=self._query_cache.get(key)
        if cached and now-cached[0]<=self.cache_ttl_seconds:self._query_cache.move_to_end(key); return list(cached[1])
        vectors=self.embed_texts([query]);
        if not vectors:raise RuntimeError("No embedding was produced for the query.")
        vector=list(vectors[0])
        if self.cache_size:self._query_cache[key]=(now,vector); self._query_cache.move_to_end(key)
        while len(self._query_cache)>self.cache_size:self._query_cache.popitem(last=False)
        return vector
    def validate_embedding(self,vector:list[float]|tuple[float,...])->None:self._validate([list(vector)],1)
    def validate_batch(self,vectors:list[list[float]])->None:self._validate(vectors,len(vectors))
    def _test_embedding(self,text:str)->list[float]:
        vector=[byte/255.0 for byte in hashlib.sha256(text.encode("utf-8")).digest()]; self._validate([vector],1); return vector
