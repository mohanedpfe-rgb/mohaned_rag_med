from __future__ import annotations

from typing import Any, Dict


class MetadataFilter:
    @staticmethod
    def build(filters: Dict[str, Any] | None) -> Dict[str, Any]:
        if not filters:
            return {"index_state": "READY"}
        clean = {
            key: value
            for key, value in filters.items()
            if value is not None and value != ""
        }
        if not clean:
            return {"index_state": "READY"}
        if "index_state" not in clean:
            return {"$and": [{"index_state": "READY"}, clean]}
        return clean
