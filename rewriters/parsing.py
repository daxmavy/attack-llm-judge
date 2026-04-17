"""Robust JSON extraction for structured-output rewriters."""
from __future__ import annotations

import json
import re


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Try several parses from most-strict to most-lenient."""
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = _FENCE_RE.search(t)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = _OBJECT_RE.search(t)
    if m:
        snippet = m.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            # Lenient: try to find "final" field specifically.
            fm = re.search(r'"final"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', snippet, re.DOTALL)
            if fm:
                return {"final": fm.group(1)}
    return None
