from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvidia_cmd.attention import resolve_backend


def test_auto_resolves_to_sdpa():
    assert resolve_backend("auto") == "sdpa"


def test_sdpa_is_explicit():
    assert resolve_backend("sdpa") == "sdpa"


def test_unknown_backend_raises():
    try:
        resolve_backend("natten")
    except ValueError as exc:
        assert "natten" in str(exc)
    else:
        raise AssertionError("expected ValueError")
