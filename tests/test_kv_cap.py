from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvidia_cmd.memory import circular_history_slots, estimate_generation_footprint


def test_circular_history_does_not_include_current_slot_before_store():
    slots = circular_history_slots(25, 21)
    assert len(slots) == 21
    assert slots[0] == 4
    assert slots[-1] == 3
    assert 25 % 21 == 4
    assert slots.count(4) == 1


def test_long_kv_uncapped_exceeds_5090_but_capped_fits():
    foot = estimate_generation_footprint(126, local_attn_size=21)
    assert foot["kv_uncapped_gb"] > 31.0
    assert foot["kv_capped_gb"] < 10.0
