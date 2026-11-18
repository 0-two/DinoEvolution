"""The stored golden traces must match the Python game.

The TypeScript test replays the same file, so this test is what keeps the two sides
anchored to one reference.
"""

from __future__ import annotations

import json

from sim.parity import FIXTURE_PATH, verify


def test_fixture_matches_game() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    assert verify(fixture) == []


def test_fixture_covers_the_interesting_cases() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    episodes = {e["name"]: e for e in fixture["episodes"]}
    deaths = [e["death_tick"] for e in episodes.values() if e["death_tick"] is not None]
    assert len(set(deaths)) >= 4, "deaths should land on several different ticks"
    ghost = episodes["noisy-ghost"]
    assert ghost["samples"][-1][7] > 0, "ghost run should record collision ticks"
    assert ghost["samples"][-1][3] == 13.0, "ghost run should reach maximum speed"
    assert len(ghost["actions"]) == 9000
