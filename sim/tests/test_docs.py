"""The README and report: templates fill completely, and the text follows the project's style
rules (no banned words, no exclamation marks, no emoji, no question headings)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sim.figures import docs

REPO = Path(__file__).resolve().parents[2]

BANNED = [
    r"unleash", r"harness", r"dive into", r"delve", r"seamless", r"cutting-edge",
    r"revolutionary", r"powerful", r"robust", r"elevate", r"journey", r"in today's world",
    r"it's worth noting", r"worth noting", r"game-changer", r"leverage", r"empower",
    r"magic", r"supercharge", r"not just\b[^.\n]{0,80}\bbut\b",
]  # fmt: skip
EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")

PROSE_FILES = [
    REPO / "README.md",
    REPO / "report" / "REPORT.md",
    REPO / "report" / "templates" / "README.template.md",
    REPO / "report" / "templates" / "REPORT.template.md",
    REPO / "web" / "index.html",
    *sorted((REPO / "web" / "src" / "pages").glob("*.ts")),
    REPO / "web" / "src" / "render" / "networkGraph.ts",
    REPO / "web" / "src" / "render" / "fitnessChart.ts",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", PROSE_FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_no_banned_words(path: Path) -> None:
    text = read(path)
    for pattern in BANNED:
        hit = re.search(pattern, text, re.IGNORECASE)
        assert hit is None, f"{path.name}: banned phrase {hit.group(0)!r}"


@pytest.mark.parametrize(
    "path",
    [p for p in PROSE_FILES if p.suffix == ".md"],
    ids=lambda p: str(p.relative_to(REPO)),
)
def test_markdown_style(path: Path) -> None:
    text = read(path)
    assert "!" not in re.sub(r"!\[", "", text), "exclamation mark in prose"
    assert EMOJI.search(text) is None, "emoji"
    for line in text.splitlines():
        if line.startswith("#"):
            assert not line.rstrip().endswith("?"), f"question heading: {line}"


def test_web_pages_have_no_exclamation_marks_or_emoji_in_user_text() -> None:
    for path in (REPO / "web" / "src" / "pages").glob("*.ts"):
        text = read(path)
        # Strings only: look for "...!" inside quotes or template text, not the != and !x operators.
        strings = re.findall(r'"([^"\n]*)"|`([^`]*)`', text)
        for a, b in strings:
            s = a or b
            assert not re.search(r"\w!(\s|$)", s), f"{path.name}: exclamation mark in {s!r}"
            assert EMOJI.search(s) is None


def test_templates_fill_without_leftover_placeholders() -> None:
    values = docs.build_values()
    for name in ("README.template.md", "REPORT.template.md"):
        out = docs.render(read(REPO / "report" / "templates" / name), values)
        assert "{{" not in out and "}}" not in out, name


def test_unknown_placeholders_are_an_error() -> None:
    with pytest.raises(KeyError, match="undefined placeholders"):
        docs.render("value: {{no.such.value}}", {"x": "1"})


def test_every_number_in_the_headline_table_is_in_the_committed_readme() -> None:
    table = read(REPO / "report" / "tables" / "headline.md").strip()
    assert table in read(REPO / "README.md")
    assert table in read(REPO / "report" / "REPORT.md")


def test_readme_follows_the_requested_order() -> None:
    headings = [ln[3:] for ln in read(REPO / "README.md").splitlines() if ln.startswith("## ")]
    assert headings == [
        "Results",
        "How it works",
        "How to reproduce",
        "Repository layout",
        "What did not work",
        "Limitations and future work",
        "References",
        "Licence",
    ]


def test_gif_is_under_the_size_limit() -> None:
    gif = REPO / "report" / "media" / "live-evolution.gif"
    assert gif.exists()
    assert gif.stat().st_size < 8 * 1024 * 1024


@pytest.mark.parametrize(
    "path", [REPO / "README.md", REPO / "report" / "REPORT.md"], ids=lambda p: p.name
)
def test_relative_links_and_images_exist(path: Path) -> None:
    text = read(path)
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    local = [t for t in targets if not t.startswith(("http://", "https://", "#", "mailto:"))]
    assert local, "expected at least one relative link or image"
    for target in local:
        assert (path.parent / target.split("#")[0]).exists(), f"{path.name}: broken link {target}"
