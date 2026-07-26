"""Public-cleanliness invariants for the shipped documentation.

Guards the release docs against internal names, machine-specific paths,
private links, and broken relative links. The parity fixture's provenance
README is exempt: it records how the golden snapshot was generated on the
original machine, absolute paths included.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXEMPT = {REPO / "tests" / "fixtures" / "parity" / "README.md"}

# Internal-only names and machine-specific literals that must never ship.
BANNED = ["simplepredict", "notion.so", "/home/rotem", "/media/rotem"]


def _shipped_docs():
    """README + every markdown file under docs/, minus exemptions."""
    files = [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md"))]
    for extra in ("CITATION.cff", "CONTRIBUTING.md"):
        p = REPO / extra
        if p.exists():
            files.append(p)
    return [f for f in files if f.exists() and f not in EXEMPT]


@pytest.mark.parametrize("banned", BANNED)
def test_no_banned_strings_in_shipped_docs(banned):
    offenders = []
    for f in _shipped_docs():
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if banned in line:
                offenders.append(f"{f.relative_to(REPO)}:{i}")
    assert not offenders, f"'{banned}' found in shipped docs: {offenders}"


def test_relative_markdown_links_resolve():
    """Every relative [text](path) link in shipped docs points at a real file."""
    link_re = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    broken = []
    for f in _shipped_docs():
        if f.suffix != ".md":
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for target in link_re.findall(line):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                path = target.split("#", 1)[0]
                if not path:
                    continue
                if not (f.parent / path).resolve().exists():
                    broken.append(f"{f.relative_to(REPO)}:{i} -> {target}")
    assert not broken, f"broken relative links: {broken}"


def test_readme_covers_release_essentials():
    """The README must carry the things a public reader needs to get started."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    required = [
        "DLOFeel-State",          # project name
        "30.52",                  # headline RMSE
        "--recurse-submodules",   # Sparsh is a submodule; clone must say so
        "dlofeel-eval",           # a runnable entrypoint
        "docs/DATA.md",           # how to get data
        "CITATION.cff",           # citation metadata
        "CC-BY-NC",               # the Sparsh licence boundary
    ]
    missing = [r for r in required if r not in text]
    assert not missing, f"README missing required content: {missing}"


def test_citation_is_parseable_with_correct_authors_and_no_venue_claim():
    """CITATION.cff must be valid CFF with the exact paper title and author list.

    The paper is under review, so the file must NOT claim a venue or DOI.
    """
    import yaml
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["cff-version"] == "1.2.0"
    assert cff["preferred-citation"]["title"] == (
        "Physics-Informed Tactile-based Shape Estimation of Deformable Linear Objects"
    )
    # No premature publication claims while the paper is under review.
    for banned in ("doi", "journal", "year"):
        assert banned not in cff["preferred-citation"], (
            f"'{banned}' must not appear until the paper is accepted"
        )
    expected = [
        ("Rotem", "Atari"),
        ("Stav", "Ashur"),
        ("Avishai", "Sintov"),
    ]
    for key in ("authors",):
        got = [(a["given-names"], a["family-names"]) for a in cff[key]]
        assert got == expected, f"{key} mismatch: {got}"
    got_paper = [
        (a["given-names"], a["family-names"])
        for a in cff["preferred-citation"]["authors"]
    ]
    assert got_paper == expected, f"preferred-citation authors mismatch: {got_paper}"
