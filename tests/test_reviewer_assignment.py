"""
Tests for the reviewer-routing stage.

Everything here runs offline — no AWS, no model calls. What is covered is the
part that decides whether a human sees the right thing: flag normalisation,
domain-name repair, cross-domain routing, and what actually lands in the
workbook a reviewer opens.
"""

import io
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
# hierarchical_taxonomy imports credentials at module scope; stub them so the
# tests never touch a real config.py.
sys.modules.setdefault("config", types.SimpleNamespace(
    AWS_ACCESS_KEY_ID="test", AWS_SECRET_ACCESS_KEY="test", AWS_REGION="us-east-1"))

import reviewer_assignment as ra  # noqa: E402


@pytest.fixture
def result_df():
    def tech(name, l1, l2, leaf):
        return {"technology_name": name, "Technology_Description": "description",
                "category_level_1": l1, "category_level_2": l2,
                "category_level_3": "", "leaf_category": leaf,
                "full_path": f"{l1} / {l2} / {leaf}"}
    return pd.DataFrame([
        tech("Topological qubit lattice", "Quantum", "Qubits", "Qubit hardware"),
        tech("Diamond vacancy magnetometry array", "Quantum", "Qubits", "Qubit hardware"),
        tech("Optogenetic retinal restoration vector", "Bio", "Neuro", "Neural restoration"),
        tech("Vision-language manipulation policy", "Robotics", "Control", "Grab bag"),
        tech("Iron-air multi-day storage cell", "Robotics", "Control", "Grab bag"),
    ])


@pytest.fixture
def domain_map():
    return {
        "Qubit hardware": {
            "primary_domain": ra.DOMAIN_1, "primary_sub_domain": "sub-a",
            "secondary_domain": None, "secondary_sub_domain": None,
            "reasoning": "core mechanism is qubit physics",
            "group_status": "tight",
            "anomalies": [{"name": "Diamond vacancy magnetometry array", "level": "yellow"}],
        },
        # A genuine bridge: its core sits in one domain, its delivery in another.
        "Neural restoration": {
            "primary_domain": ra.DOMAIN_2, "primary_sub_domain": "sub-b",
            "secondary_domain": ra.DOMAIN_4, "secondary_sub_domain": "sub-d",
            "reasoning": "gene therapy paired with an imaging device",
            "group_status": "tight", "anomalies": [],
        },
        "Grab bag": {
            "primary_domain": ra.DOMAIN_4, "primary_sub_domain": "sub-d",
            "secondary_domain": None, "secondary_sub_domain": None,
            "reasoning": "no clear majority",
            "group_status": "mixed",
            # These must be dropped: a mixed group is routed manually as a whole.
            "anomalies": [{"name": "Iron-air multi-day storage cell", "level": "red"}],
        },
    }


# ── flag normalisation ──────────────────────────────────────────────────────

def test_flag_view_reads_the_current_shape():
    status, anoms = ra._flag_view(
        {"group_status": "diverse", "anomalies": [{"name": "A", "level": "red"}]})
    assert status == "diverse"
    assert anoms == [{"name": "A", "level": "red"}]


def test_flag_view_accepts_bare_strings_as_yellow():
    """The model has been seen returning a plain list of names."""
    _, anoms = ra._flag_view({"group_status": "tight", "anomalies": ["A", "B"]})
    assert anoms == [{"name": "A", "level": "yellow"}, {"name": "B", "level": "yellow"}]


def test_flag_view_supports_the_legacy_boolean_shape():
    """Old saved domain_maps must still render rather than crash."""
    assert ra._flag_view({"incoherent": True, "anomalies": ["A"]})[0] == "mixed"
    assert ra._flag_view({"incoherent": False, "anomalies": []})[0] == "tight"


def test_unknown_status_falls_back_to_tight():
    assert ra._flag_view({"group_status": "banana"})[0] == "tight"


def test_mixed_group_drops_per_item_flags():
    """A mixed group already goes to manual routing — item flags add only noise."""
    status, anoms = ra._flag_view(
        {"group_status": "mixed", "anomalies": [{"name": "A", "level": "red"}]})
    assert status == "mixed"
    assert anoms == []


def test_blank_names_are_discarded():
    _, anoms = ra._flag_view({"group_status": "tight", "anomalies": ["", "  ", "Real"]})
    assert anoms == [{"name": "Real", "level": "yellow"}]


# ── domain-name repair ──────────────────────────────────────────────────────

def test_normalize_domain_map_passes_names_through_the_repair():
    """One deformed character would otherwise send a whole file to the wrong person."""
    fixed = ra.normalize_domain_map({
        "leaf": {"primary_domain": f"  {ra.DOMAIN_1}  ", "secondary_domain": ra.DOMAIN_2},
    })
    assert fixed["leaf"]["primary_domain"] == ra.DOMAIN_1
    assert fixed["leaf"]["secondary_domain"] == ra.DOMAIN_2


def test_fix_domain_tolerates_empty_input():
    assert ra._fix_domain(None) is None
    assert ra._fix_domain("") == ""


def test_normalize_does_not_mutate_the_input():
    original = {"leaf": {"primary_domain": f" {ra.DOMAIN_1} "}}
    ra.normalize_domain_map(original)
    assert original["leaf"]["primary_domain"] == f" {ra.DOMAIN_1} "


# ── prompt assembly ─────────────────────────────────────────────────────────

def test_user_prompt_carries_the_evidence_the_model_needs():
    prompt = ra.build_user_prompt([{
        "leaf_category": "Qubit hardware", "full_path": "A / B / Qubit hardware",
        "summary": "shared mechanism", "tech_names": ["Alpha", "Beta"],
    }])
    for expected in ("Qubit hardware", "A / B / Qubit hardware",
                     "shared mechanism", "Alpha, Beta"):
        assert expected in prompt


def test_user_prompt_survives_a_leaf_with_no_summary():
    prompt = ra.build_user_prompt([{"leaf_category": "L", "full_path": "p"}])
    assert "L" in prompt


# ── routing and export ──────────────────────────────────────────────────────

def test_cross_domain_bridge_reaches_both_reviewers(result_df, domain_map):
    files = ra.export_reviewer_files(result_df, domain_map)
    assert ra.DOMAIN_2 in files and ra.DOMAIN_4 in files

    openpyxl = pytest.importorskip("openpyxl")

    def sheet_text(domain, sheet):
        wb = openpyxl.load_workbook(io.BytesIO(files[domain]))
        return "\n".join(str(c.value) for row in wb[sheet].iter_rows()
                         for c in row if c.value)

    tech = "Optogenetic retinal restoration vector"
    assert tech in sheet_text(ra.DOMAIN_2, "טכנולוגיות")   # its home
    assert tech in sheet_text(ra.DOMAIN_4, "טכנולוגיות")   # and the referral copy
    # the referral points back home rather than pretending to own the row
    assert "הבית הראשי" in sheet_text(ra.DOMAIN_4, "טכנולוגיות")


def test_one_workbook_per_domain_with_both_sheets(result_df, domain_map):
    openpyxl = pytest.importorskip("openpyxl")
    files = ra.export_reviewer_files(result_df, domain_map)
    assert set(files) == {ra.DOMAIN_1, ra.DOMAIN_2, ra.DOMAIN_4}
    for blob in files.values():
        wb = openpyxl.load_workbook(io.BytesIO(blob))
        assert wb.sheetnames == ["סיכום עלים", "טכנולוגיות"]


def test_item_flag_is_visible_to_the_reviewer(result_df, domain_map):
    """The point of a flag is that a human sees it, so assert on the file."""
    openpyxl = pytest.importorskip("openpyxl")
    files = ra.export_reviewer_files(result_df, domain_map)
    wb = openpyxl.load_workbook(io.BytesIO(files[ra.DOMAIN_1]))
    ws = wb["טכנולוגיות"]

    flagged_rows = [r for r in ws.iter_rows()
                    if any(str(c.value or "").startswith("⚠️") for c in r)]
    assert len(flagged_rows) == 1
    names = [str(c.value) for c in flagged_rows[0] if c.value]
    assert "Diamond vacancy magnetometry array" in names


def test_mixed_group_gets_a_banner_and_no_item_flags(result_df, domain_map):
    openpyxl = pytest.importorskip("openpyxl")
    files = ra.export_reviewer_files(result_df, domain_map)
    wb = openpyxl.load_workbook(io.BytesIO(files[ra.DOMAIN_4]))

    tech_text = "\n".join(str(c.value) for row in wb["טכנולוגיות"].iter_rows()
                          for c in row if c.value)
    assert "קבוצה מעורבת" in tech_text
    # the red item flag inside a mixed group must have been suppressed
    assert "לא מתאימה בקבוצה" not in tech_text

    summary_text = "\n".join(str(c.value) for row in wb["סיכום עלים"].iter_rows()
                             for c in row if c.value)
    assert "⛔ Grab bag" in summary_text


def test_leaves_with_no_assignment_are_skipped_not_crashed(result_df):
    """A leaf the model never returned must not take the whole export down."""
    files = ra.export_reviewer_files(result_df, {})
    assert files == {} or all(isinstance(v, bytes) for v in files.values())
