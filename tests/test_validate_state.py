"""
Tests for scripts/validate_state.py — the schema/semantic validator that
gates every commit touching state.json.

These tests exercise validate_state.py's internals directly against
in-memory state dicts (no filesystem state.json involved), so they're fast
and don't touch the real project state. Run with: pytest tests/
"""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import validate_state  # noqa: E402


@pytest.fixture
def schema():
    return json.loads((ROOT / "state.schema.json").read_text())


@pytest.fixture
def valid_state():
    return json.loads((ROOT / "tests" / "fixtures" / "valid_state.json").read_text())


def run_all(state, schema):
    """Mirror validate_state.main()'s logic without touching disk."""
    structural = validate_state.validate(state, schema, schema)
    semantic = validate_state.semantic_checks(state) if not structural else []
    return structural + semantic


class TestValidFixturePasses:
    def test_valid_fixture_has_no_errors(self, valid_state, schema):
        assert run_all(valid_state, schema) == []


class TestStructuralRejections:
    def test_rejects_unknown_field(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["weeks"]["1"]["days"]["1"]["rogue_field"] = "nope"
        errors = run_all(s, schema)
        assert any("rogue_field" in e for e in errors)

    def test_rejects_bad_status_enum(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["weeks"]["1"]["days"]["1"]["status"] = "finished"
        errors = run_all(s, schema)
        assert any("status" in e and "finished" in e for e in errors)

    def test_rejects_bad_mode_enum(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["weeks"]["1"]["days"]["1"]["mode"] = "easy"
        errors = run_all(s, schema)
        assert any("mode" in e for e in errors)

    def test_rejects_bad_language_enum(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["language"] = "rust"
        errors = run_all(s, schema)
        assert any("language" in e for e in errors)

    def test_rejects_missing_required_field(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        del s["months"]["1"]["weeks"]["1"]["days"]["2"]["srs"]
        errors = run_all(s, schema)
        assert any("missing required field 'srs'" in e for e in errors)

    def test_rejects_ease_factor_below_minimum(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["weeks"]["1"]["days"]["1"]["srs"]["ease_factor"] = 1.0
        errors = run_all(s, schema)
        assert any("below minimum" in e for e in errors)

    def test_rejects_negative_time_spent(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["weeks"]["1"]["days"]["1"]["time_spent_minutes"] = -5
        errors = run_all(s, schema)
        assert any("below minimum" in e for e in errors)

    def test_rejects_wrong_type_for_tags(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["weeks"]["1"]["days"]["1"]["tags"] = "not-an-array"
        errors = run_all(s, schema)
        assert any("tags" in e for e in errors)

    def test_accepts_ease_factor_as_int(self, valid_state, schema):
        # JSON numbers that happen to be whole should still validate as "number"
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["weeks"]["1"]["days"]["1"]["srs"]["ease_factor"] = 3
        assert run_all(s, schema) == []


class TestSemanticRejections:
    def test_rejects_two_current_days(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["weeks"]["1"]["days"]["1"]["status"] = "current"
        errors = run_all(s, schema)
        assert any("exactly one day" in e for e in errors)

    def test_rejects_zero_current_days(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["weeks"]["1"]["days"]["2"]["status"] = "done"
        errors = run_all(s, schema)
        assert any("exactly one day" in e for e in errors)

    def test_rejects_current_day_mismatch(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["current_day"] = 99
        errors = run_all(s, schema)
        assert any("no day found with global_day" in e for e in errors)


class TestBadgeAndBadgeArray:
    def test_rejects_badge_missing_date(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["badges_unlocked"] = [{"id": "first-day-done"}]
        errors = run_all(s, schema)
        assert any("missing required field 'date'" in e for e in errors)


class TestGithubContribution:
    def test_null_is_valid(self, valid_state, schema):
        # fixture already has github_contribution: null on every day
        assert run_all(valid_state, schema) == []

    def test_accepts_valid_active_contribution(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["weeks"]["1"]["days"]["1"]["github_contribution"] = {
            "status": "in_progress",
            "repo": "psf/requests",
            "issue_url": "https://github.com/psf/requests/issues/1",
            "issue_title": "Fix thing",
            "pr_url": None,
        }
        assert run_all(s, schema) == []

    def test_rejects_bad_status_enum(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["weeks"]["1"]["days"]["1"]["github_contribution"] = {
            "status": "merged",  # not a real value
            "repo": "psf/requests",
            "issue_url": "https://github.com/psf/requests/issues/1",
            "issue_title": "Fix thing",
            "pr_url": None,
        }
        errors = run_all(s, schema)
        assert any("status" in e for e in errors)

    def test_rejects_missing_pr_url(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["weeks"]["1"]["days"]["1"]["github_contribution"] = {
            "status": "in_progress",
            "repo": "psf/requests",
            "issue_url": "https://github.com/psf/requests/issues/1",
            "issue_title": "Fix thing",
        }
        errors = run_all(s, schema)
        assert any("missing required field 'pr_url'" in e for e in errors)

    def test_rejects_extra_field(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        s["months"]["1"]["weeks"]["1"]["days"]["1"]["github_contribution"] = {
            "status": "in_progress",
            "repo": "psf/requests",
            "issue_url": "https://github.com/psf/requests/issues/1",
            "issue_title": "Fix thing",
            "pr_url": None,
            "extra": "nope",
        }
        errors = run_all(s, schema)
        assert any("extra" in e for e in errors)

    def test_rejects_day_missing_field_entirely(self, valid_state, schema):
        s = copy.deepcopy(valid_state)
        del s["months"]["1"]["weeks"]["1"]["days"]["2"]["github_contribution"]
        errors = run_all(s, schema)
        assert any("missing required field 'github_contribution'" in e for e in errors)


class TestNoStateFileIsNotAnError:
    def test_main_handles_missing_state_gracefully(self, tmp_path, monkeypatch):
        # Point the module at an empty temp dir with no state.json
        monkeypatch.setattr(validate_state, "STATE_PATH", tmp_path / "state.json")
        monkeypatch.setattr(validate_state, "SCHEMA_PATH", ROOT / "state.schema.json")
        assert validate_state.main() == 0
