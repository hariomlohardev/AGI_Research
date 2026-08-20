#!/usr/bin/env python3
"""
Validate state.json against state.schema.json, plus a few cross-field
semantic checks that a JSON Schema can't express on its own.

No external dependencies (stdlib only) — deliberately, so this runs
anywhere pytest can run, with no extra `pip install` required.

Usage: python3 scripts/validate_state.py
Exit code 0 = valid, 1 = invalid (or nothing to validate yet).

Every skill that writes state.json should run this before committing.
It is also run automatically by .githooks/pre-commit on any commit that
stages state.json.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "state.json"
SCHEMA_PATH = ROOT / "state.schema.json"


# ---------------------------------------------------------------------------
# Minimal structural validator: supports exactly the JSON Schema subset used
# in state.schema.json (type incl. nullable, enum, object/array, integer,
# number, required, properties, additionalProperties: false,
# patternProperties, items, minimum, and local $ref into #/definitions/<name>).
# ---------------------------------------------------------------------------

def resolve_ref(ref, schema_root):
    assert ref.startswith("#/"), f"only local refs supported, got {ref}"
    node = schema_root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def check_type(value, t):
    if t == "null":
        return value is None
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, list)
    return False


def validate(instance, schema, schema_root, path="state"):
    errors = []

    if "$ref" in schema:
        schema = resolve_ref(schema["$ref"], schema_root)

    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(check_type(instance, t) for t in types):
            errors.append(
                f"{path}: expected type {schema['type']}, got "
                f"{type(instance).__name__} ({instance!r})"
            )
            return errors  # can't usefully recurse further

    if "enum" in schema:
        if instance not in schema["enum"]:
            errors.append(f"{path}: value {instance!r} not in allowed enum {schema['enum']}")

    if "minimum" in schema and isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if instance < schema["minimum"]:
            errors.append(f"{path}: value {instance} is below minimum {schema['minimum']}")

    if isinstance(instance, dict):
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for r in required:
            if r not in instance:
                errors.append(f"{path}: missing required field '{r}'")

        additional_allowed = schema.get("additionalProperties", True)
        pattern_props = schema.get("patternProperties", {})

        for key, val in instance.items():
            subschema = None
            if key in props:
                subschema = props[key]
            else:
                for pat, ps in pattern_props.items():
                    if re.match(pat, key):
                        subschema = ps
                        break
            if subschema is not None:
                errors.extend(validate(val, subschema, schema_root, f"{path}.{key}"))
            elif additional_allowed is False:
                errors.append(f"{path}.{key}: unexpected field not defined in state.schema.json")

    elif isinstance(instance, list):
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(instance):
                errors.extend(validate(item, item_schema, schema_root, f"{path}[{i}]"))

    return errors


# ---------------------------------------------------------------------------
# Semantic checks JSON Schema can't express: current_day/current_month
# consistency, exactly one "current" day.
# ---------------------------------------------------------------------------

def semantic_checks(state):
    errors = []
    cm = str(state.get("current_month"))
    cd = state.get("current_day")
    months = state.get("months", {})

    if cm not in months:
        errors.append(f"current_month {cm!r} has no matching entry under 'months'")
        return errors

    month = months[cm]
    current_days = []
    target_day = None

    for wnum, week in month.get("weeks", {}).items():
        for dnum, day in week.get("days", {}).items():
            if not isinstance(day, dict):
                continue
            if day.get("status") == "current":
                current_days.append(f"week {wnum} day {dnum}")
            if day.get("global_day") == cd:
                target_day = (wnum, dnum, day)

    if len(current_days) != 1:
        errors.append(
            f"expected exactly one day with status 'current' in month {cm}, "
            f"found {len(current_days)}: {current_days}"
        )

    if target_day is None:
        errors.append(f"no day found with global_day == current_day ({cd}) in month {cm}")
    else:
        wnum, dnum, day = target_day
        if day.get("status") != "current":
            errors.append(
                f"day at global_day {cd} (week {wnum} day {dnum}) has status "
                f"{day.get('status')!r}, expected 'current' (it matches current_day)"
            )

    return errors


def main():
    if not STATE_PATH.exists():
        print("No state.json found yet — nothing to validate.")
        return 0

    if not SCHEMA_PATH.exists():
        print(f"ERROR: {SCHEMA_PATH} not found — cannot validate.")
        return 1

    state = json.loads(STATE_PATH.read_text())
    schema_root = json.loads(SCHEMA_PATH.read_text())

    structural_errors = validate(state, schema_root, schema_root)
    semantic_errors = semantic_checks(state) if not structural_errors else []
    all_errors = structural_errors + semantic_errors

    if not all_errors:
        print("state.json is valid against state.schema.json (structural + semantic checks passed).")
        return 0

    print(f"state.json FAILED validation — {len(all_errors)} error(s):")
    for e in all_errors:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
