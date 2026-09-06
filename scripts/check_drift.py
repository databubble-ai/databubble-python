#!/usr/bin/env python3
"""
scripts/check_drift.py

Diffs the main app's API surface against the SDK's wrapper methods — journeys
AND skills. Supersedes check_journey_drift.py, which covered journeys only and
therefore never noticed that `transformations` (7th entry in SKILL_REGISTRY)
had no SDK method at all.

Two modes:

1. Manifest mode (works in CI — preferred).
   Reads api_manifest.json from the SDK repo root. The app repo generates it
   (see platform/TRACK-B-SPEC.md, task B4) and it is committed here, so this
   check runs on every PR without a private checkout.

2. Sibling-checkout mode (local fallback, unchanged behaviour).
   Parses the app repo's source directly. Assumes:
     databubble/
       code/   <- the main app
       sdk/    <- this repo
   Override with DATABUBBLE_CODE_REPO.

Exit codes: 0 = in sync (or nothing to compare), 1 = drift detected.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parent.parent
CODE_REPO = Path(os.environ.get("DATABUBBLE_CODE_REPO", SDK_ROOT.parent / "code"))
MANIFEST = SDK_ROOT / "api_manifest.json"

ROUTE_FILE = CODE_REPO / "api" / "routes" / "journeys.py"
DISPATCH_FILE = CODE_REPO / "skills" / "dispatch.py"

SDK_JOURNEYS = SDK_ROOT / "databubble" / "journeys.py"
SDK_SKILLS = SDK_ROOT / "databubble" / "skills.py"

ROUTE_RE = re.compile(r'@journeys_router\.post\("/journeys/([a-z_]+)"')
REGISTRY_RE = re.compile(r'^\s*"([a-z_]+)":\s*_run_[a-z_]+,\s*$', re.MULTILINE)
SDK_JOURNEY_RE = re.compile(r'self\._call\("([a-z_]+)"')
# Matches both self._call("slug", ...) and self._call_first_output("slug", ...)
# — bivariate() uses the latter (skills/skills.py:_parse_skill_result_first_output)
# since bivariate_ts always returns an outputs[] list server-side.
SDK_SKILL_RE = re.compile(r'self\._call(?:_first_output)?\("([a-z_]+)"')


def _server_surface() -> tuple[set[str], set[str], str]:
    if MANIFEST.exists():
        data = json.loads(MANIFEST.read_text())
        return set(data.get("journeys", [])), set(data.get("skills", [])), f"manifest {MANIFEST.name}"

    if not ROUTE_FILE.exists():
        return set(), set(), ""

    journeys = set(ROUTE_RE.findall(ROUTE_FILE.read_text()))
    skills: set[str] = set()
    if DISPATCH_FILE.exists():
        text = DISPATCH_FILE.read_text()
        block = text.split("SKILL_REGISTRY", 1)[-1].split("}", 1)[0]
        skills = set(REGISTRY_RE.findall(block))
    return journeys, skills, f"source checkout {CODE_REPO}"


def _report(kind: str, server: set[str], wrapped: set[str]) -> int:
    missing = sorted(server - wrapped)
    extra = sorted(wrapped - server)
    if not missing and not extra:
        print(f"OK: SDK wraps all {len(server)} {kind}.")
        return 0
    if missing:
        print(f"MISSING from SDK ({len(missing)} {kind}): {', '.join(missing)}")
        print(f"  -> add a wrapper method for each.")
    if extra:
        print(f"WRAPPED but not a live {kind[:-1]} ({len(extra)}): {', '.join(extra)}")
        print("  -> renamed or removed on the app side; update or remove the wrapper.")
    return 1


def main() -> int:
    journeys, skills, source = _server_surface()
    if not source:
        print(
            f"SKIP: no {MANIFEST.name} and can't find {ROUTE_FILE}. Commit the "
            "manifest, or set DATABUBBLE_CODE_REPO to the app repo's path."
        )
        return 0

    print(f"Comparing against: {source}")
    status = 0

    wrapped_journeys = set(SDK_JOURNEY_RE.findall(SDK_JOURNEYS.read_text()))
    status |= _report("journeys", journeys, wrapped_journeys)

    if skills:
        wrapped_skills = set(SDK_SKILL_RE.findall(SDK_SKILLS.read_text()))
        status |= _report("skills", skills, wrapped_skills)
    else:
        print("SKIP: no skill registry found to compare.")

    return status


if __name__ == "__main__":
    sys.exit(main())
