# Releasing the DataBubble SDK

The publish workflow (`.github/workflows/publish.yml`) is inert until the
one-time account-level setup below is done. It uses **PyPI Trusted
Publishing (OIDC)** — no API token is ever stored in this repo or in GitHub
Actions.

## The scripted flow (recommended)

Mirrors the main app's `code/scripts/deploy_test.sh` / `promote_demo.sh`
pattern: local dev → a cheap, repeatable "test" stage → a manually-gated
real release. Nothing here fires automatically on `git push` — same
deliberate choice as the main app, for the same reason (a bad merge
shouldn't be able to reach anyone).

1. **`scripts/check_local.sh`** — build the sdist/wheel, run the test suite,
   validate the artifacts (`twine check`), and check the SDK hasn't drifted
   behind the app's `/v1/journeys/*` routes again
   (`scripts/check_journey_drift.py` — this is the exact class of gap that
   motivated writing these scripts; see CHANGELOG.md's 2026-08-16 entry).
   Pure local, touches no remote.
2. **`scripts/publish_testpypi.sh`** — uploads the *current local build* to
   test.pypi.org (needs a one-time TestPyPI API token, see the script's own
   header comment) and verifies it by installing into a throwaway venv and
   importing it. Deliberately does not use the workflow's own
   `workflow_dispatch`/`testpypi` dry-run path, since that builds from
   whatever's already pushed to GitHub, not your local changes.
3. **`scripts/publish_release.sh`** — the real release. Verifies version
   consistency and a clean tree, requires typing `PROMOTE`, then pushes
   `main`, tags `vX.Y.Z`, and creates the GitHub Release — publishing that
   release is what triggers the real PyPI upload via OIDC. This is the one
   step that's genuinely irreversible-ish (public repo, public package
   index), which is why it's the only one gated behind a typed confirmation.

The manual steps below are what these scripts automate — read them if you
want to understand exactly what's happening, if a script fails partway and
you need to finish by hand, or if you're setting up the one-time account
config for the first time.

## One-time setup (do this once, before the first release)

1. **Create the PyPI project.** Reserve the `databubble` name on
   [pypi.org](https://pypi.org) if not already done. If the name is taken,
   pick a different distribution name and update `pyproject.toml`'s
   `[project].name` before continuing — the workflow publishes whatever
   `python -m build` produces from that field.
2. **Configure the Trusted Publisher.** On the PyPI project's
   "Publishing" settings, add a new trusted publisher:
   - Owner: `databubble-ai`
   - Repository: `databubble-python`
   - Workflow filename: `publish.yml`
   - Environment name: `pypi`
3. **Repeat for TestPyPI** (optional, only needed for dry-runs), using
   environment name `testpypi` on [test.pypi.org](https://test.pypi.org).
4. **Create the GitHub environments.** In the repo's Settings → Environments,
   create `pypi` and (optionally) `testpypi`. No secrets need to be added to
   either — Trusted Publishing authenticates via OIDC, not a stored value.
   Adding required reviewers on the `pypi` environment is recommended so a
   real publish needs explicit approval.

## Cutting a release manually (what `publish_release.sh` does)

1. Bump `version` in `pyproject.toml` and `__version__` in
   `databubble/__init__.py` together — keep them in sync.
2. Update `README.md`'s version reference and any changed install
   instructions (drop the "Pre-release" note once the package is actually
   on PyPI).
3. Commit, tag (`git tag v0.X.0`), and push the tag.
4. Create a GitHub Release from that tag. **Publishing the release is what
   triggers the workflow** — nothing fires on a plain tag push or on `git
   push` to `main`.
5. Watch the Actions run: `build` job produces the sdist+wheel, then
   `publish-pypi` runs (gated on the `pypi` environment, using OIDC — no
   token needed) and uploads to PyPI.

## Verifying a release

- Check the workflow run in the Actions tab completed successfully.
- Check [pypi.org/project/databubble](https://pypi.org/project/databubble)
  shows the new version.
- In a clean environment: `pip install databubble==<version>` and run a
  quick smoke check (`from databubble import DataBubble`).

## Dry-running against TestPyPI

**Preferred: `scripts/publish_testpypi.sh`** — builds and uploads your
current local tree, so it actually tests what you just wrote rather than
whatever's already on GitHub.

**Alternative (tests already-pushed `main`, not local changes):** run the
workflow manually (Actions tab → "Publish to PyPI" → "Run workflow"), typing
`testpypi` into the confirmation field. This builds and publishes to
test.pypi.org only — it never touches the real PyPI index, and requires the
explicit confirmation string so it can't fire by accident from a blank
manual trigger.

## What the workflow deliberately does NOT do

- Does not run on `push` to any branch — only on a published release, or an
  explicitly-confirmed manual TestPyPI dry-run.
- Does not store or reference any PyPI API token — Trusted Publishing (OIDC)
  is the only auth mechanism.
- Does not make the GitHub repository public, or decide the release
  version — those are your calls, made before tagging.
