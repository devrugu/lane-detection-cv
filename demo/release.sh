#!/usr/bin/env bash
# Trigger a fresh Windows .exe build from Codespaces (or any machine with git).
#
# What it does:
#   1. Commits any pending changes (so the build reflects your latest code).
#   2. Creates and pushes a version tag.
#   3. GitHub Actions picks up the tag, builds LaneDetectionDemo.exe on a
#      Windows runner, and publishes it to the repo's Releases page.
#
# Usage:
#   ./demo/release.sh v1.0.0
#   ./demo/release.sh            # auto-bumps the patch of the latest vX.Y.Z tag
#
# After it finishes, watch progress at:  <repo>/actions
# Download the exe from:                 <repo>/releases
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# --- figure out the version ---
if [[ $# -ge 1 ]]; then
  VERSION="$1"
else
  LAST="$(git tag --list 'v*' --sort=-v:refname | head -n1)"
  if [[ -z "$LAST" ]]; then
    VERSION="v0.1.0"
  else
    base="${LAST#v}"
    IFS='.' read -r MA MI PA <<< "$base"
    VERSION="v${MA}.${MI}.$((PA + 1))"
  fi
fi
echo ">> Releasing $VERSION"

# --- commit pending work so the build uses current code ---
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo ">> Committing pending changes"
  git add -A
  git commit -m "Build $VERSION"
fi

git push origin HEAD

# --- tag and push ---
if git rev-parse "$VERSION" >/dev/null 2>&1; then
  echo "!! Tag $VERSION already exists. Pass a new version, e.g. ./demo/release.sh v1.2.3"
  exit 1
fi
git tag "$VERSION"
git push origin "$VERSION"

REMOTE="$(git config --get remote.origin.url | sed -E 's#git@github.com:#https://github.com/#; s#\.git$##')"
echo
echo ">> Pushed tag $VERSION."
echo ">> Build progress:  ${REMOTE}/actions"
echo ">> Download exe at:  ${REMOTE}/releases/tag/${VERSION}  (ready in ~3-5 min)"