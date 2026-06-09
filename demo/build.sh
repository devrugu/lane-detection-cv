#!/usr/bin/env bash
# Build the Lane Detection demo into a single executable.
#
# Runs on whatever OS invokes it:
#   • Windows runner  -> LaneDetectionDemo.exe   (the distributable demo)
#   • Linux/macOS     -> LaneDetectionDemo        (handy for local testing)
#
# Usage:  ./demo/build.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo ">> Installing build dependencies"
python -m pip install --upgrade pip
python -m pip install -r demo/requirements-demo.txt
python -m pip install pyinstaller

echo ">> Cleaning previous build"
rm -rf build dist

echo ">> Building executable"
pyinstaller demo/LaneDetectionDemo.spec --noconfirm

echo ">> Done. Artifact(s):"
ls -lh dist/