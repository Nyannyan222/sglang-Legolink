#!/usr/bin/env bash
# Rebuild the patched SGLang runtime from upstream plus the shipped patch series, and prove it is the runtime this repository was written against.
#
#   bash scripts/build_patched_sglang.sh <target-dir>
#
# Needs git and network access to the upstream repository (override with SGLANG_UPSTREAM, e.g. a local mirror). Nothing outside <target-dir> is
# touched, and an existing <target-dir> is never overwritten. The result is a git checkout on branch noc/legolink-pic-runtime whose tree id must
# equal PATCHED_SGLANG.json -> patched_tree; otherwise the script exits 2 and the build must not be used.
#
# Afterwards:  pip install -e <target-dir>/python -c configs/constraints-runtime.txt   (see docs/INSTALLATION.md)
#              python scripts/verify_runtime.py --sglang-dir <target-dir>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TARGET="${1:?usage: build_patched_sglang.sh <target-dir>}"
UPSTREAM="${SGLANG_UPSTREAM:-https://github.com/sgl-project/sglang}"
PYTHON="${PYTHON:-python3}"
META="$REPO_ROOT/PATCHED_SGLANG.json"
SPEC="$REPO_ROOT/UPSTREAM_SGLANG.json"

field() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' "$1" "$2"; }
BASE="$(field "$SPEC" base_commit_full)"
EXPECTED_TREE="$(field "$META" patched_tree)"
PATCH_COUNT="$(field "$META" patch_count)"

mapfile -t PATCHES < <(ls "$REPO_ROOT"/external_patches/sglang/*.patch | LC_ALL=C sort)
[[ "${#PATCHES[@]}" -eq "$PATCH_COUNT" ]] || { echo "ERROR: expected $PATCH_COUNT patches, found ${#PATCHES[@]}" >&2; exit 1; }
[[ ! -e "$TARGET" ]] || { echo "ERROR: $TARGET exists; refusing to overwrite" >&2; exit 1; }

git clone --quiet "$UPSTREAM" "$TARGET"
cd "$TARGET"
git checkout --quiet "$BASE"
git checkout --quiet -b noc/legolink-pic-runtime
# a throw-away committer identity: `git am` needs one, and the tree id (what is verified) does not depend on it
git -c user.name="legolink-build" -c user.email="legolink-build@localhost.invalid" am --quiet "${PATCHES[@]}"

TREE="$(git rev-parse 'HEAD^{tree}')"
echo "base commit:   $BASE"
echo "patched tree:  $TREE"
echo "expected tree: $EXPECTED_TREE"
if [[ "$TREE" != "$EXPECTED_TREE" ]]; then
  echo "ERROR: the rebuilt runtime is not byte-identical to the one this repository was written against; do not use it" >&2
  exit 2
fi
echo "OK: patched SGLang rebuilt at $TARGET (tree id matches)."
echo "Next: pip install -e '$TARGET/python' -c '$REPO_ROOT/configs/constraints-runtime.txt'"
