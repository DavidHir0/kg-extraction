#!/usr/bin/env bash
#
# Download the two binary assets the upstream commands need.
#
#   scripts/fetch_assets.sh            both
#   scripts/fetch_assets.sh jar        pdffigures2 only
#   scripts/fetch_assets.sh weights    classifier only
#
# Neither is in git: the checkpoint is 91 MB, past GitHub's 100 MB file limit
# and unwelcome in every clone's history regardless. They are release assets,
# fetched on demand into models/.
#
# NOTHING in the benchmark needs either of these. The 73 golden figures are
# already in data/golden_dataset/; `generate` and the benchmark start there.
# These two are the upstream stages that turn PDFs into figures.
#
# Every download is checksummed. An earlier version of this script tested only
# whether the file EXISTED, so an interrupted download left a truncated file
# that every later run happily skipped as "already present" -- and torch then
# failed on a corrupt checkpoint far from the cause.

set -euo pipefail

REPO="DavidHir0/kg-extraction"
TAG="assets-v1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JAR_SHA="7ce481df8f814cc877129cc1bb6d8cc1fc039249521ab5c6c19f6fe1ce6592a3"
PTH_SHA="062ca6f653e39d3a3495c654da6569bf37b3f1fbf87dda66a0691bf561827b42"

want="${1:-all}"

verify() {  # path expected_sha -> 0 if the file is present AND complete
  [ -f "$1" ] || return 1
  [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]
}

fetch() {  # asset dest_dir expected_sha
  local asset="$1" dir="$2" sha="$3" dest="$2/$1"
  mkdir -p "$dir"

  if verify "$dest" "$sha"; then
    echo "already present and verified: $dest"
    return
  fi
  if [ -f "$dest" ]; then
    echo "re-downloading $asset (present but checksum does not match)"
    rm -f "$dest"
  else
    echo "downloading $asset -> $dest"
  fi

  # Download to a temp name and move into place only once verified, so an
  # interrupted run can never leave a file that looks finished.
  local tmp="${dest}.part"
  rm -f "$tmp"
  if command -v gh >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$asset" \
      --output "$tmp" --clobber
  else
    curl -fL --progress-bar \
      "https://github.com/$REPO/releases/download/$TAG/$asset" -o "$tmp"
  fi

  if ! verify "$tmp" "$sha"; then
    rm -f "$tmp"
    echo "ERROR: $asset failed its checksum. Download interrupted, or the" >&2
    echo "release asset changed. Re-run; if it persists, open an issue." >&2
    exit 1
  fi
  mv -f "$tmp" "$dest"
  echo "verified: $dest"
}

if [ "$want" = "all" ] || [ "$want" = "jar" ]; then
  # pdffigures2, (c) Allen Institute for AI, Apache-2.0. Built from source with
  # `sbt assembly` because upstream publishes no release and is not on Maven
  # Central. It bundles other libraries; models/pdffigures2/NOTICE.md lists them
  # with their licences, and the licence texts are in models/pdffigures2/licenses/.
  fetch "pdffigures2.jar" "$ROOT/models/pdffigures2" "$JAR_SHA"
fi

if [ "$want" = "all" ] || [ "$want" = "weights" ]; then
  # ResNet50 fine-tuned on the ACL figures dataset, 18 figure classes
  # (models/classification/classes.txt). Needs the `classify` extra for torch:
  #   uv sync --extra classify
  fetch "resnet50_vanilla.pth" "$ROOT/models/classification" "$PTH_SHA"
fi

echo
echo "done. models/ now holds:"
find "$ROOT/models" -type f \( -name '*.jar' -o -name '*.pth' \) -exec ls -lh {} \; \
  | awk '{print "  " $5 "  " $9}'
