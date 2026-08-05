#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/orbslam_ws/src/ORB_SLAM3/Vocabulary"
archive="$dest/ORBvoc.txt.tar.gz"
url="https://raw.githubusercontent.com/UZ-SLAMLab/ORB_SLAM3/master/Vocabulary/ORBvoc.txt.tar.gz"
archive_sha="ff2d0e82a69a8f4c5c002e3a0dff82a00e5872e1659fac1b996f41166b92693b"
vocab_sha="f8dd027f7a6cb88129821341194d7f2c75b77b3394257ddd0d2229863d1a3570"

mkdir -p "$dest"
curl --fail --location --retry 3 "$url" --output "$archive"
printf '%s  %s\n' "$archive_sha" "$archive" | sha256sum --check --status || {
  echo "ERROR: ORB vocabulary archive checksum mismatch" >&2
  exit 1
}
tar -xzf "$archive" -C "$dest"
printf '%s  %s\n' "$vocab_sha" "$dest/ORBvoc.txt" | sha256sum --check
