#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The upstream repository contains the runnable project in its SAM-6D/
# subdirectory. Clone the repository root as sam6d_master so the wrapper sees
# sam6d_master/SAM-6D exactly once.
dest="${1:-$root/sam6d_ws/sam6d_master}"
repo="https://github.com/JiehongLin/SAM-6D.git"
revision="1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2"

if [[ -e "$dest" ]]; then
  echo "ERROR: destination already exists: $dest" >&2
  exit 1
fi
parent="$(dirname "$dest")"
mkdir -p "$parent"
temp_checkout="$(mktemp -d "$parent/.sam6d-install.XXXXXX")"
trap 'rm -rf -- "$temp_checkout"' EXIT
git clone --filter=blob:none --no-checkout "$repo" "$temp_checkout"
git -C "$temp_checkout" checkout --detach "$revision"
mv "$temp_checkout" "$dest"
trap - EXIT
echo "Installed SAM-6D at pinned revision $revision"
echo "Review upstream terms and run its prepare.sh to download checkpoints."
