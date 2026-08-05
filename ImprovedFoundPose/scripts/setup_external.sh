#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
external_dir="${root_dir}/external"
mkdir -p "${external_dir}"

install_at_revision() {
    local url="$1"
    local revision="$2"
    local destination="$3"

    if [[ -d "${destination}/.git" ]]; then
        current_revision="$(git -C "${destination}" rev-parse HEAD)"
        if [[ "${current_revision}" == "${revision}" ]]; then
            echo "Already installed at ${revision}: ${destination}"
            return 0
        fi
        echo "Existing checkout has unexpected revision: ${destination}" >&2
        return 1
    fi
    if [[ -e "${destination}" ]]; then
        echo "Existing path is not a Git checkout: ${destination}" >&2
        return 1
    fi
    git clone --filter=blob:none "${url}" "${destination}"
    git -C "${destination}" checkout --detach "${revision}"
}

install_at_revision \
    "https://github.com/thodan/bop_toolkit.git" \
    "e7ba9f2389d48d721856df8147fb133c0145e99c" \
    "${external_dir}/bop_toolkit"
install_at_revision \
    "https://github.com/facebookresearch/dinov2.git" \
    "e1277af2ba9496fbadf7aec6eba56e8d882d1e35" \
    "${external_dir}/dinov2"

echo "Installed pinned external dependencies under ${external_dir}"
