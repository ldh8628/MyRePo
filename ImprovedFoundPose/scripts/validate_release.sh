#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"
python_bin="${PYTHON:-python3}"
scan_tmp="$(mktemp -d)"
trap 'rm -rf -- "${scan_tmp}"' EXIT
export PYTHONPYCACHEPREFIX="${scan_tmp}/pycache"

required_files=(
    README.md NOTICE.md SHARE_MANIFEST.md LICENSE .gitignore
    conda_foundpose_gpu.yaml run_pipeline.py
    scripts/gen_templates.py scripts/gen_repre.py scripts/infer.py
    scripts/prepare_bop_submission.py scripts/setup_external.sh
    tools/convert_ply_gs2mesh_to_foundpose.py
    configs/gen_templates/lmo.json configs/gen_repre/lmo.json configs/infer/lmo.json
)
for path in "${required_files[@]}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done

forbidden_dirs='/(3DGS|GS2Mesh|research_pose|select_topk|scripts/research|configs/research|_bmad[^/]*)($|/)'
if find . -path ./external -prune -o -type d -print | grep -E "${forbidden_dirs}" >/dev/null; then
    echo "Forbidden directory found:" >&2
    find . -path ./external -prune -o -type d -print | grep -E "${forbidden_dirs}" >&2
    exit 1
fi

forbidden_ext='\.(pth|pt|ckpt|onnx|safetensors|ply|obj|stl|npz|npy|csv|html|log|png|jpe?g)$'
if find . -path ./external -prune -o -type f -print | grep -Ei "${forbidden_ext}" >/dev/null; then
    echo "Forbidden data/model file found:" >&2
    find . -path ./external -prune -o -type f -print | grep -Ei "${forbidden_ext}" >&2
    exit 1
fi

if find . -path ./external -prune -o -type f -size +99M -print | grep -q .; then
    echo "File at or above 100 MB found:" >&2
    find . -path ./external -prune -o -type f -size +99M -print >&2
    exit 1
fi

runtime_files=(run_pipeline.py scripts tools utils configs)
if grep -RInE --exclude-dir=__pycache__ --exclude=validate_release.sh \
    '(/home/etri/|/Users/[^/]+/|[A-Za-z]:\\Users\\|FOUNDPOSE_AUDIT_EXPECTED)' \
    "${runtime_files[@]}" >"${scan_tmp}/path_scan.txt"; then
    echo "Internal absolute path or audit hook found:" >&2
    sed -n '1,80p' "${scan_tmp}/path_scan.txt" >&2
    exit 1
fi

if grep -RInE --exclude-dir=__pycache__ --exclude-dir=external --exclude=validate_release.sh \
    '(AWS_SECRET_ACCESS_KEY|BEGIN (RSA |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,})' \
    . >"${scan_tmp}/secret_scan.txt"; then
    echo "Potential secret found:" >&2
    sed -n '1,80p' "${scan_tmp}/secret_scan.txt" >&2
    exit 1
fi

"${python_bin}" -m compileall -q run_pipeline.py scripts tools utils tests
"${python_bin}" run_pipeline.py --help >/dev/null
"${python_bin}" scripts/prepare_bop_submission.py --help >/dev/null
"${python_bin}" tools/convert_ply_gs2mesh_to_foundpose.py --help >/dev/null
"${python_bin}" -m pytest -q -p no:cacheprovider

repo_size_kib="$(du -sk --exclude=external --exclude=.git . | awk '{print $1}')"
if (( repo_size_kib > 10240 )); then
    echo "Release is larger than 10 MiB: ${repo_size_kib} KiB" >&2
    exit 1
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    tracked="$(git ls-files -- .)"
    if printf '%s\n' "${tracked}" | grep -Ei \
        '(^|/)(data|datasets|inputs?|outputs?|3DGS|GS2Mesh|_bmad)(/|$)|\.(pth|pt|ckpt|ply)$' \
        >/dev/null; then
        echo "Forbidden Git-tracked path found" >&2
        exit 1
    fi
    echo "Git tracked-file scan: PASS"
else
    echo "Git tracked-file scan: SKIP (not inside a Git worktree)"
fi

echo "Release validation: PASS (${repo_size_kib} KiB)"
