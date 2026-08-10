#!/usr/bin/env bash

set -euo pipefail

afs_root=/mnt/afs_fangwenqi
project_root="$afs_root/Lumina-T2X"
python_bin="$afs_root/miniconda3/envs/lum/bin/python"
blender_bin="$afs_root/blender/blender"
script_file="$project_root/tools/tokenlight_data/render_assets.py"
config_file="$project_root/lumina_next_t2i/config.yaml"
log_file="$afs_root/render_assets.log"
blender_deps="$afs_root/blender-deps/usr/lib/x86_64-linux-gnu"

export LD_LIBRARY_PATH="$blender_deps:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

echo "===== system dependency verification ====="
if ! ldconfig -p 2>/dev/null | awk '$1 == "libX11.so.6" { found = 1 } END { exit !found }'; then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "错误：系统缺少 libX11.so.6，且未找到 apt-get。" >&2
        exit 1
    fi

    if [[ $EUID -eq 0 ]]; then
        apt-get update
        apt-get install -y libx11-6
    elif command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y libx11-6
    else
        echo "错误：安装 libx11-6 需要 root 权限，但未找到 sudo。" >&2
        exit 1
    fi
fi

echo "===== storage verification ====="
df -h "$afs_root"
test -x "$python_bin"
test -x "$blender_bin"
test -f "$script_file"
test -f "$config_file"

echo "===== Blender dependency verification ====="
missing_libs="$(ldd "$blender_bin" | awk '/not found/ { print }')"
if [[ -n "$missing_libs" ]]; then
    echo "Blender 仍缺少以下动态库：" >&2
    echo "$missing_libs" >&2
    exit 1
fi
"$blender_bin" --version

echo "===== runtime verification ====="
"$python_bin" --version
"$python_bin" - <<'PY'
import os
import sys
import torch

print("python=", sys.executable)
print("cwd=", os.getcwd())
print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
PY

echo "===== render assets ====="
cd "$project_root"

"$python_bin" "$script_file" \
    --config "$config_file" \
    2>&1 | tee "$log_file"
