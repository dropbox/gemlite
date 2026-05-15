#!/usr/bin/env bash
# Install/uninstall the gemlite autopatch hook in vllm/engine/__init__.py.
# Needed for use with async engines that fork worker processes — those
# subprocesses re-import vllm, and this hook makes every re-import run
# gemlite.vllm.patch_vllm() which reads env vars.
#
# Usage:
#   bash patch.sh install [--vllm-engine-init /path/to/vllm/engine/__init__.py]
#   bash patch.sh uninstall
#   bash patch.sh status
#
# Then:
#   VLLM_GEMLITE_ENABLE=1 \
#   VLLM_GEMLITE_ONTHEFLY_QUANT=A8W8_FP8_DYNAMIC_BLOCK \
#   vllm serve ...

set -euo pipefail

HOOK_BEGIN="# --- gemlite autopatch ---"
HOOK_END="# --- end gemlite autopatch ---"

default_engine_init() {
  python3 -c "import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), 'engine', '__init__.py'))"
}

cmd="${1:-}"
shift || true

engine_init=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm-engine-init) engine_init="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done
if [[ -z "$engine_init" ]]; then
  engine_init="$(default_engine_init)"
fi

hook_body=$(cat <<'EOF'
# --- gemlite autopatch ---
import os as _os
if _os.environ.get("VLLM_GEMLITE_ENABLE", "0") != "0" or _os.environ.get(
    "VLLM_GEMLITE_ONTHEFLY_QUANT"
):
    try:
        from gemlite.vllm import patch_vllm as _patch_vllm
        _patch_vllm()
    except Exception as _e:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "gemlite autopatch failed: %s", _e
        )
# --- end gemlite autopatch ---
EOF
)

case "$cmd" in
  install)
    if [[ ! -f "$engine_init" ]]; then
      echo "engine init not found: $engine_init" >&2; exit 1
    fi
    if grep -qF "$HOOK_BEGIN" "$engine_init"; then
      echo "already installed in $engine_init"; exit 0
    fi
    printf "\n%s\n" "$hook_body" >> "$engine_init"
    echo "installed autopatch hook into $engine_init"
    ;;

  uninstall)
    if [[ ! -f "$engine_init" ]]; then
      echo "engine init not found: $engine_init" >&2; exit 1
    fi
    python3 - "$engine_init" "$HOOK_BEGIN" "$HOOK_END" <<'PY'
import sys, re
path, begin, end = sys.argv[1:4]
with open(path) as f: src = f.read()
new = re.sub(re.escape(begin) + r".*?" + re.escape(end) + r"\n?", "", src, flags=re.DOTALL)
with open(path, "w") as f: f.write(new)
print(f"removed autopatch hook from {path}")
PY
    ;;

  status)
    if grep -qF "$HOOK_BEGIN" "$engine_init"; then
      echo "gemlite autopatch: INSTALLED in $engine_init"
    else
      echo "gemlite autopatch: not installed ($engine_init)"
    fi
    ;;

  *)
    echo "usage: $0 {install|uninstall|status} [--vllm-engine-init PATH]" >&2
    exit 1;;
esac
