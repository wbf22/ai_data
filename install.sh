#!/usr/bin/env bash
# Install the aidata opencode capture pipeline on a machine.
#
# What it does:
#   1. (optional) copies this repo to --prefix
#   2. installs python + plugin dependencies
#   3. symlinks the opencode plugin into the global config (and/or projects)
#   4. drops an `aidata` wrapper in ~/.local/bin
#
# Usage:
#   ./install.sh [--prefix DIR] [--no-global] [--project DIR]... [--uninstall]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
CAPTURE_DIR="${AIDATA_CAPTURE_DIR:-$HOME/.local/share/opencode/aidata-capture}"
BIN_DIR="$HOME/.local/bin"
PLUGIN_REL=".opencode/plugin/aidata-capture.ts"

PREFIX=""
DO_GLOBAL=1
UNINSTALL=0
PROJECTS=()

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Install the aidata opencode capture pipeline on a machine.

Steps: (optional) copy to --prefix, install python + plugin dependencies,
symlink the opencode plugin into the global config and/or projects, and drop
an `aidata` wrapper in ~/.local/bin.

Usage:
  ./install.sh [--prefix DIR] [--no-global] [--project DIR]... [--uninstall]

Options:
  --prefix DIR     copy the tool to DIR and install from there
  --no-global      do not symlink the plugin into the global opencode config
  --project DIR    also symlink the plugin into DIR/.opencode (repeatable)
  --uninstall      remove the plugin symlink(s) and the aidata wrapper
  -h, --help       show this help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="${2:?--prefix needs a directory}"; shift 2 ;;
    --prefix=*) PREFIX="${1#*=}"; shift ;;
    --no-global) DO_GLOBAL=0; shift ;;
    --project) PROJECTS+=("${2:?--project needs a directory}"); shift 2 ;;
    --project=*) PROJECTS+=("${1#*=}"); shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

# opencode discovers plugins in a `plugins/` dir (docs). Older 1.x also read a
# legacy `plugin/` dir, so remove that when linking to avoid double-loading.
link_plugin_into() {  # <base> where <base>/plugins/ should hold the symlink
  local base="$1"
  mkdir -p "$base/plugins"
  ln -sf "$INSTALL_DIR/$PLUGIN_REL" "$base/plugins/aidata-capture.ts"
  [ -L "$base/plugin/aidata-capture.ts" ] && rm -f "$base/plugin/aidata-capture.ts"
  echo "$base/plugins/aidata-capture.ts"
}

unlink_plugin_from() {  # <base>
  rm -f "$1/plugins/aidata-capture.ts" "$1/plugin/aidata-capture.ts"
}

if [ "$UNINSTALL" = 1 ]; then
  INSTALL_DIR="${PREFIX:-$SCRIPT_DIR}"
  log "uninstalling"
  unlink_plugin_from "$CONFIG_DIR"
  rm -f "$BIN_DIR/aidata"
  for p in "${PROJECTS[@]:-}"; do
    [ -n "$p" ] && unlink_plugin_from "$p/.opencode"
  done
  log "removed plugin symlink(s) and the aidata wrapper"
  echo "install dir left in place: $INSTALL_DIR"
  exit 0
fi

# --- 1. resolve install location -------------------------------------------- #
if [ -n "$PREFIX" ] && [ "$(cd "$PREFIX" 2>/dev/null && pwd || echo)" != "$SCRIPT_DIR" ]; then
  mkdir -p "$PREFIX"
  INSTALL_DIR="$(cd "$PREFIX" && pwd)"
  if [ "$INSTALL_DIR" != "$SCRIPT_DIR" ]; then
    log "copying tool -> $INSTALL_DIR"
    tar \
      --exclude='./.git' --exclude='./data' --exclude='./src.zip' \
      --exclude='./__pycache__' --exclude='./.opencode/node_modules' \
      --exclude='./.opencode/bun.lock' \
      -cf - -C "$SCRIPT_DIR" . | tar -xf - -C "$INSTALL_DIR"
  fi
else
  INSTALL_DIR="$SCRIPT_DIR"
fi
log "install dir: $INSTALL_DIR"

# --- 2. dependencies -------------------------------------------------------- #
command -v python3 >/dev/null 2>&1 || die "python3 not found; install Python 3 first"

if command -v bun >/dev/null 2>&1; then PM=bun
elif command -v npm >/dev/null 2>&1; then PM=npm
else PM=""; fi

if [ -n "$PM" ]; then
  log "installing plugin deps with $PM ($INSTALL_DIR/.opencode)"
  ( cd "$INSTALL_DIR/.opencode" && "$PM" install )
else
  warn "neither bun nor npm found; skipping plugin dependency install"
fi

if ! python3 -c 'import pathspec' >/dev/null 2>&1; then
  log "installing python dep: pathspec"
  python3 -m pip install --user pathspec >/dev/null 2>&1 \
    || python3 -m pip install --user --break-system-packages pathspec >/dev/null 2>&1 \
    || warn "could not install pathspec; run: python3 -m pip install --user pathspec"
fi

if ! command -v opencode >/dev/null 2>&1; then
  warn "opencode CLI not found on PATH; the plugin needs opencode to run"
fi

# --- 3. plugin symlinks ----------------------------------------------------- #
LINKS=()
if [ "$DO_GLOBAL" = 1 ]; then
  LINKS+=("$(link_plugin_into "$CONFIG_DIR") (global)")
  if [ -n "$PM" ] && [ ! -e "$CONFIG_DIR/node_modules/@opencode-ai/plugin" ]; then
    log "installing @opencode-ai/plugin in $CONFIG_DIR"
    if [ ! -f "$CONFIG_DIR/package.json" ]; then
      printf '{\n  "dependencies": { "@opencode-ai/plugin": "latest" }\n}\n' > "$CONFIG_DIR/package.json"
    fi
    ( cd "$CONFIG_DIR" && "$PM" install ) || warn "global plugin dep install failed"
  fi
fi
for p in "${PROJECTS[@]:-}"; do
  [ -n "$p" ] && LINKS+=("$(link_plugin_into "$p/.opencode") (project)")
done
mkdir -p "$CAPTURE_DIR"

# --- 4. cli wrapper --------------------------------------------------------- #
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/aidata" <<EOF
#!/usr/bin/env bash
exec python3 "$INSTALL_DIR/aidata.py" "\$@"
EOF
chmod +x "$BIN_DIR/aidata"

# --- summary ---------------------------------------------------------------- #
echo
log "done"
echo "  install dir : $INSTALL_DIR"
echo "  capture dir : $CAPTURE_DIR"
echo "  wrapper     : $BIN_DIR/aidata"
for l in "${LINKS[@]:-}"; do [ -n "$l" ] && echo "  plugin      : $l"; done
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) warn "$BIN_DIR is not on PATH; add it to use the 'aidata' command" ;;
esac
echo
echo "Restart opencode so it loads the plugin, then in any repo:"
echo "  aidata start     # zip the repo per .ai_data_config"
echo "  aidata end       # export, sanitize, collect licenses, label"
