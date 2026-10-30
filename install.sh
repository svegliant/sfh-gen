#!/usr/bin/env bash
# =============================================================================
#  install.sh — installer for sfh-gen
#
#    bash install.sh            install
#    bash install.sh --dev      editable install, for working on the code
#    bash install.sh --check    report dependencies and the target environment,
#                               install nothing
#    bash install.sh --uninstall
#
#  A thin wrapper around pip. It exists to name the environment it is about to
#  write to, which pip does not, and to report missing dependencies before
#  anything is written. `pip install .` does the same job without the
#  reporting. After installing, the command `sfh-gen` is on your PATH.
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}  ✔${NC}  $*"; }
info() { echo -e "${CYAN}  ➜${NC}  $*"; }
warn() { echo -e "${YELLOW}  ⚠${NC}  $*"; }
fail() { echo -e "${RED}  ✘${NC}  $*" >&2; }
die()  { fail "$*"; exit 1; }
sep()  { echo -e "${BOLD}────────────────────────────────────────────────${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="install"          # install | dev | check | uninstall
for arg in "$@"; do
    case "$arg" in
        --dev)       MODE="dev"       ;;
        --check)     MODE="check"     ;;
        --uninstall) MODE="uninstall" ;;
        --help|-h)   sed -n '3,14p' "$0" | sed 's/^# \{0,2\}//'; exit 0 ;;
        *)           die "Unknown option: $arg  (try --help)" ;;
    esac
done

sep
echo -e "${BOLD}  sfh-gen installer${NC}"
sep

# ── Python and pip ───────────────────────────────────────────────────────────
PYTHON="$(command -v python3 || true)"
[[ -n "$PYTHON" ]] || die "python3 not found."
ok "Python $("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')  ($PYTHON)"

"$PYTHON" -m pip --version &>/dev/null \
    || die "pip is not available for $PYTHON. Install it and retry."
PIP="$PYTHON -m pip"

# ── Which environment is about to be written to ──────────────────────────────
# pip installs into whichever interpreter is first on PATH, which on a shared
# cluster is often the system one. Name it, so a mistake is visible now rather
# than weeks later.
ENV_NAME="$("$PYTHON" - <<'PY'
import os, sys
if os.environ.get("CONDA_DEFAULT_ENV"):
    print("conda environment '%s'" % os.environ["CONDA_DEFAULT_ENV"])
elif sys.prefix != sys.base_prefix:
    print("virtual environment at %s" % sys.prefix)
else:
    print("SYSTEM interpreter at %s" % sys.prefix)
PY
)"
info "Target environment: ${ENV_NAME}"
if [[ "$ENV_NAME" == SYSTEM* ]]; then
    warn "Not a virtual environment. On a managed system pip may refuse to"
    warn "write here (PEP 668); if it does, create one first:"
    warn "    python3 -m venv ~/sfh-env && source ~/sfh-env/bin/activate"
fi

# ── Uninstall ────────────────────────────────────────────────────────────────
if [[ "$MODE" == "uninstall" ]]; then
    sep
    $PIP uninstall -y sfh-gen && ok "sfh-gen uninstalled." \
        || warn "sfh-gen was not installed."
    exit 0
fi

# ── Dependencies ─────────────────────────────────────────────────────────────
# pip resolves these from pyproject.toml during the install. They are checked
# here so that --check can report what is missing before anything is written,
# which matters on a cluster node with no package index.
sep
info "Checking dependencies..."

# Import name -> the name to type after `pip install`; the two differ for pyyaml.
pypi_name() { case "$1" in yaml) echo "pyyaml" ;; *) echo "$1" ;; esac; }

MISSING=()
for pkg in numpy scipy h5py yaml; do
    if ver=$("$PYTHON" -c "import $pkg; print(getattr($pkg,'__version__','?'))" 2>/dev/null); then
        ok "$pkg  ($ver)"
    else
        fail "$pkg — NOT FOUND"
        MISSING+=("$pkg")
    fi
done

if [[ "$MODE" == "check" ]]; then
    sep
    if [[ ${#MISSING[@]} -eq 0 ]]; then
        ok "All dependencies satisfied — ready to install."
        info "Run  bash install.sh  to install."
        exit 0
    fi
    PYPI=(); for p in "${MISSING[@]}"; do PYPI+=("$(pypi_name "$p")"); done
    fail "Missing: ${MISSING[*]}"
    info "pip will fetch them during the install, or:  pip install ${PYPI[*]}"
    exit 1
fi

# ── Install ──────────────────────────────────────────────────────────────────
sep
if [[ "$MODE" == "dev" ]]; then
    info "Installing sfh-gen in editable mode..."
    $PIP install -e "$SCRIPT_DIR"
else
    info "Installing sfh-gen..."
    $PIP install "$SCRIPT_DIR"
fi

# ── Verify ───────────────────────────────────────────────────────────────────
# Run from / so that the source tree in the current directory cannot satisfy
# the import and make a failed install look successful.
sep
info "Verifying..."

INSTALLED_AT="$(cd / && "$PYTHON" -c \
    "import sfh_gen, os; print(os.path.dirname(sfh_gen.__file__))" 2>/dev/null || true)"
[[ -n "$INSTALLED_AT" ]] || die "'import sfh_gen' does not work after the install."
ok "sfh_gen at: $INSTALLED_AT"

(cd / && "$PYTHON" -c "from sfh_gen.cli import main" &>/dev/null) \
    || die "'sfh_gen.cli:main' is not importable — the command would fail."

if command -v sfh-gen &>/dev/null; then
    ok "Command 'sfh-gen' at: $(command -v sfh-gen)"
else
    warn "'sfh-gen' is not on your PATH. pip put it in its script directory;"
    warn "add that to PATH, or call it as:  python3 -m sfh_gen"
fi

sep
echo -e "${GREEN}${BOLD}  Done.${NC}"
echo ""
echo -e "  First run, in an empty directory:"
echo -e "    ${CYAN}sfh-gen${NC}              writes a commented config.yaml"
echo -e "    ${CYAN}sfh-gen config.yaml${NC}  runs the pipeline"
sep
