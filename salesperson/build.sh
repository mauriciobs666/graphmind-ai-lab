#!/usr/bin/env bash
# Build the salesperson storefront SPA to salesperson/dist/.
#
# The bundle is a pure static artifact served by falkor-chat's FastAPI process
# at /shop (that mount is why vite.config.ts pins `base: "/shop/"`). dist/ is
# gitignored by design: this script is the reproducible way to regenerate it.
#
# Usage:
#   ./build.sh                 # install deps if needed, then build
#   ./build.sh --skip-install  # build with the node_modules already present
#   ./build.sh --help
#
# Env:
#   NODE_BIN_DIR  bin directory of a specific Node toolchain to use
#   NODE_PREFIX   where install_node.sh put Node (default: $HOME/.local/node)

set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

skip_install=0
for arg in "$@"; do
  case "$arg" in
    --skip-install) skip_install=1 ;;
    -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "build.sh: unknown argument '$arg' (try --help)" >&2; exit 2 ;;
  esac
done

NODE_PREFIX="${NODE_PREFIX:-$HOME/.local/node}"
# .node-version is the *pin* — what install_node.sh installs and what a version
# manager picks up. The floor below is the *minimum supported* major, matching
# package.json's "engines" (Vite 8 needs 20.19+/22.12+). They differ on purpose:
# an existing Node 22 toolchain is fine, we just don't install one.
required_major=22
pinned="$(tr -d '[:space:]' < .node-version)"

die_no_node() {
  cat >&2 <<EOF

build.sh: no usable Node toolchain found.
$1

This component needs Node >= ${required_major} (pinned: v${pinned}). Install it, without
sudo, with the script that ships next to this one:

    ./scripts/install_node.sh

Then re-run ./build.sh. Nothing else needs to change — build.sh finds the
toolchain at \$NODE_PREFIX/current/bin (default \$HOME/.local/node/current/bin)
on its own.

EOF
  exit 1
}

# --- resolve a Node toolchain ------------------------------------------------
# Order: explicit override, then the pinned per-user install, then whatever is
# already on PATH. The pinned install wins over PATH so a build is reproducible
# regardless of the caller's shell.
node_bin=""
if [ -n "${NODE_BIN_DIR:-}" ]; then
  [ -x "$NODE_BIN_DIR/node" ] || die_no_node "NODE_BIN_DIR=$NODE_BIN_DIR has no executable 'node'."
  node_bin="$NODE_BIN_DIR"
elif [ -x "$NODE_PREFIX/current/bin/node" ]; then
  node_bin="$NODE_PREFIX/current/bin"
elif command -v node >/dev/null 2>&1; then
  node_bin="$(dirname "$(command -v node)")"
else
  # The usual WSL2 case: `npm` resolves (to the Windows shim) but `node` does not.
  if command -v npm >/dev/null 2>&1; then
    die_no_node "'node' is not on PATH, though 'npm' is ($(command -v npm)).
A Windows npm under /mnt/c cannot build this: it installs Windows-native
binaries (esbuild, rollup, lightningcss) that a Linux Vite build cannot load."
  fi
  die_no_node "Neither 'node' nor 'npm' is on PATH."
fi

export PATH="$node_bin:$PATH"

# A Node reached through /mnt/ is the Windows one under a WSL mount. It will
# fail later, deep inside the bundler, with an unrelated-looking error.
case "$(command -v node)" in
  /mnt/*) die_no_node "The only 'node' found is the Windows build at $(command -v node).
Its native modules are Windows binaries and cannot be loaded by a Linux build." ;;
esac

node_version="$(node --version)"   # e.g. v24.20.0
node_major="${node_version#v}"; node_major="${node_major%%.*}"
if [ "$node_major" -lt "$required_major" ]; then
  die_no_node "Found node ${node_version} at $(command -v node), which is older than the required major ${required_major}."
fi

command -v npm >/dev/null 2>&1 || die_no_node "node ${node_version} is present but 'npm' is not next to it."

echo "build.sh: node ${node_version} (npm v$(npm --version)) from $(dirname "$(command -v node)")"

# --- dependencies ------------------------------------------------------------
if [ "$skip_install" -eq 1 ]; then
  [ -d node_modules ] || die_no_node "--skip-install was given but node_modules/ does not exist."
  echo "build.sh: --skip-install, using the existing node_modules/"
elif [ ! -d node_modules ] || [ package-lock.json -nt node_modules/.package-lock.json ]; then
  echo "build.sh: installing dependencies (npm ci) ..."
  npm ci
else
  echo "build.sh: dependencies up to date"
fi

# --- build -------------------------------------------------------------------
echo "build.sh: building ..."
npm run build

# --- verify ------------------------------------------------------------------
[ -f dist/index.html ] || { echo "build.sh: build finished but dist/index.html is missing" >&2; exit 1; }
grep -q '/shop/' dist/index.html || {
  echo "build.sh: dist/index.html does not reference /shop/ — is vite.config.ts's base still '/shop/'?" >&2
  exit 1
}

echo
echo "build.sh: OK — bundle in $here/dist (base /shop/)"
ls -la dist
