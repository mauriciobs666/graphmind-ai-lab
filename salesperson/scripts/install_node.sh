#!/usr/bin/env bash
# Install the pinned Node toolchain for this component, per-user, without sudo.
#
# Why a tarball and not apt/nvm: this dev box (WSL2) has no passwordless sudo,
# and the only `npm` on PATH is the *Windows* one under /mnt/c/Program Files —
# a Windows npm installs Windows-native binaries (esbuild, rollup, lightningcss)
# that a Linux Vite build cannot load. A per-user Linux tarball sidesteps both.
#
# Idempotent: re-running with the version already installed is a no-op.
#
# Usage:
#   ./scripts/install_node.sh              # install the pinned version
#   NODE_PREFIX=/opt/node ./scripts/install_node.sh
#
# Env:
#   NODE_PREFIX   install root (default: $HOME/.local/node)
#   NODE_VERSION  override the pin (default: read from .node-version)

set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

NODE_PREFIX="${NODE_PREFIX:-$HOME/.local/node}"
NODE_VERSION="${NODE_VERSION:-$(tr -d '[:space:]' < "$here/.node-version")}"

case "$(uname -m)" in
  x86_64)  arch=x64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) echo "install_node.sh: unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

dist="node-v${NODE_VERSION}-linux-${arch}"
target="$NODE_PREFIX/$dist"

if [ -x "$target/bin/node" ]; then
  echo "Node v${NODE_VERSION} already installed at $target"
else
  for cmd in curl tar sha256sum; do
    command -v "$cmd" >/dev/null 2>&1 || {
      echo "install_node.sh: required command '$cmd' not found" >&2; exit 1; }
  done

  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  base="https://nodejs.org/dist/v${NODE_VERSION}"

  echo "Downloading ${dist}.tar.xz from ${base} ..."
  curl -fsSL -o "$tmp/${dist}.tar.xz" "${base}/${dist}.tar.xz"
  curl -fsSL -o "$tmp/SHASUMS256.txt" "${base}/SHASUMS256.txt"

  echo "Verifying checksum ..."
  ( cd "$tmp" && grep " ${dist}.tar.xz\$" SHASUMS256.txt | sha256sum -c - )

  mkdir -p "$NODE_PREFIX"
  tar -xJf "$tmp/${dist}.tar.xz" -C "$NODE_PREFIX"
  echo "Installed to $target"
fi

# `current` is the stable path build.sh looks for; repointing it is the upgrade.
ln -sfn "$target" "$NODE_PREFIX/current"

# npm is a #!/usr/bin/env node script, so it only runs with node on PATH.
export PATH="$NODE_PREFIX/current/bin:$PATH"

node --version >/dev/null

cat <<EOF

Node toolchain ready.
  node $(node --version)
  npm  v$(npm --version)
  path $NODE_PREFIX/current/bin

./build.sh finds this automatically. To use node/npm interactively, add to
your shell profile:

  export PATH="$NODE_PREFIX/current/bin:\$PATH"

Note: put it *before* \$PATH, not after — the Windows npm shim under
/mnt/c/Program Files/nodejs is already on PATH and must be shadowed.
EOF
