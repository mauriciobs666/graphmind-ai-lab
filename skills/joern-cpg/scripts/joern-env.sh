#!/usr/bin/env bash
# joern-env.sh — resolve the Joern + JDK environment. SOURCE this (do not exec):
#   . "$(dirname "$0")/joern-env.sh"
#
# It sets and exports JOERN_HOME, JAVA_HOME, and prepends both to PATH so the
# joern-* wrappers and `java` are found regardless of the caller's shell config.
# Nothing here hardcodes a personal home path — JOERN_HOME defaults under $HOME
# and JAVA_HOME is auto-detected from `java`, so the scripts are portable.
#
# Overrides (env): JOERN_HOME, JAVA_HOME.

# --- Joern ---
JOERN_HOME="${JOERN_HOME:-$HOME/joern/joern-cli}"
if [ ! -x "$JOERN_HOME/joern-parse" ]; then
  echo "joern-env: joern not found at JOERN_HOME=$JOERN_HOME" >&2
  echo "           set JOERN_HOME to your joern-cli dir, or install Joern (https://docs.joern.io/installation)." >&2
  return 1 2>/dev/null || exit 1
fi

# --- JDK 21+ ---
if [ -z "${JAVA_HOME:-}" ]; then
  if command -v java >/dev/null 2>&1; then
    JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
  elif [ -x /usr/lib/jvm/java-21-openjdk-amd64/bin/java ]; then
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
  fi
fi

export JOERN_HOME JAVA_HOME
export PATH="${JAVA_HOME:+$JAVA_HOME/bin:}$JOERN_HOME:$PATH"

if ! command -v java >/dev/null 2>&1; then
  echo "joern-env: java not found. Install a JDK 21 (Ubuntu: sudo apt install -y openjdk-21-jdk)" >&2
  echo "           or set JAVA_HOME to an existing JDK." >&2
  return 1 2>/dev/null || exit 1
fi
