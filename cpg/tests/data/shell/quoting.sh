#!/bin/bash
f() {
  echo "${v:-fallback}"
  echo "${PORT:-8200}"
  echo 'single $1 $(not run)'
  echo "esc \" \$HOME"
  echo a"b"c
  echo "pre-$v-post"
  echo "$(date)"
  echo ${v%.txt}
}
