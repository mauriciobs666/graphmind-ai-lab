#!/bin/bash
f() {
  local a="$1"
  local b="$a"
  local d
  d=$(date +%s)
  use_a "$b"
  use_d "$d"
  x=one
  x=two
  use_x "$x"
  if [ -n "$2" ]; then y=left; else y=right; fi
  use_y "$y"
  z="$z"
  use_z "$z"
  use_sub "$(inner "$1")"
  out=$(grep -c foo | tr -d ' ')
  use_pipe "$out"
}
