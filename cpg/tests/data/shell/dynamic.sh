#!/bin/bash
run_eval() { eval "$1"; }
indirect() { local n="$1"; echo "${!n}"; }
load() { source "$1"; }
load_lib() { . ./lib.sh; }
computed() { source "$dir/lib.sh"; }
dyncmd() { "$1" --flag; }
$tool run
