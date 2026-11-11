#!/bin/sh
first() { echo "$1"; }
pair() { cp "$1" "$2"; }
second_only() { echo "${2}"; }
rest() { echo "$1" "$@"; }
only_all() { exec "$@"; }
literals() { echo '$1' "\$1" "x$1y"; }
echo "$1" "$@"
