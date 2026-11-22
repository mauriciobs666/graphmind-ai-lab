#!/usr/bin/env bash
# usage: sentinel.sh plant | snap <label>   -> manifests of the developer-default areas (copy defaults + real defaults)
. "$(dirname "$0")/env.sh"
CW=$REPO/cpg/.ccg-work; CA=$REPO/cpg/.audit; CK=$HOME_R/.config/ccg/keys
man() { # dir -> sorted "sha256  relpath  mode" (files), "d relpath mode" (dirs)
  [ -d "$1" ] || { echo "(absent)"; return; }
  ( cd "$1" && find . -mindepth 1 \( -type f -printf '%p %m ' -exec sha256sum {} \; -o -type d -printf 'd %p %m\n' \) | sort )
}
realman() { [ -e "$1" ] && find "$1" -printf '%p %s %T@ %m\n' | sort || echo "(absent)"; }
case "$1" in
plant)
  mkdir -p $CW/bundles/$SYSTEM $CA $CK; chmod 700 $CW $CW/bundles $CW/bundles/$SYSTEM $CK
  UNIT=0123456789abcdef0123456789abcdef; B=fedcba9876543210fedcba9876543210
  echo '{"sentinel":"dev bundle for N4"}' > $CW/bundles/$SYSTEM/$UNIT.$B.bundle.jsonl; chmod 600 $CW/bundles/$SYSTEM/$UNIT.$B.bundle.jsonl
  echo '{"status":"fresh","builtCommit":"0000000000000000000000000000000000000000"}' > $CW/bundles/$SYSTEM/$UNIT.state.json 2>/dev/null; chmod 600 $CW/bundles/$SYSTEM/$UNIT.state.json
  echo sentinel-audit > $CA/sandbox.jsonl
  head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > $CK/$SYSTEM.hmac; echo >> $CK/$SYSTEM.hmac; chmod 600 $CK/$SYSTEM.hmac
  echo planted ;;
snap)
  out=$RH/log/sentinel-$2.txt
  { echo "== copy .ccg-work"; man $CW; echo "== copy .audit"; man $CA; echo "== scratch keys"; man $CK
    echo "== REAL cpg/.ccg-work"; realman $REAL/cpg/.ccg-work; echo "== REAL cpg/.audit"; realman $REAL/cpg/.audit; echo "== REAL ~/.config/ccg"; realman $HOME/.config/ccg; } > $out
  echo "$out: $(wc -l < $out) lines" ;;
esac
