#!/usr/bin/env bash
# K-066 U3: usage `qwen_watch.sh <events.log>` (polls every 10 s; every poll is appended to <events.log>.poll). Run detached; kill it when done.
# Poll LM Studio's loaded models every 10 s; if any qwen chat model is loaded, log and unload it.
LMS=/mnt/c/Users/mauri/.lmstudio/bin/lms.exe
OUT="$1"
while true; do
  ids=$(curl -s --max-time 5 localhost:1234/api/v0/models | python3 -c "import json,sys
try: print(' '.join(m['id'] for m in json.load(sys.stdin)['data'] if m.get('state')=='loaded'))
except Exception as e: print('ERR',e)")
  echo "$(date -Is) loaded: $ids" >> "$OUT.poll"
  for id in $ids; do
    case "$id" in qwen/*) echo "$(date -Is) QWEN LOADED: $id -> unloading" >> "$OUT"; "$LMS" unload "$id" >> "$OUT" 2>&1; echo "$(date -Is) after unload: $("$LMS" ps 2>&1 | tr '\n' ' ')" >> "$OUT";; esac
  done
  sleep 10
done
