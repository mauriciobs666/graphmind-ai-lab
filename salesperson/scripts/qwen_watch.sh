#!/usr/bin/env bash
# K-066 U3: usage `qwen_watch.sh <events.log>` (polls every 10 s; every poll is appended to <events.log>.poll). Run detached; kill it when done.
# Poll LM Studio's loaded models every 10 s; if any qwen chat model is loaded, log and unload it.
# LMS: path to LM Studio's `lms` CLI. Unset: $LMS_CLI (the falkor-chat demo scripts' name), then
# `lms`/`lms.exe` on PATH, then (WSL) %USERPROFILE%\.lmstudio\bin\lms.exe via cmd.exe + wslpath.
resolve_lms() {
  local c=${LMS:-${LMS_CLI:-}} up
  [[ -n $c ]] && { printf '%s\n' "$c"; return; }
  c=$(command -v lms || command -v lms.exe) && { printf '%s\n' "$c"; return; }
  if command -v cmd.exe >/dev/null && command -v wslpath >/dev/null; then
    up=$(cd /mnt/c 2>/dev/null && cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')
    [[ -n $up && $up != *%* ]] && c="$(wslpath -u "$up")/.lmstudio/bin/lms.exe" && [[ -x $c ]] && { printf '%s\n' "$c"; return; }
  fi
  echo "qwen_watch.sh: lms CLI not found; set LMS=/path/to/lms (WSL: /mnt/c/Users/<winuser>/.lmstudio/bin/lms.exe)" >&2
  return 1
}
OUT="${1:?usage: qwen_watch.sh <events.log>}"
LMS=$(resolve_lms) || exit 1
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
