# rehearsal environment (sourced by every script). Paths are in the session scratchpad; nothing here touches the real custodian or services.
export S=/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad
export REAL=/home/mauricio/prg/graphmind-ai-lab
export REPO=${REPO:-$S/repo}                         # clone of committed HEAD; all commits for the rehearsal happen here, never in $REAL
export RH=$S/rh
export HOME_R=$RH/home                      # HOME for harness processes, so ~/.local/share/ccg-eval and ~/.config/ccg are scratch
export CUST=$HOME_R/.local/share/ccg-eval/rehearsal
export PORT=${PORT:-16432}
export SYSTEM=graphmind-ai-lab
export PY=$REAL/cpg/.venv/bin/python
export CCG_CREDENTIALS_DIR=$RH/cred-$PORT   # per store: create_users rotates the passwords
export CCG_CONFIG_DIR=$RH/cfg
export GATE_ID=ac5-gate-1
export DEV_PROBE=${DEV_PROBE:-$RH/dev-probe.json}   # the attestation `gate.py dev-probe` writes from the dev rehearsal (11_dev_probe.sh); a list of dev items only needs none
run_gate() { ( cd $REPO/cpg/eval/tools && HOME=$HOME_R "$PY" gate_run.py run --gate-id $GATE_ID --custodian $CUST --repo $REPO --system $SYSTEM --host 127.0.0.1 --port $PORT --dev-probe $DEV_PROBE "$@" ); }
