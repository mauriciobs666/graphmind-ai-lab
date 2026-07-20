# WSL2 Memory Diagnostic — 16GB host, WSL2 + Docker/FalkorDB + Windows-side LM Studio

**Date:** 2026-07-18
**Author:** devops (read-only diagnostic run — no config changed, no restart, no fix applied)
**Companion doc:** `local-model-ram-budget-ml.md` (data-scientist RAM budget that raised the ballooning hypothesis)

## Scope

Read-only investigation to confirm/refute the hypothesis that **WSL2 memory ballooning** (an
uncapped or over-generous WSL2 VM starving the Windows side) is the root cause of the
memory-overload crashes that led the user to downgrade from 32GB to 16GB. Deliverable: verdict +
recommended `.wslconfig` + apply procedure. **Nothing was mutated this run.**

---

## Evidence collected (live)

| What | Observed value | Source |
|---|---|---|
| Windows physical RAM | 16,849,256,448 B ≈ **15.7 GiB (16GB host)** | `Win32_ComputerSystem.TotalPhysicalMemory` via interop |
| WSL version | **2.4.13.0**, WSLg 1.0.65, kernel 5.15.167.4 | `wsl.exe --version` |
| Windows build | 10.0.26200 (Windows 11, 24H2/25H2 line) | `wsl.exe --version` |
| RAM visible to WSL2 VM | **MemTotal 7,973,228 kB ≈ 7.6 GiB** | `/proc/meminfo` |
| WSL2 swap | 2.0 GiB (`/dev/sdb`, default) | `free -h`, `swapon --show` |
| Current WSL usage | 1.1 GiB used / 6.2 GiB free / 489 MiB cache | `free -h` |
| `.wslconfig` | **PRESENT** at `/mnt/c/Users/mauri/.wslconfig` (323 B) | direct read |
| `.wslconfig` memory cap | **NONE** — file has only `[wsl2] networkingMode=mirrored` | direct read |
| Docker engine | **Native Docker Engine inside WSL2** (`default` context = `unix:///var/run/docker.sock`, active). Docker Desktop context (`desktop-linux`) exists but is **not** selected. | `docker context ls`, `docker info` |
| Docker `MemTotal` | 8,164,585,472 B ≈ 7.6 GiB — **same pool as the WSL2 VM** (native engine, no separate utility VM) | `docker info` |
| FalkorDB running now? | **No** — `docker ps` empty; `docker stats` empty | `docker ps` |
| FalkorDB assets present | image `falkordb/falkordb:v4.18.11` (+ edge/latest) and volume `falkordb-data` exist | `docker images`, `docker volume ls` |
| vCPUs in VM | 20 (all host logical processors — WSL default) | `nproc` |
| autoMemoryReclaim supported? | **Yes** — WSL 2.4.13 (≥ 2.0.0) + `/mnt/wslg` present | version + wslg probe |

---

## Verdict: **CONFIRMED** (with one important refinement)

The ballooning hypothesis holds, with a precise correction to the mechanism:

1. **There IS a `.wslconfig`, but it sets NO `memory=` cap.** It carries only
   `networkingMode=mirrored`. With no cap, WSL2 falls back to its **default ceiling of 8GB** (50%
   of a 16GB host on modern builds). The VM currently reports **7.6 GiB total** — direct proof the
   8GB default is the live ceiling.

2. **On a 16GB host, that 8GB default is itself the overload surface.** Worst-case concurrent
   footprint:
   - WSL2 VM (Docker + FalkorDB + Python server + build spikes): up to **8 GB** (the default cap)
   - LM Studio models on Windows: **~5 GB typical, up to ~8 GB** (chat + embedding loaded)
   - Windows OS/desktop: **~3.5 GB**
   - **Total: ~16.5–19.5 GB against a 16 GB host → overcommit → the crashes.**

3. **WSL2 does not return freed memory on its own here.** `autoMemoryReclaim` is **not set**, and
   the Linux page cache balloons toward the cap and is not handed back to Windows during idle. So
   even below 8GB, WSL2's resident footprint ratchets upward over a session — the classic ballooning
   symptom — squeezing LM Studio and Windows until something OOMs. This is the refinement: it is not
   "uncapped to 50%+", it is "capped at a too-high 8GB **and** never reclaiming."

4. **Because the engine is native Docker-in-WSL2 (not Docker Desktop), FalkorDB's RAM is WSL2's
   RAM.** There is no separate Docker Desktop utility VM to account for — a single `memory=` cap on
   the WSL2 VM bounds Docker and FalkorDB together. (FalkorDB is not running right now, so it is not
   contributing to the current 1.1 GiB usage; the crash scenario is with it up and the graph loaded —
   and project docs flag "RAM is the binding constraint" as FalkorDB grows.)

**Nothing refutes the hypothesis; the observed defaults (8GB ceiling, no reclaim) on a 16GB host
that must also host LM Studio are sufficient on their own to reproduce the overload.**

---

## Recommended `.wslconfig` (proposed — NOT applied this run)

Full file for `C:\Users\mauri\.wslconfig`. **Preserves the load-bearing `networkingMode=mirrored`
line** (LM Studio reachability from WSL over localhost depends on it — see Severino notes).

```ini
# WSL2 global configuration (applies to all distros).
# Mirrored networking makes the Windows host reachable from WSL via localhost,
# so services bound on Windows (e.g. LM Studio on 127.0.0.1:1234) work without
# chasing the dynamic NAT gateway IP. Requires Windows 11 22H2+ (build 22621+).
[wsl2]
networkingMode=mirrored

# --- Memory containment (added for the 16GB host) ---
# Cap the WSL2 VM so the Windows side keeps enough RAM for LM Studio (~5-8GB)
# plus the OS (~3.5GB). The default was 8GB (50% of 16GB), which overcommits the
# host once LM Studio is also loaded. 6GB leaves ~10GB for Windows.
memory=6GB

# Disk-backed safety valve. With a tight RAM cap, transient spikes (pip installs,
# pytest, FalkorDB bulk loads) swap instead of OOM-killing dockerd/FalkorDB.
swap=4GB

# Optional, NOT a memory fix: fewer vCPUs reduces scheduler contention and the
# memory spikes of highly parallel builds. Remove this line to use all 20.
processors=8

[experimental]
# Return freed WSL2 page cache to Windows during idle. Supported on WSL 2.0+
# (you run 2.4.13). Without it the VM cache balloons toward the cap and never
# gives RAM back — the core of the ballooning problem even with a cap set.
autoMemoryReclaim=gradual

# Optional disk hygiene (not RAM): let the WSL VHDX shrink as data is freed.
sparseVhd=true
```

### Why these numbers

- **`memory=6GB`** — The single most important line. It drops the ceiling from 8GB to 6GB, leaving
  ~10GB for Windows (OS ~3.5GB + LM Studio ~5GB typical = ~8.5GB, comfortable; even a ~8GB worst-case
  model load leaves Windows to page gracefully rather than hard-overcommit the whole host). 6GB is
  still ample for native Docker + FalkorDB (~1–2GB) + the `falkorchat` Python server + `pip`/`pytest`
  build spikes.
  - **Trade-off / knobs:** If you routinely load the *largest* LM Studio models and still see
    pressure, drop to **`memory=5GB`** (leaves 11GB for Windows). If instead FalkorDB grows large and
    you rarely run the biggest models, **`memory=7GB`** is acceptable — but that re-tightens the
    Windows side, so only move up if you confirm LM Studio still fits. Start at 6GB.
- **`swap=4GB`** — A larger swap than the 2GB default deliberately pairs with the tighter RAM cap:
  transient Linux-side spikes hit disk-backed swap (slower but survivable) instead of triggering an
  OOM kill of FalkorDB or dockerd. Swap lives in a VHDX on the Windows disk; it costs disk, not RAM.
- **`processors=8`** — Optional and unrelated to the RAM fix; it curbs scheduler contention and the
  memory bursts of 20-way parallel builds. Safe to omit if you want full CPU.
- **`autoMemoryReclaim=gradual`** — The companion to the cap. `gradual` reclaims idle page cache
  slowly (safe, low perf impact). `dropcache` is more aggressive (can hurt warm-cache perf);
  `disabled` is the current implicit behavior that lets the balloon persist. Use `gradual`.

---

## Apply procedure (for the user to run — do NOT run this diagnostic session)

1. **Back up the current file** (it is tiny; keep the original in case):
   - From Windows: copy `C:\Users\mauri\.wslconfig` to `C:\Users\mauri\.wslconfig.bak`.
   - Or from WSL: `cp /mnt/c/Users/mauri/.wslconfig /mnt/c/Users/mauri/.wslconfig.bak`
2. **Edit `C:\Users\mauri\.wslconfig`** to match the recommended contents above (Notepad, VS Code,
   or any editor). Ensure `networkingMode=mirrored` stays.
3. **Stop anything important first** — this restarts *all* WSL distros and any Docker containers in
   them (FalkorDB included). Save work; note the `falkordb-data` volume persists across the restart.
4. **Shut down WSL** from a Windows PowerShell/CMD prompt:
   ```
   wsl --shutdown
   ```
5. Wait ~10 seconds, then **reopen your WSL terminal** (or run `wsl`). The new limits take effect on
   next boot.
6. **Verify the cap took:**
   ```bash
   free -h              # MemTotal should now read ~6.0Gi (was ~7.6Gi)
   grep MemTotal /proc/meminfo
   cat /proc/swaps      # swap ~4GB
   nproc                # 8 if you kept processors=8
   ```
7. **Restart FalkorDB** when needed: `cd falkor-chat && ./scripts/start_falkordb.sh` (add `-d` for
   headless). The `falkordb-data` volume is untouched by the WSL restart.

---

## Risks & open questions

- **Not root-caused by observation, only by configuration + budget math.** FalkorDB was **not
  running** during this diagnostic, so no live crash was reproduced. The verdict rests on the
  demonstrated defaults (8GB ceiling + no reclaim) versus the 16GB budget, which is sufficient to
  explain the overload — but if crashes persist after the cap, capture `free -h` on the WSL side and
  Windows Task Manager memory at crash time to confirm which side actually OOMs.
- **6GB vs FalkorDB growth.** Project docs call RAM "the binding constraint" for FalkorDB. If the
  workspace graph plus vector indexes grow past a couple of GB, 6GB could get tight for the Linux
  side. Monitor `docker stats --no-stream` for the FalkorDB container; if it approaches ~4GB
  resident, reconsider the cap split (or trim embedding dimensions / workspace count).
- **LM Studio worst-case (~8GB) + OS (~3.5GB) = ~11.5GB > the ~10GB left by a 6GB WSL cap.** Under
  simultaneous peak on both sides Windows will page rather than crash — acceptable, but if the user
  keeps the largest models resident continuously, prefer `memory=5GB`.
- **Do not remove `networkingMode=mirrored`.** Dropping it breaks WSL→LM Studio over localhost and
  would send you back to chasing the NAT gateway IP.
- **`[experimental]` keys can change across WSL releases.** `autoMemoryReclaim`/`sparseVhd` are
  documented under `[experimental]` as of WSL 2.x; if a future WSL rejects them, they are ignorable
  (the `[wsl2]` caps still apply). Re-verify against Microsoft's `.wslconfig` docs on major WSL
  upgrades.
- **Docker Desktop is installed too** (the `desktop-linux` context exists). This diagnostic and the
  cap target the **native** engine currently in use. If the user ever switches Docker to the Desktop
  backend, Docker Desktop runs its own utility VM with separate memory settings (Docker Desktop's
  Settings → Resources, which also reads `.wslconfig`) — revisit sizing then.
