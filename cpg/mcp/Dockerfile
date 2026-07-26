# The `cpg` MCP server as a container: ONE stdio process per session, launched by
# cpg/mcp/docker-run.sh (see the repo-root .mcp.json) — NOT a long-lived service.
# No EXPOSE and no HEALTHCHECK on purpose: it listens on nothing and no orchestrator
# polls it. See docs/plans/cpg-mcp-containerization.md §3.7.
#
# Build context is cpg/mcp/ (narrow, deliberately — keeps cpg/.cpg-artifacts/ out of
# the build tar). Build with cpg/mcp/build.sh, never by hand: the image TAG is a
# content hash of the build inputs and build.sh is what computes it (§3.3, §3.6).
#
# IMPORTANT: every path COPYed below must be covered by image-tag.sh — a FILE operand
# by cpg_mcp_input_files, a DIRECTORY operand by cpg_mcp_input_dirs (which is walked) —
# or the hash gate will not notice a change to it. `build.sh --verify-inputs` enforces
# both rules.

FROM python:3.12-slim AS base

LABEL cpg-mcp=1

# FALKORDB_HOST default: 127.0.0.1 would mean *this container*. The host-gateway
# alias is supplied by docker-run.sh's --add-host. See §3.1.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FALKORDB_HOST=host.docker.internal \
    FALKORDB_PORT=6379

WORKDIR /app

# Before the COPYs, so a source edit does not re-run it.
RUN useradd --create-home --shell /usr/sbin/nologin appuser

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py server.py

# --- test stage: dev deps + the suite. Never shipped as the runtime image. ----
FROM base AS test
COPY requirements-dev.txt requirements-dev.txt
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY pytest.ini pytest.ini
COPY tests tests
# pytest writes .pytest_cache into its rootdir; without this, appuser cannot, and the
# suite emits a PytestCacheWarning that makes the output differ from the host baseline
# for no reason (measured — see §3.4).
RUN chown appuser:appuser /app
USER appuser
CMD ["python", "-m", "pytest", "tests", "-q"]

# --- runtime stage: server + runtime deps only. No pytest, no tests. ---------
FROM base AS runtime
USER appuser
CMD ["python", "server.py"]
