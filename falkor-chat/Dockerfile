# falkor-chat M1 server — REST + MCP (Streamable-HTTP) + static web UI on one
# uvicorn process. Build context is falkor-chat/ (the component root): app.py
# resolves the browser client as a *sibling* of server/ (`parents[2]/web`), so
# the image preserves the server/ + web/ layout and installs the package
# editable to keep that path valid.
#
#   docker compose up            # preferred — wires FalkorDB too (compose.yaml)
#   docker build -t falkorchat . # image only

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY server/pyproject.toml server/pyproject.toml
COPY server/falkorchat server/falkorchat
COPY web web

RUN pip install --no-cache-dir -e ./server

# Non-root runtime user (install stays root-owned and read-only to appuser).
RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 8000

# /health returns 200 only when the process is up AND FalkorDB answers (K-006).
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

CMD ["uvicorn", "falkorchat.app:app", "--host", "0.0.0.0", "--port", "8000"]
