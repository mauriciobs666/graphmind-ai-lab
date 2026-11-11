ARG BASE_PORT=8300
FROM python:3.12 AS build
ENV BUILD_ONLY=1
CMD ["build"]
FROM python:3.12-slim
ENV APP_PORT=8300 \
    LOG_LEVEL=info
EXPOSE 8300/tcp 9300
# a comment
ENTRYPOINT ["tini", "--"]
CMD uvicorn job.trigger_server:app --host 0.0.0.0 --port ${APP_PORT:-8300}
