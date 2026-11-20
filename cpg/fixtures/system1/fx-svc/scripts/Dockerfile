FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install .
CMD ["uvicorn", "mcp.server:app", "--host", "127.0.0.1", "--port", "8200"]
