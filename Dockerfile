# TSA FastAPI backend (SQLite version) — Docker image.
#
# Build from the repo root:
#   docker build -t tsa-api-sqlite .
# Run (host network, so the API is on localhost:8000):
#   docker run --rm --network=host -e TSA_HOST=127.0.0.1 -e TSA_PORT=8000 tsa-api-sqlite
# Or with port mapping (needs the container to listen on 0.0.0.0):
#   docker run --rm -p 8000:8000 -e TSA_HOST=0.0.0.0 -e TSA_PORT=8000 tsa-api-sqlite
#
# The SQLite database (data/tsa.db, ~172 MB) is baked into the image.
# It is read-only at runtime: the API never writes to the database
# (all writes happen offline via the import/compute scripts).
# To refresh the data, rebuild the image with a new data/tsa.db.
# No LLM key is baked in; set GEMINI_API_KEY at run time if needed.
# No Postgres, no DATABASE_URL: the runtime reads SQLITE_PATH
# (default /app/data/tsa.db).

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Runtime code + the SQLite database.
COPY scripts/ /app/scripts/
COPY data/tsa.db /app/data/tsa.db

ENV SQLITE_PATH=/app/data/tsa.db
ENV TSA_HOST=0.0.0.0
ENV TSA_PORT=8000

EXPOSE 8000

CMD ["python", "scripts/api.py"]
