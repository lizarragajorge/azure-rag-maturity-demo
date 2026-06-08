# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8000 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Drop root: create a non-privileged user and own the app dir
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 app \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/_stcore/health', timeout=3).status == 200 else 1)"

# Security notes:
#   * XSRF protection stays ON (Streamlit default) — never pass
#     `--server.enableXsrfProtection=false` on a public deploy.
#   * CORS is disabled because Container Apps ingress is the single origin;
#     Streamlit docs require one of {XSRF on, CORS off} on a reverse proxy.
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8000", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--browser.gatherUsageStats=false"]
