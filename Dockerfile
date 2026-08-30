FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal: HiGHS ships prebuilt wheels via highspy, so there
# is no compiler and no commercial solver licence in this image.
COPY pyproject.toml ./
COPY src ./src
COPY app ./app

RUN pip install --no-cache-dir -e .

EXPOSE 8501

# The healthcheck uses Python rather than curl: python:3.12-slim ships no curl,
# and adding it would mean an apt layer for a single request.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", \
            "--server.headless=true", "--browser.gatherUsageStats=false"]
