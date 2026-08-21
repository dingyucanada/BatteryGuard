FROM python:3.11-slim AS runtime

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BATTERYGUARD_OFFLINE=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYBAMM_DISABLE_TELEMETRY=true \
    UV_LINK_MODE=copy \
    HOME=/home/batteryguard \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --no-install-project

COPY README.md LICENSE NOTICE ./
COPY src ./src
COPY apps ./apps
COPY configs ./configs
COPY .streamlit ./.streamlit
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev && \
  groupadd --gid "${APP_GID}" batteryguard && \
  useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home \
    --shell /usr/sbin/nologin batteryguard && \
  mkdir -p /app/artifacts /app/reports /app/data/raw /app/data/interim /app/data/processed && \
  chown -R "${APP_UID}:${APP_GID}" /app/artifacts /app/reports /app/data

USER batteryguard

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "batteryguard.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
