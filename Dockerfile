# syntax=docker/dockerfile:1
FROM node:22.15.0-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.10.14-slim-bookworm AS python-build
ARG TORCH_VARIANT=cpu
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 MAX_JOBS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY requirements-build.txt requirements-server.txt requirements-model.txt requirements-lock.txt ./
RUN pip install -r requirements-build.txt && \
    pip install "torch==1.13.1+${TORCH_VARIANT}" "torchvision==0.14.1+${TORCH_VARIANT}" \
      -c requirements-lock.txt \
      --extra-index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" && \
    pip install -r requirements-model.txt
RUN CUDA_VISIBLE_DEVICES='' FORCE_CUDA=0 pip install --no-build-isolation --no-deps \
    'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@0ae803b1449cd2d3f8fa1b7c0f59356db10b3083' && pip check
COPY scripts/setup-dicom.sh scripts/setup-dicom.sh
RUN KPRISM_PYTHON=/opt/venv/bin/python bash scripts/setup-dicom.sh

FROM python:3.10.14-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 1000 prism && useradd --uid 1000 --gid 1000 --create-home prism
COPY --from=python-build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    KPRISM_PYTHON=/opt/venv/bin/python \
    KPRISM_HOST=0.0.0.0 KPRISM_PORT=8502 \
    KPRISM_WORKSPACE=/data/workspace KPRISM_DEVICE=auto \
    VERSE_CHECKPOINT=/app/weight/cardiac_achilleas_epoch10.pth \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY server/ server/
COPY Verse_2_MOE_ui/ Verse_2_MOE_ui/
COPY scripts/start.sh scripts/start.sh
COPY scripts/check_environment.py scripts/check_environment.py
COPY start.sh ./
COPY --from=frontend /build/frontend/dist frontend/dist
COPY --from=python-build /build/runtime/tools runtime/tools
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.txt ./
COPY licenses/ licenses/
RUN chmod -R a+rX /app && chmod +x start.sh scripts/start.sh && \
    mkdir -p /data/workspace /app/weight /app/demo-data/images /app/demo-data/references && \
    chown -R prism:prism /data /app/weight /app/demo-data
USER prism
EXPOSE 8502
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8502/api/health', timeout=5).read()"
ENTRYPOINT ["/app/start.sh"]

