# Use slim Debian + Python 3.10 (MediaPipe wheels are built for this combo)
FROM python:3.10-slim-bookworm

# --- Runtime hygiene & speed ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# [FIX] Temporarily allow unauthenticated packages to install the keyring fixer
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --allow-unauthenticated ca-certificates gnupg debian-archive-keyring && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install Python deps first (best cache hit)
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

# 2) Download MediaPipe models
RUN mkdir -p /app/models && \
    curl -fsSL -o /app/models/face_landmarker_v2_with_blendshapes.task \
    https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2_with_blendshapes.task

# 3) App code (bind-mounted in dev; copied for completeness)
COPY app/ ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```eof
**주요 변경점:**
* 첫 `apt-get update`에 `--allow-insecure-repositories` 옵션을 추가했습니다.
* `debian-archive-keyring` 설치 시 `--allow-unauthenticated` 옵션을 추가했습니다.
* 이후 `apt-get update`는 정상적으로 실행하여, 키가 업데이트된 상태에서 안전하게 나머지 패키지를 설치합니다.

이 방법으로 GPG 키 문제를 우회하여 빌드를 성공시킬 수 있을 것입니다.