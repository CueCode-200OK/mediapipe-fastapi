#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status.

# This script is intended to be run from the root of the mediapipe-fastapi repository.

# === Configuration ===
# These variables should be set as environment variables in your CI/CD system
# NCP_CR_URL: e.g., cuecode.kr.ncr.ntruss.com
# NCP_CR_ACCESS_KEY_ID: Your access key
# NCP_CR_SECRET_KEY: Your secret key

# Image repository name
IMAGE_REPO_NAME="cuecode"

# Image name for this service
IMAGE_NAME="mediapipe-fastapi"

# Tag for the image. Use git commit hash for uniqueness. Defaults to "latest" if git command fails.
IMAGE_TAG=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")

# Full image tag
FULL_IMAGE_TAG="${NCP_CR_URL}/${IMAGE_REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"
LATEST_TAG="${NCP_CR_URL}/${IMAGE_REPO_NAME}/${IMAGE_NAME}:latest"


# === Build Process ===
echo "======================================================"
echo "Building and Pushing MediaPipe FastAPI Service"
echo "======================================================"

echo "Logging in to Naver Cloud Container Registry..."
docker login -u "${NCP_CR_ACCESS_KEY_ID}" -p "${NCP_CR_SECRET_KEY}" "${NCP_CR_URL}"

echo "\nBuilding Docker image..."
echo "Tag: ${FULL_IMAGE_TAG}"
docker build -t "${FULL_IMAGE_TAG}" .

echo "\nPushing image with tag: ${IMAGE_TAG}"
docker push "${FULL_IMAGE_TAG}"

echo "\nTagging image as 'latest'..."
docker tag "${FULL_IMAGE_TAG}" "${LATEST_TAG}"

echo "Pushing image with tag: 'latest'"
docker push "${LATEST_TAG}"

echo "\n======================================================"
echo "Build and push complete for ${IMAGE_NAME}."
echo "======================================================"
