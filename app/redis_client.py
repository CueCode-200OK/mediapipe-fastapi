import os
import redis

# Read configuration from environment
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6380"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_TIMEOUT = float(os.getenv("REDIS_TIMEOUT", "2.0"))

# Example resolution hints:
# - If your app runs in a pod: set REDIS_HOST=motion-redisdb and REDIS_PORT=6379
# - If your app runs in a Docker container on Windows and you port-forward on host: set REDIS_HOST=host.docker.internal and REDIS_PORT=6380
# - If your app runs on the host directly: use 127.0.0.1:6380

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    socket_connect_timeout=REDIS_TIMEOUT,
    socket_timeout=REDIS_TIMEOUT,
    decode_responses=True,
)

# optional simple health check helper
def ping() -> bool:
    try:
        return redis_client.ping()
    except Exception:
        return False

# convenience wrapper used in nodes.py (lrange returning list)
def lrange(key: str, start: int = 0, end: int = -1):
    try:
        return redis_client.lrange(key, start, end) or []
    except Exception:
        return []
