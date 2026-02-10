# CueCode Integrated Data Structure - mediapipe-fastapi MSA

## 1. Service Overview

| Item | Value |
|------|-------|
| **Service Name** | `mediapipe-fastapi` |
| **Role** | Motion Detection + AAC Sentence Generation |
| **Port** | `8000` |
| **K8s Service** | `ClusterIP` (`mediapipe-fastapi:8000`) |
| **Container Registry** | `cuecoderegistry.azurecr.io/mediapipe-fastapi` |
| **Tech Stack** | FastAPI, MediaPipe, LangGraph, OpenAI GPT, Google Gemini |

---

## 2. API Endpoints & Data Structures

### 2-1. `GET /health`

```json
{ "status": "ok" }
```

### 2-2. `POST /api/process-motion`

**Request** (`multipart/form-data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `phrase` | `string` | Y | Word/phrase corresponding to the motion |
| `detectionArea` | `string` | Y | `"face"` / `"eyes"` / `"hand"` / `"hands"` |
| `videoFile` | `File (mp4)` | Y | Recorded video file |

**Response - Face** (`detectionArea="face"`):

```json
{
  "phrase": "string",
  "detectionArea": "face",
  "motion_data": {
    "face_blendshapes": [
      {
        "timestamp_ms": 0,
        "values": {
          "browDownLeft": 0.0,
          "browDownRight": 0.0,
          "browInnerUp": 0.0,
          "eyeBlinkLeft": 0.0,
          "eyeBlinkRight": 0.0,
          "jawOpen": 0.0,
          "mouthSmileLeft": 0.0,
          "mouthSmileRight": 0.0
        }
      }
    ]
  }
}
```

> 52 facial blendshapes total, float values 0~1

**Response - Eyes** (`detectionArea="eyes"`):

```json
{
  "phrase": "string",
  "detectionArea": "eyes",
  "motion_data": {
    "eye_landmarks": [
      {
        "timestamp_ms": 0,
        "left_eye": [[x, y, z], ...],
        "right_eye": [[x, y, z], ...]
      }
    ]
  }
}
```

> 16 points per eye, float coordinates 0~1

**Response - Hands** (`detectionArea="hand"/"hands"`):

```json
{
  "phrase": "string",
  "detectionArea": "hand",
  "motion_data": {
    "hand_landmarks": [
      {
        "timestamp_ms": 0,
        "left_hand": [[x, y, z], ...],
        "right_hand": [[x, y, z], ...]
      }
    ]
  }
}
```

> 21 landmarks per hand, float coordinates 0~1, null if hand not detected

### 2-3. `POST /api/sentence/generate`

**Request** (`application/json`):

```json
{
  "user_id": "string"
}
```

**Response**:

```json
{
  "sentence": "string",
  "debug_trace": [
    {
      "step": "load_recent_phrases",
      "user_id": "string",
      "redis_key": "phrases:{user_id}",
      "raw_phrases": ["string"],
      "count": 0,
      "used_list_mode": true
    },
    {
      "step": "normalize_phrases",
      "raw_phrases": ["string"],
      "normalized_phrases": ["string"]
    },
    {
      "step": "intent_classifier",
      "phrases": ["string"],
      "final_intent": "EMERGENCY | REQUEST | STATUS | OTHER",
      "is_emergency": false
    },
    {
      "step": "normal_generate | emergency_generate",
      "draft_sentence": "string"
    },
    {
      "step": "refine_sentence",
      "refined_sentence": "string"
    },
    {
      "step": "normal_check | emergency_check",
      "rule_status": "OK | REWRITE | TOO_LONG | NOT_POLITE | UNCLEAR",
      "final_sentence": "string"
    }
  ]
}
```

---

## 3. Internal State Model (GraphState)

```
GraphState (TypedDict)
├── user_id: str                     # CueCode user unique ID
├── raw_phrases: List[str]           # Raw tokens from Redis
├── normalized_phrases: List[str]    # Deduplicated tokens
├── intent: Intent                   # "EMERGENCY" | "REQUEST" | "STATUS" | "OTHER"
├── draft_sentence: str              # LLM first-pass sentence
├── refined_sentence: str            # LLM refined sentence
├── final_sentence: str              # Final output sentence
├── rule_status: RuleStatus          # "OK" | "REWRITE" | "TOO_LONG" | "NOT_POLITE" | "UNCLEAR"
├── is_emergency: bool               # Emergency branch flag
├── error_message: Optional[str]     # Error message
└── debug_trace: List[Any]           # Step-by-step debug log
```

---

## 4. LangGraph Workflow

```
[load_recent_phrases] --> [normalize_phrases]
                               |
                     +--- empty ---> END
                     |
                  continue
                     |
               [intent_classifier]
                     |
           +-- EMERGENCY --+          +-- OTHER/REQUEST/STATUS --+
           v                           v
   [emergency_generate]         [normal_generate]
           |                           |
           v                           v
   [emergency_check]            [refine_sentence]
       |         |                     |
    OK -> END   REWRITE                v
       (max 2 retries)          [normal_check]
                                   |         |
                                OK -> END   NOT_OK
                                         (retry refine)
```

---

## 5. External Integrations (CueCode MSA)

### 5-1. Redis (Queue-based Communication)

| Item | Value |
|------|-------|
| **Key Pattern** | `phrases:{user_id}` |
| **Type** | Redis LIST (LPOP consume) / JSON string fallback |
| **Direction** | Other MSA -> Redis -> mediapipe-fastapi consumes |
| **K8s Secret** | `motion-redis-secret` |
| **Port** | `6380` (host) / `6379` (K8s pod) |

```
[Frontend / Other MSA]
        |
        |  RPUSH phrases:{user_id} "token1" "token2" ...
        v
    +--------+
    | Redis  |  (phrases:{user_id} = LIST)
    +--------+
        |
        |  LPOP (batch up to 200)
        v
[mediapipe-fastapi]
```

### 5-2. LLM External APIs

| Service | Model | Purpose | K8s Secret |
|---------|-------|---------|------------|
| OpenAI GPT | `gpt-4.1-2025-04-14` | Sentence generation/refinement | `llm-api-keys.OPENAI_API_KEY` |
| OpenAI GPT | `gpt-4.1-mini-2025-04-14` | Intent classification | `llm-api-keys.OPENAI_API_KEY` |
| Google Gemini | `gemini-2.5-flash` | Rule validation | `llm-api-keys.GEMINI_API_KEY` |

### 5-3. MediaPipe Models

| Model | File | Purpose |
|-------|------|---------|
| FaceLandmarker v2 | `face_landmarker_v2_with_blendshapes.task` | Face 478-point mesh + 52 blendshapes |
| Hands (Solutions) | Built-in | Hand 21 landmarks (both hands) |

---

## 6. CueCode Infrastructure Deployment

```
[GitHub CueCode-200OK/mediapipe-fastapi]
        |
        | push to master
        v
[GitHub Actions CI/CD]
        |
        +-- Docker Build (python:3.10-slim-bullseye)
        +-- Push -> cuecoderegistry.azurecr.io/mediapipe-fastapi:{sha}
        |
        v
[Azure AKS: cuecode-k8s]
        |
        +-- Deployment: mediapipe-fastapi (1 replica)
        |     +-- CPU: 200m~1000m
        |     +-- Memory: 512Mi~2Gi
        |     +-- Liveness: /health (10s interval)
        |     +-- Readiness: /health (5s interval)
        |     +-- Secrets: llm-api-keys, motion-db-secret, motion-redis-secret
        |
        +-- Service: ClusterIP -> :8000
```

---

## 7. End-to-End Data Flow (CueCode Integrated View)

```
+---------------------------------------------------------------------+
|                      CueCode Platform (AKS)                          |
|                                                                      |
|  [Client App / Frontend]                                             |
|       |                    |                                         |
|       | Video Upload       | RPUSH phrases:{uid}                    |
|       v                    v                                         |
|  +------------------+  +--------+                                    |
|  | mediapipe-fastapi |  | Redis  |                                   |
|  | (this service)    |<-| LIST   |                                   |
|  |                   |  +--------+                                   |
|  |  /api/process-    |                                               |
|  |    motion         |-> { phrase, detectionArea, motion_data }      |
|  |                   |     (face 52bs / eye 16pt*2 / hand 21lm*2)   |
|  |  /api/sentence/   |                                               |
|  |    generate       |-> { sentence, debug_trace }                   |
|  |                   |     (LangGraph: GPT gen -> Gemini validate)   |
|  +------------------+                                                |
|       |         |                                                    |
|       v         v                                                    |
|  [OpenAI API] [Gemini API]                                           |
|                                                                      |
+----------------------------------------------------------------------+
```

---

## 8. Environment Variables

### LLM Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI authentication |
| `GPT_MODEL` | `gpt-4.1-2025-04-14` | Sentence generation model |
| `GPT_INTENT_MODEL` | `gpt-4.1-mini-2025-04-14` | Intent classification model |
| `GEMINI_API_KEY` | - | Google Gemini authentication |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Rule validation model |

### Redis Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `127.0.0.1` | Redis host address |
| `REDIS_PORT` | `6380` | Redis port |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_PASSWORD` | - | Redis password (optional) |
| `REDIS_TIMEOUT` | `2.0` | Connection timeout (seconds) |

### Model Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/face_landmarker_v2_with_blendshapes.task` | MediaPipe face model path |
