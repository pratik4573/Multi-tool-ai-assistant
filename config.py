"""
Central configuration for Pratik AI.

Every tunable value, model name, and filesystem path used anywhere in the
project is defined here and nowhere else. Modules must import from this
file instead of hardcoding paths or model identifiers.

Values can be overridden via environment variables (see .env.example),
which makes it easy to run the same code on different machines without
editing source files.
"""

from pathlib import Path
from dotenv import load_dotenv
import os


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
EXAMPLE_ENV_PATH = BASE_DIR / ".env.example"

# Load real environment variables first, then fall back to the example file
# so the app still works if a user forgets to copy .env.example to .env.
loaded_real_env = load_dotenv(ENV_PATH, override=False)
loaded_example_env = load_dotenv(EXAMPLE_ENV_PATH, override=False)
loaded_env_source = (
    ".env"
    if loaded_real_env
    else ".env.example"
    if loaded_example_env
    else "none"
)

# --------------------------------------------------------------------------
# Project root / base paths
# --------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
PERSONAL_DOCS_DIR: Path = DATA_DIR / "personal_docs"
PERSONAL_NOTES_DIR: Path = PERSONAL_DOCS_DIR / "notes"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
CHROMA_DB_DIR: Path = DATA_DIR / "chroma_db"
SQLITE_DB_PATH: Path = DATA_DIR / "memory.sqlite3"
LOG_DIR: Path = BASE_DIR / "logs"

for _dir in (
    DATA_DIR,
    PERSONAL_DOCS_DIR,
    PERSONAL_NOTES_DIR,
    UPLOADS_DIR,
    CHROMA_DB_DIR,
    LOG_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Application identity
# --------------------------------------------------------------------------
APP_NAME: str = "Pratik AI"
APP_VERSION: str = "1.0.0"

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("PRATIK_LOG_LEVEL", "INFO")
LOG_FILE: Path = LOG_DIR / "pratik_ai.log"

# --------------------------------------------------------------------------
# Audio (shared by STT / VAD / playback)
# --------------------------------------------------------------------------
SAMPLE_RATE: int = 16000          # required by Whisper + Silero VAD
CHANNELS: int = 1
BLOCK_DURATION_MS: int = 30       # audio frame size fed to VAD (10/20/30 ms only)
SILENCE_TIMEOUT_SEC: float = 1.0  # silence duration that ends an utterance
MAX_UTTERANCE_SEC: int = 30       # hard cap so recording never hangs forever
VAD_THRESHOLD: float = 0.5        # Silero VAD speech-probability threshold

# --------------------------------------------------------------------------
# STT (faster-whisper)
# --------------------------------------------------------------------------
WHISPER_MODEL_SIZE: str = os.environ.get("PRATIK_WHISPER_MODEL", "small.en")
WHISPER_DEVICE: str = os.environ.get("PRATIK_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE: str = os.environ.get("PRATIK_WHISPER_COMPUTE", "int8")
WHISPER_LANGUAGE: str = "en"
WHISPER_BEAM_SIZE: int = 1

# --------------------------------------------------------------------------
# TTS
# --------------------------------------------------------------------------
# "piper" is the default, fully offline engine. "xtts" is a stub for later.
TTS_ENGINE: str = os.environ.get("PRATIK_TTS_ENGINE", "piper")

MODELS_DIR: Path = BASE_DIR / "models"
PIPER_MODEL_PATH: str = os.environ.get(
    "PIPER_MODEL_PATH", str(MODELS_DIR / "en_US-lessac-medium.onnx")
)
PIPER_CONFIG_PATH: str = os.environ.get(
    "PIPER_CONFIG_PATH", str(MODELS_DIR / "en_US-lessac-medium.onnx.json")
)
TTS_OUTPUT_DIR: Path = DATA_DIR / "tts_output"
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# XTTS v2 (not yet implemented, reserved for future voice cloning)
XTTS_MODEL_NAME: str = os.environ.get("XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
XTTS_SPEAKER_WAV: str = os.environ.get("XTTS_SPEAKER_WAV", "")  # path to a reference voice sample

# --------------------------------------------------------------------------
# LLM (Ollama - fully local)
# --------------------------------------------------------------------------
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
LLM_TEMPERATURE: float = float(os.environ.get("PRATIK_LLM_TEMPERATURE", "0.4"))
LLM_MAX_TOOL_HOPS: int = int(os.environ.get("PRATIK_LLM_MAX_TOOL_HOPS", "4"))
LLM_REQUEST_TIMEOUT_SEC: int = 600
LLM_CONTEXT_WINDOW: int = int(os.environ.get("PRATIK_LLM_CONTEXT_WINDOW", "8192"))

SYSTEM_PROMPT: str ="""You are Pratik AI, a personal AI assistant that knows the user
personally through their documents and long-term memory. You run fully offline.

Guidelines:
- For questions about the USER (their background, projects, preferences, history, etc.),
  rely only on the RELEVANT PERSONAL CONTEXT and RELEVANT MEMORIES provided to you.
  Do not invent facts about the user that are not supported by that context or memory.
- For general knowledge questions that are NOT about the user (e.g. facts, people,
  places, concepts, definitions), answer confidently and directly using your own
  knowledge. Do not refuse or hedge just because personal context or memory wasn't
  provided for these — that restriction only applies to claims about the user.
- When a user's request matches one of your available tools, call the tool instead
  of guessing or hallucinating an answer.
- Keep spoken answers concise (1-4 sentences) since they may be read aloud via
  text-to-speech. Text chat answers can be longer and use Markdown formatting.
- Be warm, direct, and helpful.
"""

# --------------------------------------------------------------------------
# RAG (personal knowledge retrieval)
# --------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = os.environ.get("PRATIK_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE: str = os.environ.get("PRATIK_EMBEDDING_DEVICE", "cpu")

VECTOR_STORE_BACKEND = os.environ.get("PRATIK_VECTOR_BACKEND", "faiss")  # "chroma" or "faiss"
CHROMA_COLLECTION_NAME: str = "pratik_personal_knowledge"

RAG_CHUNK_SIZE: int = 500          # characters per chunk
RAG_CHUNK_OVERLAP: int = 50        # characters of overlap between chunks
RAG_TOP_K: int = 4                 # number of chunks retrieved per query

# Minimum cosine similarity score to include a chunk. NOTE: all-MiniLM-L6-v2
# routinely produces genuinely-relevant matches in the ~0.2-0.45 range, not
# the 0.6-0.8+ range people expect from larger embedding models. The old
# value of 0.35 was silently discarding relevant chunks (e.g. resume.md /
# skills.md matches scoring 0.25-0.33) and left only 1 of 4 retrieved chunks
# ever reaching the LLM. Lowered so top_k does the real work of bounding
# context size, and the threshold just filters out true noise.
RAG_SCORE_THRESHOLD: float = 0.15  # minimum similarity score to include a chunk

# Supported personal document filenames (informational; loader accepts any
# .md/.txt/.pdf placed under PERSONAL_DOCS_DIR, this list is just what the
# spec calls out explicitly)
PERSONAL_DOC_FILENAMES = [
    "resume.md",
    "projects.md",
    "skills.md",
    "experience.md",
    "education.md",
    "about_me.md",
    "career.md",
    "goals.md",
    "certificates.md",
]

# --------------------------------------------------------------------------
# Memory (SQLite long-term memory)
# --------------------------------------------------------------------------
MEMORY_TOP_K: int = 5                 # number of memories injected per turn
MEMORY_EXTRACTION_ENABLED: bool = True  # auto-extract facts/preferences after each turn
MEMORY_KINDS = (
    "fact",
    "preference",
    "goal",
    "name",
    "reminder",
    "project",
)

# --------------------------------------------------------------------------
# Vision (placeholder for future image understanding)
# --------------------------------------------------------------------------
VISION_ENABLED: bool = False
VISION_MODEL_NAME: str = os.environ.get("PRATIK_VISION_MODEL", "llava:7b")

# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
WEB_SEARCH_ENABLED: bool = os.environ.get("PRATIK_WEB_SEARCH_ENABLED", "false").lower() == "true"
WEB_SEARCH_API_KEY: str = os.environ.get("SERPAPI_API_KEY", "")
WEATHER_API_KEY: str = os.environ.get("OPENWEATHER_API_KEY", "")
FILE_SEARCH_ROOTS = [str(Path.home())]

# Debug (temporary)
print("Loaded .env:", loaded_real_env)
print("Loaded .env.example:", loaded_example_env)
print("Weather Key present:", bool(WEATHER_API_KEY))
print("SerpAPI Key present:", bool(WEB_SEARCH_API_KEY))
print("Web Search enabled:", WEB_SEARCH_ENABLED)
# --------------------------------------------------------------------------
# API (FastAPI)
# --------------------------------------------------------------------------
API_HOST: str = os.environ.get("PRATIK_API_HOST", "0.0.0.0")
API_PORT: int = int(os.environ.get("PRATIK_API_PORT", "8000"))
API_CORS_ORIGINS = ["*"]

# --------------------------------------------------------------------------
# Frontend (Streamlit)
# --------------------------------------------------------------------------
STREAMLIT_API_BASE_URL = "http://127.0.0.1:57851/"

# --------------------------------------------------------------------------
# Wake word (optional, off by default)
# --------------------------------------------------------------------------
USE_WAKE_WORD: bool = False
WAKE_WORD: str = "hey pratik"