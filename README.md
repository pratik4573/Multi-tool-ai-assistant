# Pratik AI — Intelligent Multi-Tool Personal AI Assistant

Pratik AI is a modular, offline-first personal AI assistant that supports
text and voice conversation, knows you personally through your own documents
(RAG) and long-term memory (SQLite), and can call tools to get things done.

See `docs` in this README for setup, architecture, and extension points.

## Architecture

```
frontend/ (Streamlit)  <--HTTP-->  src/api/ (FastAPI)
                                        |
        +-------------------------------+-------------------------------+
        |               |               |              |                |
     src/llm        src/rag         src/memory     src/speech       src/tools
  (Ollama +        (ChromaDB +      (SQLite         (mic/VAD/       (calculator,
   tool loop)       bge-small)      long-term        stt/tts)        weather, ...)
                                    memory)
```

## Quick start

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env

# Pull the default local model
ollama pull qwen2.5:3b
ollama serve

# Add your personal docs (resume.md, projects.md, etc.) to:
#   data/personal_docs/

# Run the backend
python main.py

# In another terminal, run the UI
streamlit run frontend/streamlit_app.py
```

## Project layout

See the top-level directory tree in the project plan. Key modules:

- `config.py` — every path/model/setting, nothing hardcoded elsewhere.
- `src/llm/orchestrator.py` — builds the prompt (system + RAG + memory),
  runs the Ollama tool-calling loop.
- `src/rag/` — embeds and retrieves your personal documents with ChromaDB.
- `src/memory/` — SQLite-backed long-term memory (facts, preferences, goals,
  reminders, projects) plus full chat history.
- `src/tools/registry.py` — central tool registry; each tool lives in its
  own file under `src/tools/`.
- `src/tts/base.py` + `piper_tts.py` + `xtts_tts.py` — swappable TTS engines.
- `src/api/` — FastAPI endpoints: `/chat`, `/voice`, `/upload`, `/search`,
  `/history`, `/memory`, `/tools`.
- `frontend/streamlit_app.py` — the chat UI.

## Adding a new tool

1. Create `src/tools/my_tool.py` with a `run(args: dict) -> str` function.
2. Add a JSON-schema spec + register it in `src/tools/registry.py`.
3. The LLM will call it automatically when relevant based on the description.

## Swapping models

- LLM: change `OLLAMA_MODEL` in `.env` or `config.py`.
- STT: change `PRATIK_WHISPER_MODEL` (tiny.en → large-v3).
- TTS: implement `src/tts/xtts_tts.py` fully and set `PRATIK_TTS_ENGINE=xtts`.
- Embeddings: change `PRATIK_EMBEDDING_MODEL` (must be a sentence-transformers
  compatible model).
