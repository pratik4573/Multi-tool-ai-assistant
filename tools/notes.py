"""Notes tool — lets the assistant save and recall quick user notes.

Notes are stored as 'fact' memories with a distinguishing prefix so they
show up in normal memory search but stay easy to filter separately later.
"""

from __future__ import annotations

from src.memory.memory_store import MemoryStore

SPEC = {
    "name": "manage_notes",
    "description": (
        "Save a quick note for the user to recall later, or list existing notes. "
        "Use action='add' with 'content' to save a note, or action='list' to see recent notes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "list"]},
            "content": {"type": "string", "description": "Note text (required for action='add')."},
        },
        "required": ["action"],
    },
}

_NOTE_PREFIX = "[note] "


def run(args: dict) -> str:
    action = str(args.get("action", "")).strip().lower()
    store = MemoryStore()

    if action == "add":
        content = str(args.get("content", "")).strip()
        if not content:
            return "Error: no note content provided."
        store.add_memory(kind="fact", content=f"{_NOTE_PREFIX}{content}")
        return "Note saved."

    if action == "list":
        memories = store.list_memories(kind="fact", limit=200)
        notes = [m.content[len(_NOTE_PREFIX):] for m in memories if m.content.startswith(_NOTE_PREFIX)]
        if not notes:
            return "No notes saved yet."
        return "\n".join(f"- {n}" for n in notes[:20])

    return "Error: action must be 'add' or 'list'."
