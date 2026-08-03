"""File search tool — finds files by name pattern within allowed roots."""

from __future__ import annotations

import fnmatch
import os

import config

SPEC = {
    "name": "file_search",
    "description": (
        "Search for files by name pattern on the user's machine, within "
        "allowed directories (e.g. '*.pdf' or 'resume*')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Filename glob pattern, e.g. '*.pdf'."},
            "max_results": {"type": "integer", "description": "Maximum number of results (default 20)."},
        },
        "required": ["pattern"],
    },
}


def run(args: dict) -> str:
    pattern = str(args.get("pattern", "")).strip()
    max_results = int(args.get("max_results", 20))
    if not pattern:
        return "Error: no filename pattern provided."

    matches: list[str] = []
    for root_dir in config.FILE_SEARCH_ROOTS:
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if fnmatch.fnmatch(filename.lower(), pattern.lower()):
                    matches.append(os.path.join(dirpath, filename))
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break
        if len(matches) >= max_results:
            break

    if not matches:
        return f"No files matching '{pattern}' were found."
    return "\n".join(matches)
