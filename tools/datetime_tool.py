"""Current date/time tool."""

from __future__ import annotations

import datetime

SPEC = {
    "name": "get_datetime",
    "description": "Get the current date and time on the user's machine.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def run(args: dict) -> str:
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")
