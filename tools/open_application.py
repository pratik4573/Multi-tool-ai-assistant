"""Open-application tool — launches a local application by name."""

from __future__ import annotations

import platform
import subprocess

SPEC = {
    "name": "open_application",
    "description": "Open/launch a desktop application on the user's machine by name (e.g. 'notepad', 'calculator').",
    "parameters": {
        "type": "object",
        "properties": {
            "application_name": {"type": "string", "description": "Name of the application to open."}
        },
        "required": ["application_name"],
    },
}


def run(args: dict) -> str:
    app_name = str(args.get("application_name", "")).strip()
    if not app_name:
        return "Error: no application name provided."

    system = platform.system()
    try:
        if system == "Windows":
            subprocess.Popen(["start", "", app_name], shell=True)
        elif system == "Darwin":
            subprocess.Popen(["open", "-a", app_name])
        else:
            subprocess.Popen([app_name])
        return f"Opened '{app_name}'."
    except Exception as exc:  # noqa: BLE001
        return f"Error: could not open '{app_name}' ({exc})."
