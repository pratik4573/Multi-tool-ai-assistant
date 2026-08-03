"""
System information tool.

Returns information about the LOCAL COMPUTER running Pratik AI.
Only use this tool when the user explicitly asks about the computer,
operating system, CPU, RAM, disk, or system resource usage.
"""

from __future__ import annotations

import platform

SPEC = {
    "name": "system_info",
    "description": (
        "Get information about the local computer including operating "
        "system, CPU usage, RAM usage, and disk usage."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def run(args: dict) -> str:
    """
    Return a snapshot of the local machine's system information.
    """

    try:
        import psutil
    except ImportError:
        return (
            "The system information tool requires the 'psutil' package "
            "to be installed."
        )

    cpu = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()

    # Works correctly on Windows, Linux, and macOS
    disk = psutil.disk_usage(str(platform.system() == "Windows" and "C:\\" or "/"))

    return (
        f"Operating System: {platform.system()} {platform.release()}\n"
        f"CPU Usage: {cpu:.1f}%\n"
        f"RAM Usage: {memory.percent:.1f}% "
        f"({memory.used // (1024 ** 2)} MB / {memory.total // (1024 ** 2)} MB)\n"
        f"Disk Usage: {disk.percent:.1f}% "
        f"({disk.used // (1024 ** 3)} GB / {disk.total // (1024 ** 3)} GB)"
    )