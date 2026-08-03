"""
Central tool registry for Pratik AI.
"""

from __future__ import annotations

import config
from src.tools import (
    calculator,
    datetime_tool,
    file_search,
    notes,
    open_application,
    read_pdf,
    system_info,
    weather,
    web_search,
)
from src.utils.exceptions import ToolExecutionError, ToolNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TOOL_MODULES = [
    calculator,
    datetime_tool,
    weather,
    file_search,
    system_info,
    notes,
    open_application,
    read_pdf,
]

if config.WEB_SEARCH_ENABLED and config.WEB_SEARCH_API_KEY:
    _TOOL_MODULES.append(web_search)
else:
    logger.info(
        "web_search tool disabled (WEB_SEARCH_ENABLED/API key not set); hiding from LLM."
    )

_REGISTRY = {
    module.SPEC["name"]: module
    for module in _TOOL_MODULES
}

TOOL_SPECS = [
    {
        "type": "function",
        "function": module.SPEC,
    }
    for module in _TOOL_MODULES
]


def list_tool_names() -> list[str]:
    return list(_REGISTRY.keys())


def run_tool(name: str, args: dict) -> str:
    module = _REGISTRY.get(name)

    if module is None:
        raise ToolNotFoundError(
            f"Tool '{name}' is not registered."
        )

    try:
        return module.run(args)

    except Exception as exc:
        logger.exception(
            "Tool '%s' raised an exception.",
            name,
        )
        raise ToolExecutionError(
            f"Tool '{name}' failed: {exc}"
        ) from exc