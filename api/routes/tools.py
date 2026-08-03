"""GET /tools, POST /tools/execute — introspect and manually invoke tools."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import ToolExecuteRequest
from src.tools.registry import TOOL_SPECS, list_tool_names, run_tool
from src.utils.exceptions import ToolExecutionError, ToolNotFoundError

router = APIRouter(tags=["tools"])


@router.get("/tools")
def get_tools() -> dict:
    return {"tools": list_tool_names(), "specs": TOOL_SPECS}


@router.post("/tools/execute")
def execute_tool(request: ToolExecuteRequest) -> dict:
    try:
        output = run_tool(request.name, request.arguments)
    except ToolNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ToolExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"name": request.name, "output": output}
