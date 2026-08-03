"""Calculator tool — safely evaluates arithmetic expressions."""

from __future__ import annotations

import ast
import operator

SPEC = {
    "name": "calculator",
    "description": (
        "Evaluate a basic arithmetic expression (numbers, +, -, *, /, %, **, "
        "parentheses). Use this for any math the user asks for instead of "
        "computing it yourself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The arithmetic expression to evaluate, e.g. '(3 + 4) * 2'.",
            }
        },
        "required": ["expression"],
    },
}

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported or unsafe expression.")


def run(args: dict) -> str:
    expression = str(args.get("expression", "")).strip()
    if not expression:
        return "Error: no expression provided."
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Error: could not evaluate expression ({exc})."
