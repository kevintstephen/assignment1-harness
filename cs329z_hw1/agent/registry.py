"""Tool registry for Problem 5 (tool_registry)."""

from __future__ import annotations

from jsonschema import Draft202012Validator

from cs329z_hw1.types import ToolResult, ToolSpec


class ToolRegistry:
    """Tools in registration order, validated before they run."""

    def __init__(self, tools: list[ToolSpec]):
        self._tools = list(tools)
        self._by_name = {tool.name: tool for tool in self._tools}

    def schemas(self) -> list[dict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools
        ]

    def validate(self, name: str, args: dict) -> ToolResult | None:
        return self._check(name, args)

    def execute(self, name: str, args: dict) -> ToolResult:
        try:
            rejected = self._check(name, args)
            if rejected is not None:
                return rejected
            value = self._by_name[name].fn(**args)
            return ToolResult(status="ok", content=str(value))
        except Exception as exc:
            return ToolResult(status="tool_error", content=str(exc))

    def _check(self, name: str, args: dict) -> ToolResult | None:
        tool = self._by_name.get(name)
        if tool is None:
            return ToolResult(status="unknown_tool", content=f"Unknown tool: {name!r}.")
        validator = Draft202012Validator(tool.parameters)
        errors = sorted(validator.iter_errors(args), key=lambda error: list(error.path))
        if not errors:
            return None
        detail = "; ".join(error.message for error in errors)
        return ToolResult(status="invalid_args", content=detail or "Invalid arguments.")
