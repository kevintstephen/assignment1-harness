"""Text tool-call format for Problem 6 (tool_protocol)."""

from __future__ import annotations

import json

from cs329z_hw1.types import ParsedResponse, ToolCall

OPEN = "<tool_call>"
CLOSE = "</tool_call>"

_INSTRUCTIONS = """\
Your entire reply is one tool call and nothing else. Do not compute the result, echo the text, wait, or write the tool's output yourself. An empty reply is wrong. Do not wrap the call in a Markdown fence.

<tool_call>
{"name": "add", "arguments": {"a": 17, "b": 25}}
</tool_call>

arguments is one JSON object. Use JSON types: 0.5 and 3 are numbers, not strings; true, false, and null are literals. A line break inside a string is \\n. Copy strings exactly, including quotes and backslashes. Use only parameter names from that tool's schema. If a parameter lists an enum, the value must be one of those strings. A tool with no parameters is called with "arguments": {}. For big_output, pass n_lines and do not write the lines.
"""


def format_tool_call(text: str, call: ToolCall) -> str:
    payload = json.dumps(
        {"name": call.name, "arguments": call.args},
        ensure_ascii=False,
    )
    block = f"{OPEN}\n{payload}\n{CLOSE}"
    prose = text.strip()
    if prose:
        return f"{prose}\n{block}"
    return block


def parse_response(text: str) -> ParsedResponse:
    try:
        return _parse_response(text if isinstance(text, str) else str(text))
    except Exception as exc:
        shown = text.strip() if isinstance(text, str) else ""
        message = str(exc).strip() or "Could not parse the reply."
        return ParsedResponse(text=shown, tool_call=None, error=message)


def _parse_response(text: str) -> ParsedResponse:
    start = text.find(OPEN)
    if start < 0:
        bare = _first_bare_call(text)
        if bare is not None:
            return bare
        return ParsedResponse(text=text.strip(), tool_call=None, error=None)

    rest = text[start + len(OPEN) :]
    decoder = json.JSONDecoder()
    index = 0
    while index < len(rest) and rest[index].isspace():
        index += 1
    try:
        payload, end = decoder.raw_decode(rest, index)
    except json.JSONDecodeError:
        return ParsedResponse(
            text=text.strip(),
            tool_call=None,
            error="Tool call JSON is incomplete or invalid.",
        )

    tail = rest[end:]
    if OPEN in tail:
        return ParsedResponse(
            text=text.strip(),
            tool_call=None,
            error="A reply can contain only one tool call.",
        )

    if not isinstance(payload, dict):
        return _bad_shape(text)
    name = payload.get("name")
    args = payload.get("arguments")
    if not isinstance(name, str) or not isinstance(args, dict):
        return _bad_shape(text)

    closer = tail.lstrip()
    if closer.startswith(CLOSE):
        tail = closer[len(CLOSE) :]
    prose = (text[:start] + tail).strip()
    return ParsedResponse(text=prose, tool_call=ToolCall(name=name, args=args), error=None)


def _first_bare_call(text: str) -> ParsedResponse | None:
    """A model often writes the JSON object and skips the marker."""
    decoder = json.JSONDecoder()
    index = 0
    while True:
        brace = text.find("{", index)
        if brace < 0:
            return None
        try:
            payload, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            index = brace + 1
            continue
        name = payload.get("name") if isinstance(payload, dict) else None
        args = payload.get("arguments") if isinstance(payload, dict) else None
        if isinstance(name, str) and isinstance(args, dict):
            prose = (text[:brace] + text[end:]).strip()
            return ParsedResponse(text=prose, tool_call=ToolCall(name=name, args=args), error=None)
        index = end


def _bad_shape(text: str) -> ParsedResponse:
    return ParsedResponse(
        text=text.strip(),
        tool_call=None,
        error="Tool call must be a JSON object with a string name and an arguments object.",
    )


def render_tools(schemas: list[dict]) -> str:
    blocks = [_INSTRUCTIONS.rstrip(), "", "Tools:"]
    for schema in schemas:
        name = str(schema.get("name", ""))
        description = str(schema.get("description", "")).strip()
        parameters = schema.get("parameters") or {}
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        names = ", ".join(properties) if isinstance(properties, dict) and properties else "(none)"
        dumped = json.dumps(parameters, ensure_ascii=False, indent=2)
        blocks.append(f"\n## {name}\n{description}\nParameter names: {names}\n{dumped}")
    return "\n".join(blocks).strip() + "\n"
