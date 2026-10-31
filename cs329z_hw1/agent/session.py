"""Conversation session for Problem 7 (agent_loop)."""

from __future__ import annotations

from cs329z_hw1.agent.compact import fit
from cs329z_hw1.agent.protocol import parse_response, render_tools
from cs329z_hw1.agent.registry import ToolRegistry
from cs329z_hw1.agent.tools import tools_for_session
from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.tokens import count_message_tokens, count_tokens
from cs329z_hw1.types import AgentConfig, AgentResult, ToolResult, ToolSpec

_MALFORMED_STATUSES = ("unknown_tool", "invalid_args")

_SYSTEM = """\
You are an assistant. When you need a tool, your reply is only that tool call. When the tool results contain the answer, reply with the answer and no tool call. A count, a date, or a listing over a data file requires a shell command in that same reply. Do not say you cannot determine it, and do not ask the user to paste a document.
"""


class AgentSession:
    """One conversation. A second session does not share these messages."""

    def __init__(self, lm, tools: list[ToolSpec] | None, config: AgentConfig):
        self.lm = lm
        self.config = config
        self.aux_lm = config.aux_lm if config.aux_lm is not None else lm
        self.registry = ToolRegistry(tools_for_session(tools, config))
        self.messages: list[dict] = [
            {"role": "system", "content": _SYSTEM + render_tools(self.registry.schemas())}
        ]
        self.transcript: list[dict] = []
        self.mode = config.mode
        self._user_requests: list[str] = []

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def send(self, user_message: str) -> AgentResult:
        self._event({"type": "user", "content": user_message})
        self.messages.append({"role": "user", "content": user_message})
        self._user_requests.append(user_message)

        calls_made = 0
        tokens_spent = 0
        malformed_streak = 0

        while True:
            if calls_made >= self.config.max_turns:
                return self._finish(
                    "max_turns",
                    "Stopped: reached the maximum number of model calls for this message.",
                )
            if tokens_spent >= self.config.token_budget:
                return self._finish(
                    "token_budget",
                    "Stopped: reached the token budget for this message.",
                )

            self._take_steering()
            self.messages = fit(
                self.messages,
                self.config.context_budget,
                user_message,
                self.aux_lm,
                self._user_requests,
            )
            sent = self.messages
            try:
                reply = self.lm(sent)
            except OutputTruncated as exc:
                calls_made += 1
                tokens_spent += count_message_tokens(sent) + count_tokens(exc.text or "")
                visible = (exc.text or "").strip()
                feedback = (
                    f"The model reply was cut off: {visible}"
                    if visible
                    else "The model reply was cut off before it finished."
                )
                self._event({"type": "error", "content": feedback})
                self.messages.append({"role": "user", "content": feedback})
                malformed_streak += 1
                if malformed_streak >= 3:
                    return self._malformed()
                continue

            calls_made += 1
            tokens_spent += count_message_tokens(sent) + count_tokens(reply)
            self._event({"type": "assistant", "content": reply})
            self.messages.append({"role": "assistant", "content": reply})

            parsed = parse_response(reply)
            if parsed.error:
                feedback = parsed.error.strip() or "The reply could not be parsed."
                self._event({"type": "error", "content": feedback})
                self.messages.append({"role": "user", "content": feedback})
                malformed_streak += 1
                if malformed_streak >= 3:
                    return self._malformed()
                continue

            if parsed.tool_call is None:
                text = (parsed.text or "").strip() or reply.strip()
                if not text:
                    feedback = "The reply was empty. Send the answer, or one tool call."
                    self._event({"type": "error", "content": feedback})
                    self.messages.append({"role": "user", "content": feedback})
                    malformed_streak += 1
                    if malformed_streak >= 3:
                        return self._malformed()
                    continue
                return self._finish("done", text)

            call = parsed.tool_call
            self._event({"type": "tool_call", "name": call.name, "args": dict(call.args)})
            result = self._run_tool(call)
            self._event(
                {
                    "type": "tool_result",
                    "name": call.name,
                    "status": result.status,
                    "content": result.content,
                }
            )
            self.messages.append({"role": "user", "content": result.content})
            if result.status in _MALFORMED_STATUSES:
                malformed_streak += 1
                if malformed_streak >= 3:
                    return self._malformed()
            else:
                malformed_streak = 0

    def _take_steering(self) -> None:
        user = self.config.user
        if user is None:
            return
        incoming = user.pending_message()
        if not incoming:
            return
        self._event({"type": "user", "content": incoming})
        self.messages.append({"role": "user", "content": incoming})

    def _run_tool(self, call) -> ToolResult:
        rejected = self.registry.validate(call.name, call.args)
        if rejected is not None:
            return rejected
        if self._needs_approval(call.name):
            decision = self.config.user.approve(call)
            if not decision.approved:
                reason = decision.reason.strip() or "The user declined this action."
                return ToolResult(status="denied", content=reason)
        return self.registry.execute(call.name, call.args)

    def _needs_approval(self, name: str) -> bool:
        if self.mode != "confirm" or self.config.user is None:
            return False
        tool = self.registry.get(name)
        return tool is not None and not tool.read_only

    def _malformed(self) -> AgentResult:
        return self._finish(
            "malformed",
            "Stopped: the model produced three malformed replies in a row.",
        )

    def _event(self, event: dict) -> None:
        self.transcript.append(event)

    def _finish(self, status: str, text: str) -> AgentResult:
        if not text.strip():
            text = "Stopped."
        return AgentResult(text=text, status=status, transcript=list(self.transcript))
