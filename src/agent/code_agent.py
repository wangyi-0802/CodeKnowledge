"""Single agent implementation with ReAct loop for code understanding.
Uses a clean function-calling loop (not LangGraph) for reliability.
"""
from __future__ import annotations
import json
from typing import Any
from src.agent.tools import CodeKnowledgeTools
from src.agent.prompts import SYSTEM_PROMPT
from src.utils.logger import get_logger
logger = get_logger(__name__)


class CodeKnowledgeAgent:
    """ReAct agent for code repository understanding.
    Uses a simple loop: think -> call tool -> observe -> repeat.
    """

    def _init_client(self, provider: str, api_key: str, base_url: str) -> Any:
        from openai import OpenAI
        if provider == "deepseek":
            return OpenAI(api_key=api_key or None, base_url="https://api.deepseek.com")
        return OpenAI(api_key=api_key or None, base_url=base_url or None)

    def run(self, query: str, thread_id: str = "default") -> str:
        if thread_id not in self._conversations:
            self._conversations[thread_id] = []

        history = self._conversations[thread_id][-6:]  # Last 3 turns max
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": query})
        tool_defs = self.tools.get_tool_definitions()
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.debug("Agent iteration %d/%d", iteration, self.max_iterations)

            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tool_defs,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=4096,
                )
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                return f"Error communicating with LLM: {e}"

            msg = response.choices[0].message

            # Check if LLM wants to call a tool
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    logger.info("Tool call: %s(%s)", tool_name, args)
                    tool_result = self._execute_tool(tool_name, args)

                    # Add assistant message with tool call
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        ],
                    })

                    # Add tool response
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result[:4000],  # Truncate long results
                    })

                # Continue loop for next LLM call
                continue

            # No tool call: this is the final answer
            answer = msg.content or "I could not find relevant information."
            if thread_id in self._conversations:
                self._conversations[thread_id].insert(0, {"role": "user", "content": query})
            return answer

        return "The agent reached the maximum number of iterations. Please try a more specific question."

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool and return formatted results."""
        tool_map = {
            "search_code": lambda: self.tools.search_code(**args),
            "read_file": lambda: self.tools.read_file(**args),
            "list_files": lambda: self.tools.list_files(**args),
        }
        if tool_name not in tool_map:
            return f"Unknown tool: {tool_name}"
        try:
            result = tool_map[tool_name]()
            if isinstance(result, list):
                parts = []
                for r in result[:5]:
                    loc = f"{r['file_path']}:L{r['start_line']}-L{r['end_line']}"
                    parts.append(
                        f"[{loc}] ({r['symbol_kind']}: {r['symbol_name']}, "
                        f"score: {r['relevance_score']:.2f})\n{r['content']}"
                    )
                return "\n---\n".join(parts) if parts else "No results found."
            return str(result)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            return f"Error: {e}"

    def __init__(self, tools, llm_provider="openai", model_name="gpt-4o-mini",
                 max_iterations=6, api_key="", base_url=""):
        self.tools = tools
        self.max_iterations = max_iterations
        self.model_name = model_name
        self._client = self._init_client(llm_provider, api_key, base_url)
        self._conversations: dict[str, list[dict]] = {}

    def reset_conversation(self, thread_id: str = "default") -> None:
        self._conversations[thread_id] = []

    @classmethod
    def create_with_defaults(
        cls, tools: CodeKnowledgeTools, settings: Any | None = None
    ) -> CodeKnowledgeAgent:
        if settings is None:
            from src.config.settings import get_settings
            settings = get_settings()
        llm_config = settings.llm_config
        return cls(
            tools=tools,
            llm_provider=settings.llm_provider,
            model_name=llm_config.get("model", "gpt-4o-mini"),
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url", ""),
        )