"""A self-contained LangGraph example for a small agentic AI loop.

The file intentionally defines its own contracts, planners, tools, graph, and
CLI so it can be read independently from the custom loop example.

High-level flow:
1. The user gives a goal, such as "How many paid leave days do I get?"
2. A planner decides the next action: call a tool, or return a final answer.
3. If the planner chooses a tool, the LangGraph tool node executes that tool.
4. The tool output is saved as an observation in graph state.
5. The graph loops back to the planner with the new observation.
6. Once observations are enough, the planner returns the special
   ``final_answer`` action and LangGraph stops.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable, Literal, Protocol, TypedDict
from zoneinfo import ZoneInfo

from langgraph.graph import END, StateGraph


LOGGER = logging.getLogger("agentic_ai_langgraph")

# Default paths are resolved relative to the repository root so this script can
# be run from the project directory without passing extra command-line flags.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "agentic_ai_config.json"
DEFAULT_LOCAL_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "flan-t5-small"
DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "agentic_ai_langgraph.log"

# Logs are displayed in IST because the intended user/debugging timezone is IST.
IST = ZoneInfo("Asia/Kolkata")


class RealtimeFileHandler(logging.FileHandler):
    """File logger that forces each record to disk immediately."""

    def emit(self, record: logging.LogRecord) -> None:
        # ``FileHandler`` already writes the record, but editors may not show it
        # immediately unless Python flushes and the OS commits it to disk.
        super().emit(record)
        self.flush()
        if self.stream is not None:
            os.fsync(self.stream.fileno())


class ISTFormatter(logging.Formatter):
    """Logging formatter that displays timestamps in Indian Standard Time."""

    def formatTime(
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        # ``record.created`` is a UNIX timestamp. Convert it to IST before the
        # normal logging formatter turns it into text.
        record_time = datetime.fromtimestamp(record.created, tz=IST)
        if datefmt is not None:
            return record_time.strftime(datefmt)
        return record_time.isoformat()


@dataclass(frozen=True)
class ToolResult:
    """Standard output from a tool.

    ``ok`` tells the agent whether the tool succeeded, and ``content`` is the
    observation that will be added back into the graph state.
    """

    ok: bool
    content: str


@dataclass(frozen=True)
class Tool:
    """Metadata plus the Python function for one callable agent tool."""

    name: str
    description: str
    run: Callable[[dict[str, str]], ToolResult]


@dataclass(frozen=True)
class AgentAction:
    """The planner's decision for the next graph step.

    ``tool_name`` can be a real tool or the special ``final_answer`` action.
    """

    thought: str
    tool_name: str
    tool_input: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    """Runtime limits for the graph agent."""

    max_steps: int = 5


@dataclass
class PlannerState:
    """Small state object passed into planners.

    This is separate from ``AgentGraphState`` so planners only see the user goal
    and observations, not LangGraph's execution bookkeeping.
    """

    user_goal: str
    observations: list[str] = field(default_factory=list)


class ToolRegistry:
    """Keeps graph tool lookup explicit and easy to audit."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        """Index tools by name so graph nodes can call them safely."""

        # The planner returns a string tool name. The graph uses this dictionary
        # to turn that string into the actual Python function to execute.
        self._tools = {tool.name: tool for tool in tools}
        LOGGER.info("[ToolRegistry.__init__] registered tools: %s", ", ".join(self.names()))

    def get(self, name: str) -> Tool | None:
        """Return a registered tool by name, or ``None`` if it is unknown."""

        tool = self._tools.get(name)
        LOGGER.info(
            "[ToolRegistry.get] requested=%s found=%s",
            name,
            tool is not None,
        )
        return tool

    def names(self) -> list[str]:
        """Return all available tool names in stable sorted order."""

        return sorted(self._tools)


class Planner(Protocol):
    """Shared contract for rule-based and LLM-based planners."""

    def next_action(self, state: PlannerState) -> AgentAction:
        """Choose the next graph action from the current state."""


class RuleBasedPlanner:
    """Deterministic planner used to keep the LangGraph example runnable."""

    def next_action(self, state: PlannerState) -> AgentAction:
        """Choose the next tool with simple keyword rules.

        On the first step it routes the request to one tool. After any
        observation exists, it stops the graph with a ``final_answer`` action.
        """

        goal = state.user_goal.lower()
        LOGGER.info(
            "[RuleBasedPlanner.next_action] goal=%r observations=%d",
            state.user_goal,
            len(state.observations),
        )

        # First decision: if no tool has run yet, choose the most relevant tool
        # from the user's goal. This keeps the first graph loop simple:
        # planner -> one tool.
        if not state.observations:
            # Leave/vacation/PTO questions should use the small policy search
            # knowledge base because the answer is stored there.
            if re.search(r"\b(leave|vacation|pto)\b", goal):
                LOGGER.info(
                    "[RuleBasedPlanner.next_action] routing to search_company_policy"
                )
                return AgentAction(
                    thought="The goal asks about leave policy, so retrieve policy context first.",
                    tool_name="search_company_policy",
                    tool_input={"query": state.user_goal},
                )

            # Date/deadline questions need date arithmetic, so the planner
            # routes to the calculator tool instead of policy search.
            if any(keyword in goal for keyword in ("days until", "deadline", "date")):
                LOGGER.info(
                    "[RuleBasedPlanner.next_action] routing to calculate_days_until"
                )
                return AgentAction(
                    thought="The goal needs date arithmetic, so use the date calculator.",
                    tool_name="calculate_days_until",
                    tool_input={"query": state.user_goal},
                )

            # If the rule planner cannot confidently classify the request, it
            # creates a support ticket. This is the human-triage fallback path.
            LOGGER.info(
                "[RuleBasedPlanner.next_action] routing to create_support_ticket"
            )
            return AgentAction(
                thought="The goal is unclear, so create a triage ticket for a human.",
                tool_name="create_support_ticket",
                tool_input={"summary": state.user_goal},
            )

        # Second decision: once any observation exists, this demo assumes the
        # agent has enough evidence to stop. A larger production agent might do
        # more reasoning here and possibly call another tool.
        LOGGER.info("[RuleBasedPlanner.next_action] observations found, finalizing")
        return AgentAction(
            thought="The available observations are enough to answer the user.",
            tool_name="final_answer",
            tool_input={"answer": self._build_answer(state)},
        )

    @staticmethod
    def _build_answer(state: PlannerState) -> str:
        """Turn collected observations into the final user-facing answer."""

        LOGGER.info(
            "[RuleBasedPlanner._build_answer] building answer from %d observations",
            len(state.observations),
        )
        # Observations are the facts returned by tools. The final answer simply
        # presents those facts and reminds the user to verify critical decisions.
        evidence = "\n".join(f"- {observation}" for observation in state.observations)
        return (
            "Based on the tools I used, here is the answer:\n"
            f"{evidence}\n\n"
            "Next step: verify this against the latest HR or operations system "
            "before making a business-critical decision."
        )


class LLMPlanner:
    """Planner that asks an LLM which graph action should run next."""

    def __init__(
        self,
        tools: ToolRegistry,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        api_env_var: str = "OPENAI_API_KEY",
        provider_name: str = "OpenAI",
    ) -> None:
        """Create an OpenAI-compatible client used only for planning.

        ``base_url`` lets the same planner work with providers that expose an
        OpenAI-compatible chat completions API, such as Gemini.
        """

        LOGGER.info(
            "[LLMPlanner.__init__] initializing %s planner model=%s",
            provider_name,
            model,
        )
        # The API key can come from the JSON config file or from the environment.
        # This lets local development keep secrets out of source code.
        resolved_api_key = api_key or os.getenv(api_env_var)
        if not resolved_api_key:
            raise ValueError(
                f"{provider_name} API key not found. Add it to "
                f"agentic_ai_config.json or set {api_env_var}."
            )

        from openai import OpenAI

        # Gemini is called through an OpenAI-compatible base URL, while OpenAI
        # itself uses the default client URL. The rest of the planner code does
        # not need to know which provider is behind the client.
        if base_url:
            self.client = OpenAI(api_key=resolved_api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=resolved_api_key)
        self.model = model
        self.tools = tools
        self.provider_name = provider_name
        LOGGER.info("[LLMPlanner.__init__] %s client initialized", provider_name)

    def next_action(self, state: PlannerState) -> AgentAction:
        """Ask the configured chat model to return the next ``AgentAction``."""

        LOGGER.info(
            "[LLMPlanner.next_action] requesting action from %s model=%s observations=%d",
            self.provider_name,
            self.model,
            len(state.observations),
        )
        # The LLM is not asked to answer the user directly. It is only asked to
        # produce structured JSON describing the next graph action.
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_goal": state.user_goal,
                            "observations": state.observations,
                        }
                    ),
                },
            ],
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM planner returned an empty response.")

        LOGGER.info("[LLMPlanner.next_action] received planner JSON: %s", content)
        return self._parse_action(content)

    def _system_prompt(self) -> str:
        """Build the planner instructions, including available tool names."""

        LOGGER.info("[LLMPlanner._system_prompt] building planner system prompt")
        tool_descriptions = "\n".join(
            f"- {name}: {self.tools.get(name).description}"
            for name in self.tools.names()
            if self.tools.get(name) is not None
        )
        # ``final_answer`` is not a real tool in ToolRegistry. It is a reserved
        # action name that tells the graph to stop.
        available_tools = ", ".join([*self.tools.names(), "final_answer"])

        return (
            "You are an agent planner inside a LangGraph workflow. "
            "Choose exactly one next action.\n"
            "Return only valid JSON with this schema:\n"
            "{"
            '"thought": "short reason", '
            '"tool_name": "tool name", '
            '"tool_input": {"key": "value"}'
            "}\n\n"
            "Available tools:\n"
            f"{tool_descriptions}\n"
            "- final_answer: Use only when observations are enough to answer.\n\n"
            f"Allowed tool_name values: {available_tools}\n"
            "If there are no useful observations yet, choose the best tool. "
            "If there are observations, usually return final_answer with an "
            "answer field inside tool_input."
        )

    @staticmethod
    def _parse_action(content: str) -> AgentAction:
        """Validate LLM JSON and convert it into an ``AgentAction`` object."""

        LOGGER.info("[LLMPlanner._parse_action] parsing planner response")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM planner returned invalid JSON: {content}") from exc

        thought = data.get("thought")
        tool_name = data.get("tool_name")
        tool_input = data.get("tool_input", {})

        if not isinstance(thought, str) or not thought:
            raise ValueError("LLM planner response must include a non-empty thought.")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("LLM planner response must include a non-empty tool_name.")
        if not isinstance(tool_input, dict):
            raise ValueError("LLM planner tool_input must be a JSON object.")

        # Tool inputs must be simple strings because the demo tools all accept
        # ``dict[str, str]``. This keeps the planner/tool contract predictable.
        normalized_input = {str(key): str(value) for key, value in tool_input.items()}
        return AgentAction(
            thought=thought,
            tool_name=tool_name,
            tool_input=normalized_input,
        )


class LocalLLMPlanner:
    """Planner that uses a small local Hugging Face model for tool selection."""

    def __init__(
        self,
        tools: ToolRegistry,
        model: str = "google/flan-t5-small",
        local_files_only: bool = False,
    ) -> None:
        """Load a local text-to-text model for offline tool selection."""

        LOGGER.info(
            "[LocalLLMPlanner.__init__] loading model=%s local_files_only=%s",
            model,
            local_files_only,
        )
        # Imports stay inside the constructor so rule-based and API-based runs
        # do not require transformers/torch unless the local planner is used.
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # The tokenizer converts text into token IDs. The model consumes those
        # token IDs and generates token IDs for the selected tool name.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            local_files_only=local_files_only,
        )
        self.model_runner = AutoModelForSeq2SeqLM.from_pretrained(
            model,
            local_files_only=local_files_only,
        )
        self.model = model
        self.tools = tools
        # Small local models can produce unexpected text. The deterministic
        # planner is the safety net when tool-name extraction fails.
        self.rule_fallback = RuleBasedPlanner()
        LOGGER.info("[LocalLLMPlanner.__init__] local model loaded")

    def next_action(self, state: PlannerState) -> AgentAction:
        """Use the local model to select a tool, then finalize after observation."""

        LOGGER.info(
            "[LocalLLMPlanner.next_action] goal=%r observations=%d",
            state.user_goal,
            len(state.observations),
        )
        # Local LLM planning is only used before a tool has run. After a tool
        # returns an observation, the planner has evidence and can finalize.
        if state.observations:
            LOGGER.info("[LocalLLMPlanner.next_action] observations found, finalizing")
            return AgentAction(
                thought="The available observations are enough to answer the user.",
                tool_name="final_answer",
                tool_input={"answer": RuleBasedPlanner._build_answer(state)},
            )

        # The prompt lists the exact tool names and their meanings. The model is
        # asked to return only one of those names, so the output can be mapped
        # back to ToolRegistry.
        available_tool_names = ", ".join(self.tools.names())
        prompt = (
            "Use the tool name and tool meaning to choose the best tool for the given user goal.\n"
            "Return only the tool name, with no extra words.\n\n"
            f"Available tool names: {available_tool_names}\n\n"
            "Tool meanings:\n"
            "- search_company_policy: company policy, leave, vacation, PTO, "
            "hybrid work, security, 2FA\n"
            "- calculate_days_until: deadlines, date arithmetic, days until "
            "a YYYY-MM-DD date\n"
            "- create_support_ticket: unclear requests or human triage\n\n"
            f"User goal: {state.user_goal}\n"
            "Tool name:"
        )
        LOGGER.info("[LocalLLMPlanner.next_action] final prompt sent to local LLM:\n%s", prompt)
        # Tokenize the final prompt and ask the model to generate a short answer.
        # ``do_sample=False`` makes generation deterministic for easier debugging.
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model_runner.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()
        LOGGER.info(
            "[LocalLLMPlanner.next_action] generated tool text=%r",
            generated_text,
        )
        tool_name = self._extract_tool_name(generated_text)

        # If the local model returns text that does not match a known tool, keep
        # the agent moving by falling back to the deterministic rule planner.
        if tool_name is None:
            LOGGER.warning(
                "[LocalLLMPlanner.next_action] unclear local model output; "
                "falling back to RuleBasedPlanner model=%s generated=%r",
                self.model,
                generated_text,
            )
            return self.rule_fallback.next_action(state)

        LOGGER.info("[LocalLLMPlanner.next_action] selected tool=%s", tool_name)
        # Convert the selected tool name into the input shape expected by that
        # specific tool, then return it as the next graph action.
        return AgentAction(
            thought=f"Local model selected {tool_name}.",
            tool_name=tool_name,
            tool_input=self._tool_input_for(tool_name, state.user_goal),
        )

    def _extract_tool_name(self, generated_text: str) -> str | None:
        """Find one known tool name in the local model's generated text."""

        LOGGER.info("[LocalLLMPlanner._extract_tool_name] extracting tool name")
        normalized_text = generated_text.lower()
        tool_names = self.tools.names()
        LOGGER.info(
            "[LocalLLMPlanner._extract_tool_name] available tools=%s",
            ", ".join(tool_names),
        )
        # First try exact matching. This is the preferred path because the model
        # was instructed to return one exact tool name.
        for tool_name in tool_names:
            if tool_name in normalized_text:
                LOGGER.info(
                    "[LocalLLMPlanner._extract_tool_name] exact match=%s",
                    tool_name,
                )
                return tool_name

        # If exact matching fails, try broad keywords. This helps when the model
        # returns "policy" instead of "search_company_policy", or "days" instead
        # of "calculate_days_until".
        if "policy" in normalized_text:
            LOGGER.info("[LocalLLMPlanner._extract_tool_name] keyword match=policy")
            return "search_company_policy"
        if "date" in normalized_text or "days" in normalized_text:
            LOGGER.info("[LocalLLMPlanner._extract_tool_name] keyword match=date/days")
            return "calculate_days_until"
        if "ticket" in normalized_text or "support" in normalized_text:
            LOGGER.info(
                "[LocalLLMPlanner._extract_tool_name] keyword match=ticket/support"
            )
            return "create_support_ticket"

        LOGGER.info("[LocalLLMPlanner._extract_tool_name] no tool match")
        return None

    @staticmethod
    def _tool_input_for(tool_name: str, user_goal: str) -> dict[str, str]:
        """Create the input dictionary expected by the selected tool."""

        LOGGER.info("[LocalLLMPlanner._tool_input_for] building input for %s", tool_name)
        # Ticket creation expects a ``summary`` key. The other tools search or
        # parse the user's original text, so they receive it under ``query``.
        if tool_name == "create_support_ticket":
            return {"summary": user_goal}
        return {"query": user_goal}


class AgentGraphState(TypedDict, total=False):
    """State passed between LangGraph nodes.

    LangGraph nodes read and write this dictionary as the workflow moves from
    planning, to tool execution, and finally to an answer.
    """

    user_goal: str
    # Tool outputs collected so far. Planners use this to decide whether they
    # should call another tool or create the final answer.
    observations: list[str]
    # Number of tool executions completed. This is used to enforce max_steps.
    step: int
    # Maximum number of tool executions allowed for one user goal.
    max_steps: int
    # Most recent planner decision. The route function reads this field.
    action: AgentAction
    # Populated only when the planner emits the special final_answer action.
    final_answer: str


class LangGraphAgent:
    """Graph-backed agent with a planner node and a tool node.

    The graph shape is: planner -> tools -> planner, until the planner emits
    ``final_answer`` and the graph routes to ``END``.
    """

    def __init__(
        self,
        planner: Planner,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
    ) -> None:
        """Store dependencies and compile the LangGraph workflow."""

        self.planner = planner
        self.tools = tools
        self.config = config or AgentConfig()
        self.graph = self._build_graph()
        LOGGER.info(
            "[LangGraphAgent.__init__] initialized with planner=%s max_steps=%d",
            type(planner).__name__,
            self.config.max_steps,
        )

    def run(self, user_goal: str) -> str:
        """Start the graph for one user goal and return the final answer."""

        LOGGER.info("[LangGraphAgent.run] starting graph for goal=%r", user_goal)
        # This is the first graph state. There are no observations yet because
        # no tool has run, and step starts at zero.
        initial_state: AgentGraphState = {
            "user_goal": user_goal,
            "observations": [],
            "step": 0,
            "max_steps": self.config.max_steps,
        }
        # The recursion limit protects the LangGraph loop from running forever.
        # The graph can visit planner and tool nodes for each step, plus a final
        # planner visit to return final_answer.
        result = self.graph.invoke(
            initial_state,
            config={"recursion_limit": self.config.max_steps * 2 + 2},
        )
        answer = result.get("final_answer")
        # If the graph ended without a final_answer, treat that as a bounded-loop
        # failure. In normal runs, _planner_node sets this field before END.
        if not answer:
            raise TimeoutError(
                f"Agent stopped after {self.config.max_steps} steps without a final answer."
            )

        LOGGER.info("[LangGraphAgent.run] final answer ready")
        return answer

    def _build_graph(self):
        """Create and compile the LangGraph state machine."""

        graph = StateGraph(AgentGraphState)
        # Node 1: planner decides what to do next.
        graph.add_node("planner", self._planner_node)
        # Node 2: tools executes the selected tool and records its observation.
        graph.add_node("tools", self._tool_node)
        # Every run starts by asking the planner what action to take.
        graph.set_entry_point("planner")
        # After planning, inspect the chosen action. If the planner chose the
        # special final_answer action, route to END. Otherwise, run a tool.
        graph.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {
                "tools": "tools",
                "end": END,
            },
        )
        # After a tool runs, loop back to the planner with the new observation.
        graph.add_edge("tools", "planner")
        return graph.compile()

    def _planner_node(self, state: AgentGraphState) -> AgentGraphState:
        """LangGraph node that asks the planner what should happen next."""

        # Copy observations out of graph state into PlannerState. This keeps
        # planner code independent from LangGraph-specific fields like step.
        observations = list(state.get("observations", []))
        planner_state = PlannerState(
            user_goal=state["user_goal"],
            observations=observations,
        )
        LOGGER.info(
            "[LangGraphAgent._planner_node] step=%d observations=%d",
            state.get("step", 0) + 1,
            len(observations),
        )

        action = self.planner.next_action(planner_state)
        LOGGER.info(
            "[LangGraphAgent._planner_node] planner chose tool=%s thought=%s",
            action.tool_name,
            action.thought,
        )

        # ``final_answer`` is the stop signal. The graph stores the answer in
        # state so LangGraphAgent.run can return it after graph.invoke finishes.
        if action.tool_name == "final_answer":
            answer = action.tool_input.get("answer")
            if answer is None:
                raise ValueError("final_answer action must include an answer field.")
            return {
                "action": action,
                "final_answer": answer,
            }

        # For normal tool actions, only store the planner decision. The route
        # function will send the graph to _tool_node next.
        return {"action": action}

    def _tool_node(self, state: AgentGraphState) -> AgentGraphState:
        """LangGraph node that executes the selected tool and records output."""

        # The planner node must have written an AgentAction. If not, the graph is
        # in an invalid state and should fail loudly.
        action = state.get("action")
        if action is None:
            raise ValueError("Planner node did not produce an action.")

        # Convert the planner's string tool name into an actual Tool object.
        tool = self.tools.get(action.tool_name)
        if tool is None:
            available_tools = ", ".join(self.tools.names())
            raise ValueError(
                f"Planner requested unknown tool '{action.tool_name}'. "
                f"Available tools: {available_tools}"
            )

        LOGGER.info(
            "[LangGraphAgent._tool_node] executing tool=%s input=%s",
            tool.name,
            action.tool_input,
        )
        result = tool.run(action.tool_input)
        # Even failed tool calls become observations. This lets the planner see
        # what happened instead of silently losing the failure.
        observation = result.content if result.ok else f"Tool failed: {result.content}"
        # LangGraph state updates are returned as a new partial state.
        observations = [*state.get("observations", []), observation]
        step = state.get("step", 0) + 1
        LOGGER.info(
            "[LangGraphAgent._tool_node] tool=%s ok=%s step=%d observation=%r",
            tool.name,
            result.ok,
            step,
            observation,
        )

        # Stop runaway execution. This demo raises immediately when the maximum
        # number of tool calls is reached without a final answer.
        if step >= state.get("max_steps", self.config.max_steps):
            raise TimeoutError(
                f"Agent stopped after {self.config.max_steps} steps without a final answer."
            )

        return {
            "observations": observations,
            "step": step,
        }

    @staticmethod
    def _route_after_planner(state: AgentGraphState) -> Literal["tools", "end"]:
        """Choose whether the graph should execute a tool or stop."""

        action = state.get("action")
        # The planner communicates "I am done" by returning tool_name
        # "final_answer". LangGraph then routes to END.
        if action is not None and action.tool_name == "final_answer":
            return "end"
        # Any other action means a real tool should run next.
        return "tools"


def search_company_policy(tool_input: dict[str, str]) -> ToolResult:
    """Search a tiny in-memory policy knowledge base for relevant snippets."""

    query = tool_input.get("query", "").lower()
    LOGGER.info("[search_company_policy] searching policy for query=%r", query)
    # This dictionary stands in for a real knowledge base or vector search. Each
    # document has keywords used for matching and text returned as evidence.
    policy_documents = {
        "leave": {
            "keywords": ("leave", "vacation", "pto", "paid time off"),
            "text": "Employees are entitled to 20 days of paid leave per year.",
        },
        "hybrid": {
            "keywords": ("hybrid", "remote", "office", "work policy"),
            "text": "The company follows a hybrid work policy.",
        },
        "security": {
            "keywords": ("security", "2fa", "two-factor", "internal tools"),
            "text": "Security policies require 2FA for all internal tools.",
        },
    }

    # Match any policy document whose keywords appear in the user's query.
    matches = [
        policy["text"]
        for policy in policy_documents.values()
        if any(keyword in query for keyword in policy["keywords"])
    ]

    # A miss is still a valid tool result, but ``ok=False`` tells the agent this
    # observation came from an unsuccessful lookup.
    if not matches:
        LOGGER.info("[search_company_policy] no matching policy found")
        return ToolResult(ok=False, content="No matching policy document found.")

    LOGGER.info("[search_company_policy] found %d matching policies", len(matches))
    return ToolResult(ok=True, content=" ".join(matches))


def calculate_days_until(tool_input: dict[str, str]) -> ToolResult:
    """Extract a YYYY-MM-DD date from the query and calculate days remaining."""

    query = tool_input.get("query", "")
    LOGGER.info("[calculate_days_until] parsing date from query=%r", query)
    # Keep the date parser intentionally strict so the example is predictable.
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)

    if not match:
        LOGGER.info("[calculate_days_until] no YYYY-MM-DD date found")
        return ToolResult(
            ok=False,
            content="Please include a target date in YYYY-MM-DD format.",
        )

    # Convert the matched string to a date and compare it with today's date.
    target_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
    days_remaining = (target_date - date.today()).days
    LOGGER.info(
        "[calculate_days_until] target_date=%s days_remaining=%d",
        target_date.isoformat(),
        days_remaining,
    )

    return ToolResult(
        ok=True,
        content=f"There are {days_remaining} days until {target_date.isoformat()}.",
    )


def create_support_ticket(tool_input: dict[str, str]) -> ToolResult:
    """Create a mock support ticket for requests needing human follow-up."""

    summary = tool_input.get("summary", "No summary provided.")
    LOGGER.info("[create_support_ticket] creating ticket for summary=%r", summary)
    # This is a fake ticket payload. In a real system this would call a ticketing
    # API such as Jira, ServiceNow, Zendesk, or an internal operations service.
    ticket = {
        "ticket_id": "SUP-1001",
        "summary": summary,
        "status": "created",
        "owner": "operations",
    }
    LOGGER.info("[create_support_ticket] created ticket_id=%s", ticket["ticket_id"])
    return ToolResult(ok=True, content=json.dumps(ticket, indent=2))


def load_agentic_ai_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, str]:
    """Load optional local config for LLM planner settings.

    Missing config files are allowed so the rule-based planner can run without
    setup. Values are converted to strings because planner configs are simple
    names, paths, URLs, and API keys.
    """

    LOGGER.info("[load_agentic_ai_config] loading config path=%s", config_path)
    # Config is optional because the rule planner and already-downloaded local
    # planner can run without API keys.
    if not config_path.exists():
        LOGGER.info("[load_agentic_ai_config] config file not found; using defaults")
        return {}

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")

    # Avoid logging secret-looking key names while still showing which non-secret
    # settings were loaded.
    safe_keys = [key for key in config if "key" not in str(key).lower()]
    LOGGER.info("[load_agentic_ai_config] loaded config keys excluding secrets=%s", safe_keys)
    return {str(key): str(value) for key, value in config.items()}


def download_local_model(
    model_name: str = "google/flan-t5-small",
    output_path: Path = DEFAULT_LOCAL_MODEL_PATH,
) -> None:
    """Download and save a Hugging Face model for offline local planner use."""

    LOGGER.info(
        "[download_local_model] downloading model=%s output_path=%s",
        model_name,
        output_path,
    )
    # Transformers are imported lazily so users who only run rule or API planner
    # modes do not need Hugging Face dependencies at import time.
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Save both tokenizer and model weights so later local-llm runs can load from
    # disk with local_files_only=True.
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
    LOGGER.info("[download_local_model] model saved")
    print(f"Saved local model to {output_path}")


def build_tools() -> ToolRegistry:
    """Create the three demo tools and register them by name."""

    LOGGER.info("[build_tools] creating tool registry")
    # Each Tool has a stable name, a human/LLM-readable description, and the
    # Python function that actually performs the work.
    return ToolRegistry(
        tools=[
            Tool(
                name="search_company_policy",
                description="Search internal company policy snippets.",
                run=search_company_policy,
            ),
            Tool(
                name="calculate_days_until",
                description="Calculate days until a YYYY-MM-DD date found in the query.",
                run=calculate_days_until,
            ),
            Tool(
                name="create_support_ticket",
                description="Create a support ticket when the request needs human triage.",
                run=create_support_ticket,
            ),
        ]
    )


def build_langgraph_agent(
    planner_type: str = "rule",
    model: str | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> LangGraphAgent:
    """Build a self-contained LangGraph agent for the selected planner type.

    This is the main factory used by the CLI. It wires together tools, config,
    planner choice, and the compiled LangGraph agent.
    """

    LOGGER.info(
        "[build_langgraph_agent] planner_type=%s model=%s config_path=%s",
        planner_type,
        model,
        config_path,
    )
    tools = build_tools()
    config = load_agentic_ai_config(config_path)

    # Planner selection is the main switch for this example. The graph shape
    # stays the same; only the planner implementation changes.
    if planner_type == "rule":
        LOGGER.info("[build_langgraph_agent] using RuleBasedPlanner")
        planner: Planner = RuleBasedPlanner()
    elif planner_type == "llm":
        # The API planner supports OpenAI by default and Gemini when config says
        # llm_provider=gemini.
        llm_provider = config.get("llm_provider", "openai").lower()
        if llm_provider == "gemini":
            configured_model = config.get("gemini_model")
            LOGGER.info(
                "[build_langgraph_agent] using Gemini-compatible LLMPlanner model=%s",
                configured_model or model or "gemini-2.0-flash",
            )
            planner = LLMPlanner(
                tools=tools,
                model=configured_model or model or "gemini-2.0-flash",
                api_key=config.get("gemini_api_key"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_env_var="GEMINI_API_KEY",
                provider_name="Gemini",
            )
        else:
            configured_model = config.get("openai_model")
            LOGGER.info(
                "[build_langgraph_agent] using OpenAI LLMPlanner model=%s",
                configured_model or model or "gpt-4o-mini",
            )
            planner = LLMPlanner(
                tools=tools,
                model=configured_model or model or "gpt-4o-mini",
                api_key=config.get("openai_api_key"),
            )
    elif planner_type == "local-llm":
        # Prefer an explicitly configured local model path, then the default
        # downloaded model folder, then a Hugging Face model name that may be
        # downloaded by transformers.
        configured_model_path = config.get("local_model_path")
        configured_model = config.get("local_model")
        local_model_path = Path(configured_model_path) if configured_model_path else None

        if local_model_path is not None and local_model_path.exists():
            model_for_local_planner = str(local_model_path)
            local_files_only = True
            LOGGER.info(
                "[build_langgraph_agent] using configured local_model_path=%s",
                model_for_local_planner,
            )
        elif DEFAULT_LOCAL_MODEL_PATH.exists():
            model_for_local_planner = str(DEFAULT_LOCAL_MODEL_PATH)
            local_files_only = True
            LOGGER.info(
                "[build_langgraph_agent] using default downloaded local model=%s",
                model_for_local_planner,
            )
        else:
            model_for_local_planner = configured_model or model or "google/flan-t5-small"
            local_files_only = False
            LOGGER.info(
                "[build_langgraph_agent] local model directory missing; model may be downloaded=%s",
                model_for_local_planner,
            )

        # local_files_only=True prevents network access when we know a local
        # model directory exists. Otherwise transformers may download the model.
        planner = LocalLLMPlanner(
            tools=tools,
            model=model_for_local_planner,
            local_files_only=local_files_only,
        )
    else:
        raise ValueError("planner_type must be 'rule', 'llm', or 'local-llm'.")

    # At this point all dependencies are wired. LangGraphAgent compiles the graph
    # in its constructor.
    return LangGraphAgent(planner=planner, tools=tools)


def configure_logging(log_file: Path) -> None:
    """Send readable flow logs to both console and a local log file."""

    # Ensure the target directory exists before FileHandler opens the log file.
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = ISTFormatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RealtimeFileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Replace any prior root handlers so repeated runs in notebooks or tests do
    # not duplicate every log line.
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    LOGGER.info("[configure_logging] writing logs to %s", log_file)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for running this file directly."""

    parser = argparse.ArgumentParser(description="Run a LangGraph agentic AI example.")
    parser.add_argument(
        "goal",
        nargs="?",
        default="How many paid leave days do employees get?",
        help="User goal for the agent to solve.",
    )
    parser.add_argument(
        "--planner",
        choices=["rule", "llm", "local-llm"],
        default="rule",
        help="Planner to use. local-llm runs without an API key.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model used by the selected LLM planner.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to agentic AI config JSON used by LLM planners.",
    )
    parser.add_argument(
        "--download-local-model",
        action="store_true",
        help="Download the local Hugging Face model and exit.",
    )
    parser.add_argument(
        "--local-model-path",
        type=Path,
        default=DEFAULT_LOCAL_MODEL_PATH,
        help="Directory used by --download-local-model and offline local-llm.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path where method-by-method flow logs are written.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint: configure logging, build the agent, and print the answer."""

    args = parse_args()
    configure_logging(args.log_file)
    LOGGER.info("[main] starting LangGraph agentic AI example")
    LOGGER.info(
        "[main] args planner=%s model=%s config=%s download_local_model=%s log_file=%s",
        args.planner,
        args.model,
        args.config,
        args.download_local_model,
        args.log_file,
    )

    # Optional one-time helper path: download the local model and exit without
    # running the agent loop.
    if args.download_local_model:
        config = load_agentic_ai_config(args.config)
        download_local_model(
            model_name=args.model or config.get("local_model") or "google/flan-t5-small",
            output_path=args.local_model_path,
        )
        return

    # Normal CLI path: build the requested planner/graph and solve the goal.
    answer = build_langgraph_agent(
        planner_type=args.planner,
        model=args.model,
        config_path=args.config,
    ).run(args.goal)
    LOGGER.info("[main] printing final answer")
    print(answer)

if __name__ == "__main__":
    main()
