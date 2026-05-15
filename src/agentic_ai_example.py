"""A small, production-style example of an agentic AI loop.

This example is intentionally self-contained so it can run without API keys.
In a real system, the ``RuleBasedPlanner`` would usually be replaced by an LLM
planner that returns the same structured ``AgentAction`` contract.
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
from typing import Callable, Iterable, Protocol


LOGGER = logging.getLogger("agentic_ai_example")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "agentic_ai_config.json"
DEFAULT_LOCAL_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "flan-t5-small"
DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "agentic_ai_example.log"


@dataclass(frozen=True)
class ToolResult:
    """Standard response envelope returned by every tool."""

    ok: bool
    content: str


@dataclass(frozen=True)
class Tool:
    """A callable capability the agent can use."""

    name: str
    description: str
    run: Callable[[dict[str, str]], ToolResult] # this means the run function is a callable that takes a dictionary of strings and returns a ToolResult


@dataclass(frozen=True)
class AgentAction:
    """Structured planner output.

    ``tool_name='final_answer'`` is reserved as the terminal action.
    """

    thought: str
    tool_name: str
    tool_input: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 5

@dataclass
class AgentState:
    user_goal: str
    observations: list[str] = field(default_factory=list)


class ToolRegistry:
    """Keeps tool lookup explicit and easy to audit."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        LOGGER.info("[ToolRegistry.__init__] registered tools: %s", ", ".join(self.names()))

    def get(self, name: str) -> Tool | None:
        tool = self._tools.get(name)
        LOGGER.info(
            "[ToolRegistry.get] requested=%s found=%s",
            name,
            tool is not None,
        )
        return tool

    def names(self) -> list[str]:
        return sorted(self._tools)


class Planner(Protocol):
    """Shared contract for rule-based and LLM-based planners."""

    def next_action(self, state: AgentState) -> AgentAction:
        """Choose the next agent action from the current state."""


class RuleBasedPlanner:
    """Deterministic planner used to keep the example runnable everywhere."""

    def next_action(self, state: AgentState) -> AgentAction:
        goal = state.user_goal.lower()
        LOGGER.info(
            "[RuleBasedPlanner.next_action] goal=%r observations=%d",
            state.user_goal,
            len(state.observations),
        )

        if not state.observations:
            if re.search(r"\b(leave|vacation|pto)\b", goal):
                LOGGER.info(
                    "[RuleBasedPlanner.next_action] routing to search_company_policy"
                )
                return AgentAction(
                    thought="The goal asks about leave policy, so retrieve policy context first.",
                    tool_name="search_company_policy",
                    tool_input={"query": state.user_goal},
                )

            if any(keyword in goal for keyword in ("days until", "deadline", "date")):
                LOGGER.info(
                    "[RuleBasedPlanner.next_action] routing to calculate_days_until"
                )
                return AgentAction(
                    thought="The goal needs date arithmetic, so use the date calculator.",
                    tool_name="calculate_days_until",
                    tool_input={"query": state.user_goal},
                )

            LOGGER.info(
                "[RuleBasedPlanner.next_action] routing to create_support_ticket"
            )
            return AgentAction(
                thought="The goal is unclear, so create a triage ticket for a human.",
                tool_name="create_support_ticket",
                tool_input={"summary": state.user_goal},
            )

        LOGGER.info("[RuleBasedPlanner.next_action] observations found, finalizing")
        return AgentAction(
            thought="The available observations are enough to answer the user.",
            tool_name="final_answer",
            tool_input={"answer": self._build_answer(state)},
        )

    @staticmethod
    def _build_answer(state: AgentState) -> str:
        LOGGER.info(
            "[RuleBasedPlanner._build_answer] building answer from %d observations",
            len(state.observations),
        )
        evidence = "\n".join(f"- {observation}" for observation in state.observations)
        return (
            "Based on the tools I used, here is the answer:\n"
            f"{evidence}\n\n"
            "Next step: verify this against the latest HR or operations system "
            "before making a business-critical decision."
        )


class LLMPlanner:
    """Planner that asks an LLM which tool the agent should use next."""

    def __init__(
        self,
        tools: ToolRegistry,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        LOGGER.info("[LLMPlanner.__init__] initializing OpenAI planner model=%s", model)
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OpenAI API key not found. Add it to agentic_ai_config.json "
                "or set OPENAI_API_KEY."
            )

        from openai import OpenAI

        self.client = OpenAI(api_key=resolved_api_key)
        self.model = model
        self.tools = tools
        LOGGER.info("[LLMPlanner.__init__] OpenAI client initialized")

    def next_action(self, state: AgentState) -> AgentAction:
        LOGGER.info(
            "[LLMPlanner.next_action] requesting action from OpenAI model=%s observations=%d",
            self.model,
            len(state.observations),
        )
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
        LOGGER.info("[LLMPlanner._system_prompt] building planner system prompt")
        tool_descriptions = "\n".join(
            f"- {name}: {self.tools.get(name).description}"
            for name in self.tools.names()
            if self.tools.get(name) is not None
        )
        available_tools = ", ".join([*self.tools.names(), "final_answer"])

        return (
            "You are an agent planner. Choose exactly one next action.\n"
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
        LOGGER.info(
            "[LocalLLMPlanner.__init__] loading model=%s local_files_only=%s",
            model,
            local_files_only,
        )
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
        self.rule_fallback = RuleBasedPlanner()
        LOGGER.info("[LocalLLMPlanner.__init__] local model loaded")

    def next_action(self, state: AgentState) -> AgentAction:
        LOGGER.info(
            "[LocalLLMPlanner.next_action] goal=%r observations=%d",
            state.user_goal,
            len(state.observations),
        )
        if state.observations:
            LOGGER.info("[LocalLLMPlanner.next_action] observations found, finalizing")
            return AgentAction(
                thought="The available observations are enough to answer the user.",
                tool_name="final_answer",
                tool_input={"answer": RuleBasedPlanner._build_answer(state)},
            )

        prompt = (
            "Choose the best tool for this user goal.\n"
            "Return exactly one tool name from this list:\n"
            "search_company_policy, calculate_days_until, create_support_ticket.\n\n"
            "Tool meanings:\n"
            "- search_company_policy: company policy, leave, vacation, PTO, "
            "hybrid work, security, 2FA\n"
            "- calculate_days_until: deadlines, date arithmetic, days until "
            "a YYYY-MM-DD date\n"
            "- create_support_ticket: unclear requests or human triage\n\n"
            f"User goal: {state.user_goal}\n"
            "Tool name:"
        )
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

        if tool_name is None:
            LOGGER.warning(
                "[LocalLLMPlanner.next_action] unclear local model output; "
                "falling back to RuleBasedPlanner model=%s generated=%r",
                self.model,
                generated_text,
            )
            return self.rule_fallback.next_action(state)

        LOGGER.info("[LocalLLMPlanner.next_action] selected tool=%s", tool_name)
        return AgentAction(
            thought=f"Local model selected {tool_name}.",
            tool_name=tool_name,
            tool_input=self._tool_input_for(tool_name, state.user_goal),
        )

    def _extract_tool_name(self, generated_text: str) -> str | None:
        LOGGER.info("[LocalLLMPlanner._extract_tool_name] extracting tool name")
        normalized_text = generated_text.lower()
        for tool_name in self.tools.names():
            if tool_name in normalized_text:
                LOGGER.info(
                    "[LocalLLMPlanner._extract_tool_name] exact match=%s",
                    tool_name,
                )
                return tool_name

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
        LOGGER.info("[LocalLLMPlanner._tool_input_for] building input for %s", tool_name)
        if tool_name == "create_support_ticket":
            return {"summary": user_goal}
        return {"query": user_goal}


class Agent:
    """Simple plan-act-observe loop with bounded execution."""

    def __init__(
        self,
        planner: Planner,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
    ) -> None:
        self.planner = planner
        self.tools = tools
        self.config = config or AgentConfig()
        LOGGER.info(
            "[Agent.__init__] initialized with planner=%s max_steps=%d",
            type(planner).__name__,
            self.config.max_steps,
        )

    def run(self, user_goal: str) -> str:
        LOGGER.info("[Agent.run] starting agent for goal=%r", user_goal)
        state = AgentState(user_goal=user_goal)

        for step in range(1, self.config.max_steps + 1):
            LOGGER.info("[Agent.run] step=%d asking planner for next action", step)
            action = self.planner.next_action(state)
            LOGGER.info(
                "[Agent.run] step=%d planner chose tool=%s thought=%s",
                step,
                action.tool_name,
                action.thought,
            )

            if action.tool_name == "final_answer":
                answer = action.tool_input.get("answer")
                if answer is None:
                    raise ValueError("final_answer action must include an answer field.")
                LOGGER.info("[Agent.run] final answer ready")
                return answer

            tool = self.tools.get(action.tool_name)
            if tool is None:
                available_tools = ", ".join(self.tools.names())
                raise ValueError(
                    f"Planner requested unknown tool '{action.tool_name}'. "
                    f"Available tools: {available_tools}"
                )

            LOGGER.info("[Agent.run] executing tool=%s input=%s", tool.name, action.tool_input)
            result = tool.run(action.tool_input)
            observation = result.content if result.ok else f"Tool failed: {result.content}"
            LOGGER.info(
                "[Agent.run] tool=%s ok=%s observation=%r",
                tool.name,
                result.ok,
                observation,
            )
            state.observations.append(observation)

        raise TimeoutError(
            f"Agent stopped after {self.config.max_steps} steps without a final answer."
        )


def search_company_policy(tool_input: dict[str, str]) -> ToolResult:
    query = tool_input.get("query", "").lower()
    LOGGER.info("[search_company_policy] searching policy for query=%r", query)
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

    matches = [
        policy["text"]
        for policy in policy_documents.values()
        if any(keyword in query for keyword in policy["keywords"])
    ]

    if not matches:
        LOGGER.info("[search_company_policy] no matching policy found")
        return ToolResult(ok=False, content="No matching policy document found.")

    LOGGER.info("[search_company_policy] found %d matching policies", len(matches))
    return ToolResult(ok=True, content=" ".join(matches))


def calculate_days_until(tool_input: dict[str, str]) -> ToolResult:
    query = tool_input.get("query", "")
    LOGGER.info("[calculate_days_until] parsing date from query=%r", query)
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)

    if not match:
        LOGGER.info("[calculate_days_until] no YYYY-MM-DD date found")
        return ToolResult(
            ok=False,
            content="Please include a target date in YYYY-MM-DD format.",
        )

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
    summary = tool_input.get("summary", "No summary provided.")
    LOGGER.info("[create_support_ticket] creating ticket for summary=%r", summary)
    ticket = {
        "ticket_id": "SUP-1001",
        "summary": summary,
        "status": "created",
        "owner": "operations",
    }
    LOGGER.info("[create_support_ticket] created ticket_id=%s", ticket["ticket_id"])
    return ToolResult(ok=True, content=json.dumps(ticket, indent=2))


def load_agentic_ai_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, str]:
    """Load optional local config for LLM planner settings."""

    LOGGER.info("[load_agentic_ai_config] loading config path=%s", config_path)
    if not config_path.exists():
        LOGGER.info("[load_agentic_ai_config] config file not found; using defaults")
        return {}

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")

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
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
    LOGGER.info("[download_local_model] model saved")
    print(f"Saved local model to {output_path}")


def build_tools() -> ToolRegistry:
    LOGGER.info("[build_tools] creating tool registry")
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


def build_agent(
    planner_type: str = "rule",
    model: str | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Agent:
    LOGGER.info(
        "[build_agent] planner_type=%s model=%s config_path=%s",
        planner_type,
        model,
        config_path,
    )
    tools = build_tools()
    config = load_agentic_ai_config(config_path)

    if planner_type == "rule":
        LOGGER.info("[build_agent] using RuleBasedPlanner")
        planner: Planner = RuleBasedPlanner()
    elif planner_type == "llm":
        api_key = config.get("openai_api_key")
        configured_model = config.get("openai_model")
        LOGGER.info(
            "[build_agent] using LLMPlanner model=%s",
            configured_model or model or "gpt-4o-mini",
        )
        planner = LLMPlanner(
            tools=tools,
            model=configured_model or model or "gpt-4o-mini",
            api_key=api_key,
        )
    elif planner_type == "local-llm":
        configured_model_path = config.get("local_model_path")
        configured_model = config.get("local_model")
        local_model_path = Path(configured_model_path) if configured_model_path else None

        if local_model_path is not None and local_model_path.exists():
            model_for_local_planner = str(local_model_path)
            local_files_only = True
            LOGGER.info(
                "[build_agent] using configured local_model_path=%s",
                model_for_local_planner,
            )
        elif DEFAULT_LOCAL_MODEL_PATH.exists():
            model_for_local_planner = str(DEFAULT_LOCAL_MODEL_PATH)
            local_files_only = True
            LOGGER.info(
                "[build_agent] using default downloaded local model=%s",
                model_for_local_planner,
            )
        else:
            model_for_local_planner = configured_model or model or "google/flan-t5-small"
            local_files_only = False
            LOGGER.info(
                "[build_agent] local model directory missing; model may be downloaded=%s",
                model_for_local_planner,
            )

        planner = LocalLLMPlanner(
            tools=tools,
            model=model_for_local_planner,
            local_files_only=local_files_only,
        )
    else:
        raise ValueError("planner_type must be 'rule', 'llm', or 'local-llm'.")

    return Agent(planner=planner, tools=tools)


def configure_logging(log_file: Path) -> None:
    """Send readable flow logs to both console and a local log file."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    LOGGER.info("[configure_logging] writing logs to %s", log_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small agentic AI example.")
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
    args = parse_args()
    configure_logging(args.log_file)
    LOGGER.info("[main] starting agentic AI example")
    LOGGER.info(
        "[main] args planner=%s model=%s config=%s download_local_model=%s log_file=%s",
        args.planner,
        args.model,
        args.config,
        args.download_local_model,
        args.log_file,
    )
    if args.download_local_model:
        config = load_agentic_ai_config(args.config)
        download_local_model(
            model_name=args.model or config.get("local_model") or "google/flan-t5-small",
            output_path=args.local_model_path,
        )
        return

    answer = build_agent(
        planner_type=args.planner,
        model=args.model,
        config_path=args.config,
    ).run(args.goal)
    LOGGER.info("[main] printing final answer")
    print(answer)


if __name__ == "__main__":
    main()
