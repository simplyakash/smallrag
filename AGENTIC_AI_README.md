# Agentic AI Example

This project includes a small, working agentic AI example in
`src/agentic_ai_example.py`.

The example runs locally without API keys when using the default rule-based
planner or the local Hugging Face planner. It also includes an optional OpenAI
planner that returns the same structured `AgentAction`.

## What It Demonstrates

- Agent state tracking through `AgentState`
- Structured planner output through `AgentAction`
- Standard tool responses through `ToolResult`
- Tool registration and lookup through `ToolRegistry`
- A bounded plan-act-observe loop inside `Agent`
- Simple logging for each agent step
- Clear fallback behavior when the request needs human triage

## File Structure

```text
src/agentic_ai_example.py
├── Data contracts
│   ├── ToolResult
│   ├── Tool
│   ├── AgentAction
│   ├── AgentConfig
│   └── AgentState
├── ToolRegistry
├── RuleBasedPlanner
├── LLMPlanner
├── LocalLLMPlanner
├── Agent
├── Tools
│   ├── search_company_policy
│   ├── calculate_days_until
│   └── create_support_ticket
├── build_agent
└── CLI entrypoint
```

## How It Works

The agent follows this flow:

```text
User Goal
  -> Planner decides the next action
  -> Agent executes the selected tool
  -> Tool returns an observation
  -> Agent stores the observation in state
  -> Planner creates the final answer
```

The loop is bounded by `AgentConfig.max_steps`, which prevents endless
agent execution.

## Planner Options

`RuleBasedPlanner` is the default. It decides which tool to use with simple
deterministic rules:

- Leave, vacation, or PTO questions use `search_company_policy`
- Deadline or date questions use `calculate_days_until`
- Unclear requests use `create_support_ticket`
- After one observation is collected, the planner returns `final_answer`

`LLMPlanner` asks an OpenAI model to choose the next action. It receives the
user goal, current observations, and available tool descriptions. The model must
return JSON matching the `AgentAction` structure:

```json
{
  "thought": "short reason",
  "tool_name": "search_company_policy",
  "tool_input": {
    "query": "How many paid leave days do employees get?"
  }
}
```

`LocalLLMPlanner` uses a small local Hugging Face text-to-text model for tool
selection. By default it uses `google/flan-t5-small`, so no API key is needed.
If the local model output is unclear, it falls back to the rule-based planner.

## Available Tools

`search_company_policy`
: Searches a small in-memory company policy knowledge base.

`calculate_days_until`
: Finds a `YYYY-MM-DD` date in the user query and calculates days remaining.

`create_support_ticket`
: Creates a mock support ticket when the request should be handled by a human.

## Run The Example

Run the default goal:

```bash
python src/agentic_ai_example.py
```

Choose the rule-based planner explicitly:

```bash
python src/agentic_ai_example.py --planner rule "How many paid leave days do employees get?"
```

Use the local LLM planner without an API key:

```bash
python src/agentic_ai_example.py --planner local-llm "How many paid leave days do employees get?"
```

Download the local model once, then run without Hugging Face network requests:

```bash
python src/agentic_ai_example.py --download-local-model
python src/agentic_ai_example.py --planner local-llm "How many paid leave days do employees get?"
```

By default this saves the model under `models/flan-t5-small`. The `models/`
folder is ignored by Git.

Use the LLM planner:

First edit `agentic_ai_config.json`:

```json
{
  "openai_api_key": "your_api_key_here",
  "openai_model": "gpt-4o-mini",
  "local_model": "google/flan-t5-small"
}
```

Then run:

```bash
python src/agentic_ai_example.py --planner llm "How many paid leave days do employees get?"
```

Use a different config file:

```bash
python src/agentic_ai_example.py --planner llm --config ./agentic_ai_config.json "Need help with onboarding laptop access"
```

Use a different local Hugging Face model:

```bash
python src/agentic_ai_example.py --planner local-llm --model google/flan-t5-base "Need help with onboarding laptop access"
```

Download a different local model path:

```bash
python src/agentic_ai_example.py --download-local-model --model google/flan-t5-base --local-model-path models/flan-t5-base
python src/agentic_ai_example.py --planner local-llm --config ./agentic_ai_config.json "Need help with onboarding laptop access"
```

Run a policy lookup:

```bash
python src/agentic_ai_example.py "How many paid leave days do employees get?"
```

Run date calculation:

```bash
python src/agentic_ai_example.py "How many days until the 2026-12-31 deadline?"
```

Run fallback triage:

```bash
python src/agentic_ai_example.py "Need help with onboarding laptop access"
```

## Logging

The script writes the full method-by-method flow to
`logs/agentic_ai_example.log` by default. The same log messages are also printed
to the console.

Use a custom log file:

```bash
python src/agentic_ai_example.py --planner local-llm --log-file logs/my_run.log "How many paid leave days do employees get?"
```

The `logs/` folder is ignored by Git.

## Why This Matches Industry Patterns

Real agentic systems usually separate these responsibilities:

- Planner: decides what to do next
- Tools: perform external actions or lookups
- State: stores the goal and observations
- Executor loop: runs the planner and tools safely
- Contracts: keep tool inputs and outputs predictable
- Limits: prevent infinite loops and runaway cost
- Planner choice: allows deterministic, local LLM, or API-backed behavior

This example keeps those responsibilities separate while staying small enough
for learning.

## How To Extend It

To add a new capability:

1. Create a function that accepts `dict[str, str]` and returns `ToolResult`.
2. Register it in `build_agent()`.
3. Update `RuleBasedPlanner.next_action()` to route matching requests to it.
4. The LLM planner will automatically see the new tool description from
   `ToolRegistry`.

For production, keep the same separation but add stronger validation, retries,
tool-level authorization, observability, and evaluation tests.

## Local Config

`agentic_ai_config.json` is for local secrets and model settings:

```json
{
  "openai_api_key": "your_api_key_here",
  "openai_model": "gpt-4o-mini",
  "local_model": "google/flan-t5-small"
}
```

This file is ignored by Git through `.gitignore`, so your real API key should
stay local. If the config file does not contain a key, the code falls back to
the `OPENAI_API_KEY` environment variable.

The `local_model` value is used only by `--planner local-llm` and does not need
an API key. The first run may download the model from Hugging Face, then reuse
the local cache.

For fully local loading, run `--download-local-model` once. After
`models/flan-t5-small` exists, `--planner local-llm` loads from that directory
with `local_files_only=True`.

python src/agentic_ai_example.py --planner local-llm "How many paid leave days do employees get?"
