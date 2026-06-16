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
  "llm_provider": "openai",
  "gemini_api_key": "your_gemini_api_key_here",
  "gemini_model": "gemini-2.0-flash",
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

## Run The LangGraph Variant

Install the extra graph dependency when you want to run the LangGraph version:

```bash
pip install -r requirements-agentic-langgraph.txt
```

Then run the parallel implementation:

```bash
python src/agentic_ai_langgraph.py --planner rule "How many paid leave days do employees get?"
python src/agentic_ai_langgraph.py --planner rule "How many days until the 2026-12-31 deadline?"
python src/agentic_ai_langgraph.py --planner rule "Need help with onboarding laptop access"
```

The LangGraph file defines its own tools and planner contracts locally, then
represents the executor loop as a graph:

```text
User Goal
  -> planner node chooses AgentAction
  -> route to END when tool_name is final_answer
  -> otherwise run the tool node
  -> append observation and loop back to the planner node
```

The LLM and local LLM planners use the same config and model flags as
`src/agentic_ai_example.py`.

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

To use Gemini instead of OpenAI, set:

```json
{
  "llm_provider": "gemini",
  "gemini_api_key": "your_gemini_api_key_here",
  "gemini_model": "gemini-2.0-flash"
}
```

`GEMINI_API_KEY` is also supported as an environment variable.

The `local_model` value is used only by `--planner local-llm` and does not need
an API key. The first run may download the model from Hugging Face, then reuse
the local cache.

For fully local loading, run `--download-local-model` once. After
`models/flan-t5-small` exists, `--planner local-llm` loads from that directory
with `local_files_only=True`.

python src/agentic_ai_example.py --planner local-llm "How many paid leave days do employees get?"

Deploy:
uvicorn src.agentic_ai_service:app --host 0.0.0.0 --port 8000

# 🧠 Advantages of LangGraph Over Normal Python Code for Agentic AI

A common interview question is:

> "Why use LangGraph? Can't we just write agents in Python?"

The answer is:

```text
Yes, you can build agents using plain Python.

However, as workflows become more complex,
LangGraph provides structure, reliability,
state management, observability, and control.
```

---

# Normal Python Agent

Example:

```python
def agent(query):
    plan = planner(query)

    if needs_search(plan):
        result = search_tool(plan)

    if needs_calculator(plan):
        result = calculator_tool(plan)

    answer = llm(result)

    return answer
```

Works fine for:

```text
Simple Agent
1-2 Tools
Linear Workflow
```

---

# Problem as Complexity Grows

Suppose you need:

```text
Planner
    ↓
Retriever
    ↓
Evaluator
    ↓
Replan if failure
    ↓
Tool Calls
    ↓
Human Approval
    ↓
Final Response
```

Now code becomes:

```python
if ...
    while ...
        try ...
            if ...
                ...
```

Soon you have:

```text
Nested ifs
Nested loops
Retries
Branching
State passing
Checkpointing
```

which becomes hard to maintain.

---

# LangGraph Approach

Instead of writing workflow logic manually:

```text
Node A
   ↓
Node B
   ↓
Node C
```

You explicitly define a graph.

```text
Planner
   ↓
Retriever
   ↓
Evaluator
  ↙     ↘
Pass    Fail
  ↓       ↓
Answer  Replan
```

---

# Advantage 1: Explicit Workflow Graph

Normal Python:

```python
if ...
while ...
for ...
```

Need to mentally trace execution.

---

LangGraph:

```text
Planner
   ↓
Retriever
   ↓
Evaluator
```

Workflow is visually obvious.

---

# Advantage 2: State Management

Agents maintain:

```text
Conversation History
Retrieved Docs
Tool Outputs
Intermediate Reasoning
Memory
```

In Python:

```python
state = {}
state["history"] = ...
state["docs"] = ...
state["tools"] = ...
```

Becomes messy.

---

LangGraph:

```python
class AgentState(TypedDict):
    messages: list
    documents: list
    answer: str
```

Shared state automatically flows through nodes.

---

# Advantage 3: Cycles and Replanning

Agentic systems often require:

```text
Plan
   ↓
Execute
   ↓
Evaluate
   ↓
Replan
```

Normal Python:

```python
while not success:
    plan()
    execute()
    evaluate()
```

Hard to debug.

---

LangGraph:

```text
Planner
   ↓
Executor
   ↓
Evaluator
   ↓
Planner
```

Cycles are first-class citizens.

---

# Advantage 4: Human-in-the-Loop

Example:

```text
Generate SQL
      ↓
Human Approval
      ↓
Execute
```

Python:

```python
input()
callbacks
manual code
```

---

LangGraph:

```text
Node
   ↓
Pause
   ↓
Human Review
   ↓
Resume
```

Built into the framework.

---

# Advantage 5: Checkpointing

Suppose an agent runs for:

```text
10 minutes
20 tool calls
```

and crashes at step 18.

---

Normal Python:

```text
Start over
```

---

LangGraph:

```text
Resume from checkpoint
```

Only re-execute failed nodes.

---

# Advantage 6: Persistence

State can survive:

```text
Server restart
Pod restart
Crash
```

because checkpoints are stored in:

```text
Redis
Postgres
SQLite
S3
```

---

# Advantage 7: Multi-Agent Systems

Example:

```text
Research Agent
      ↓
Coding Agent
      ↓
Reviewer Agent
```

Python:

```python
research_agent()
coding_agent()
review_agent()
```

Quickly becomes difficult with:

```text
State Sharing
Routing
Retries
Memory
```

---

LangGraph:

```text
Agent A
  ↓
Agent B
  ↓
Agent C
```

Agents become graph nodes.

---

# Advantage 8: Conditional Routing

Example:

```text
Question
    ↓
Need Search?
   / \
 Yes No
 /     \
Search  Answer
```

Python:

```python
if search_required:
```

works initially but becomes messy with many branches.

---

LangGraph:

```python
builder.add_conditional_edges(...)
```

Routing is explicit.

---

# Advantage 9: Streaming

Agent execution can stream:

```text
Node Started
Node Finished
Tool Output
Intermediate Result
```

Useful for:

```text
ChatGPT-like UIs
Agent Monitoring
Debugging
```

---

# Advantage 10: Observability

With LangGraph + :contentReference[oaicite:0]{index=0}:

```text
See every node
See every tool call
See every prompt
See every token
See state transitions
```

Normal Python:

```text
print()
logging()
manual tracing
```

---

# Example Comparison

## Plain Python

```text
User Query
    ↓
Planner
    ↓
Search
    ↓
Evaluate
    ↓
Retry?
    ↓
Answer
```

Implemented as:

```python
while True:
    ...
```

---

## LangGraph

```text
User Query
    ↓
Planner
    ↓
Search
    ↓
Evaluator
   /   \
Pass   Fail
  ↓      ↓
Answer  Planner
```

The workflow itself is the code.

---

# When Plain Python Is Enough

Use plain Python if:

```text
✅ Simple chatbot
✅ Single tool
✅ Small prototype
✅ Few workflow steps
```

LangGraph may be overkill.

---

# When LangGraph Helps

Use LangGraph if:

```text
✅ Multi-step agents
✅ Tool calling
✅ Replanning loops
✅ Human approval
✅ Long-running workflows
✅ Multi-agent systems
✅ Checkpointing
✅ Production observability
```

---

# Interview Answer

LangGraph provides a graph-based orchestration framework for agentic AI systems. While the same functionality can be implemented using normal Python control flow, LangGraph offers explicit workflow representation, shared state management, conditional routing, cycles for replanning, checkpointing, persistence, human-in-the-loop support, multi-agent orchestration, and observability. These features make complex agent workflows easier to build, debug, monitor, and maintain compared to manually managing the same logic with nested loops, conditionals, and state dictionaries in plain Python.

