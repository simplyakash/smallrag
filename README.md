# smallrag
# Run rag_persistent.py
# Run ingest_agnews.py
# Run query_agnews.py

## Agentic AI example

Run a self-contained agentic AI example:

```bash
python src/agentic_ai_example.py "How many paid leave days do employees get?"
```

Choose the planner:

```bash
python src/agentic_ai_example.py --planner rule "How many paid leave days do employees get?"
python src/agentic_ai_example.py --planner local-llm "How many paid leave days do employees get?"
python src/agentic_ai_example.py --planner llm "How many paid leave days do employees get?"
```

The LLM planner reads `openai_api_key` from `agentic_ai_config.json`.
`OPENAI_API_KEY` is still supported as a fallback.
The local LLM planner uses `local_model` from `agentic_ai_config.json` and does
not need an API key.

Download the local model once to avoid future Hugging Face requests:

```bash
python src/agentic_ai_example.py --download-local-model
python src/agentic_ai_example.py --planner local-llm "How many paid leave days do employees get?"
```

Flow logs are written to `logs/agentic_ai_example.log` by default:

```bash
python src/agentic_ai_example.py --planner local-llm --log-file logs/my_run.log "How many paid leave days do employees get?"
```

Try a tool-routing example with date arithmetic:

```bash
python src/agentic_ai_example.py "How many days until the 2026-12-31 deadline?"
```
