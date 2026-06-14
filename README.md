# smallrag
# Run rag_persistent.py
# Run ingest_agnews.py
# Run query_agnews.py

## Setup

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` is the lightweight service/API install. Install
`requirements-rag.txt` only when you need AG News ingestion and Chroma storage.
Install `requirements-local-llm.txt` only on a machine with enough disk for
Torch and Hugging Face local generation.

For constrained environments, ingest AG News in small batches:

```bash
python -m pip install -r requirements-rag.txt
AGNEWS_SAMPLE_SIZE=20 AGNEWS_BATCH_SIZE=5 python -m src.ingest_agnews
python -m src.query_agnews
```

## MLflow model registry

Install only the lightweight dependencies needed for the registry example:

```bash
python3 -m pip install -r requirements-mlflow.txt
```

Register an already trained sklearn model file in a local MLflow Model
Registry and assign the `champion` alias:

```bash
python src/mlflow_model_registry.py \
  --model-name my_model \
  --model-path path/to/trained_model.pkl
```

Register and log regression metrics:

```bash
python src/mlflow_model_registry.py \
  --model-name my_model \
  --model-path models/sample_regression_model_v2.pkl \
  --eval-features "[[10.0, 4.0], [2.0, 3.0], [5.0, 1.0]]" \
  --eval-targets "[46.0, 15.0, 29.0]"
```

You can also register an existing MLflow model URI:

```bash
python src/mlflow_model_registry.py \
  --model-name my_model \
  --model-uri runs:/<run_id>/model
```

Load the current registered alias:

```bash
python src/mlflow_model_registry.py --model-name my_model --load
```

Run inference from the registered model:

```bash
python src/mlflow_inference.py \
  --model-name my_model \
  --alias champion \
  --features "[[10.0, 4.0]]"
```

Each inference call is logged to the `mlflow_inference_logs` experiment by
default. The log includes the model URI, input features, predictions, and
targets/metrics when targets are provided.

Run the built-in regression test and log inference metrics:

```bash
python src/mlflow_inference.py \
  --model-name my_model \
  --alias champion \
  --run-test
```

Skip logging for a one-off local prediction:

```bash
python src/mlflow_inference.py \
  --model-name my_model \
  --alias champion \
  --features "[[10.0, 4.0]]" \
  --no-log-inference
```

Open the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

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
For Gemini, set `llm_provider` to `gemini` and add `gemini_api_key` plus
`gemini_model` in `agentic_ai_config.json`; `GEMINI_API_KEY` is also supported
as a fallback.
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

Try the expanded Agentic AI design examples:

```bash
python src/agentic_ai_example.py "Design a multi-agent handoff for a RAG support assistant."
python src/agentic_ai_example.py "How should this agent manage memory and context windows?"
python src/agentic_ai_example.py "Create a production deployment plan for this agentic AI app."
```

Run the production-style API with monitoring:

```bash
python -m pip install -r requirements-agentic-service.txt
uvicorn src.agentic_ai_service:app --host 0.0.0.0 --port 8000
docker compose up --build
```

See `AGENTIC_AI_README.md` for API, Docker, Prometheus, and production
deployment steps.
