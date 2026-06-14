"""FastAPI service wrapper for the agentic AI example.

Run from the repository root:
    uvicorn src.agentic_ai_service:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.agentic_ai_example import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LOG_PATH,
    build_agent,
    build_tools,
    configure_logging,
)


LOGGER = logging.getLogger("agentic_ai_service")

HTTP_REQUESTS = Counter(
    "agentic_ai_http_requests_total",
    "Total HTTP requests handled by the agentic AI service.",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "agentic_ai_http_request_seconds",
    "HTTP request latency for the agentic AI service.",
    ["method", "path"],
)
AGENT_RUNS = Counter(
    "agentic_ai_runs_total",
    "Agent runs handled by the service.",
    ["planner", "status"],
)
AGENT_LATENCY = Histogram(
    "agentic_ai_run_seconds",
    "Agent run latency by planner type.",
    ["planner"],
)
LAST_SUCCESSFUL_RUN = Gauge(
    "agentic_ai_last_successful_run_timestamp_seconds",
    "Unix timestamp of the last successful agent run.",
)


class AgentRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=8_000)
    planner: Literal["rule", "llm", "local-llm"] = "rule"
    model: str | None = Field(default=None, max_length=200)
    config_path: str | None = Field(default=None, max_length=500)


class AgentResponse(BaseModel):
    request_id: str
    planner: str
    answer: str
    elapsed_ms: int


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready"]
    tools: list[str]


def create_app() -> FastAPI:
    log_path = Path(os.getenv("AGENTIC_AI_LOG_FILE", str(DEFAULT_LOG_PATH)))
    configure_logging(log_path)

    app = FastAPI(
        title="SmallRAG Agentic AI Service",
        version="1.0.0",
        description="API wrapper for src.agentic_ai_example.",
    )

    @app.middleware("http")
    async def record_http_metrics(request, call_next):  # type: ignore[no-untyped-def]
        started_at = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            elapsed = time.perf_counter() - started_at
            route = request.scope.get("route")
            path = route.path if route else request.url.path
            HTTP_REQUESTS.labels(request.method, path, status).inc()
            HTTP_LATENCY.labels(request.method, path).observe(elapsed)

    @app.get("/", response_model=HealthResponse)
    async def root() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadinessResponse)
    async def ready() -> ReadinessResponse:
        return ReadinessResponse(status="ready", tools=build_tools().names())

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/ask", response_model=AgentResponse)
    async def ask(request: AgentRequest) -> AgentResponse:
        request_id = str(uuid.uuid4())
        started_at = time.perf_counter()
        config_path = Path(request.config_path) if request.config_path else DEFAULT_CONFIG_PATH

        LOGGER.info(
            "[ask] request_id=%s planner=%s model=%s goal=%r",
            request_id,
            request.planner,
            request.model,
            request.goal,
        )

        try:
            with AGENT_LATENCY.labels(request.planner).time():
                agent = build_agent(
                    planner_type=request.planner,
                    model=request.model,
                    config_path=config_path,
                )
                answer = await run_in_threadpool(agent.run, request.goal)
        except Exception as exc:
            AGENT_RUNS.labels(request.planner, "error").inc()
            LOGGER.exception("[ask] request_id=%s failed", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        AGENT_RUNS.labels(request.planner, "success").inc()
        LAST_SUCCESSFUL_RUN.set(time.time())
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)

        return AgentResponse(
            request_id=request_id,
            planner=request.planner,
            answer=answer,
            elapsed_ms=elapsed_ms,
        )

    return app


app = create_app()
