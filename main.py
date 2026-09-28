import os
import time
import json
import logging
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# OpenTelemetry Imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Configuration
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://opentelemetry-collector.monitoring.svc.cluster.local:4318")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama.ai.svc.cluster.local:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:1.5b")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [trace_id=%(otelTraceID)s] %(message)s")
logger = logging.getLogger("ai-service")

class TraceIdFilter(logging.Filter):
    def filter(self, record):
        span = trace.get_current_span()
        record.otelTraceID = f"{span.get_span_context().trace_id:032x}" if span.is_recording() else "0"
        return True

logger.addFilter(TraceIdFilter())

# 1. Setup OpenTelemetry Resource & Providers
resource = Resource.create({"service.name": "sample-ai-service", "service.version": "1.0.0"})

trace_provider = TracerProvider(resource=resource)
trace_exporter = OTLPSpanExporter(endpoint=f"{OTEL_ENDPOINT}/v1/traces")
trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
trace.set_tracer_provider(trace_provider)
tracer = trace.get_tracer("genai.tracer")

metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{OTEL_ENDPOINT}/v1/metrics"), export_interval_millis=5000)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter("genai.meter")

# Custom GenAI Metrics (CNCF Semantic Conventions)
token_counter = meter.create_counter(
    "gen_ai.client.token.usage",
    unit="{token}",
    description="Number of prompt and completion tokens used"
)
ttft_histogram = meter.create_histogram(
    "gen_ai.client.time_to_first_token",
    unit="s",
    description="Time to first token (TTFT)"
)

# 2. FastAPI Application
app = FastAPI(title="AI Native Platform Microservice")
FastAPIInstrumentor.instrument_app(app, tracer_provider=trace_provider)

class PromptRequest(BaseModel):
    prompt: str

@app.get("/healthz")
async def health():
    return {"status": "ok"}

@app.post("/api/v1/generate")
async def generate(req: PromptRequest):
    start_time = time.time()
    first_token_time = None
    response_text = ""
    prompt_tokens = len(req.prompt.split()) # Estimation simple si non fourni
    completion_tokens = 0

    logger.info(f"Received inference request for model: {MODEL_NAME}")

    with tracer.start_as_current_span(f"chat {MODEL_NAME}") as span:
        span.set_attribute("gen_ai.system", "ollama")
        span.set_attribute("gen_ai.request.model", MODEL_NAME)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": MODEL_NAME, "prompt": req.prompt, "stream": True}
                ) as resp:
                    if resp.status_code != 200:
                        raise HTTPException(status_code=500, detail="Ollama backend error")

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if first_token_time is None and chunk.get("response"):
                            first_token_time = time.time()
                            ttft = first_token_time - start_time
                            span.set_attribute("gen_ai.client.time_to_first_token", ttft)
                            ttft_histogram.record(ttft, {"model": MODEL_NAME})

                        response_text += chunk.get("response", "")
                        if chunk.get("done"):
                            prompt_tokens = chunk.get("prompt_eval_count", prompt_tokens)
                            completion_tokens = chunk.get("eval_count", len(response_text.split()))

            # Record telemetry
            span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
            span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
            token_counter.add(prompt_tokens, {"type": "prompt", "model": MODEL_NAME})
            token_counter.add(completion_tokens, {"type": "completion", "model": MODEL_NAME})

            logger.info(f"Completed inference: {completion_tokens} tokens generated")
            return {
                "model": MODEL_NAME,
                "response": response_text,
                "metrics": {
                    "ttft_seconds": round(first_token_time - start_time, 4) if first_token_time else 0,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                }
            }
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            logger.error(f"Inference failure: {e}")
            raise HTTPException(status_code=500, detail=str(e))
