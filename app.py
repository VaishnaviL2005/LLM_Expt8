import logging, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from support_rag import answer_question, BackendError

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("support-api")

app = FastAPI(title="Customer Support RAG API")

class Question(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def check(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty")
        if len(v) > 1000:
            raise ValueError("Question must be at most 1000 characters")
        return v

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(body: Question):
    start = time.perf_counter()
    try:
        answer = answer_question(body.question)
    except BackendError as e:
        ms = (time.perf_counter() - start) * 1000
        log.error("ask FAILED latency_ms=%.0f reason=%s", ms, type(e).__name__)
        raise HTTPException(status_code=500, detail="Internal server error")
    except Exception:
        ms = (time.perf_counter() - start) * 1000
        log.exception("ask unexpected error latency_ms=%.0f", ms)
        raise HTTPException(status_code=500, detail="Internal server error")
    ms = (time.perf_counter() - start) * 1000
    log.info("ask OK latency_ms=%.0f q_len=%d", ms, len(body.question))
    return {"answer": answer, "response_time_ms": round(ms, 1)}
