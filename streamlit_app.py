import os
import time
import logging
import streamlit as st

# Copy Streamlit Cloud secrets into environment variables.
# This must run BEFORE importing support_rag.
try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass  # locally there is no secrets file; .env is used instead

from support_rag import answer_question, BackendError, LLM_BACKEND, GROQ_MODEL, OLLAMA_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("support-ui")

st.set_page_config(page_title="BrewNest Support Assistant", page_icon="☕")
st.title("☕ BrewNest Customer Support Assistant")
st.caption(
    "RAG assistant that answers only from the BrewNest FAQ. "
    "Do not enter passwords, OTPs or card details."
)

model_name = GROQ_MODEL if LLM_BACKEND == "groq" else OLLAMA_MODEL
st.caption(f"Backend: {LLM_BACKEND} | Model: {model_name}")

question = st.text_area("Ask your question:", max_chars=1000, height=100)

if st.button("Ask"):
    q = question.strip()
    if not q:
        st.warning("Please enter a question.")
    elif len(q) > 1000:
        st.warning("Question must be at most 1000 characters.")
    else:
        start = time.perf_counter()
        try:
            with st.spinner("Searching the FAQ..."):
                answer = answer_question(q)
            ms = (time.perf_counter() - start) * 1000
            log.info("ask OK latency_ms=%.0f q_len=%d", ms, len(q))
            st.success(answer)
            st.caption(f"Response time: {ms:.0f} ms")
        except BackendError as e:
            ms = (time.perf_counter() - start) * 1000
            log.error("ask FAILED latency_ms=%.0f reason=%s detail=%s",
                      ms, type(e).__name__, str(e)[:250])
            st.error("Sorry, the assistant is temporarily unavailable. Please try again later.")
        except Exception:
            log.exception("unexpected error")
            st.error("Something went wrong. Please try again later.")
