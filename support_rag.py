import os
import re
import requests
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ---------- Configuration (.env locally, Streamlit Secrets in the cloud) ----------
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")  # "groq" or "ollama"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")


class BackendError(Exception):
    pass


# ---------- Knowledge base ----------
support_faq = """
BrewNest Customer Support FAQ

Order Tracking
After your order is packed and handed to the courier, a tracking number is sent by SMS and email. You can also see shipment progress under My Orders in your BrewNest account. Most orders are delivered within 3 to 6 business days. If the tracking status has not changed for 72 hours, write to support with your order ID. The assistant cannot see live order or courier status.

Shipping and Delivery Charges
Delivery is free on orders above Rs. 999. Orders below this amount have a flat delivery fee of Rs. 79. Cash on Delivery is available for orders up to Rs. 5,000 in selected pin codes. The delivery address can be changed only before the order is dispatched.

Returns and Refunds
Unopened coffee beans and unused accessories in original packaging can be returned within 10 days of delivery. Opened coffee packs cannot be returned for hygiene reasons unless they arrived damaged. To start a return, go to My Orders, select the item, and tap Request Return. Once the returned item passes inspection, the refund is credited to the original payment method within 7 business days. Cash on Delivery refunds are sent to a bank account added by the customer. The assistant cannot approve or speed up refunds.

Warranty and Repairs
Espresso machines and grinders come with a 1-year manufacturer warranty from the date of delivery. The warranty covers manufacturing defects but not damage caused by drops, water spills, or descaling liquids not recommended in the user manual. To raise a warranty claim, open My Orders, choose the product, select Warranty Claim, and upload the invoice with a short video of the problem.

Coffee Subscriptions
BrewNest offers a monthly coffee bean subscription. Subscribers can pause, skip a month, or cancel from the Subscriptions page at any time before the 25th of the month. Changes made after the 25th apply from the following month.

Account Access
If you forgot your password, select Forgot Password on the login screen and follow the reset link sent to your registered email. The reset link expires after 30 minutes. If the email does not arrive, check your spam folder and confirm that the email address is correct. BrewNest staff will never ask for your password or OTP.

Billing and Payments
BrewNest accepts UPI, credit cards, debit cards, net banking, and Cash on Delivery. If money was deducted but the order was not confirmed, the amount is usually auto-reversed by the bank within 5 to 7 business days. If you see a double charge, do not pay again; contact billing support with the transaction ID and date. Never share your full card number, CVV, or UPI PIN.

App and Website Issues
If the BrewNest app is crashing or not loading, update the app to the latest version, check your internet connection, and restart your phone. On the website, clear the browser cache or try an incognito window. When contacting support, mention your device, the error message, and the steps that caused it.

Contact Support
For questions not answered here, use the Help Centre in the app or website to chat with an agent between 9 AM and 9 PM, Monday to Saturday. Share your order ID and a short description of the issue. Do not share passwords, OTPs, CVV, or UPI PIN with anyone.
"""

# ---------- Load ONCE at startup: chunks, embeddings, FAISS ----------
faq_doc = Document(page_content=support_faq, metadata={"source": "support_faq.txt"})
splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60)
chunks = splitter.split_documents([faq_doc])

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# ---------- LLM backend: Groq (cloud) or Ollama (local) ----------
def clean_answer(text: str) -> str:
    """Remove <think>...</think> reasoning text some Qwen models include."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return text.strip()


def call_groq(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}",
               "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2048,  # room for reasoning + answer
    }
    # First try with reasoning switched off (faster); retry without it if rejected
    r = requests.post(GROQ_URL, headers=headers,
                      json={**payload, "reasoning_effort": "none"}, timeout=60)
    if r.status_code == 400:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise BackendError(f"Groq HTTP {r.status_code}: {r.text[:200]}")
    text = clean_answer(r.json()["choices"][0]["message"].get("content"))
    if not text:
        raise BackendError("Groq returned an empty answer")
    return text


def call_ollama(prompt: str) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL,
              "messages": [{"role": "user", "content": prompt}],
              "stream": False,
              "options": {"temperature": 0, "num_predict": 160}},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def call_llm(prompt: str) -> str:
    try:
        if LLM_BACKEND == "groq":
            return call_groq(prompt)
        return call_ollama(prompt)
    except BackendError:
        raise
    except Exception as e:
        raise BackendError(str(e))


# ---------- LangGraph workflow ----------
class SupportState(TypedDict, total=False):
    question: str
    intent: str
    search_query: str
    context: str
    sources: List[str]
    draft: str
    retry_count: int
    final_answer: str


def classify_intent(state: SupportState):
    q = state["question"].lower()
    if any(w in q for w in ["refund", "return", "exchange"]):
        intent = "returns_refunds"
    elif any(w in q for w in ["warranty", "repair", "defect", "broken", "stopped working"]):
        intent = "warranty_repairs"
    elif any(w in q for w in ["subscription", "subscribe", "pause", "skip a month"]):
        intent = "subscriptions"
    elif any(w in q for w in ["delivery fee", "delivery charge", "free delivery", "delivery free", "shipping", "cash on delivery", "address"]):
        intent = "shipping_charges"
    elif any(w in q for w in ["track", "tracking", "shipment", "delivery", "order status", "live status"]):
        intent = "order_tracking"
    elif any(w in q for w in ["password", "sign in", "log in", "login", "account", "otp", "verification"]):
        intent = "account_access"
    elif any(w in q for w in ["charged", "payment", "billing", "deducted", "upi", "card", "invoice"]):
        intent = "billing_payments"
    elif any(w in q for w in ["website", "error", "loading", "crash", "browser", " app", "technical"]):
        intent = "technical_support"
    else:
        intent = "general_support"
    return {"intent": intent, "search_query": state["question"],
            "retry_count": state.get("retry_count", 0)}


def retrieve_context(state: SupportState):
    docs = retriever.invoke(state["search_query"])
    return {"context": "\n\n".join(d.page_content for d in docs),
            "sources": [d.metadata.get("source", "unknown") for d in docs]}


def draft_answer(state: SupportState):
    prompt = f"""You are a customer-support assistant for BrewNest, an online coffee store.
You know NOTHING except the FAQ CONTEXT below. Never use outside knowledge and never mention who created you.
Answer only if the FAQ CONTEXT clearly answers the CUSTOMER QUESTION.
If the question is not about BrewNest or the context does not clearly answer it,
reply with exactly this one word and nothing else: INSUFFICIENT_CONTEXT
Do not invent order status, refund approvals, timelines, or policies.
Do not ask for passwords, one-time codes, or full payment-card details.
Keep the answer short.

FAQ CONTEXT:
{state.get("context", "")}

CUSTOMER QUESTION:
{state["question"]}

ANSWER:"""
    return {"draft": call_llm(prompt)}


def review_answer(state: SupportState):
    draft = state.get("draft", "").strip()
    if "insufficient_context" in draft.lower().replace(" ", "_"):
        return {"final_answer": ""}
    return {"final_answer": draft}


def route_after_review(state: SupportState):
    if state.get("final_answer"):
        return "finish"
    if state.get("retry_count", 0) < 1:
        return "retry"
    return "fallback"


def rewrite_query(state: SupportState):
    terms = {
        "returns_refunds": "return eligibility refund process inspection",
        "warranty_repairs": "warranty claim repair manufacturing defect",
        "subscriptions": "coffee subscription pause skip cancel",
        "shipping_charges": "delivery fee free delivery cash on delivery address",
        "order_tracking": "order dispatch tracking shipment status",
        "account_access": "account login password reset",
        "billing_payments": "billing payment transaction double charge",
        "technical_support": "app website crash troubleshooting",
        "general_support": "customer support help contact",
    }
    expanded = state["question"] + " " + terms.get(state.get("intent", "general_support"), "customer support")
    return {"search_query": expanded, "retry_count": state.get("retry_count", 0) + 1}


def fallback_answer(state: SupportState):
    return {"final_answer": (
        "I couldn't find a reliable answer in the available FAQ. "
        "Please contact the support team through the Help Centre. "
        "Do not share your password, OTP, CVV, or UPI PIN."
    )}


g = StateGraph(SupportState)
g.add_node("classify_intent", classify_intent)
g.add_node("retrieve", retrieve_context)
g.add_node("draft_answer", draft_answer)
g.add_node("review_answer", review_answer)
g.add_node("rewrite_query", rewrite_query)
g.add_node("fallback", fallback_answer)
g.add_edge(START, "classify_intent")
g.add_edge("classify_intent", "retrieve")
g.add_edge("retrieve", "draft_answer")
g.add_edge("draft_answer", "review_answer")
g.add_conditional_edges("review_answer", route_after_review,
                        {"finish": END, "retry": "rewrite_query", "fallback": "fallback"})
g.add_edge("rewrite_query", "retrieve")
g.add_edge("fallback", END)
support_graph = g.compile()


# ---------- Function used by FastAPI and Streamlit ----------
def answer_question(question: str) -> str:
    result = support_graph.invoke({"question": question, "retry_count": 0})
    return result.get("final_answer", "")
