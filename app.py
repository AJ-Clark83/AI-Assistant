import os
import base64
import re
from urllib.parse import urlparse, unquote
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import pandas as pd

# -----------------------------
# Config / Environment
# -----------------------------
load_dotenv()

def get_secret(name, default=None):
    """Load variable from st.secrets if present, else from environment."""
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)

SEARCH_ENDPOINT = get_secret("AZURE_SEARCH_ENDPOINT")
SEARCH_API_KEY = get_secret("AZURE_SEARCH_API_KEY")
SEARCH_INDEX = get_secret("AZURE_SEARCH_INDEX")

CONTENT_FIELD = get_secret("AZURE_SEARCH_CONTENT_FIELD", "content")
VECTOR_FIELD = get_secret("AZURE_SEARCH_VECTOR_FIELD", "contentVector")

OPENAI_ENDPOINT = get_secret("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = get_secret("AZURE_OPENAI_API_KEY")
OPENAI_DEPLOYMENT = get_secret("AZURE_OPENAI_DEPLOYMENT")
EMBED_DEPLOYMENT = get_secret("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


if not all(
    [
        SEARCH_ENDPOINT,
        SEARCH_API_KEY,
        SEARCH_INDEX,
        OPENAI_ENDPOINT,
        OPENAI_API_KEY,
        OPENAI_DEPLOYMENT,
        EMBED_DEPLOYMENT,
    ]
):
    st.warning("Missing one or more required .env settings. Check your environment variables.")

# Azure Search client
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_API_KEY),
)

# Azure OpenAI client
client = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version="2024-12-01-preview",
    azure_endpoint=OPENAI_ENDPOINT,
)

# -----------------------------
# Helpers (your original logic)
# -----------------------------
def embed_text(text: str):
    resp = client.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input=[text],
    )
    return resp.data[0].embedding


def _decode_base64_url(b64_str: str):
    if not b64_str:
        return None, None

    padding = "=" * (-len(b64_str) % 4)
    candidate = b64_str + padding

    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            url = decoder(candidate.encode("utf-8")).decode("utf-8")
            parsed = urlparse(url)
            filename = unquote(parsed.path.split("/")[-1]) if parsed.path else None
            return url, filename
        except Exception:
            continue

    return None, None


def decode_text_document_id(text_doc_id: str):
    if not text_doc_id:
        return None, None
    return _decode_base64_url(text_doc_id)


def decode_from_content_id(content_id: str):
    if not content_id:
        return None, None

    base = content_id

    if "_pages_" in base:
        base = base.rsplit("_pages_", 1)[0]

    parts = base.split("_", 1)
    if len(parts) < 2:
        return None, None

    b64_str = parts[1]
    b64_str = re.sub(r"\d+$", "", b64_str)

    return _decode_base64_url(b64_str)


def extract_page_from_content_id(content_id: str):
    if not content_id:
        return None
    m = re.search(r"_pages_(\d+)$", content_id)
    return int(m.group(1)) if m else None


def retrieve_docs(question: str, k: int = 5, use_hybrid: bool = True):
    embedding = embed_text(question)

    vq = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=k,
        fields=VECTOR_FIELD,
    )

    results = search_client.search(
        search_text=question if use_hybrid else None,
        vector_queries=[vq],
        top=k,
    )

    docs = []
    for r in results:
        row = dict(r)

        content = row.get(CONTENT_FIELD, "") or ""
        content_id = row.get("content_id")
        text_doc_id = row.get("text_document_id")
        page = extract_page_from_content_id(content_id)

        doc_url = None
        doc_name = None

        title = row.get("document_title")
        if title:
            doc_name = title

        if not doc_name and text_doc_id:
            url, filename = decode_text_document_id(text_doc_id)
            if filename:
                doc_name = filename
            if url:
                doc_url = url

        if not doc_name and content_id:
            url2, filename2 = decode_from_content_id(content_id)
            if filename2:
                doc_name = filename2
            if url2 and not doc_url:
                doc_url = url2

        if not doc_name:
            doc_name = row.get("content_id") or "Unknown document"

        docs.append(
            {
                "content": content,
                "source_id": content_id,
                "doc_url": doc_url,
                "doc_name": doc_name,
                "page": page,
                "score": row.get("@search.score"),
            }
        )

    return docs


def build_context(docs, max_chars: int = 4000):
    pieces = []
    total = 0

    for d in docs:
        text = d["content"]
        if not text:
            continue

        label = d.get("doc_name", "Unknown document")
        page = d.get("page")
        if page is not None:
            label = f"{label}, page {page}"

        block = f"[Source: {label}]\n{text.strip()}\n"
        if total + len(block) > max_chars:
            break

        pieces.append(block)
        total += len(block)

    return "\n\n---\n\n".join(pieces)


def answer_with_rag(
    question: str,
    chat_history=None,
    k: int = 5,
    system_prompt_override: str = None,
    max_chars: int = 4000,
):
    if chat_history is None:
        chat_history = []

    docs = retrieve_docs(question, k=k, use_hybrid=True)
    context = build_context(docs, max_chars=max_chars)

    if not context:
        answer = (
            "I couldn't find anything in the documentation for that question. "
            "Please speak to your supervisor or HSE for guidance."
        )
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        return answer, chat_history, docs

    default_system_prompt = (
        "You are an internal support assistant for MACA frontline workers. "
        "Use ONLY the provided documentation (labelled [Source: ...]) to answer questions. "
        "If the answer is not explicitly contained in the documentation, say you do not know "
        "and advise the user to refer to their supervisor, HSE, or the relevant operating procedure. "
        "Do NOT invent new procedures, policies, or safety guidance. "
        "If the question appears to ask for judgement outside the documentation (e.g., priority rules, shortcuts, "
        "permissions, or safety critical decisions), clearly state that this must be referred to a supervisor or HSE."
    )

    system_prompt = system_prompt_override.strip() if system_prompt_override else default_system_prompt

    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Relevant documentation:\n{context}\n\n"
        "Answer the user's question using ONLY the documentation above. "
        "If the documentation does not directly answer the question, explicitly say so."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *chat_history,  # <-- keeps conversational awareness
        {"role": "user", "content": user_prompt},
    ]

    resp = client.chat.completions.create(
        model=OPENAI_DEPLOYMENT,
        messages=messages,
    )

    answer = resp.choices[0].message.content

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, chat_history, docs


def is_human_readable_source(doc_name: str):
    """
    Filter out ugly IDs like fb024fccaa92_aHR0cHM6Ly9...
    We keep sources that look like normal filenames or titles.
    """
    if not doc_name:
        return False
    if "aHR0cHM" in doc_name:
        return False
    if re.match(r"^[0-9a-f]{8,}_", doc_name.lower()):
        return False
    return True


# -----------------------------
# Streamlit UI / State
# -----------------------------
st.set_page_config(page_title="MACA & Thiess Worker Support Agent (POC)", layout="wide")

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": "...", "content": "..."}
if "messages_for_ui" not in st.session_state:
    st.session_state.messages_for_ui = []  # list of (role, text, sources)
if "system_prompt_override" not in st.session_state:
    st.session_state.system_prompt_override = None
if "k" not in st.session_state:
    st.session_state.k = 5
if "max_chars" not in st.session_state:
    st.session_state.max_chars = 4000
if "search_logs" not in st.session_state:
    st.session_state.search_logs = []  # list of dicts per query

DEFAULT_PROMPT = (
    "You are an internal support assistant for MACA frontline workers."
    "Use ONLY the provided documentation (labelled [Source: ...]) to answer questions."
    "If the answer is not explicitly contained in the documentation, or contained within the appendix of the document, say you do not know the answer and advise the user to refer to their supervisor, HSE, or the relevant operating procedure."
    "Do NOT invent new procedures, policies, or safety guidance. If the question appears to ask for judgement outside the documentation (e.g., priority rules, shortcuts, permissions, or safety critical decisions), clearly state that this must be referred to a supervisor or HSE." 
    "Answer as thoroughly as possible using the main document content, images, and data in the appendix to prevent the user needing to make follow up requests to obtain the answer that they seek. "
    "For follow-up questions, assume context from earlier user questions and previous answers unless the user clearly indicates a new topic"
    "When the user asks a high-level conceptual question (e.g. 'Who has right of way?'), search for related procedures even if specific vehicle combinations are not mentioned"
    "If the documents imply the answer but do not explicitly state it, summarise what is relevant and explain limitations."
)

# Sidebar navigation
with st.sidebar:
    st.title("MACA POC")
    page = st.radio("Go to", ["Chat", "Settings", "Search log"], index=0)

# -----------------------------
# Chat page
# -----------------------------
if page == "Chat":
    st.title("MACA and Thiess Frontline Worker Support (POC)")
    st.caption("Ask questions about mining SOPs / terminology / technology etc. Answers are grounded in indexed documents.")

    # Render chat so far
    for role, text, sources in st.session_state.messages_for_ui:
        with st.chat_message(role):
            st.markdown(text)
            if role == "assistant" and sources:
                with st.expander("Sources used"):
                    for d in sources:
                        label = d["doc_name"]
                        if d["page"] is not None:
                            label += f", page {d['page']}"
                        score = d["score"]
                        url = d["doc_url"]

                        if url:
                            st.markdown(f"- **{label}** (score: {score:.4f})  \n  {url}")
                        else:
                            st.markdown(f"- **{label}** (score: {score:.4f})")

    # Chat input
    question = st.chat_input("Ask a question about MACA SOPs, mine operations, or terminology...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching docs and drafting answer..."):
                answer, new_history, docs_used = answer_with_rag(
                    question,
                    chat_history=st.session_state.chat_history,
                    k=st.session_state.k,
                    system_prompt_override=st.session_state.system_prompt_override or DEFAULT_PROMPT,
                    max_chars=st.session_state.max_chars,
                )

                # filter sources for display
                docs_used_filtered = [
                    d for d in docs_used if is_human_readable_source(d.get("doc_name"))
                ]

                st.markdown(answer)

        # persist chat history and messages for UI
        st.session_state.chat_history = new_history
        st.session_state.messages_for_ui.append(("user", question, None))
        st.session_state.messages_for_ui.append(("assistant", answer, docs_used_filtered))

        # determine if an answer was actually given (vs fallback)
        fallback_prefix = "I couldn't find anything in the documentation for that question."
        answered = not answer.startswith(fallback_prefix)

        top_source_name = docs_used_filtered[0]["doc_name"] if docs_used_filtered else None

        # log the search (in-memory only)
        st.session_state.search_logs.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": question,
                "answered": answered,
                "num_sources": len(docs_used),
                "top_source": top_source_name,
            }
        )

# -----------------------------
# Settings page
# -----------------------------
elif page == "Settings":
    st.title("Settings")

    st.subheader("System prompt")
    system_prompt_override = st.text_area(
        "System prompt (editable for testing)",
        value=st.session_state.system_prompt_override or DEFAULT_PROMPT,
        height=220,
    )
    st.session_state.system_prompt_override = system_prompt_override

    st.subheader("Retrieval parameters")
    st.session_state.k = st.slider(
        "Number of sources (k)",
        min_value=1,
        max_value=10,
        value=st.session_state.k,
    )

    st.session_state.max_chars = st.slider(
        "Max context characters",
        1000,
        8000,
        st.session_state.max_chars,
        step=500,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reset prompt"):
            st.session_state.system_prompt_override = DEFAULT_PROMPT
            st.success("System prompt reset to default.")
    with col_b:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.session_state.messages_for_ui = []
            st.session_state.search_logs = []
            st.success("Chat & logs cleared.")

# -----------------------------
# Search log page
# -----------------------------
elif page == "Search log":
    st.title("Search log (this session)")

    logs = st.session_state.search_logs

    if not logs:
        st.info("No searches have been logged in this session yet.")
    else:
        df = pd.DataFrame(logs)
        st.dataframe(df, use_container_width=True)

        total = len(df)
        answered = int(df["answered"].sum())
        unanswered = total - answered

        st.markdown("### Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total searches", total)
        c2.metric("Answered (context found)", answered)
        c3.metric("Unanswered", unanswered)
