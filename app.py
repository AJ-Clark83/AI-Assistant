import os
import re
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import pandas as pd
import hmac

# -----------------------------
# Config / Environment
# -----------------------------
load_dotenv()

def get_secret(name, default=None):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

APP_PASSWORD = get_secret("APP_PASSWORD")

def require_password():
    if not APP_PASSWORD:
        st.error("APP_PASSWORD is not set in secrets or environment.")
        st.stop()

    # Persist auth across reruns
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if st.session_state.auth_ok:
        return

    st.title("MACA POC")
    st.caption("Enter password to continue")

    pw = st.text_input("Password", type="password")

    if st.button("Unlock"):
        # Use constant-time compare
        if hmac.compare_digest(pw, APP_PASSWORD):
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Incorrect password")

    st.stop()

require_password()

SEARCH_ENDPOINT = get_secret("AZURE_SEARCH_ENDPOINT")
SEARCH_API_KEY = get_secret("AZURE_SEARCH_API_KEY")
SEARCH_INDEX = get_secret("AZURE_SEARCH_INDEX")

CONTENT_FIELD = get_secret("AZURE_SEARCH_CONTENT_FIELD", "chunk")
VECTOR_FIELD = get_secret("AZURE_SEARCH_VECTOR_FIELD", "text_vector")

OPENAI_ENDPOINT = get_secret("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = get_secret("AZURE_OPENAI_API_KEY")
OPENAI_DEPLOYMENT = get_secret("AZURE_OPENAI_DEPLOYMENT")
EMBED_DEPLOYMENT = get_secret("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

EXPECTED_EMBED_DIM = 3072  # matches your index text_vector.dimensions

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
# Helpers
# -----------------------------
def embed_text(text: str):
    resp = client.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input=[text],
    )
    vec = resp.data[0].embedding

    # Guardrail: vector field expects 3072 dims now
    if len(vec) != EXPECTED_EMBED_DIM:
        raise ValueError(
            f"Embedding dim mismatch: got {len(vec)}, expected {EXPECTED_EMBED_DIM}. "
            "Check AZURE_OPENAI_EMBEDDING_DEPLOYMENT and the index vector field dimensions."
        )

    return vec


def extract_page_from_chunk_id(chunk_id: str):
    """
    Optional: only works if chunk_id contains a suffix like _pages_12
    (your new pipeline may or may not include this).
    """
    if not chunk_id:
        return None
    m = re.search(r"_pages_(\d+)$", chunk_id)
    return int(m.group(1)) if m else None


def retrieve_docs(question: str, k: int = 5, use_hybrid: bool = True):
    embedding = embed_text(question)

    vq = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=k,
        fields=VECTOR_FIELD,
    )

    # Select only fields that exist in your new index schema
    results = search_client.search(
        search_text=question,  # always set for semantic
        vector_queries=[vq] if use_hybrid else None,
        top=k,
        query_type="semantic",
        semantic_configuration_name="maca-large-rag-1765859572456-semantic-configuration",
        select=[
            "chunk_id", "parent_id", "chunk", "title", "source_url",
            "header_1", "header_2", "header_3",
        ],
    )

    docs = []
    for r in results:
        row = dict(r)

        chunk_id = row.get("chunk_id")
        parent_id = row.get("parent_id")

        content = row.get(CONTENT_FIELD, "") or ""
        page = extract_page_from_chunk_id(chunk_id)

        # New schema: use title + source_url directly
        doc_name = row.get("title") or parent_id or chunk_id or "Unknown document"
        doc_url = row.get("source_url")

        docs.append(
            {
                "content": content,
                "source_id": chunk_id,     # key in your index
                "doc_url": doc_url,        # now comes from source_url
                "doc_name": doc_name,      # now comes from title
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
        "If the provided documentation appears to contain only a table of contents or headings for a section, and not the clause text itself, explicitly state this and request additional relevant chunks instead of concluding the section has no content"
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
        *chat_history,
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
    With the new index, doc_name should usually be the filename from metadata_storage_name,
    so this filter is generally fine to keep.
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages_for_ui" not in st.session_state:
    st.session_state.messages_for_ui = []
if "system_prompt_override" not in st.session_state:
    st.session_state.system_prompt_override = None
if "k" not in st.session_state:
    st.session_state.k = 5
if "max_chars" not in st.session_state:
    st.session_state.max_chars = 4000
if "search_logs" not in st.session_state:
    st.session_state.search_logs = []

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

with st.sidebar:
    st.title("MACA POC")
    page = st.radio("Go to", ["Chat", "Settings", "Search log"], index=0)

# -----------------------------
# Chat page
# -----------------------------
if page == "Chat":
    st.title("MACA and Thiess Frontline Worker Support (POC)")
    st.caption("Ask questions about mining SOPs / terminology / technology etc. Answers are grounded in indexed documents.")

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

                docs_used_filtered = [
                    d for d in docs_used if is_human_readable_source(d.get("doc_name"))
                ]

                st.markdown(answer)

        st.session_state.chat_history = new_history
        st.session_state.messages_for_ui.append(("user", question, None))
        st.session_state.messages_for_ui.append(("assistant", answer, docs_used_filtered))

        fallback_prefix = "I couldn't find anything in the documentation for that question."
        answered = not answer.startswith(fallback_prefix)

        top_source_name = docs_used_filtered[0]["doc_name"] if docs_used_filtered else None

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
        max_value=15,
        value=st.session_state.k,
    )

    st.session_state.max_chars = st.slider(
        "Max context characters",
        1000,
        12000,
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