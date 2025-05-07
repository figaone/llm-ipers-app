import streamlit as st
from openai import OpenAI
from elasticsearch import Elasticsearch
from keybert import KeyBERT
import tiktoken
import os
from datetime import datetime
import json
from streamlit_feedback import streamlit_feedback

# ---- Config ----
es = Elasticsearch(st.secrets["ES_CLOUD_URL"], api_key=st.secrets["ES_API_KEY"])
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
index_name = "ocr_documents_pages"
enc = tiktoken.encoding_for_model("gpt-4")
kw_model = KeyBERT(model="all-MiniLM-L6-v2")
FEEDBACK_LOG_PATH = "feedback_log.json"

# ---- Helpers ----
def extract_keywords(text, top_k=15):
    return [kw[0].lower() for kw in kw_model.extract_keywords(text, top_n=top_k, stop_words="english")]

def query_elasticsearch(question, max_docs=200, max_tokens=90000):
    keywords = extract_keywords(question)
    should = [{"match": {"content": kw}} for kw in keywords]
    res = es.search(
        index=index_name,
        size=max_docs,
        query={"bool": {"should": should, "minimum_should_match": 1}}
    )
    if not res["hits"]["hits"]:
        res = es.search(index=index_name, size=max_docs, query={"match": {"content": {"query": question}}})
    if not res["hits"]["hits"]:
        res = es.search(index=index_name, size=max_docs, query={"match_all": {}})
    total_tokens, docs = 0, []
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        content = src.get("content","")
        tokens = len(enc.encode(content))
        if total_tokens + tokens > max_tokens:
            break
        meta = (
            f"Investment Period: {src.get('investment_period','N/A')}, "
            f"Document Group: {src.get('document_group','N/A')}, "
            f"Document ID: {src.get('document_id','N/A')}, "
            f"Pages: {src.get('page_numbers','N/A')}"
        )
        docs.append(f"{meta}\n{content}")
        total_tokens += tokens
    return "\n\n".join(docs)

def ask_gpt_with_context(prompt):
    context = query_elasticsearch(prompt)
    system = f"""You are an AI assistant specialized in analyzing OCR-processed investment documents for IPERS.

Use the full content below to answer the user's question concisely and accurately.

--- Context ---
{context}

--- Question ---
{prompt}

**Answer briefly and clearly:**"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":system}],
        max_tokens=1000,
    )
    return resp.choices[0].message.content.strip()

def append_feedback_log(entry: dict):
    if os.path.exists(FEEDBACK_LOG_PATH):
        with open(FEEDBACK_LOG_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(FEEDBACK_LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)

# ---- UI ----
st.set_page_config(page_title="IPERS OCR Chatbot", layout="wide")
st.sidebar.title("Feedback log")
if os.path.exists(FEEDBACK_LOG_PATH):
    with open(FEEDBACK_LOG_PATH, "r") as f:
        log_content = f.read()
    st.sidebar.download_button("⬇️ Download feedback log", log_content, file_name="feedback_log.json")

st.title("📄 IPERS Document Chatbot")
st.caption("Ask questions about OCR’d IPERS investment documents")

if "history" not in st.session_state:
    st.session_state.history = []

# render chat history
for idx, m in enumerate(st.session_state.history):
    st.chat_message("user").write(m["prompt"])
    st.chat_message("assistant").write(m["answer"])
    # attach feedback widget if not yet given
    feedback = m.get("feedback", None)
    st.session_state[f"feedback_{idx}"] = feedback
    st.feedback(
        "thumbs",
        key=f"feedback_{idx}",
        disabled=feedback is not None,
        on_change=lambda i=idx: save_feedback(i),
        args=[idx],
    )

def save_feedback(index):
    fb = st.session_state.get(f"feedback_{index}")
    if fb is None:
        return
    # update in-memory history
    msg = st.session_state.history[index]
    msg["feedback"] = fb
    # append to disk
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": msg["prompt"],
        "answer": msg["answer"],
        "feedback": fb
    }
    append_feedback_log(entry)
    st.toast("✅ Feedback recorded!")

# new user input
if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").write(prompt)
    answer = ask_gpt_with_context(prompt)
    st.chat_message("assistant").write(answer)

    # add to history
    st.session_state.history.append({"prompt": prompt, "answer": answer})
    idx = len(st.session_state.history) - 1

    # immediate feedback widget for the new response
    st.session_state[f"feedback_{idx}"] = None
    st.feedback(
        "thumbs",
        key=f"feedback_{idx}",
        on_change=lambda i=idx: save_feedback(i),
        args=[idx],
    )
