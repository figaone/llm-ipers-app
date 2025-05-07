import streamlit as st
from openai import OpenAI
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from keybert import KeyBERT
import tiktoken
import os
import json



# Use secrets instead of dotenv
es = Elasticsearch(st.secrets["ES_CLOUD_URL"], api_key=st.secrets["ES_API_KEY"])
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Create ES client
# es = Elasticsearch(
#     ES_CLOUD_URL,
#     api_key=ES_API_KEY,
# )


# client = OpenAI(api_key=OPENAI_API_KEY)
index_name = "ocr_documents_pages"
enc = tiktoken.encoding_for_model("gpt-4")
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

# ---- Feedback log file ----
FEEDBACK_LOG_PATH = "feedback_log.json"

# ---- Utility Functions ----
def extract_keywords(text, top_k=15):
    return [kw[0].lower() for kw in kw_model.extract_keywords(text, top_n=top_k, stop_words='english')]

def query_elasticsearch(question, max_docs=200, max_tokens=90000):
    keywords = extract_keywords(question)
    should_clauses = [{"match": {"content": kw}} for kw in keywords]

    res = es.search(index=index_name, size=max_docs, query={"bool": {"should": should_clauses, "minimum_should_match": 1}})
    if len(res["hits"]["hits"]) == 0:
        res = es.search(index=index_name, size=max_docs, query={"match": {"content": {"query": question}}})
    if len(res["hits"]["hits"]) == 0:
        res = es.search(index=index_name, size=max_docs, query={"match_all": {}})

    total_tokens, docs = 0, []
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        meta = (
            f"Investment Period: {src.get('investment_period', 'N/A')}, "
            f"Document Group: {src.get('document_group', 'N/A')}, "
            f"Document ID: {src.get('document_id', 'N/A')}, "
            f"Pages: {src.get('page_numbers', 'N/A')}"
        )
        content = src.get("content", "")
        tokens = len(enc.encode(content))
        if total_tokens + tokens > max_tokens:
            break
        docs.append(f"{meta}\n{content}")
        total_tokens += tokens
    return "\n\n".join(docs)

def ask_gpt_with_context(question):
    context = query_elasticsearch(question)
    prompt = f"""
You are an AI assistant specialized in analyzing OCR-processed investment documents for IPERS.

Use the full content below to answer the user's question concisely and accurately.

--- Context ---
{context}

--- Question ---
{question}

**Answer briefly and clearly:**
"""
    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], max_tokens=1000,
    )
    return response.choices[0].message.content.strip()

def save_feedback(prompt, answer, rating):
    feedback_entry = {"prompt": prompt, "answer": answer, "rating": rating}
    if os.path.exists(FEEDBACK_LOG_PATH):
        with open(FEEDBACK_LOG_PATH, "r") as f:
            history = json.load(f)
    else:
        history = []
    history.append(feedback_entry)
    with open(FEEDBACK_LOG_PATH, "w") as f:
        json.dump(history, f, indent=2)

# ---- Streamlit UI ----
st.set_page_config(page_title="IPERS OCR Chatbot", layout="wide")
st.title("📄 IPERS Document Chatbot")
st.caption("Ask questions about OCR'd IPERS investment documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for idx, msg in enumerate(st.session_state.chat_history):
    st.chat_message("user").write(msg["question"])
    st.chat_message("assistant").write(msg["answer"])

    # Rating input
    rating = st.radio(
        f"Rate this answer (Prompt #{idx + 1})", ["👍", "👎"], key=f"rating_{idx}", horizontal=True
    )
    if f"saved_{idx}" not in st.session_state:
        save_feedback(msg["question"], msg["answer"], rating)
        st.session_state[f"saved_{idx}"] = True
        st.success("✅ Feedback saved")

question = st.chat_input("Ask a question about IPERS documents...")

if question:
    st.chat_message("user").write(question)
    answer = ask_gpt_with_context(question)
    st.chat_message("assistant").write(answer)

    # Store question-answer pair temporarily for rating
    qa_id = len(st.session_state.chat_history)
    st.session_state.chat_history.append({"question": question, "answer": answer})

    # Prompt for feedback immediately
    rating = st.radio(
        f"Rate this answer (Prompt #{qa_id + 1})", ["👍", "👎"], key=f"rating_{qa_id}", horizontal=True
    )

    # Save immediately once a rating is chosen
    if f"saved_{qa_id}" not in st.session_state and rating in ["👍", "👎"]:
        save_feedback(question, answer, rating)
        st.session_state[f"saved_{qa_id}"] = True
        # st.success("✅ Feedback saved")