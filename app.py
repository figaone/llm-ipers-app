import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from elasticsearch import Elasticsearch
from keybert import KeyBERT
import tiktoken
import os
from datetime import datetime
import json
from streamlit_feedback import streamlit_feedback
import pyrebase
import firebase_admin
from firebase_admin import credentials, auth as admin_auth, firestore


import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, auth as admin_auth, firestore
import json

# ─────────────────────────────────────────────────────────────
# 1️⃣ This MUST be the very first Streamlit call
st.set_page_config(page_title="IPERS OCR Chatbot", layout="wide")
# ─────────────────────────────────────────────────────────────

# 2️⃣ Firebase Admin + Pyrebase setup
sa       = json.loads(st.secrets["firebase"]["firebase_service_account"])
if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate(sa))
firebase_db = firestore.client()

pb_cfg = {
    "apiKey":            st.secrets["FIREBASE"]["apiKey"],
    "authDomain":        st.secrets["FIREBASE"]["authDomain"],
    "databaseURL":       st.secrets["FIREBASE"]["databaseURL"],
    "projectId":         st.secrets["FIREBASE"]["projectId"],
    "storageBucket":     st.secrets["FIREBASE"]["storageBucket"],
    "messagingSenderId": st.secrets["FIREBASE"]["messagingSenderId"],
    "appId":             st.secrets["FIREBASE"]["appId"],
}
pb      = pyrebase.initialize_app(pb_cfg)
fb_auth = pb.auth()

## ─── LOGIN / SIGNUP GATE ───
if "user" not in st.session_state:
    st.title("🔐 IPERS OCR Chatbot — Login / Sign-up")
    mode  = st.radio("Action", ["Login", "Sign up"])
    email = st.text_input("Email", key="auth_email")
    pwd   = st.text_input("Password", type="password", key="auth_pwd")

    if mode == "Sign up":
        if st.button("Create account"):
            try:
                fb_auth.create_user_with_email_and_password(email, pwd)
                st.success("Account created! Now switch to **Login** above.")
            except Exception as e:
                st.error(f"Sign-up failed: {e}")

    else:  # Login
        if st.button("Login"):
            try:
                user     = fb_auth.sign_in_with_email_and_password(email, pwd)
                id_token = user["idToken"]
                decoded  = admin_auth.verify_id_token(id_token)

                # ─── NEW: allow-list check ───
                allowed = st.secrets["ALLOW"]["users"]
                user_email = decoded.get("email", "")
                if user_email not in allowed:
                    st.error("⛔ You are not authorized to use this app.")
                    st.stop()

                # ─── success: persist & rerun ───
                st.session_state.user     = decoded
                st.session_state.id_token = id_token
                st.rerun()

            except Exception as e:
                st.error(f"Login failed: {e}")

    st.stop()


# 4️⃣ LOGOUT BUTTON
st.sidebar.write(f"👤 {st.session_state.user['email']}")
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# ─── GUARDRAIL POPUP ───
if "guard_shown" not in st.session_state:
    st.markdown("### ⚠️ Please note")
    st.markdown(
        """
        • **Data coverage:** January 1, 2018 – December 31, 2021  
        • **Source:** OCR-processed IPERS investment documents  
        • **Privacy:** The chatbot will not surface any sensitive or private information beyond what’s in the scanned documents.  

        **This is an early-stage prototype**  
        - The model may occasionally produce incorrect, incomplete, or misleading answers.  
        - We are actively working to expand the dataset and refine the AI so its accuracy improves over time.  

        **What it can do:**  
        - Answer factual questions about document content, metadata, investment periods, document groups, etc.  

        **What it can’t do:**  
        - Provide legal, financial, or personal advice  
        - Speculate beyond the OCR’d text or outside the 2018–2021 window  
        - Retrieve any data published after 2021  
        - Display private or sensitive data not present in the OCR’d documents  

        **Tip:** To get the most accurate and relevant answers, please narrow your question by adding specific filters—such as date ranges, document group names, or keywords—to give the model clearer context.  

        Thank you for your patience and understanding as we continue to enhance this prototype.
        """
    )
    if st.button("I Understand, Continue"):
        st.session_state.guard_shown = True
        st.rerun()   # <-- immediately restart so we skip this block on this same click
    st.stop()

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
# st.set_page_config(page_title="IPERS OCR Chatbot", layout="wide")
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

def append_feedback_log(entry: dict):
    # 1) existing local‐file log
    if os.path.exists(FEEDBACK_LOG_PATH):
        with open(FEEDBACK_LOG_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(FEEDBACK_LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)

    # 2) new: push to Firestore
    try:
        # you can name your collection anything; here "feedback_logs"
        firebase_db.collection("feedback_logs").add(entry)
    except Exception as e:
        st.error(f"⚠️ Failed to push feedback to Firestore: {e}")

def save_feedback(index):
    fb = st.session_state.get(f"feedback_{index}")
    if fb is None:
        return
    msg = st.session_state.history[index]
    msg["feedback"] = fb

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_email": st.session_state.user.get("email"),
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
