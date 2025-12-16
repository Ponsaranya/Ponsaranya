import streamlit as st
import pandas as pd
import pickle
import sqlite3
import os
from datetime import datetime

# IMPORT MODULES
from utils.recommendation_engine import get_recommendation
from utils.faq_engine import get_faq_answer
from utils.detect import detect_module

# DATABASE SETUP
DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(sender, message):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO chat (sender, message, timestamp) VALUES (?, ?, ?)",
        (sender, message, local_time)
    )
    conn.commit()
    conn.close()

init_db()

# LOAD FAQ MODEL FILES
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
model = pickle.load(open("models/model.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

df_faq = pd.read_excel("data/FAQ_dataset.xlsx")

# FAQ CHATBOT RESPONSE
def chatbot_faq(user_query):
    try:
        query_vec = vectorizer.transform([user_query])
        probs = model.predict_proba(query_vec)[0]
        if max(probs) < 0.20:
            return None

        pred = model.predict(query_vec)[0]
        intent = label_encoder.inverse_transform([pred])[0]
        answers = df_faq[df_faq["intent"] == intent]["answer"]

        if answers.empty:
            return None

        return answers.sample(1).values[0]

    except:
        return None

# Load CSS
def load_css_try(paths):
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            return True
    return False

load_css_try(["style.css", os.path.join("static", "style.css")])

# Page config
st.set_page_config(page_title="Banking Chatbot", layout="wide")

# IMPORTANT: Start with EMPTY UI (do NOT load DB chat)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# SUBMIT HANDLER
def submit_message():
    msg = st.session_state.get("chat_input", "")
    if not msg or not msg.strip():
        return

    parts = detect_module(msg)
    responses = []

    if parts.get("faq"):
        faq_reply = chatbot_faq(parts["faq"])
        if faq_reply:
            responses.append(faq_reply)

    if parts.get("recommendation"):
        rec_reply = get_recommendation(parts["recommendation"])
        if rec_reply:
            responses.append(rec_reply)

    if not responses:
        responses.append("Please ask something related to banking or product suggestions.")

    final_reply = "\n\n".join(responses)

    st.session_state.chat_history.append(("You", msg))
    st.session_state.chat_history.append(("Bot", final_reply))

    save_to_db("You", msg)
    save_to_db("Bot", final_reply)

    st.session_state["chat_input"] = ""

# HEADER
st.markdown("<div class='page-header'>", unsafe_allow_html=True)
st.markdown("<h1 class='app-title'>Banking Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-desc'>Ask about banking and product suggestions</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# CONVERSATION
st.markdown('<div class="conversation-card">', unsafe_allow_html=True)
st.markdown("<div class='conv-title-row'><h2 class='conv-title'>Conversation</h2></div>", unsafe_allow_html=True)
st.markdown('<div class="chat-window">', unsafe_allow_html=True)

for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(
            f"<div class='msg-row user-row'><div class='bubble user-bubble'>{message}</div></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='msg-row bot-row'><div class='bubble bot-bubble'>{message}</div></div>",
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# INPUT ROW
st.markdown('<div class="input-row">', unsafe_allow_html=True)
left_col, right_col = st.columns([4, 1], gap="small")

with left_col:
    st.text_input(
        "",
        key="chat_input",
        placeholder="Type your message...",
        on_change=submit_message,
        label_visibility="collapsed"
    )
    if st.button("Send"):
        submit_message()

with right_col:
    st.markdown("<div style='margin-top:22px'>", unsafe_allow_html=True)
    if st.button("Reset"):
        st.session_state.chat_history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='color:green;font-size:12px;margin-top:6px'>"
        "If you want to reset the conversation, click this button."
        "</div>",
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)