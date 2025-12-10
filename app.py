import sqlite3
import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load saved ML model files
# ---------------------------
vectorizer = pickle.load(open(r"C:\\Users\\SHRUTI\\Downloads\\BankingChatbot\\vectorizer.pkl", "rb"))
model = pickle.load(open(r"C:\\Users\\SHRUTI\\Downloads\\BankingChatbot\\model.pkl", "rb"))
label_encoder = pickle.load(open(r"C:\\Users\\SHRUTI\\Downloads\\BankingChatbot\\label_encoder.pkl", "rb"))
df = pd.read_excel(r"C:\\Users\\SHRUTI\\Downloads\\BankingChatbot\\banking_chatbot_full_dataset.xlsx")

# ---------------------------
# DATABASE SETUP
# ---------------------------
DB_PATH = r"C:\\Users\\SHRUTI\\Downloads\\BankingChatbot\\chatbot_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT,
            intent TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------------------
# SAVE TO DATABASE
# ---------------------------
def save_to_db(user_msg, intent, bot_msg):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chats (user_query, intent, bot_response) VALUES (?, ?, ?)",
        (user_msg, intent, bot_msg)
    )
    conn.commit()
    conn.close()

# ---------------------------
# CHATBOT RESPONSE FUNCTION
# ---------------------------
def chatbot_response(user_query):
    query_vec = vectorizer.transform([user_query])

    # Get probabilities of each intent
    probs = model.predict_proba(query_vec)[0]
    max_prob = max(probs)

    # Get top intent
    pred = model.predict(query_vec)[0]
    intent = label_encoder.inverse_transform([pred])[0]

    # -------- CONFIDENCE CHECK --------
    if max_prob < 0.40:
        return (
            "Sorry! I don't have an answer for that yet. Please try asking something related to Banking.",
            "unknown"
        )

    # Fetch the answer from dataset
    answers = df[df["intent"] == intent]["answer"]

    if answers.empty:
        response = "Sorry! I don't have an answer for that yet. Please try asking something related to Banking."
    else:
        response = answers.sample(1).values[0]

    return response, intent

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Banking Chatbot")

# Store chat history inside Streamlit
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask me anything about banking:")

if st.button("Send"):
    if user_input.strip() != "":
        bot_reply, intent = chatbot_response(user_input)

        # Save to UI chat history
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("Bot", bot_reply))

        # Save into SQL database
        save_to_db(user_input, intent, bot_reply)

# Display chat history
for sender, msg in st.session_state.chat_history:
    st.write(f"**{sender}:** {msg}")
