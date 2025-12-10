import streamlit as st
import pandas as pd
import pickle
from sqlalchemy import create_engine, text

# ---------------------------------------------------
# 1️⃣ LOAD ML FILES
# ---------------------------------------------------
vectorizer = pickle.load(open(r"C:\\Users\\SHRUTI\\Downloads\\BankingChatbot\\vectorizer.pkl", "rb"))
model = pickle.load(open(r"C:\\Users\\SHRUTI\\Downloads\\BankingChatbot\\model.pkl", "rb"))
label_encoder = pickle.load(open(r"C:\\Users\\SHRUTI\\Downloads\\BankingChatbot\\label_encoder.pkl", "rb"))
df = pd.read_excel(r"C:\\Users\\SHRUTI\\Downloads\\BankingChatbot\\banking_chatbot_full_dataset.xlsx")

# ---------------------------------------------------
# 2️⃣ DATABASE (POSTGRESQL)
# ---------------------------------------------------
# CHANGE THIS TO YOUR DATABASE NAME / USER / PASSWORD
engine = create_engine("postgresql://postgres:yourpassword@localhost:5432/yourdbname")

def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chats (
                id SERIAL PRIMARY KEY,
                user_query TEXT,
                intent TEXT,
                bot_response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()

init_db()

def save_to_db(user_msg, intent, bot_msg):
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO chats (user_query, intent, bot_response) VALUES (:u, :i, :b)"),
            {"u": user_msg, "i": intent, "b": bot_msg}
        )
        conn.commit()

# ---------------------------------------------------
# 3️⃣ CHATBOT RESPONSE FUNCTION
# ---------------------------------------------------
def chatbot_response(user_query):
    query_vec = vectorizer.transform([user_query])

    # Probability for confidence checking
    probs = model.predict_proba(query_vec)[0]
    max_prob = max(probs)

    # Predict intent
    pred = model.predict(query_vec)[0]
    intent = label_encoder.inverse_transform([pred])[0]

    # Low confidence → fallback message
    if max_prob < 0.40:
        return (
            "Sorry! I don’t have an answer for that yet. Please ask something related to Banking.",
            "unknown"
        )

    # Fetch answer from dataset
    answers = df[df["intent"] == intent]["answer"]

    if answers.empty:
        response = "Sorry! I don’t have an answer for that yet. Please ask something related to Banking."
    else:
        response = answers.sample(1).values[0]

    return response, intent

# ---------------------------------------------------
# 4️⃣ STREAMLIT UI
# ---------------------------------------------------
st.title("💬 Banking Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask me anything about banking:")

if st.button("Send"):
    if user_input.strip() != "":
        bot_reply, intent = chatbot_response(user_input)

        # Save to streamlit UI
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("Bot", bot_reply))

        # Save to DATABASE
        save_to_db(user_input, intent, bot_reply)

# Chat display
st.write("### Chat History")
for sender, msg in st.session_state.chat_history:
    st.write(f"**{sender}:** {msg}")
