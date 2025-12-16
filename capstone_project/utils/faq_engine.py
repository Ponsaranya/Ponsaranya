import pandas as pd
import pickle
# Load ML model files
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
model = pickle.load(open("models/model.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
df = pd.read_excel("data/FAQ_dataset.xlsx")
# FAQ Answer With Confidence
def get_faq_answer(text):
    try:
        # Convert query into TF-IDF vector
        vec = vectorizer.transform([text])

        # Get model confidence
        probs = model.predict_proba(vec)[0]
        max_prob = max(probs)
        
        if max_prob < 0.40:   # 40% threshold – change to 0.50 for stricter
            return None       # Returning None tells app.py to use fallback

        # Predict intent
        pred = model.predict(vec)[0]
        intent = label_encoder.inverse_transform([pred])[0]

        # Fetch answer from dataset
        answers = df[df["intent"] == intent]["answer"]

        if answers.empty:
            return None  # No matching answer → fallback

        return answers.sample(1).values[0]

    except Exception as e:
        print("FAQ Engine Error:", e)
        return None
