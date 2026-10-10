
from flask import Flask, render_template, request
import joblib
import re
import math
import os

app = Flask(__name__)

# Project folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and vectorizer
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
VECTORIZER_PATH = os.path.join(
    BASE_DIR, "models", "tfidf_vectorizer.pkl"
)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_email(email_text):
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]

    # Estimated confidence from the SVM decision score
    confidence = 50.0

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(email_vector)[0])
        confidence = (
            1 / (1 + math.exp(-min(abs(score), 700)))
        ) * 100

    # Find words in the trained vocabulary
    email_words = set(
        re.findall(r"\b[a-zA-Z]{2,}\b", email_text.lower())
    )
    vocabulary = set(vectorizer.get_feature_names_out())

    keywords = [
        word for word in email_words
        if word in vocabulary
    ][:8]

    return {
        "prediction": "SPAM" if int(prediction) == 1 else "HAM",
        "confidence": round(confidence, 2),
        "keywords": keywords
    }


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    email_text = ""

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()

        if email_text:
            result = predict_email(email_text)
        else:
            result = {
                "error": "Please enter an email message before checking."
            }

    return render_template(
        "index.html",
        result=result,
        email_text=email_text
    )


if __name__ == "__main__":
    app.run(debug=True)