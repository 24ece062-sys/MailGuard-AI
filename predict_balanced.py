import joblib
import numpy as np

# Load Balanced SVM and TF-IDF
model = joblib.load("models/balanced_svm.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# Enter a completely new email here
email = input("\nEnter the email content:\n")


# Convert email into TF-IDF features
email_vector = vectorizer.transform([email])


# Predict
prediction = model.predict(email_vector)[0]


# Decision score
decision_score = model.decision_function(email_vector)[0]

confidence = 1 / (1 + np.exp(-abs(decision_score)))
confidence = confidence * 100


# Display result
print("\n===== EMAIL ANALYSIS =====")

if prediction == 1:
    print("Prediction : SPAM")
else:
    print("Prediction : HAM")

print(f"Confidence : {confidence:.2f}%")
print(f"Decision Score : {decision_score:.4f}")