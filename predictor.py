import joblib
import numpy as np

# Load trained model and TF-IDF vectorizer
model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("=" * 60)
print("              EMAIL SPAM DETECTOR")
print("=" * 60)

# Get new email
email = input("\nEnter the email content:\n")

# Convert email into TF-IDF features
email_vector = vectorizer.transform([email])

# Predict
prediction = model.predict(email_vector)[0]

# Decision score from SVM
decision_score = model.decision_function(email_vector)[0]

# Convert decision score into an easy-to-understand confidence estimate
confidence = 1 / (1 + np.exp(-abs(decision_score)))
confidence = confidence * 100

print("\n" + "=" * 60)

if prediction == 1:
    print("Prediction       : SPAM")
else:
    print("Prediction       : HAM")

print(f"Confidence       : {confidence:.2f}%")

# Find important words/phrases
feature_names = vectorizer.get_feature_names_out()
tfidf_values = email_vector.toarray()[0]
coefficients = model.coef_[0]

# Contribution of each feature
contributions = tfidf_values * coefficients

# Get important features
if prediction == 1:
    important_indices = np.argsort(contributions)[::-1]
else:
    important_indices = np.argsort(contributions)

print("\nImportant Keywords/Phrases:")

count = 0

for index in important_indices:
    if tfidf_values[index] > 0:
        print("  •", feature_names[index])
        count += 1

        if count == 8:
            break

print("=" * 60)