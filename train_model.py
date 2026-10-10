import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("dataset/cleaned_emails.csv")

# Load additional representative training examples
additional_df = pd.read_csv("dataset/additional_training_emails.csv")

# Combine original dataset with additional examples
df = pd.concat([df, additional_df], ignore_index=True)

print("========== DATASET LOADED ==========")
print("Original dataset shape:", (len(df) - len(additional_df), 2))
print("Additional examples:", len(additional_df))
print("Combined dataset shape:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df["text"]
y = df["spam"]

print("\nEmail samples:", len(X))
print("Labels:", y.unique())


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN-TEST SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. TF-IDF FEATURE EXTRACTION
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\n========== TF-IDF ==========")
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# ==========================================
# 5. CREATE MODELS
# ==========================================

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "SVM": LinearSVC()
}


# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

results = {}

for name, model in models.items():

    print("\n======================================")
    print("Training:", name)
    print("======================================")

    # Train model
    model.fit(X_train_tfidf, y_train)

    # Prediction
    y_pred = model.predict(X_test_tfidf)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ==========================================
# 7. MODEL COMPARISON
# ==========================================

print("\n\n========== MODEL COMPARISON ==========")

results_df = pd.DataFrame(results).T

print(results_df)


# ==========================================
# 8. FIND BEST MODEL
# ==========================================

best_model_name = results_df["F1 Score"].idxmax()

print("\nBest Model:", best_model_name)
print(
    "Best F1 Score:",
    results_df.loc[best_model_name, "F1 Score"]
)

# ==========================================
# 9. SAVE TRAINED MODELS
# ==========================================

os.makedirs("models", exist_ok=True)

# Save all trained models
joblib.dump(models["Naive Bayes"], "models/naive_bayes.pkl")
joblib.dump(models["Logistic Regression"], "models/logistic_regression.pkl")
joblib.dump(models["SVM"], "models/svm.pkl")

# Save the best model
best_model = models[best_model_name]
joblib.dump(best_model, "models/best_model.pkl")

# Save TF-IDF vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Save model comparison results
results_df.to_csv("models/model_comparison.csv")

print("\n========== MODELS SAVED ==========")
print("Best model:", best_model_name)
print("All models and TF-IDF vectorizer saved successfully!")

print("\n========== TRAINING COMPLETED ==========")