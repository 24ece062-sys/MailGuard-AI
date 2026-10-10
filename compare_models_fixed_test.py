import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. LOAD ORIGINAL CLEANED DATASET
# ==========================================

df = pd.read_csv("dataset/cleaned_emails.csv")

X = df["text"]
y = df["spam"]


# ==========================================
# 2. RECREATE ORIGINAL FIXED TEST SET
# ==========================================

X_train_original, X_test, y_train_original, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("========== FIXED TEST SET ==========")
print("Test samples:", len(X_test))


# ==========================================
# 3. TRAIN ORIGINAL SVM AGAIN
# ==========================================

original_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_original_tfidf = original_vectorizer.fit_transform(
    X_train_original
)

X_test_original_tfidf = original_vectorizer.transform(
    X_test
)

original_model = LinearSVC()

original_model.fit(
    X_train_original_tfidf,
    y_train_original
)

original_pred = original_model.predict(
    X_test_original_tfidf
)


# ==========================================
# 4. LOAD NEW MODEL + ITS VECTORIZER
# ==========================================

new_model = joblib.load("models/best_model.pkl")

new_vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

new_test_tfidf = new_vectorizer.transform(
    X_test
)

new_pred = new_model.predict(
    new_test_tfidf
)


# ==========================================
# 5. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    print("\n======================================")
    print(name)
    print("======================================")

    print(
        f"Accuracy : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision: {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall   : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nFalse Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")


# ==========================================
# 6. COMPARE BOTH MODELS
# ==========================================

evaluate_model(
    "ORIGINAL SVM",
    y_test,
    original_pred
)

evaluate_model(
    "NEW SVM WITH ADDITIONAL TRAINING",
    y_test,
    new_pred
)


print("\n======================================")
print("FAIR FIXED-TEST COMPARISON COMPLETED")
print("======================================")