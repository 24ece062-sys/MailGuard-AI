import pandas as pd
import joblib

from sklearn.model_selection import train_test_split


# ============================================================
# 1. Load cleaned dataset
# ============================================================

df = pd.read_csv("dataset/cleaned_emails.csv")

X = df["text"]
y = df["spam"]


# ============================================================
# 2. Recreate EXACTLY the same train-test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. Load trained SVM and TF-IDF
# ============================================================

model = joblib.load("models/svm.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# ============================================================
# 4. Convert test emails to TF-IDF
# ============================================================

X_test_tfidf = vectorizer.transform(X_test)


# ============================================================
# 5. Predict
# ============================================================

y_pred = model.predict(X_test_tfidf)


# ============================================================
# 6. Create results table
# ============================================================

results = pd.DataFrame({
    "email": X_test.values,
    "actual": y_test.values,
    "predicted": y_pred
})


# ============================================================
# 7. Find errors
# ============================================================

false_positives = results[
    (results["actual"] == 0) &
    (results["predicted"] == 1)
]

false_negatives = results[
    (results["actual"] == 1) &
    (results["predicted"] == 0)
]


# ============================================================
# 8. Display summary
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

print(f"\nTest emails       : {len(results)}")
print(f"False Positives   : {len(false_positives)}")
print(f"False Negatives   : {len(false_negatives)}")


# ============================================================
# 9. Display False Positives
# ============================================================

print("\n" + "=" * 70)
print("FALSE POSITIVES")
print("(Actual HAM -> Predicted SPAM)")
print("=" * 70)

for i, row in false_positives.iterrows():

    print(f"\nEmail {i}:")
    print(row["email"])


# ============================================================
# 10. Display False Negatives
# ============================================================

print("\n" + "=" * 70)
print("FALSE NEGATIVES")
print("(Actual SPAM -> Predicted HAM)")
print("=" * 70)

for i, row in false_negatives.iterrows():

    print(f"\nEmail {i}:")
    print(row["email"])


# ============================================================
# 11. Save error files
# ============================================================

false_positives.to_csv(
    "dataset/false_positives.csv",
    index=False
)

false_negatives.to_csv(
    "dataset/false_negatives.csv",
    index=False
)


print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETED")
print("=" * 70)

print("\nSaved:")
print("dataset/false_positives.csv")
print("dataset/false_negatives.csv")