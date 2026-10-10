import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("dataset/cleaned_emails.csv")

# Count Spam and Ham emails
spam_count = (df["spam"] == 1).sum()
ham_count = (df["spam"] == 0).sum()

print("Spam emails:", spam_count)
print("Ham emails:", ham_count)


# ==========================================
# CREATE VISUALIZATION FOLDER
# ==========================================

os.makedirs("visualizations", exist_ok=True)


# ==========================================
# SPAM VS HAM BAR CHART
# ==========================================

labels = ["Ham", "Spam"]
counts = [ham_count, spam_count]

plt.figure(figsize=(8, 5))

plt.bar(labels, counts)

plt.title("Spam vs Ham Email Distribution")
plt.xlabel("Email Type")
plt.ylabel("Number of Emails")

# Display values above bars
for i, count in enumerate(counts):
    plt.text(i, count + 50, str(count), ha="center")

plt.tight_layout()

# Save graph
plt.savefig("visualizations/spam_ham_distribution.png")

plt.show()

print("\nSpam vs Ham visualization saved successfully!")