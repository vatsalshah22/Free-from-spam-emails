import tkinter as tk
from tkinter import messagebox

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --------------------------
# Load Dataset
# --------------------------

try:
    data = pd.read_csv("spam.csv")
except:
    messagebox.showerror("Error", "spam.csv file not found")
    exit()

# --------------------------
# Prepare Data
# --------------------------

X = data["message"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------
# Train Model
# --------------------------

model = MultinomialNB()

model.fit(X_train, y_train)

# --------------------------
# Prediction Function
# --------------------------

def check_email():

    text = email_box.get("1.0", tk.END).strip()

    if text == "":
        messagebox.showwarning("Warning", "Please enter an email.")
        return

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector).max()

    if prediction == "spam":
        result.config(
            text=f"⚠ SPAM EMAIL\nConfidence: {probability*100:.2f}%",
            fg="red"
        )

    else:
        result.config(
            text=f"✓ SAFE EMAIL\nConfidence: {probability*100:.2f}%",
            fg="green"
        )


# --------------------------
# GUI
# --------------------------

root = tk.Tk()

root.title("Email Spam Detection")
root.geometry("600x450")
root.resizable(False, False)

title = tk.Label(
    root,
    text="Email Spam Detection using Machine Learning",
    font=("Arial", 16, "bold")
)

title.pack(pady=10)

label = tk.Label(
    root,
    text="Enter Email Message:",
    font=("Arial", 12)
)

label.pack()

email_box = tk.Text(
    root,
    width=60,
    height=10
)

email_box.pack(pady=10)

button = tk.Button(
    root,
    text="Check Email",
    font=("Arial", 12),
    bg="blue",
    fg="white",
    command=check_email
)

button.pack(pady=10)

result = tk.Label(
    root,
    text="",
    font=("Arial", 14, "bold")
)

result.pack(pady=20)

root.mainloop()

