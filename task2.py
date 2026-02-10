# Simple Support Ticket Classification (No NLTK)

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Load data
data = pd.read_csv("support_tickets.csv")

# Input and Output
X = data["ticket"]
y_cat = data["category"]
y_pri = data["priority"]

# Convert text to numbers
vectorizer = CountVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)


# -------- Category Model --------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_cat, test_size=0.2, random_state=42
)

model_cat = MultinomialNB()
model_cat.fit(X_train, y_train)

pred_cat = model_cat.predict(X_test)

print("Category Accuracy:", accuracy_score(y_test, pred_cat))


# -------- Priority Model --------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_pri, test_size=0.2, random_state=42
)

model_pri = MultinomialNB()
model_pri.fit(X_train, y_train)

pred_pri = model_pri.predict(X_test)

print("Priority Accuracy:", accuracy_score(y_test, pred_pri))


# -------- Test New Ticket --------
def predict_ticket(text):

    vec = vectorizer.transform([text])

    cat = model_cat.predict(vec)[0]
    pri = model_pri.predict(vec)[0]

    print("\nTicket:", text)
    print("Category:", cat)
    print("Priority:", pri)


# Example
predict_ticket("My payment failed again")
predict_ticket("Internet is very slow")
c:\Users\BORRA USHA\Videos\Captures\task2.py - FUTURE_ML_02 - Visual Studio Code 2026-02-10 18-55-39.mp4