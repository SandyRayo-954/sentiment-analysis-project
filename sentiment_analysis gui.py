

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
print("Loading dataset...")
df = pd.read_csv('C:/Users/Sandeep/Desktop/training.1600000.processed.noemoticon.csv',
                 encoding='ISO-8859-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Sample for quick processing
print("Sampling dataset...")
df = df.sample(5000, random_state=42)

# Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    words = [word for word in tokens if word.isalpha()]
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

print("Preprocessing text...")
df['clean_text'] = df['text'].apply(preprocess_text)
df['label'] = df['target'].map({0: 0, 2: 1, 4: 2})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

# Train the model
print("Training model...")
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
model.fit(X_train, y_train)

# Prediction function with confidence
def predict_sentiment(text):
    cleaned = preprocess_text(text)
    probs = model.predict_proba([cleaned])[0]
    predicted_class = probs.argmax()
    confidence = probs[predicted_class]
    sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}[predicted_class]
    return sentiment, confidence

# Store counts of analyzed sentiments in current session
sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}

# GUI functions
def analyze_sentiment():
    user_input = entry.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return
    sentiment, confidence = predict_sentiment(user_input)
    result_label.config(text=f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
    color_map = {"Positive": "#2e7d32", "Neutral": "#f9a825", "Negative": "#c62828"}
    result_label.config(fg=color_map.get(sentiment, "black"))

    # Update counts
    sentiment_counts[sentiment] += 1

def save_result():
    user_input = entry.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return
    sentiment, confidence = predict_sentiment(user_input)
    with open("sentiment_results.txt", "a", encoding='utf-8') as file:
        file.write(f"Text: {user_input}\nSentiment: {sentiment} (Confidence: {confidence:.2f})\n\n")
    messagebox.showinfo("Saved", "Result saved to 'sentiment_results.txt'.")

def clear_text():
    entry.delete("1.0", tk.END)
    result_label.config(text="Sentiment: ", fg="black")

def show_pie_chart():
    if sum(sentiment_counts.values()) == 0:
        messagebox.showinfo("No Data", "Analyze some texts first to see the pie chart.")
        return

    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()
    colors = ['#2e7d32', '#f9a825', '#c62828']  # positive-green, neutral-yellow, negative-red

    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title("Sentiment Distribution of Analyzed Texts")
    plt.axis('equal')  # Equal aspect ratio ensures pie is circular.
    plt.show()

# Build GUI with style
root = tk.Tk()
root.title("Sentiment Analysis Tool")
root.geometry("700x500")
root.configure(bg="#f0f4f8")

style = ttk.Style(root)
style.theme_use('clam')

style.configure('TButton', font=('Segoe UI', 11, 'bold'), padding=8, foreground="#ffffff", background="#4a90e2")
style.map('TButton',
          background=[('active', '#357ABD')],
          foreground=[('disabled', '#a3a3a3')])

frame_title = tk.Frame(root, bg="#4a90e2", pady=10)
frame_title.pack(fill=tk.X)

title_label = tk.Label(frame_title, 
                       text="Sentiment Analysis Project",
                       font=("Segoe UI", 18, "bold"),
                       fg="white",
                       bg="#4a90e2")
title_label.pack()

subtitle_label = tk.Label(root,
                          text="Predict if text sentiment is Positive, Neutral, or Negative.",
                          font=("Segoe UI", 12),
                          bg="#f0f4f8")
subtitle_label.pack(pady=(10, 20))

frame_input = tk.Frame(root, bg="#f0f4f8")
frame_input.pack(pady=5)

input_label = tk.Label(frame_input,
                       text="Enter text to analyze:",
                       font=("Segoe UI", 12, "bold"),
                       bg="#f0f4f8")
input_label.pack(anchor="w")

entry = tk.Text(frame_input, height=6, width=75, font=("Segoe UI", 11), bd=2, relief="groove")
entry.pack(pady=5)

frame_buttons = tk.Frame(root, bg="#f0f4f8")
frame_buttons.pack(pady=10)

analyze_btn = ttk.Button(frame_buttons, text="Analyze", command=analyze_sentiment)
analyze_btn.grid(row=0, column=0, padx=10)

save_btn = ttk.Button(frame_buttons, text="Save Result", command=save_result)
save_btn.grid(row=0, column=1, padx=10)

clear_btn = ttk.Button(frame_buttons, text="Clear", command=clear_text)
clear_btn.grid(row=0, column=2, padx=10)

pie_btn = ttk.Button(frame_buttons, text="Show Pie Chart", command=show_pie_chart)
pie_btn.grid(row=0, column=3, padx=10)

result_label = tk.Label(root, text="Sentiment: ", font=("Segoe UI", 14, "bold"), bg="#f0f4f8")
result_label.pack(pady=15)

print("Ready. Launching GUI...")
root.mainloop()


