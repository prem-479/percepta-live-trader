import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from GoogleNews import GoogleNews

BASE = os.path.join("colab_training", "models", "sentiment")
os.makedirs(BASE, exist_ok=True)

def fetch_google_news(query="stock market", limit=20):
    googlenews = GoogleNews(lang='en')
    googlenews.search(query)
    results = googlenews.result()

    cleaned = []
    for r in results[:limit]:
        cleaned.append({
            "title": r.get("title", ""),
            "description": r.get("desc", ""),
            "url": r.get("link", "")
        })
    return pd.DataFrame(cleaned)

def load_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def get_sentiment(tokenizer, model, text):
    if not text:
        return "neutral", 0.0

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1).numpy()[0]
    labels = ["positive", "negative", "neutral"]
    return labels[np.argmax(probs)], float(probs.max())

def run():
    print("Fetching news...")

    df = fetch_google_news("stock market", 20)

    if df.empty:
        print("No news fetched!")
        return

    tokenizer, model = load_finbert()

    results = []

    for _, row in df.iterrows():
        text = f"{row['title']}. {row['description']}"
        sentiment, score = get_sentiment(tokenizer, model, text)

        results.append({
            "title": row["title"],
            "description": row["description"],
            "sentiment": sentiment,
            "confidence": round(score, 4),
            "url": row["url"]
        })

    df_sent = pd.DataFrame(results)
    df_sent["impact_score"] = df_sent["sentiment"].map({
        "positive": 1,
        "neutral": 0,
        "negative": -1
    })

    df_sent.to_csv("colab_training/data/sentiment_output.csv", index=False)
    print("Saved sentiment_output.csv")

    tokenizer.save_pretrained(BASE)
    model.save_pretrained(BASE)
    print(f"Model saved to: {BASE}")

if __name__ == "__main__":
    run()
