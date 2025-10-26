"""
Kumparan Topic Extractor Model (FastText + Logistic Regression Version)
Author: Heydar (Data Scientist)
Compatible with kumparanian ds verify
"""

import re
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import kumparanian as kn

# Ensure NLTK dependencies are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


class Model:
    def __init__(self):
        print("🔤 Initializing FastText + Logistic Regression model (no Sastrawi)...")
        self.ft_model = None
        self.clf = None
        self.report = {}
        self.stop_words = set(stopwords.words('indonesian'))

    def light_stem(self, word: str) -> str:
        """Lightweight heuristic Indonesian stemmer (no external lib)."""
        for suf in ['kan', 'annya', 'annya', 'nya', 'lah', 'kah', 'an', 'i']:
            if word.endswith(suf) and len(word) > len(suf) + 2:
                word = word[: -len(suf)]
        for pre in ['meng', 'meny', 'men', 'mem', 'me', 'ber', 'ter', 'ke', 'se', 'di']:
            if word.startswith(pre) and len(word) > len(pre) + 2:
                word = word[len(pre):]
        return word

    def clean_and_tokenize(self, text):
        """Normalize, tokenize, remove stopwords, and lightly stem tokens."""
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = [w for w in word_tokenize(text) if w not in self.stop_words]
        tokens = [self.light_stem(w) for w in tokens]
        return tokens

    def get_sentence_vector(self, tokens):
        """Average FastText word vectors for a given token list."""
        if not tokens or not self.ft_model:
            return np.zeros(300)
        vectors = [self.ft_model.wv[w] for w in tokens if w in self.ft_model.wv]
        if len(vectors) == 0:
            return np.zeros(self.ft_model.vector_size)
        return np.mean(vectors, axis=0)

    def train(self, data_path="data.csv"):
        start_time = time.time()
        print("=== Loading dataset ===")
        df = pd.read_csv(data_path)
        print(f"Dataset shape: {df.shape}")

        # Validation
        required_cols = {"article_content", "article_topic"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Dataset must contain columns: {required_cols}")

        df.dropna(subset=["article_content", "article_topic"], inplace=True)
        df.drop_duplicates(subset=["article_content"], inplace=True)
        print(f"Dataset after cleaning: {df.shape}")

        if len(df) > 20000:
            df = df.sample(10000, random_state=42)
            print("📉 Using 10,000 samples for faster FastText training.")

        print("🧹 Cleaning and tokenizing text...")
        df["tokens"] = df["article_content"].apply(self.clean_and_tokenize)

        X_train_tokens, X_val_tokens, y_train, y_val = train_test_split(
            df["tokens"], df["article_topic"],
            test_size=0.2, random_state=42, stratify=df["article_topic"]
        )

        print("⚙️ Training FastText embeddings (from scratch)...")
        self.ft_model = FastText(
            sentences=X_train_tokens,
            vector_size=300,
            window=10,
            min_count=2,
            sg=1,
            epochs=25,
            workers=4
        )

        print("🔢 Generating sentence embeddings...")
        X_train_vecs = np.array([self.get_sentence_vector(t) for t in X_train_tokens])
        X_val_vecs = np.array([self.get_sentence_vector(t) for t in X_val_tokens])

        print("🤖 Training Logistic Regression classifier...")
        clf = LogisticRegression(max_iter=500, solver='saga', n_jobs=-1)
        clf.fit(X_train_vecs, y_train)
        self.clf = clf

        print("📊 Evaluating model performance...")
        y_pred = clf.predict(X_val_vecs)
        acc = accuracy_score(y_val, y_pred)
        cls_report = classification_report(y_val, y_pred)

        duration = round(time.time() - start_time, 2)
        print(f"✅ Validation Accuracy: {acc:.4f}")
        print(cls_report)

        self.report = {
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": df.shape,
            "unique_topics": len(df["article_topic"].unique()),
            "validation_accuracy": round(acc, 4),
            "training_duration_sec": duration,
            "classification_report": cls_report
        }

        self._save_report()
        print(f"✅ Training complete in {duration} seconds.")

    def _save_report(self, path="training_report.txt"):
        with open(path, "w", encoding="utf-8") as f:
            f.write("📘 FASTTEXT + LOGISTIC REGRESSION TOPIC EXTRACTOR REPORT\n")
            f.write("=" * 60 + "\n")
            for k, v in self.report.items():
                if k != "classification_report":
                    f.write(f"{k}: {v}\n")
            f.write("\n=== Classification Report ===\n")
            f.write(self.report["classification_report"])
            f.write("\n" + "=" * 60 + "\n")
        print(f"📝 Report saved to {path}")

    def predict(self, text):
        """Predict topic for a given article text."""
        if not self.ft_model or not self.clf:
            raise ValueError("Model not trained yet.")
        tokens = self.clean_and_tokenize(text)
        vector = self.get_sentence_vector(tokens).reshape(1, -1)
        return str(self.clf.predict(vector)[0])

    def save_pickle(self, path="model.pickle"):
        """Save entire model (FastText + classifier) in a single file for Kumparan."""
        with open(path, "wb") as f:
            pickle.dump({
                "fasttext_model": self.ft_model,
                "classifier": self.clf,
                "predict": self.predict
            }, f)
        print(f"💾 Model saved to {path} (compatible with kumparanian verify)")

    def load_pickle(self, path="model.pickle"):
        """Reload model for inference."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.ft_model = data["fasttext_model"]
        self.clf = data["classifier"]
        print("✅ Model loaded successfully from pickle.")


if __name__ == "__main__":
    model = Model()
    model.train("data.csv")

    # Simpan model ke model.pickle (HARUS dengan pickle.dump)
    with open("model.pickle", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved as model.pickle")