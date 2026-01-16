# ============================================================
# Assignment 2 - Spam Email Detection
# AI & ML for Cybersecurity - Midterm Retake
# ============================================================

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# -------------------------
# Configuration
# -------------------------
CSV_PATH = "n_bukuri25_11221924.csv"

FEATURES = ["words", "links", "capital_words", "spam_word_count"]
LABEL = "is_spam"

RANDOM_STATE = 42
TEST_SIZE = 0.30  # 70/30 split


# -------------------------
# 1) Data loading & processing
# -------------------------
def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = set(FEATURES + [LABEL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    for col in FEATURES + [LABEL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [LABEL]).copy()
    df[LABEL] = df[LABEL].astype(int)
    df = df[df[LABEL].isin([0, 1])].copy()

    return df


# -------------------------
# 2) Model creation & training
# -------------------------
def train_model(df: pd.DataFrame):
    # ✅ KEEP AS DATAFRAME (fixes warning)
    X = df[FEATURES]
    y = df[LABEL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000,
            solver="lbfgs"
        ))
    ])

    model.fit(X_train, y_train)
    return model, (X_train, X_test, y_train, y_test)


def print_model_coefficients(model: Pipeline):
    lr = model.named_steps["lr"]
    print("\n--- Logistic Regression Coefficients ---")
    print("Intercept:", lr.intercept_[0])
    print("Coefficients:")
    for name, coef in zip(FEATURES, lr.coef_[0]):
        print(f"  {name:16s}: {coef:.6f}")


# -------------------------
# 3) Validation
# -------------------------
def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Validation ---")
    print(f"Accuracy: {acc:.6f}")
    print("Confusion Matrix (rows=Actual, cols=Predicted):")
    print(cm)

    return acc, cm


# -------------------------
# 4) Email feature extraction
# -------------------------
SPAM_WORDS = {
    "free", "winner", "urgent", "prize", "click", "offer", "money",
    "cash", "credit", "limited", "now", "exclusive", "claim", "bonus"
}

WORD_RE = re.compile(r"[A-Za-z']+")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

def extract_features_from_text(text: str) -> pd.DataFrame:
    words = WORD_RE.findall(text)
    links = URL_RE.findall(text)

    return pd.DataFrame([{
        "words": len(words),
        "links": len(links),
        "capital_words": sum(1 for w in words if w.isupper() and len(w) > 1),
        "spam_word_count": sum(1 for w in words if w.lower() in SPAM_WORDS)
    }])


def predict_email(model: Pipeline, text: str):
    feats = extract_features_from_text(text)
    pred = int(model.predict(feats)[0])
    prob = float(model.predict_proba(feats)[0][1])
    return pred, prob, feats


# -------------------------
# 5) Visualizations
# -------------------------
def plot_class_distribution(df: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    df[LABEL].value_counts().sort_index().plot(kind="bar")
    plt.title("Class Distribution (0 = Legitimate, 1 = Spam)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.show()


def plot_confusion_matrix(cm: np.ndarray):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Spam"],
        yticklabels=["Legitimate", "Spam"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


# -------------------------
# MAIN
# -------------------------
def main():
    df = load_dataset(CSV_PATH)
    print("Loaded dataset shape:", df.shape)

    model, (X_train, X_test, y_train, y_test) = train_model(df)
    print("Train size:", len(X_train), "Test size:", len(X_test))

    print_model_coefficients(model)
    acc, cm = evaluate_model(model, X_test, y_test)

    plot_class_distribution(df)
    plot_confusion_matrix(cm)

    spam_email = """URGENT WINNER NOTICE!!!

YOU HAVE WON A FREE CASH PRIZE AND EXCLUSIVE BONUS.
CLAIM YOUR MONEY NOW — LIMITED TIME OFFER!!!

CLICK NOW:
https://free-cash-prize.example/claim
http://exclusive-offer.example/winner
www.fast-bonus.example

ACT NOW! CLICK! FREE PRIZE! WINNER! URGENT!
"""

    legit_email = """Subject: Meeting notes from today

Hi team,
Please find the meeting notes from today's sync attached.
Let me know if you have any questions.

Best regards,
Nino
"""

    for label, text in [("SPAM", spam_email), ("LEGITIMATE", legit_email)]:
        pred, prob, feats = predict_email(model, text)
        print(f"\n--- {label} EMAIL ---")
        print(feats.to_string(index=False))
        print(f"Prediction is_spam={pred}, probability_spam={prob:.3f}")

    print("\nSaved images:")
    print("- class_distribution.png")
    print("- confusion_matrix.png")


if __name__ == "__main__":
    main()
