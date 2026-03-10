"""
BotTrainer NLU Engine  –  preprocessing, training, evaluation, entity extraction
"""

import re, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder

# ── Stopwords ────────────────────────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and","or","but",
    "with","my","me","i","you","we","they","this","that","are","was","be","do",
    "did","will","can","please","some","any","all","there","from","up","out","so",
}

# ── Algorithms ───────────────────────────────────────────────────────────────
ALGORITHMS = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
    "Linear SVM":           CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=1000)),
    "Naive Bayes":          MultinomialNB(alpha=1.0),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# ── Text preprocessing ───────────────────────────────────────────────────────
def preprocess(text: str, rm_stopwords: bool = False) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if rm_stopwords:
        text = " ".join(t for t in text.split() if t not in STOPWORDS)
    return text

# ── Entity extraction ────────────────────────────────────────────────────────
_ENTITY_PATTERNS = [
    ("DATE",     r"\b(today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next\s+\w+|this\s+\w+|\d{1,2}[\/\-]\d{1,2}(?:[\/\-]\d{2,4})?)\b"),
    ("TIME",     r"\b(\d{1,2}:\d{2}(?:\s?[ap]m)?|\d{1,2}\s?(?:am|pm)|midnight|noon)\b"),
    ("DURATION", r"\b(\d+\s+(?:second|minute|hour|day|week|month|year)s?)\b"),
    ("NUMBER",   r"\b(\d+(?:\.\d+)?)\b"),
    ("PHONE",    r"\b(\+?1?\s?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})\b"),
    ("EMAIL",    r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b"),
    ("CURRENCY", r"\$\d+(?:\.\d{2})?|\b\d+\s?(?:dollars?|euros?|pounds?)\b"),
    ("LOCATION", r"\b(new york|london|paris|tokyo|berlin|sydney|chicago|los angeles|san francisco|beijing|mumbai|dubai|miami|toronto|delhi|bangkok|singapore)\b"),
]

def extract_entities(text: str) -> list:
    out = []
    for etype, pat in _ENTITY_PATTERNS:
        for m in re.finditer(pat, text.lower()):
            out.append({"type": etype, "value": m.group(0), "start": m.start(), "end": m.end()})
    return sorted(out, key=lambda e: e["start"])

# ── Dataset helpers ──────────────────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        df = pd.DataFrame(data)
    elif p.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")
    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'intent' columns.")
    return df[["text","intent"]].dropna().reset_index(drop=True)

def dataset_stats(df: pd.DataFrame) -> dict:
    ic = df["intent"].value_counts()
    return {
        "total":       len(df),
        "num_intents": df["intent"].nunique(),
        "intents":     df["intent"].unique().tolist(),
        "counts":      ic.to_dict(),
        "avg_per_intent":  ic.mean(),
        "min_per_intent":  ic.min(),
        "max_per_intent":  ic.max(),
        "avg_words":   df["text"].str.split().str.len().mean(),
        "imbalance":   round(ic.max() / max(ic.min(), 1), 2),
    }

# ── Training ─────────────────────────────────────────────────────────────────
def build_pipeline(algo: str, ngram=(1,2), max_feat=5000, rm_sw=False) -> Pipeline:
    from sklearn.preprocessing import FunctionTransformer
    pre = FunctionTransformer(lambda texts: [preprocess(t, rm_sw) for t in texts])
    tfidf = TfidfVectorizer(ngram_range=ngram, max_features=max_feat, sublinear_tf=True)
    clf = ALGORITHMS[algo]
    return Pipeline([("pre", pre), ("tfidf", tfidf), ("clf", clf)])

def train_model(df: pd.DataFrame, algo="Logistic Regression", test_size=0.2,
                ngram=(1,2), max_feat=5000, rm_sw=False, seed=42) -> dict:
    X, y = df["text"].tolist(), df["intent"].tolist()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=test_size, random_state=seed, stratify=y_enc
    )
    pipe = build_pipeline(algo, ngram, max_feat, rm_sw)
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    train_time = time.time() - t0

    y_pred = pipe.predict(X_te)
    y_prob  = pipe.predict_proba(X_te) if hasattr(pipe["clf"], "predict_proba") else None

    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_te, y_pred)

    # CV
    cv_res = cross_validate(pipe, X, y_enc, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                             scoring=["accuracy","f1_macro"], return_train_score=False)

    pred_df = pd.DataFrame({
        "text":      X_te,
        "actual":    le.inverse_transform(y_te),
        "predicted": le.inverse_transform(y_pred),
        "correct":   y_te == y_pred,
        "confidence":[float(np.max(y_prob[i])) if y_prob is not None else None
                      for i in range(len(y_pred))],
    })

    return {
        "pipeline":    pipe,
        "label_encoder": le,
        "classes":     le.classes_.tolist(),
        "algo":        algo,
        "accuracy":    acc,
        "report":      report,
        "cm":          cm,
        "cv_acc":      cv_res["test_accuracy"],
        "cv_f1":       cv_res["test_f1_macro"],
        "pred_df":     pred_df,
        "train_n":     len(X_tr),
        "test_n":      len(X_te),
        "train_time":  train_time,
    }

def predict(pipe, le, text: str) -> dict:
    clean = preprocess(text)
    idx = pipe.predict([text])[0]
    intent = le.inverse_transform([idx])[0]
    proba = pipe.predict_proba([text])[0] if hasattr(pipe["clf"], "predict_proba") else None
    all_intents = []
    if proba is not None:
        all_intents = sorted(
            [{"intent": le.classes_[i], "conf": float(proba[i])} for i in range(len(proba))],
            key=lambda x: -x["conf"]
        )
    return {
        "text": text, "clean": clean,
        "intent": intent,
        "conf": float(np.max(proba)) if proba is not None else 1.0,
        "all_intents": all_intents,
        "entities": extract_entities(text),
    }

def compare_all(df: pd.DataFrame, test_size=0.2) -> pd.DataFrame:
    X, y = df["text"].tolist(), df["intent"].tolist()
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=test_size, random_state=42, stratify=y_enc)
    rows = []
    for name in ALGORITHMS:
        try:
            pipe = build_pipeline(name)
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="weighted", zero_division=0)
            cv = cross_validate(pipe, X, y_enc, cv=3, scoring="accuracy")
            rows.append({"Algorithm": name, "Accuracy": acc, "Precision": p,
                         "Recall": r, "F1": f1, "CV Mean": cv["test_score"].mean(),
                         "CV Std": cv["test_score"].std()})
        except Exception as e:
            rows.append({"Algorithm": name, "Error": str(e)})
    return pd.DataFrame(rows)
