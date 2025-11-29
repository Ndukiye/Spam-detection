from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import warnings
import joblib
from src.data import get_sms_spam_df, get_spam_assassin_df
from src.text import normalize

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")

def to_dense(X):
    try:
        return X.toarray()
    except Exception:
        return X

def _best_threshold(y_true, y_scores, min_precision: float = 0.8):
    if y_scores is None:
        return None, None
    qs = np.linspace(0, 1, 101)
    thresholds = np.quantile(y_scores, qs)
    best = None
    best_metrics = None
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        if p >= min_precision:
            if best is None or r > best_metrics["recall"]:
                a = accuracy_score(y_true, y_pred)
                f = f1_score(y_true, y_pred, zero_division=0)
                best = t
                best_metrics = {"accuracy": a, "precision": p, "recall": r, "f1": f}
    return best, best_metrics

def train_and_evaluate_model(model_name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test, fold: int = 0, reports_dir: Path | None = None, optimize_recall: bool = False, min_precision: float = 0.8):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = 0.0
    y_scores = None
    try:
        if hasattr(pipeline, "predict_proba"):
            y_scores = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_scores)
        elif hasattr(pipeline, "decision_function"):
            y_scores = pipeline.decision_function(X_test)
            auc = roc_auc_score(y_test, y_scores)
    except Exception:
        auc = 0.0
    if reports_dir is not None and y_scores is not None:
        try:
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            prec, rec, _ = precision_recall_curve(y_test, y_scores)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(reports_dir / f"roc_{model_name.lower().replace(' ', '_')}_fold_{fold}.png")
            plt.close()
            plt.figure()
            plt.plot(rec, prec)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.tight_layout()
            plt.savefig(reports_dir / f"pr_{model_name.lower().replace(' ', '_')}_fold_{fold}.png")
            plt.close()
        except Exception:
            pass
    base_metrics = {"fold": fold, "model": model_name, "variant": "base", "accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc, "threshold": None}
    chosen_metrics = base_metrics
    if optimize_recall and y_scores is not None:
        t, m = _best_threshold(y_test, y_scores, min_precision=min_precision)
        if t is not None and m is not None:
            tuned_metrics = {"fold": fold, "model": model_name, "variant": "tuned", "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "auc": auc, "threshold": float(t)}
            chosen_metrics = tuned_metrics
            print(f"{model_name} (Fold {fold} Base): accuracy={acc:.4f} f1={f1:.4f} precision={precision:.4f} recall={recall:.4f} auc={auc:.4f}")
            print(f"{model_name} (Fold {fold} Tuned): accuracy={m['accuracy']:.4f} f1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} auc={auc:.4f} threshold={t:.4f}")
        else:
            print(f"{model_name} (Fold {fold}): accuracy={acc:.4f} f1={f1:.4f} precision={precision:.4f} recall={recall:.4f} auc={auc:.4f}")
    else:
        print(f"{model_name} (Fold {fold}): accuracy={acc:.4f} f1={f1:.4f} precision={precision:.4f} recall={recall:.4f} auc={auc:.4f}")
    out_path = MODELS_DIR / f"sms_spam_pipeline_{model_name.lower().replace(' ', '_')}_fold_{fold}.joblib"
    joblib.dump(pipeline, out_path)
    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        dfm = pd.DataFrame([base_metrics] + ([chosen_metrics] if chosen_metrics is not base_metrics else []))
        dfm.to_csv(reports_dir / f"metrics_{model_name.lower().replace(' ', '_')}_fold_{fold}.csv", index=False)
        cm = confusion_matrix(y_test, y_pred)
        pd.DataFrame(cm, columns=["pred_0", "pred_1"]).assign(actual=[0, 1]).to_csv(reports_dir / f"confusion_{model_name.lower().replace(' ', '_')}_fold_{fold}.csv", index=False)
    print(f"{model_name} model (Fold {fold}) saved to: {out_path}")
    return {"accuracy": chosen_metrics["accuracy"], "f1": chosen_metrics["f1"], "precision": chosen_metrics["precision"], "recall": chosen_metrics["recall"], "auc": chosen_metrics["auc"]}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--max_folds", type=int)
    parser.add_argument("--models", type=str)
    parser.add_argument("--dataset", type=str, default="spamassassin")
    parser.add_argument("--optimize_recall", type=int, default=1)
    parser.add_argument("--min_precision", type=float, default=0.8)
    parser.add_argument("--reports_dir", type=str, default=str(Path(__file__).resolve().parent.parent / "reports"))
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.dataset.lower() == "sms":
        df = get_sms_spam_df()
    else:
        df = get_spam_assassin_df()
    y = df["label"].map({"ham": 0, "spam": 1})
    X = df["text"].astype(str)

    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    vec = TfidfVectorizer(preprocessor=normalize, stop_words=None, strip_accents="unicode", ngram_range=(1, 2), min_df=2)

    models = []
    models.append(("Logistic Regression", Pipeline([("tfidf", vec), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])))
    models.append(("Naive Bayes", Pipeline([("tfidf", vec), ("clf", MultinomialNB())])))
    models.append(("Support Vector Machine", Pipeline([("tfidf", vec), ("clf", SVC(probability=True, class_weight="balanced"))])))
    models.append(("Random Forest", Pipeline([("tfidf", vec), ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))])))
    models.append(("Bernoulli Naive Bayes", Pipeline([("tfidf", vec), ("clf", BernoulliNB())])))
    models.append(("Complement Naive Bayes", Pipeline([("tfidf", vec), ("clf", ComplementNB())])))
    models.append(("Linear SVM", Pipeline([("tfidf", vec), ("clf", LinearSVC(class_weight="balanced"))])))
    models.append(("SGD Classifier", Pipeline([("tfidf", vec), ("clf", SGDClassifier(max_iter=1000, class_weight="balanced"))])))
    models.append(("Passive Aggressive", Pipeline([("tfidf", vec), ("clf", PassiveAggressiveClassifier(max_iter=1000, class_weight="balanced"))])))
    models.append(("Decision Tree", Pipeline([("tfidf", vec), ("to_dense", FunctionTransformer(to_dense, accept_sparse=True)), ("clf", DecisionTreeClassifier())])))
    models.append(("AdaBoost", Pipeline([("tfidf", vec), ("to_dense", FunctionTransformer(to_dense, accept_sparse=True)), ("clf", AdaBoostClassifier())])))
    models.append(("Gradient Boosting", Pipeline([("tfidf", vec), ("to_dense", FunctionTransformer(to_dense, accept_sparse=True)), ("clf", GradientBoostingClassifier())])))

    if args.models:
        sel = set([m.strip().lower() for m in args.models.split(",") if m.strip()])
        models = [(name, pipe) for (name, pipe) in models if name.lower().split()[0] in sel or name.lower() in sel]

    metrics_accumulator = {name: [] for name, _ in models}
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        for name, pipe in models:
            m = train_and_evaluate_model(name, pipe, X_train, X_test, y_train, y_test, fold, reports_dir=reports_dir, optimize_recall=bool(args.optimize_recall), min_precision=args.min_precision)
            metrics_accumulator[name].append(m)
        if args.max_folds is not None and fold + 1 >= args.max_folds:
            break

    rows = []
    for name, entries in metrics_accumulator.items():
        if entries:
            acc = sum(e["accuracy"] for e in entries) / len(entries)
            f1 = sum(e["f1"] for e in entries) / len(entries)
            precision = sum(e["precision"] for e in entries) / len(entries)
            recall = sum(e["recall"] for e in entries) / len(entries)
            auc = sum(e["auc"] for e in entries) / len(entries)
            print(f"{name} (Avg): accuracy={acc:.4f} f1={f1:.4f} precision={precision:.4f} recall={recall:.4f} auc={auc:.4f}")
            rows.append({"model": name, "accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "auc": auc})
    if rows:
        pd.DataFrame(rows).to_csv(reports_dir / "metrics_avg.csv", index=False)

if __name__ == "__main__":
    main()
