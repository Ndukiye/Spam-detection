import argparse
from pathlib import Path
import sys
import joblib
import pandas as pd
import warnings

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")

def load_pipeline(model_path: str | None = None):
    p = Path(model_path) if model_path else MODELS_DIR / "sms_spam_pipeline_support_vector_machine_fold_0.joblib"
    if not p.exists():
        print("model_not_found")
        sys.exit(1)
    return joblib.load(p)

def _resolve_tuned_threshold(model_path: Path, reports_dir: Path) -> float | None:
    try:
        base = model_path.stem
        # Expect pattern: sms_spam_pipeline_<model>_fold_<n>
        parts = base.split("sms_spam_pipeline_")[-1]
        if "_fold_" not in parts:
            return None
        model_id, fold = parts.rsplit("_fold_", 1)
        metrics_file = reports_dir / f"metrics_{model_id}_fold_{fold}.csv"
        if not metrics_file.exists():
            return None
        df = pd.read_csv(metrics_file)
        tuned = df[df.get("variant", "base") == "tuned"]
        if tuned.empty:
            return None
        t = tuned.iloc[0]["threshold"]
        try:
            return float(t)
        except Exception:
            return None
    except Exception:
        return None

def _parse_model_info(model_path: Path):
    base = model_path.stem
    if "sms_spam_pipeline_" in base and "_fold_" in base:
        part = base.split("sms_spam_pipeline_")[-1]
        model_id, fold = part.rsplit("_fold_", 1)
        names = {
            "logistic_regression": "Logistic Regression",
            "support_vector_machine": "Support Vector Machine",
            "linear_svm": "Linear SVM",
            "sgd_classifier": "SGD Classifier",
            "passive_aggressive": "Passive Aggressive",
            "naive_bayes": "Naive Bayes",
            "bernoulli_naive_bayes": "Bernoulli Naive Bayes",
            "complement_naive_bayes": "Complement Naive Bayes",
            "random_forest": "Random Forest",
            "decision_tree": "Decision Tree",
            "gaussian_naive_bayes": "Gaussian Naive Bayes",
            "adaboost": "AdaBoost",
            "gradient_boosting": "Gradient Boosting",
        }
        display = names.get(model_id, model_id.replace("_", " ").title())
        return model_id, fold, display
    return None, None, base

def _infer_display_name(pipe):
    try:
        est = None
        if hasattr(pipe, "named_steps"):
            if "clf" in pipe.named_steps:
                est = pipe.named_steps["clf"]
            elif "classifier" in pipe.named_steps:
                est = pipe.named_steps["classifier"]
            elif "estimator" in pipe.named_steps:
                est = pipe.named_steps["estimator"]
        if est is None and hasattr(pipe, "steps"):
            try:
                est = pipe.steps[-1][1]
            except Exception:
                est = None
        if est is None:
            est = pipe
        cls = est.__class__.__name__
        names = {
            "LogisticRegression": "Logistic Regression",
            "MultinomialNB": "Naive Bayes",
            "SVC": "Support Vector Machine",
            "RandomForestClassifier": "Random Forest",
            "BernoulliNB": "Bernoulli Naive Bayes",
            "ComplementNB": "Complement Naive Bayes",
            "LinearSVC": "Linear SVM",
            "SGDClassifier": "SGD Classifier",
            "PassiveAggressiveClassifier": "Passive Aggressive",
            "DecisionTreeClassifier": "Decision Tree",
            "GaussianNB": "Gaussian Naive Bayes",
            "AdaBoostClassifier": "AdaBoost",
            "GradientBoostingClassifier": "Gradient Boosting",
        }
        return names.get(cls, cls.replace("_", " ").title())
    except Exception:
        return None

def _load_metrics_row(model_id: str, fold: str, reports_dir: Path, prefer_tuned: bool):
    try:
        mf = reports_dir / f"metrics_{model_id}_fold_{fold}.csv"
        if not mf.exists():
            return None
        df = pd.read_csv(mf)
        if prefer_tuned and "variant" in df.columns:
            dft = df[df["variant"] == "tuned"]
            if not dft.empty:
                return dft.iloc[0].to_dict()
        if "variant" in df.columns:
            dfb = df[df["variant"] == "base"]
            if not dfb.empty:
                return dfb.iloc[0].to_dict()
        return df.iloc[0].to_dict()
    except Exception:
        return None

def _load_confusion(model_id: str, fold: str, reports_dir: Path):
    try:
        cf = reports_dir / f"confusion_{model_id}_fold_{fold}.csv"
        if not cf.exists():
            return None
        df = pd.read_csv(cf)
        cm = df[["pred_0", "pred_1"]].values
        return cm
    except Exception:
        return None

def _npv_from_cm(cm):
    try:
        tn = float(cm[0][0])
        fn = float(cm[1][0])
        d = tn + fn
        if d == 0:
            return None
        return tn / d
    except Exception:
        return None

def _predict_with_threshold(pipe, texts, threshold: float | None):
    if threshold is None:
        return pipe.predict(texts)
    try:
        if hasattr(pipe, "predict_proba"):
            scores = pipe.predict_proba(texts)[:, 1]
        elif hasattr(pipe, "decision_function"):
            scores = pipe.decision_function(texts)
        else:
            return pipe.predict(texts)
        return (scores >= threshold).astype(int)
    except Exception:
        return pipe.predict(texts)

def _predict_score(pipe, texts):
    try:
        if hasattr(pipe, "predict_proba"):
            return pipe.predict_proba(texts)[:, 1]
        if hasattr(pipe, "decision_function"):
            return pipe.decision_function(texts)
    except Exception:
        return None
    return None

def predict_text(pipe, text: str) -> int:
    return int(pipe.predict([text])[0])

def predict_file(pipe, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise RuntimeError("missing_text_column")
    df["prediction"] = pipe.predict(df["text"].astype(str))
    return df

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--body", type=str)
    parser.add_argument("--file", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--use_tuned", type=int, default=0)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--reports_dir", type=str, default=str(DEFAULT_REPORTS_DIR))
    args = parser.parse_args()
    pipe = load_pipeline(args.model)
    display_name = _infer_display_name(pipe)
    tuned_threshold = None
    if args.use_tuned and args.model:
        tuned_threshold = _resolve_tuned_threshold(Path(args.model), Path(args.reports_dir))
    if args.threshold is not None:
        tuned_threshold = args.threshold
    model_id = None
    fold = None
    if args.model:
        mid, fld, parsed_display = _parse_model_info(Path(args.model))
        model_id = mid
        fold = fld
        if parsed_display:
            display_name = parsed_display
    if args.text:
        pred = int(_predict_with_threshold(pipe, [args.text], tuned_threshold)[0])
        score = _predict_score(pipe, [args.text])
        metrics = None
        cm = None
        conf = None
        if model_id and fold:
            metrics = _load_metrics_row(model_id, fold, Path(args.reports_dir), bool(args.use_tuned))
            cm = _load_confusion(model_id, fold, Path(args.reports_dir))
            if pred == 1 and metrics and "precision" in metrics:
                conf = float(metrics["precision"]) * 100.0
            elif pred == 0 and cm is not None:
                npv = _npv_from_cm(cm)
                conf = npv * 100.0 if npv is not None else None
        label = "spam" if pred == 1 else "not spam"
        model_out = display_name or "Pipeline"
        if conf is not None:
            if score is not None:
                print(f"{label} | model={model_out} | confidence={conf:.2f}% | score={float(score[0]):.4f}")
            else:
                print(f"{label} | model={model_out} | confidence={conf:.2f}%")
        else:
            if score is not None:
                print(f"{label} | model={model_out} | score={float(score[0]):.4f}")
            else:
                print(f"{label} | model={model_out}")
        return
    if args.subject and args.body:
        text = f"{args.subject} {args.body}"
        pred = int(_predict_with_threshold(pipe, [text], tuned_threshold)[0])
        score = _predict_score(pipe, [text])
        metrics = None
        cm = None
        conf = None
        if model_id and fold:
            metrics = _load_metrics_row(model_id, fold, Path(args.reports_dir), bool(args.use_tuned))
            cm = _load_confusion(model_id, fold, Path(args.reports_dir))
            if pred == 1 and metrics and "precision" in metrics:
                conf = float(metrics["precision"]) * 100.0
            elif pred == 0 and cm is not None:
                npv = _npv_from_cm(cm)
                conf = npv * 100.0 if npv is not None else None
        label = "spam" if pred == 1 else "not spam"
        model_out = display_name or "Pipeline"
        if conf is not None:
            if score is not None:
                print(f"{label} | model={model_out} | confidence={conf:.2f}% | score={float(score[0]):.4f}")
            else:
                print(f"{label} | model={model_out} | confidence={conf:.2f}%")
        else:
            if score is not None:
                print(f"{label} | model={model_out} | score={float(score[0]):.4f}")
            else:
                print(f"{label} | model={model_out}")
        return
    if args.file:
        df = pd.read_csv(Path(args.file))
        if "text" not in df.columns:
            raise RuntimeError("missing_text_column")
        preds = _predict_with_threshold(pipe, df["text"].astype(str), tuned_threshold)
        scores = _predict_score(pipe, df["text"].astype(str))
        labels = ["spam" if int(x) == 1 else "not spam" for x in preds]
        confs = []
        mrow = None
        cm = None
        if model_id and fold:
            mrow = _load_metrics_row(model_id, fold, Path(args.reports_dir), bool(args.use_tuned))
            cm = _load_confusion(model_id, fold, Path(args.reports_dir))
        for i, p in enumerate(preds):
            if int(p) == 1 and mrow and "precision" in mrow:
                confs.append(float(mrow["precision"]) * 100.0)
            elif int(p) == 0 and cm is not None:
                npv = _npv_from_cm(cm)
                confs.append(npv * 100.0 if npv is not None else None)
            else:
                confs.append(None)
        df["label"] = labels
        if scores is not None:
            df["score"] = [float(s) for s in scores]
        df["confidence_pct"] = confs
        df["model"] = display_name or "Pipeline"
        print(df.to_csv(index=False))
        return
    print("no_input")

if __name__ == "__main__":
    main()
