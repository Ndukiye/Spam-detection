Spam Detection Project

Overview

- Binary text classification for spam vs ham across two datasets: UCI SMS Spam Collection and SpamAssassin public email corpus.
- End-to-end pipeline: data acquisition, preprocessing, model training with cross-validation, recall-first threshold tuning, prediction CLI, and reporting (metrics, confusion matrices, ROC/PR curves).

Project Structure

- `src/data.py`: Dataset download/load, email parsing, and text cleaning helpers.
- `src/text.py`: NLP preprocessing (`normalize`) with URL/email/html removal, letter-only filtering, lowercasing, tokenization, Porter stemming, and stop-word removal (stemmed English stop words).
- `src/train.py`: Training CLI with TF‑IDF features, multiple models, k-fold CV, threshold tuning, and report artifacts.
- `src/predict.py`: Prediction CLI supporting single text, subject+body, CSV batch, tuned thresholds, and model selection.
- `models/`: Saved pipelines (`.joblib`) per model/fold and canonical pipeline.
- `reports/`: CSV metrics per fold, aggregate `metrics_avg.csv`, confusion matrices, and ROC/PR plots.

Datasets

- SMS: UCI SMS Spam Collection (`SMSSpamCollection`) auto-downloaded to `data/smsspamcollection.zip`.
- SpamAssassin: Emails downloaded/extracted into `data/spam_assassin/{ham,spam}`.

Installation

- `python -m pip install -r requirements.txt`
- Notes:
  - NLTK is used for Porter stemming (no corpora downloads required).
  - Matplotlib/Seaborn used for plots.

Preprocessing

- `normalize(text)`: removes URLs/emails/HTML, non-letters, lowercases, tokenizes, applies Porter stemming, removes stemmed English stop words, and rejoins tokens.
- TF‑IDF: `stop_words=None`, `strip_accents='unicode'`, `ngram_range=(1,2)`, `min_df=2` (stop-word removal handled in preprocessing).
- Stop-word warnings: suppressed and avoided by preprocessing; CLI output is clean.

Training Usage

- Basic (SpamAssassin):
  - `python -m src.train --n_splits 5 --dataset spamassassin --reports_dir reports`
- SMS dataset:
  - `python -m src.train --n_splits 5 --dataset sms --reports_dir reports`
- Quick run (subset of models and folds):
  - `python -m src.train --n_splits 2 --max_folds 1 --models "logistic regression,support vector machine" --dataset spamassassin --optimize_recall 1 --min_precision 0.8 --reports_dir reports`

Model Catalog

- Implemented: Logistic Regression, MultinomialNB, BernoulliNB, ComplementNB, GaussianNB, SVC (probabilities), LinearSVC, SGDClassifier, PassiveAggressiveClassifier, DecisionTree, RandomForest, AdaBoost, GradientBoosting.
- Saved to `models/sms_spam_pipeline_<model>_fold_<n>.joblib` and canonical `models/sms_spam_pipeline.joblib`.

Threshold Tuning (Recall-First)

- Enabled via `--optimize_recall 1` and `--min_precision <value>`.
- Per-fold tuned threshold saved in `reports/metrics_<model>_fold_<n>.csv` (row with `variant=tuned` and `threshold`).
- Aggregate metrics saved to `reports/metrics_avg.csv`.

Reports and Visualization

- Per-fold CSVs:
  - `metrics_<model>_fold_<n>.csv` with base and tuned metrics.
  - `confusion_<model>_fold_<n>.csv` confusion matrix.
- Plots:
  - `roc_<model>_fold_<n>.png` (ROC curve).
  - `pr_<model>_fold_<n>.png` (Precision-Recall curve).
- Aggregate: `reports/metrics_avg.csv` across selected folds/models.

Prediction Usage

- Single text (canonical model):
  - `python -m src.predict --text "Win a free prize now"`
  - Output: `spam|not spam | model=<name> | confidence=<pct> | score=<float>`
- Subject and body:
  - `python -m src.predict --subject "Congratulations" --body "You won a lottery"`
- Specify model file:
  - `python -m src.predict --model models/sms_spam_pipeline_logistic_regression_fold_0.joblib --text "free prize win cash"`
- Use tuned threshold automatically:
  - `python -m src.predict --model models/sms_spam_pipeline_logistic_regression_fold_0.joblib --use_tuned 1 --text "..."`
- Override threshold manually:
  - `python -m src.predict --model <path> --threshold 0.4 --text "..."`
- Batch CSV:
  - Input CSV must contain a `text` column.
  - `python -m src.predict --model <path> --file path/to/input.csv > path/to/output.csv`

Testing & Reproducibility

- Quick training smoke test:
  - `python -m src.train --n_splits 2 --max_folds 1 --models lr,svm --dataset spamassassin --optimize_recall 1 --min_precision 0.8 --reports_dir reports`
  - Verify models in `models/` and artifacts in `reports/`.
- Prediction smoke test:
  - `python -m src.predict --model models/sms_spam_pipeline_logistic_regression_fold_0.joblib --use_tuned 1 --text "free prize win cash"`
  - Expected output example: `spam | model=Logistic Regression | confidence=82.64% | score=0.4971`

Notes & Limitations

- Coverage vs specification:
  - Preprocessing: tokenization, stop-words, stemming (done); lemmatization (pending if spaCy is desired).
  - Models: broad classical coverage implemented; k‑NN, Gaussian Process, Extra Trees, and external boosters (XGBoost/LightGBM/CatBoost) are not included yet.
  - Deep learning models are intentionally omitted.
  - Visualization included for ROC/PR; confusion matrices saved as CSV; heatmaps can be added.
  - Experiment tracking currently via CSVs/PNGs; MLflow/notebooks optional.

Extending the Project

- Add lemmatization (spaCy) and compare against stemming.
- Add k‑NN, GaussianProcessClassifier, ExtraTreesClassifier, XGBoost/LightGBM/CatBoost.
- Add confusion matrix heatmaps and an HTML summary page.
- Optional: experiment tracking (MLflow), notebooks for exploratory analysis.

Troubleshooting

- `ModuleNotFoundError: src` when loading models: use module entry points (`python -m src.train` / `python -m src.predict`) so the `src` package is importable.
- Stop-word inconsistency warnings: eliminated. Preprocessing removes stemmed English stop-words and warnings are suppressed.

Prediction Details

- Label: prints `spam` or `not spam`.
- Model: friendly model name parsed from the saved pipeline path.
- Confidence:
  - For `spam`: tuned Precision (%) from `reports/metrics_<model>_fold_<n>.csv`.
  - For `not spam`: Negative Predictive Value (NPV, %) computed from `reports/confusion_<model>_fold_<n>.csv`.
- Score: per-sample posterior probability (`predict_proba`) or decision value (`decision_function`) when available.
