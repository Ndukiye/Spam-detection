Spam Detection System — Comprehensive Documentation

Overview

- Purpose: End-to-end binary text classification for spam vs ham across SMS and email datasets, emphasizing recall-first detection while preserving minimum precision.
- Scope: Data acquisition, preprocessing, model training with stratified k-fold cross-validation, threshold tuning, prediction CLI, and reporting artifacts (CSV metrics, confusion matrices, ROC/PR curves).
- Primary modules:
  - `src/data.py`: dataset fetching and loaders, email parsing helpers.
  - `src/text.py`: text normalization pipeline.
  - `src/train.py`: training, evaluation, model persistence, reporting.
  - `src/predict.py`: prediction CLI, confidence computation, model-name inference.

Architecture

- Data Layer (`src/data.py`):
  - SMS dataset download and load (`get_sms_spam_df` in c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\data.py:160).
  - SpamAssassin corpus download, extraction, and loader (`download_spam_assassin_corpus` in c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\data.py:101; `get_spam_assassin_df` in c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\data.py:144).
  - Email parsing and cleaning via `EmailIterator` and `SimpleEmail` (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\data.py:43,64).
- Text Preprocessing (`src/text.py`):
  - `normalize` performs URL/email/HTML removal, letter-only filtering, lowercasing, tokenization, Porter stemming, and stop-word removal using stemmed English stop words (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\text.py:8).
- Training (`src/train.py`):
  - Builds TF‑IDF features and trains multiple classical ML models with stratified k-fold CV (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:127,129,131-144).
  - Recall-first threshold tuning with minimum precision constraint via `_best_threshold` (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:26).
  - Saves pipelines and reporting artifacts per model/fold (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:97,100-105).
- Prediction (`src/predict.py`):
  - CLI supports single text, subject+body, and CSV batch modes (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:179-187, 231-235, 259-265).
  - Confidence computed from metrics/confusion, model-name inferred from pipeline or filename (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:104-140, 43-65, 67-102).

Directory Layout

- `src/`: source code.
- `data/`: downloaded datasets and extracted email corpora.
- `models/`: saved model pipelines (`.joblib`).
- `reports/`: CSV metrics, confusion matrices, ROC/PR plots.
- `requirements.txt`: Python dependencies.

Datasets

- UCI SMS Spam Collection:
  - Auto-downloaded zip (`smsspamcollection.zip`) and loaded from `SMSSpamCollection` (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\data.py:18-30).
  - Fallback small sample provided if download fails (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\data.py:32-41).
- SpamAssassin Public Corpus:
  - Downloads multiple `*.tar.bz2` archives and extracts ham/spam into `data/spam_assassin/{ham,spam}` (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\data.py:101-143).
  - Builds a DataFrame by iterating ham/spam directories and cleaning text via `SimpleEmail.clean` (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\data.py:144-157).

Text Preprocessing

- Steps in `normalize` (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\text.py:8):
  - Replace URLs and email addresses with spaces (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\text.py:11-12).
  - Strip HTML tags (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\text.py:13).
  - Keep letters only (`[^A-Za-z]+`), lowercase, split into tokens (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\text.py:14-16).
  - Porter stem each token and remove stemmed stop words built from `ENGLISH_STOP_WORDS` (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\text.py:2,5-7,17-18).
- Rationale:
  - Normalizing reduces noise and consistent token space improves model generalization.
  - Stemming clusters morphological variants; custom stop-word removal avoids scikit-learn stop-word inconsistencies.

Features & Vectorization

- TF‑IDF via `TfidfVectorizer` configured with our preprocessor and no internal stop words (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:129):
  - `preprocessor=normalize` ensures identical preprocessing across train/test.
  - `stop_words=None` to rely on our stemmed stop-word list.
  - `strip_accents="unicode"` for normalized accents.
  - `ngram_range=(1, 2)` for unigrams and bigrams.
  - `min_df=2` to prune rare terms.

Models

- Implemented classifiers (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:131-144):
  - Logistic Regression (`class_weight="balanced"`), Naive Bayes (Multinomial/Bernoulli/Complement), SVM (`SVC` with `probability=True`, `class_weight="balanced"`), Random Forest (`class_weight="balanced"`), Linear SVM, SGD Classifier (`class_weight="balanced"`), Passive Aggressive (`class_weight="balanced"`), Decision Tree (with dense conversion), Gaussian NB (dense), AdaBoost (dense), Gradient Boosting (dense).
- Pipelines use named steps, with classifier under `clf` in most cases.

Training Procedure

- Stratified k-fold CV (`StratifiedKFold`) to preserve label proportions across folds (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:127).
- For each model and fold:
  - Fit pipeline and predict on held-out fold (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:45-52).
  - Compute metrics: Accuracy, Precision, Recall, F1, and AUC when scores available (`predict_proba` or `decision_function`) (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:53-63).
  - Save ROC and PR curves as PNGs (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:65-81).
  - Persist pipeline to `models/sms_spam_pipeline_<model>_fold_<n>.joblib` (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:97).
  - Write CSV metrics per fold and confusion matrices (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:100-105).
- Aggregate average metrics across folds to `reports/metrics_avg.csv` (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:163-175).

Recall-First Threshold Tuning

- Goal: Maximize Recall subject to minimum Precision constraint.
- Implementation: `_best_threshold` sweeps thresholds via quantiles of the score distribution and selects the threshold with highest Recall among candidates with Precision ≥ `min_precision` (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:26-43).
- Usage: enable with `--optimize_recall 1` and set `--min_precision` (default `0.8`) (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:114-115).
- Reporting: tuned metrics saved with `variant=tuned` and `threshold` value; base metrics saved with `variant=base` (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:84-93,100-103).

Reporting & Visualization

- Per-fold CSV metrics: `reports/metrics_<model>_fold_<n>.csv` contain accuracy, precision, recall, f1, auc, variant, and threshold.
- Confusion matrices: `reports/confusion_<model>_fold_<n>.csv` with columns `pred_0`, `pred_1` and an `actual` row indicator (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:103-105).
- Plots: ROC and PR curve images per model/fold (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:65-81).
- Averages: `reports/metrics_avg.csv` summarizing cross-fold performance (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:163-175).

Prediction CLI

- Entry point: `python -m src.predict`.
- Inputs:
  - `--text`: single text message (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:180).
  - `--subject` + `--body`: email-like input combined (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:231-235).
  - `--file`: CSV file with a `text` column for batch prediction (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:259-265).
- Model selection and threshold:
  - `--model`: path to a saved pipeline. If omitted, loads canonical `models/sms_spam_pipeline.joblib` (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:13-18).
  - `--use_tuned 1`: prefer tuned metrics when computing confidence (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:104-118, 210-217, 239-245, 269-277).
  - `--threshold`: override decision threshold for classification (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:186, 191-196, 144-156).
  - `--reports_dir`: directory to read metrics and confusion CSVs (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:187, 210-217).
- Output format:
  - `spam|not spam | model=<name> | confidence=<pct> | score=<float>` depending on availability of confidence and score (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:218-229, 246-257).
- Confidence computation:
  - If predicted `spam` (1): uses Precision (%) from metrics CSV (tuned or base) (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:210-215).
  - If predicted `not spam` (0): uses Negative Predictive Value (NPV, %) computed from confusion matrix (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:215-217, 122-131, 133-142).
  - Requires the model path to encode `<model>_fold_<n>` to locate CSVs; otherwise only label/score are shown.
- Model-name inference:
  - Parsed from file name when available (`_parse_model_info` in c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:43-65).
  - Inferred from classifier class when using canonical pipeline (`_infer_display_name` in c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:67-102).

Warnings & Consistency

- Suppresses scikit-learn stop-word inconsistency `UserWarning` in both training and prediction (c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:24; c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:11).
- Uses custom stemmed stop-word removal to avoid mismatches between preprocessing and vectorizer.

Installation

- Dependencies (`requirements.txt`): `scikit-learn`, `pandas`, `numpy`, `joblib`, `nltk`, `matplotlib`, `seaborn`.
- Install: `python -m pip install -r requirements.txt`.
- Recommended execution via module entries so `src` package is importable: `python -m src.train`, `python -m src.predict`.

Usage Examples

- Train on SpamAssassin with recall tuning:
  - `python -m src.train --dataset spamassassin --n_splits 5 --optimize_recall 1 --min_precision 0.8 --reports_dir reports`
- Train a subset of models (e.g., Logistic Regression and Naive Bayes):
  - `python -m src.train --models "logistic regression, naive bayes" --dataset sms --optimize_recall 1`
- Predict with a per-fold model and tuned threshold:
  - `python -m src.predict --model models/sms_spam_pipeline_logistic_regression_fold_0.joblib --use_tuned 1 --text "free prize win cash"`
- Predict using canonical pipeline:
  - `python -m src.predict --text "Win a free prize now"`

Design Rationale

- Recall-first: prioritize catching spam with constraint on precision to reduce false alarms.
- Class balancing: `class_weight="balanced"` on applicable classifiers to mitigate label imbalance.
- Preprocessing: strong normalization and stemming to compress vocabulary and reduce noise across disparate sources (SMS vs email).
- Vectorization: TF‑IDF with bigrams captures short phrases highly predictive of spam.
- Reporting: CSV artifacts and basic plots enable quick experiment tracking without external tooling.

Extending the Project

- Add more classifiers (e.g., ExtraTrees, XGBoost/LightGBM/CatBoost) and compare.
- Introduce lemmatization (spaCy) and evaluate vs Porter stemming.
- Add confusion heatmaps and an HTML summary for reports.
- Integrate experiment tracking (MLflow) for parameter sweeps and artifacts.

Glossary

- TF‑IDF: Term Frequency–Inverse Document Frequency weighting for text features.
- Stratified K‑Fold CV: cross-validation that preserves label distribution across folds.
- Precision: fraction of predicted spam that is truly spam.
- Recall: fraction of actual spam that is correctly detected.
- F1: harmonic mean of precision and recall.
- AUC: area under ROC curve; model’s ability to rank positives higher than negatives.
- ROC Curve: True Positive Rate vs False Positive Rate across thresholds.
- PR Curve: Precision vs Recall across thresholds.
- NPV: Negative Predictive Value; fraction of predicted not spam that is truly not spam.
- Threshold Tuning: choosing a decision threshold on model scores to optimize a target metric under constraints.
- Class Weighting: adjusting loss contributions to counter class imbalance.

Reference Index

- `normalize` text pipeline: c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\text.py:8.
- SMS loader: c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\data.py:160.
- SpamAssassin downloader: c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\data.py:101.
- SpamAssassin DataFrame builder: c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\data.py:144.
- TF‑IDF configuration: c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:129.
- Threshold tuning function: c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:26.
- Training loop and reporting: c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:154-175.
- Pipeline saving: c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\train.py:97.
- Prediction CLI arguments: c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:179-187.
- Confidence loading and computation: c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:104-142.
- Model-name parsing and inference: c:\Users\oruka\Documents\Workspace\more projects\Spam detection\src\predict.py:43-65, 67-102.

