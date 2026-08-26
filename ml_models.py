import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    make_scorer
)


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

FILE_PATH = "cleaned_support_tickets.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# Number of random hyperparameter combinations
N_ITER_SEARCH = 15


# ==============================================================================
# 2. LOAD DATASET
# ==============================================================================

print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

df = pd.read_csv(FILE_PATH)

print("Original dataset shape:", df.shape)

required_columns = [
    "subject",
    "body",
    "type",
    "priority"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ==============================================================================
# 3. DATA CLEANING
# ==============================================================================

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")

# Normalize target labels
df["type"] = (
    df["type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["priority"] = (
    df["priority"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Combine subject and body
df["text"] = (
    df["subject"].astype(str)
    + " "
    + df["body"].astype(str)
).str.strip()

# Remove empty text
df = df[df["text"].str.len() > 0].copy()

# Remove missing labels
df = df.dropna(
    subset=["type", "priority"]
).copy()


# ==============================================================================
# 4. DATASET SUMMARY
# ==============================================================================

print("\nUsable tickets:", len(df))

print("\nIssue Type distribution:")
print(df["type"].value_counts())

print("\nPriority distribution:")
print(df["priority"].value_counts())


# ==============================================================================
# 5. PRIORITY KEYWORD ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("PRIORITY KEYWORD ANALYSIS")
print("=" * 80)

keywords = [
    "urgent",
    "immediate",
    "critical",
    "severe",
    "emergency",
    "outage",
    "down",
    "blocked",
    "blocking",
    "unable",
    "failure",
    "failed",
    "security",
    "unauthorized"
]

for keyword in keywords:

    mask = df["text"].str.lower().str.contains(
        keyword,
        regex=False,
        na=False
    )

    if mask.sum() > 0:

        print(f"\nKeyword: {keyword}")
        print("Number of tickets:", mask.sum())

        print(
            df.loc[mask, "priority"]
            .value_counts()
            .to_string()
        )


# ==============================================================================
# 6. COST-SENSITIVE PRIORITY EVALUATION
# ==============================================================================

"""
Priority labels:

    high
    medium
    low

Rows    = Actual label
Columns = Predicted label

Business-oriented penalty matrix:

Actual High:
    High   -> 0
    Medium -> 20
    Low    -> 100  <-- MOST SERIOUS ERROR

Actual Medium:
    High   -> 5
    Medium -> 0
    Low    -> 20

Actual Low:
    High   -> 5
    Medium -> 5
    Low    -> 0

The highest penalty is assigned to:

    HIGH -> LOW

because this means that an urgent ticket was missed.
"""

PRIORITY_LABELS = [
    "high",
    "medium",
    "low"
]

COST_MATRIX = np.array([
    [0, 20, 100],   # Actual High
    [5, 0, 20],     # Actual Medium
    [5, 5, 0]       # Actual Low
])


def custom_priority_cost(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=PRIORITY_LABELS
    )

    return np.sum(
        cm * COST_MATRIX
    )


cost_scorer = make_scorer(
    custom_priority_cost,
    greater_is_better=False
)


# ==============================================================================
# 7. MODELS
# ==============================================================================
#
# MODEL-SELECTION DESIGN
# -----------------------
# - BASELINE MODEL  : DummyClassifier (majority-class rule). It ignores the
#                     text entirely and always predicts the most frequent
#                     class. It is the minimum bar any tuned model must beat
#                     to be considered useful. It is NOT tuned and NOT
#                     eligible to win the final model-selection step.
# - CANDIDATE MODELS: Logistic Regression, Linear SVM, Multinomial Naive
#                     Bayes, SGD Classifier, Complement Naive Bayes. These
#                     are tuned via RandomizedSearchCV and compete for the
#                     "winner" title.
# ==============================================================================

BASELINE_MODEL_NAME = "Baseline (Majority Class)"

baseline_model = DummyClassifier(
    strategy="most_frequent",
    random_state=RANDOM_STATE
)

candidate_models = {

    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        dual="auto",
        random_state=RANDOM_STATE
    ),

    "Multinomial Naive Bayes": MultinomialNB(),

    "SGD Classifier": SGDClassifier(
        loss="hinge",
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE
    ),

    "Complement Naive Bayes": ComplementNB()
}


# ==============================================================================
# 8. HYPERPARAMETER SEARCH SPACES (candidate models only)
# ==============================================================================
#
# FEATURE PIPELINE
# ----------------
# Every candidate model shares the same feature-extraction stage:
#   TfidfVectorizer -> (lowercase, min_df, max_df, ngram_range, sublinear_tf)
# tuned jointly with the classifier's own hyperparameters inside one
# sklearn Pipeline (steps: "tfidf" -> "model"). This ensures the
# vectorizer is refit correctly inside every CV fold (no leakage).
#
# TUNING STRATEGY
# ----------------
# RandomizedSearchCV, N_ITER_SEARCH=15 random combinations per model,
# scored under 5-fold StratifiedKFold cross-validation. Test data is
# held out and never touched during tuning.
# ==============================================================================

param_distributions = {

    "Logistic Regression": {

        "tfidf__max_df": [0.90, 0.95, 1.0],
        "tfidf__min_df": [1, 2, 3, 5],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__sublinear_tf": [True, False],

        "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    },

    "Linear SVM": {

        "tfidf__max_df": [0.90, 0.95, 1.0],
        "tfidf__min_df": [1, 2, 3, 5],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__sublinear_tf": [True, False],

        "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    },

    "Multinomial Naive Bayes": {

        "tfidf__max_df": [0.90, 0.95, 1.0],
        "tfidf__min_df": [1, 2, 3, 5],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__sublinear_tf": [True, False],

        "model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    },

    "SGD Classifier": {

        "tfidf__max_df": [0.90, 0.95, 1.0],
        "tfidf__min_df": [1, 2, 3, 5],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__sublinear_tf": [True, False],

        "model__alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    },

    "Complement Naive Bayes": {

        "tfidf__max_df": [0.90, 0.95, 1.0],
        "tfidf__min_df": [1, 2, 3, 5],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__sublinear_tf": [True, False],

        "model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    }
}


# ==============================================================================
# 9. HELPER: FULL PER-CLASS METRICS (all classes, not just "high")
# ==============================================================================

def per_class_metrics_dict(y_true, y_pred, labels, prefix=""):
    """
    Returns a flat dict with precision/recall/F1 for EVERY class in `labels`,
    e.g. for labels=["high","medium","low"] it returns:
        high_precision, high_recall, high_f1,
        medium_precision, medium_recall, medium_f1,
        low_precision, low_recall, low_f1
    """

    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )

    out = {}

    for i, label in enumerate(labels):

        key_base = f"{prefix}{label}"

        out[f"{key_base}_precision"] = precisions[i]
        out[f"{key_base}_recall"] = recalls[i]
        out[f"{key_base}_f1"] = f1s[i]

    return out


# ==============================================================================
# 10. MAIN CLASSIFICATION FUNCTION
# ==============================================================================

def run_classification(X, y, target_name):

    print("\n")
    print("=" * 80)
    print(f"STARTING CLASSIFICATION: {target_name}")
    print("=" * 80)

    # ==========================================================================
    # TRAIN / TEST SPLIT
    # ==========================================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # ==========================================================================
    # STRATIFIED CROSS VALIDATION
    # ==========================================================================

    cv_strategy = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # ==========================================================================
    # LABELS FOR PER-CLASS METRICS (task-appropriate)
    # ==========================================================================

    class_labels = PRIORITY_LABELS if target_name == "Priority" else sorted(y.unique())

    # ==========================================================================
    # OPTIMIZATION OBJECTIVE
    # ==========================================================================

    if target_name == "Priority":
        scoring_metric = cost_scorer
        print("\nOptimization objective: Minimize Cost-Sensitive Priority Penalty")
    else:
        scoring_metric = "f1_macro"
        print("\nOptimization objective: Maximize Macro-F1")

    results = []
    trained_models = {}

    # ==========================================================================
    # 10a. BASELINE MODEL (fit once, NOT tuned, NOT eligible to win)
    # ==========================================================================

    print("\n" + "-" * 80)
    print(f"Fitting baseline: {BASELINE_MODEL_NAME}")
    print("-" * 80)

    baseline_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),   # kept for interface parity; unused by DummyClassifier
        ("model", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE))
    ])

    baseline_pipeline.fit(X_train, y_train)

    baseline_pred = baseline_pipeline.predict(X_test)

    baseline_accuracy = accuracy_score(y_test, baseline_pred)

    baseline_precision, baseline_recall, baseline_f1, _ = precision_recall_fscore_support(
        y_test, baseline_pred, average="macro", zero_division=0
    )

    baseline_row = {
        "Model": BASELINE_MODEL_NAME,
        "Role": "Baseline",
        "CV Score": np.nan,
        "Accuracy": baseline_accuracy,
        "Macro Precision": baseline_precision,
        "Macro Recall": baseline_recall,
        "Macro F1": baseline_f1,
    }

    if target_name == "Priority":
        baseline_row["Test Penalty Cost"] = custom_priority_cost(y_test, baseline_pred)
        baseline_row["Cost Per Ticket"] = baseline_row["Test Penalty Cost"] / len(y_test)

    baseline_row.update(
        per_class_metrics_dict(y_test, baseline_pred, class_labels)
    )

    results.append(baseline_row)

    print("Baseline Accuracy:", round(baseline_accuracy, 4))
    print("Baseline Macro F1:", round(baseline_f1, 4))
    if target_name == "Priority":
        print("Baseline Penalty Cost:", baseline_row["Test Penalty Cost"])

    # ==========================================================================
    # 10b. CANDIDATE MODELS (tuned via RandomizedSearchCV)
    # ==========================================================================

    for model_name, model in candidate_models.items():

        print("\n" + "-" * 80)
        print(f"Training candidate: {model_name}")
        print("-" * 80)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, min_df=1, max_df=1.0, sublinear_tf=True)),
            ("model", model)
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions[model_name],
            n_iter=N_ITER_SEARCH,
            scoring=scoring_metric,
            cv=cv_strategy,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            verbose=0
        )

        # Test data is NOT used during tuning.
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        trained_models[model_name] = best_model

        best_cv_score = search.best_score_
        cv_penalty_cost = -best_cv_score if target_name == "Priority" else np.nan

        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        row = {
            "Model": model_name,
            "Role": "Candidate",
            "CV Score": best_cv_score,
            "CV Penalty Cost": cv_penalty_cost,
            "Accuracy": accuracy,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1": f1,
            "Best Parameters": search.best_params_
        }

        if target_name == "Priority":
            test_penalty_cost = custom_priority_cost(y_test, y_pred)
            row["Test Penalty Cost"] = test_penalty_cost
            row["Cost Per Ticket"] = test_penalty_cost / len(y_test)

        # Per-class precision/recall/F1 for EVERY class (not just "high")
        row.update(
            per_class_metrics_dict(y_test, y_pred, class_labels)
        )

        results.append(row)

        print("\nBest CV Score:", best_cv_score)
        print("Best Parameters:", search.best_params_)
        print("Test Accuracy:", round(accuracy, 4))
        print("Test Macro F1:", round(f1, 4))

        if target_name == "Priority":
            print("Test Penalty Cost:", row["Test Penalty Cost"])
            print("Cost Per Ticket:", round(row["Cost Per Ticket"], 4))

    # ==========================================================================
    # RESULTS DATAFRAME (baseline + all candidates, side by side)
    # ==========================================================================

    results_df = pd.DataFrame(results)

    # ==========================================================================
    # FINAL MODEL SELECTION
    # ==========================================================================
    #
    # SELECTION CRITERION
    # --------------------
    # The baseline is excluded from winner selection — it exists only as a
    # reference floor for reporting, never as a candidate.
    #
    # Issue Type : highest CV Macro-F1 wins.
    # Priority   : lowest CV cost wins (highest CV Score, since the cost
    #              scorer is negated); Macro-F1 is the tiebreaker.
    # ==========================================================================

    candidates_df = results_df[results_df["Role"] == "Candidate"].copy()

    if target_name == "Priority":
        candidates_df = candidates_df.sort_values(
            by=["CV Score", "Macro F1"], ascending=[False, False]
        ).reset_index(drop=True)
    else:
        candidates_df = candidates_df.sort_values(
            by="CV Score", ascending=False
        ).reset_index(drop=True)

    winner_name = candidates_df.iloc[0]["Model"]
    winner_model = trained_models[winner_name]

    winner_predictions = winner_model.predict(X_test)

    final_accuracy = accuracy_score(y_test, winner_predictions)

    final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(
        y_test, winner_predictions, average="macro", zero_division=0
    )

    final_row = {
        "Target": target_name,
        "Winner Model": winner_name,
        "Baseline Model": BASELINE_MODEL_NAME,
        "Accuracy": final_accuracy,
        "Baseline Accuracy": baseline_accuracy,
        "Macro Precision": final_precision,
        "Macro Recall": final_recall,
        "Macro F1": final_f1,
        "Baseline Macro F1": baseline_f1,
    }

    if target_name == "Priority":
        final_penalty = custom_priority_cost(y_test, winner_predictions)
        final_row["Final Test Penalty Cost"] = final_penalty
        final_row["Cost Per Ticket"] = final_penalty / len(y_test)
        final_row["Baseline Penalty Cost"] = baseline_row["Test Penalty Cost"]
        final_row["Baseline Cost Per Ticket"] = baseline_row["Cost Per Ticket"]

    final_row.update(
        per_class_metrics_dict(y_test, winner_predictions, class_labels, prefix="winner_")
    )

    final_summary = pd.DataFrame([final_row])

    # ==========================================================================
    # PRINTED REPORTS
    # ==========================================================================

    print("\n" + "=" * 80)
    print(f"FINAL MODEL COMPARISON (Baseline vs Candidates) - {target_name}")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"WINNER MODEL: {winner_name}  (baseline: {BASELINE_MODEL_NAME})")
    print("=" * 80)

    print("\n" + "=" * 80)
    print(f"CLASSIFICATION REPORT - {target_name} (winner model)")
    print("=" * 80)
    print(classification_report(y_test, winner_predictions, zero_division=0))

    cm = confusion_matrix(y_test, winner_predictions, labels=class_labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{l}" for l in class_labels],
        columns=[f"Predicted_{l}" for l in class_labels]
    )

    print("\n" + "=" * 80)
    print(f"CONFUSION MATRIX - {target_name} (winner model)")
    print("=" * 80)
    print(cm_df)

    return results_df, final_summary, cm_df, winner_model


# ==============================================================================
# 11. RUN
# ==============================================================================

X_data = df["text"]

(type_results, type_summary, type_cm, type_winner_model) = run_classification(
    X_data, df["type"], "Issue Type"
)

type_results.to_csv("final_tuned_type_results.csv", index=False)
type_summary.to_csv("final_type_summary.csv", index=False)
type_cm.to_csv("final_type_confusion_matrix.csv")

(priority_results, priority_summary, priority_cm, priority_winner_model) = run_classification(
    X_data, df["priority"], "Priority"
)

priority_results.to_csv("final_tuned_priority_results.csv", index=False)
priority_summary.to_csv("final_priority_summary.csv", index=False)
priority_cm.to_csv("final_priority_confusion_matrix.csv")

print("\n" + "=" * 80)
print("FINAL SYSTEM SUMMARY")
print("=" * 80)
print("\nIssue Type Classification:")
print(type_summary.to_string(index=False))
print("\nPriority Classification:")
print(priority_summary.to_string(index=False))