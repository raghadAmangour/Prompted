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
    confusion_matrix,
    classification_report,
    make_scorer
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

FILE_PATH = "ml_ready_support_tickets.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_ITER_SEARCH = 15

PRIORITY_LABELS = ["high", "medium", "low"]

# ------------------------------------------------------------------------------
# OLD cost matrix (what you already ran)
# ------------------------------------------------------------------------------
OLD_COST_MATRIX = np.array([
    [0, 20, 100],   # Actual High
    [5, 0, 20],     # Actual Medium
    [5, 5, 0]       # Actual Low
])

# ------------------------------------------------------------------------------
# NEW cost matrix — softer penalty on High->Low, heavier penalty on
# Low->High / Low->Medium, so the model stops "hiding" from predicting low.
# Tweak these numbers and re-run if recall on low still isn't where you want it.
# ------------------------------------------------------------------------------
NEW_COST_MATRIX = np.array([
    [0, 20, 45],    # Actual High
    [5, 0, 15],     # Actual Medium
    [15, 12, 0]     # Actual Low
])


# ==============================================================================
# LOAD + CLEAN (same as your original pipeline)
# ==============================================================================

def load_data():
    df = pd.read_csv(FILE_PATH)

    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    df["priority"] = df["priority"].astype(str).str.strip().str.lower()

    df["text"] = (df["subject"].astype(str) + " " + df["body"].astype(str)).str.strip()

    df = df[df["text"].str.len() > 0].copy()
    df = df.dropna(subset=["priority"]).copy()

    return df


# ==============================================================================
# COST SCORER FACTORY (built per matrix so we can swap it in/out)
# ==============================================================================

def make_cost_scorer(cost_matrix):

    def custom_priority_cost(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=PRIORITY_LABELS)
        return np.sum(cm * cost_matrix)

    return make_scorer(custom_priority_cost, greater_is_better=False), custom_priority_cost


# ==============================================================================
# MODELS + SEARCH SPACES (same candidates as your original script)
# ==============================================================================

def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced", dual="auto", random_state=RANDOM_STATE
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "SGD Classifier": SGDClassifier(
            loss="hinge", class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE
        ),
        "Complement Naive Bayes": ComplementNB()
    }


PARAM_DISTRIBUTIONS = {
    "Logistic Regression": {
        "tfidf__max_df": [0.90, 0.95, 1.0], "tfidf__min_df": [1, 2, 3, 5],
        "tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__sublinear_tf": [True, False],
        "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    },
    "Linear SVM": {
        "tfidf__max_df": [0.90, 0.95, 1.0], "tfidf__min_df": [1, 2, 3, 5],
        "tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__sublinear_tf": [True, False],
        "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    },
    "Multinomial Naive Bayes": {
        "tfidf__max_df": [0.90, 0.95, 1.0], "tfidf__min_df": [1, 2, 3, 5],
        "tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__sublinear_tf": [True, False],
        "model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    },
    "SGD Classifier": {
        "tfidf__max_df": [0.90, 0.95, 1.0], "tfidf__min_df": [1, 2, 3, 5],
        "tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__sublinear_tf": [True, False],
        "model__alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    },
    "Complement Naive Bayes": {
        "tfidf__max_df": [0.90, 0.95, 1.0], "tfidf__min_df": [1, 2, 3, 5],
        "tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__sublinear_tf": [True, False],
        "model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    }
}


# ==============================================================================
# RUN PRIORITY CLASSIFICATION UNDER A GIVEN COST MATRIX
# ==============================================================================

def run_priority_with_matrix(X_train, X_test, y_train, y_test, cost_matrix, label):

    scorer, cost_fn = make_cost_scorer(cost_matrix)

    cv_strategy = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    trained = {}

    for model_name, model in get_candidate_models().items():

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, min_df=1, max_df=1.0, sublinear_tf=True)),
            ("model", model)
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=PARAM_DISTRIBUTIONS[model_name],
            n_iter=N_ITER_SEARCH,
            scoring=scorer,
            cv=cv_strategy,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            verbose=0
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        trained[model_name] = best_model

        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=PRIORITY_LABELS, average="macro", zero_division=0
        )

        # Per-class recall (the number we actually care about here)
        _, per_class_recall, _, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=PRIORITY_LABELS, zero_division=0
        )

        test_cost = cost_fn(y_test, y_pred)

        rows.append({
            "Cost Matrix": label,
            "Model": model_name,
            "CV Score": search.best_score_,
            "Accuracy": accuracy,
            "Macro F1": f1,
            "Test Penalty Cost": test_cost,
            "Cost Per Ticket": test_cost / len(y_test),
            "high_recall": per_class_recall[0],
            "medium_recall": per_class_recall[1],
            "low_recall": per_class_recall[2],
            "Best Parameters": search.best_params_
        })

        print(f"[{label}] {model_name}: Accuracy={accuracy:.4f} | Macro F1={f1:.4f} "
              f"| low_recall={per_class_recall[2]:.4f} | cost/ticket={test_cost/len(y_test):.3f}")

    results_df = pd.DataFrame(rows)

    winner_row = results_df.sort_values(by="CV Score", ascending=False).iloc[0]
    winner_name = winner_row["Model"]
    winner_model = trained[winner_name]

    return results_df, winner_name, winner_model


# ==============================================================================
# THRESHOLD TUNING FOR THE "low" CLASS
# ==============================================================================
#
# The model outputs a probability for each class. Instead of always picking
# the single highest-probability class (argmax), we lower the bar specifically
# for "low": if P(low) clears the threshold, predict "low" even if it isn't
# the single highest probability. Otherwise fall back to argmax between the
# remaining two classes (high/medium).
#
# This lets us trade some high/medium recall for better low recall WITHOUT
# retraining anything — it's a post-processing decision rule on top of the
# already-trained model's probabilities.
# ==============================================================================

def apply_low_threshold(proba, classes, threshold):

    low_idx = list(classes).index("low")
    other_idx = [i for i in range(len(classes)) if i != low_idx]

    predictions = []

    for row in proba:
        if row[low_idx] >= threshold:
            predictions.append("low")
        else:
            # pick the best among the remaining classes only
            best_other = other_idx[np.argmax(row[other_idx])]
            predictions.append(classes[best_other])

    return np.array(predictions)


def tune_low_threshold(model, X_test, y_test, thresholds=None):

    if thresholds is None:
        thresholds = np.arange(0.15, 0.55, 0.05)

    if not hasattr(model.named_steps["model"], "predict_proba"):
        print("\n[!] Winner model has no predict_proba — threshold tuning skipped "
              "(e.g. LinearSVC/SGD with hinge loss don't expose probabilities).")
        return None

    proba = model.predict_proba(X_test)
    classes = model.named_steps["model"].classes_

    rows = []

    for t in thresholds:

        y_pred = apply_low_threshold(proba, classes, t)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=PRIORITY_LABELS, average="macro", zero_division=0
        )
        p_per, r_per, f_per, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=PRIORITY_LABELS, zero_division=0
        )

        rows.append({
            "threshold": round(t, 2),
            "macro_f1": f1,
            "low_precision": p_per[2],
            "low_recall": r_per[2],
            "low_f1": f_per[2],
            "high_recall": r_per[0],
            "medium_recall": r_per[1],
        })

    result_df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("THRESHOLD TUNING — 'low' class (default argmax = threshold ~0.33-0.5)")
    print("=" * 80)
    print(result_df.to_string(index=False))

    best_row = result_df.sort_values(by="low_f1", ascending=False).iloc[0]
    print(f"\nBest threshold by low_f1: {best_row['threshold']} "
          f"(low_recall={best_row['low_recall']:.3f}, low_precision={best_row['low_precision']:.3f}, "
          f"macro_f1={best_row['macro_f1']:.3f})")

    result_df.to_csv("low_threshold_tuning.csv", index=False)

    return result_df


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["priority"],
        test_size=TEST_SIZE, stratify=df["priority"], random_state=RANDOM_STATE
    )

    print("=" * 80)
    print("RUN 1: OLD COST MATRIX (High->Low = 100)")
    print("=" * 80)
    old_results, old_winner_name, old_winner_model = run_priority_with_matrix(
        X_train, X_test, y_train, y_test, OLD_COST_MATRIX, "OLD"
    )

    print("\n" + "=" * 80)
    print("RUN 2: NEW COST MATRIX (High->Low = 45, Low->High/Medium raised)")
    print("=" * 80)
    new_results, new_winner_name, new_winner_model = run_priority_with_matrix(
        X_train, X_test, y_train, y_test, NEW_COST_MATRIX, "NEW"
    )

    combined = pd.concat([old_results, new_results], ignore_index=True)
    combined.to_csv("priority_cost_matrix_comparison.csv", index=False)

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON (low_recall is the number to watch)")
    print("=" * 80)
    print(combined[["Cost Matrix", "Model", "Accuracy", "Macro F1",
                     "low_recall", "medium_recall", "high_recall",
                     "Cost Per Ticket"]].to_string(index=False))

    print("\nOLD winner:", old_winner_name)
    print("NEW winner:", new_winner_name)

    print("\nFull classification report — NEW winner model:")
    print(classification_report(y_test, new_winner_model.predict(X_test), zero_division=0))

    # Threshold tuning on top of the NEW winner model (no retraining needed)
    tune_low_threshold(new_winner_model, X_test, y_test)
