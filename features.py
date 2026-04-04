from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score # stratified k-fold cross-validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # lda
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier # random forest classifier

try:
    from mne.decoding import CSP  # type: ignore
except ModuleNotFoundError:
    # Fall back to a minimal local CSP implementation so the repo runs without mne installed.
    from csp import CSP  # common spatial patterns for feature extraction

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "analysis_results"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

N_SPLITS = 5
RANDOM_STATE = 42

def subject_key(path):
    """Extract subject key from filename"""
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return int(nums[0])
    return path.stem

def find_xy_files(data_dir):
    """Find all X and y files in the data directory"""
    x_files = sorted([p for p in data_dir.glob("X_*.npy")], key=subject_key)
    y_files = sorted([p for p in data_dir.glob("y_*.npy")], key=subject_key)

    if len(x_files) != len(y_files):
        raise ValueError(f"Mismatch: {len(x_files)} X files but {len(y_files)} y files")
    pairs = list(zip(x_files, y_files)) # create pairs of (X, y) files for each subject
    return pairs

def load_subject(x_path, y_path):
    """Load X and y for a single subject"""
    X = np.load(x_path)
    y = np.load(y_path)

    le = LabelEncoder() # encode class labels as integers
    y = le.fit_transform(y)

    return X, y

def build_models():
    models = { #clf = classifier, csp = common spatial patterns for feature extraction
        "LDA": Pipeline([
            ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
            ('clf', LinearDiscriminantAnalysis())
        ]),
        "SVM": Pipeline([
            ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale"))
        ]),
        "RF": Pipeline([
            ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
    }
    return models

def evaluate_subject(X, y, models):
    """Evaluate all models for a single subject using stratified k-fold cross-validation"""
    results = {}
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=1)
        results[f"{name}_mean_acc"] = scores.mean()
        results[f"{name}_std_acc"] = scores.std()
    return results

def main():
    pairs = find_xy_files(DATA_DIR)
    models = build_models()

    all_results = []
    for x_path, y_path in pairs:
        subject_id = subject_key(x_path)
        print(f"Evaluating Subject {subject_id}...")

        X, y = load_subject(x_path, y_path)
        class_counts = dict(zip(*np.unique(y, return_counts=True)))

        print(f"X shape: {X.shape} | y shape: {y.shape} | class distribution: {class_counts}")

        try:
            results = evaluate_subject(X, y, models)
        except Exception as e:
            print(f"Error evaluating Subject {subject_id}: {e}")
            continue

        results["subject"] = subject_id
        results["n_trials"] = X.shape[0]
        results["n_channels"] = X.shape[1]
        results["n_timepoints"] = X.shape[2]
        results["n_classes"] = len(np.unique(y))
        all_results.append(results)

    results_df = pd.DataFrame(all_results)
    if results_df.empty: # fallback
        print("No results to save.")
        return
    
    mean_cols = [c for c in results_df.columns if c.endswith("_mean_acc")]
    results_df["best_model"] = results_df[mean_cols].idxmax(axis=1).str.replace("_mean_acc", "", regex=False)
    results_df["best_acc"] = results_df[mean_cols].max(axis=1)

    results_df = results_df.sort_values("subject")
    results_df.to_csv(OUTPUT_DIR / "classification_results.csv", index=False)

    summary = pd.DataFrame([{
        "LDA_mean": results_df["LDA_mean_acc"].mean(),
        "SVM_mean": results_df["SVM_mean_acc"].mean(),
        "RF_mean": results_df["RF_mean_acc"].mean(),
        "LDA_subject_std": results_df["LDA_mean_acc"].std(),
        "SVM_subject_std": results_df["SVM_mean_acc"].std(),
        "RF_subject_std": results_df["RF_mean_acc"].std(),
    }])
    summary.to_csv(OUTPUT_DIR / "classification_summary.csv", index=False)
    print(f"Results saved to {OUTPUT_DIR / 'classification_results.csv'}")
    print(f"Summary saved to {OUTPUT_DIR / 'classification_summary.csv'}")

if __name__ == "__main__":
    main()
