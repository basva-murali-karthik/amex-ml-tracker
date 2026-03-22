import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import time
import itertools

# ── LOAD DATA ──────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"\nFraud vs Normal:")
print(df['Class'].value_counts())

# ── PREPARE DATA ───────────────────────────────────────────────
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = X_train.copy()
X_test  = X_test.copy()
X_train[['Amount','Time']] = scaler.fit_transform(X_train[['Amount','Time']])
X_test[['Amount','Time']]  = scaler.transform(X_test[['Amount','Time']])

print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing samples  : {len(X_test):,}")
print("\n Data ready!\n")

# ── EXPERIMENT GRID ────────────────────────────────────────────
param_grid = {
    "n_estimators" : [50, 100, 200],
    "max_depth"    : [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3]
}

all_combinations = list(itertools.product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["learning_rate"]
))

print(f"Total experiments to run: {len(all_combinations)}\n")

# ── IMPORT TRACKER ─────────────────────────────────────────────
from tracker import log_experiment

# ── RUN EXPERIMENTS ────────────────────────────────────────────
results = []

for i, (n_est, depth, lr) in enumerate(all_combinations):

    exp_id = f"EXP_{str(i+1).zfill(3)}"
    print(f"[{exp_id}] trees={n_est}, depth={depth}, lr={lr} ", end="")

    try:
        start = time.time()

        model = xgb.XGBClassifier(
            n_estimators  = n_est,
            max_depth     = depth,
            learning_rate = lr,
            eval_metric   = 'logloss',
            random_state  = 42
        )
        model.fit(X_train, y_train)

        elapsed  = round(time.time() - start, 2)
        y_pred   = model.predict(X_test)
        y_prob   = model.predict_proba(X_test)[:, 1]
        accuracy = round(accuracy_score(y_test, y_pred), 4)
        auc      = round(roc_auc_score(y_test, y_prob), 4)

        print(f"→  accuracy={accuracy} | auc={auc} | time={elapsed}s")

        log_experiment(
            exp_id        = exp_id,
            n_estimators  = n_est,
            max_depth     = depth,
            learning_rate = lr,
            accuracy      = accuracy,
            auc_score     = auc,
            time_seconds  = elapsed,
            status        = "success"
        )

        results.append({
            "exp_id"        : exp_id,
            "n_estimators"  : n_est,
            "max_depth"     : depth,
            "learning_rate" : lr,
            "accuracy"      : accuracy,
            "auc_score"     : auc,
            "time_seconds"  : elapsed,
            "status"        : "success",
            "failure_reason": None
        })

    except Exception as e:
        print(f"→  FAILED | reason={str(e)}")

        log_experiment(
            exp_id        = exp_id,
            n_estimators  = n_est,
            max_depth     = depth,
            learning_rate = lr,
            accuracy      = None,
            auc_score     = None,
            time_seconds  = None,
            status        = "failed",
            failure_reason= str(e)
        )

        results.append({
            "exp_id"        : exp_id,
            "n_estimators"  : n_est,
            "max_depth"     : depth,
            "learning_rate" : lr,
            "accuracy"      : None,
            "auc_score"     : None,
            "time_seconds"  : None,
            "status"        : "failed",
            "failure_reason": str(e)
        })

# ── SUMMARY ────────────────────────────────────────────────────
results_df = pd.DataFrame(results)

print("\n\n====== EXPERIMENT SUMMARY ======")
print(results_df[[
    "exp_id","n_estimators","max_depth",
    "learning_rate","status","accuracy",
    "auc_score","time_seconds"
]])

print(f"\nTotal      : {len(results_df)}")
print(f"Successful : {(results_df['status']=='success').sum()}")
print(f"Failed     : {(results_df['status']=='failed').sum()}")
print(f"\nBest AUC   : {results_df['auc_score'].max()}")

best = results_df.loc[results_df['auc_score'].idxmax()]
print(f"\nBest Experiment : {best['exp_id']}")
print(f"  Trees         : {best['n_estimators']}")
print(f"  Depth         : {best['max_depth']}")
print(f"  Learning Rate : {best['learning_rate']}")
print(f"  AUC Score     : {best['auc_score']}")

from database import get_summary
get_summary()

print("\n Phase 1 Complete!")