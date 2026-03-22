from database import save_experiment, create_database

def log_experiment(exp_id, n_estimators, max_depth,
                   learning_rate, accuracy, auc_score,
                   time_seconds, status, failure_reason=None):
    
    # make sure database exists before saving
    create_database()

    # save to database
    save_experiment(
        exp_id         = exp_id,
        n_estimators   = n_estimators,
        max_depth      = max_depth,
        learning_rate  = learning_rate,
        accuracy       = accuracy,
        auc_score      = auc_score,
        time_seconds   = time_seconds,
        status         = status,
        failure_reason = failure_reason
    )

    # print confirmation
    if status == "success":
        print(f"   Saved [{exp_id}] → "
              f"auc={auc_score} | time={time_seconds}s")
    else:
        print(f"   Saved [{exp_id}] → "
              f"FAILED | reason={failure_reason}")


if __name__ == "__main__":
    print("Testing tracker...\n")

    log_experiment(
        exp_id        = "TEST_001",
        n_estimators  = 100,
        max_depth     = 3,
        learning_rate = 0.1,
        accuracy      = 0.9995,
        auc_score     = 0.9443,
        time_seconds  = 1.11,
        status        = "success"
    )

    print("\n Tracker working!")