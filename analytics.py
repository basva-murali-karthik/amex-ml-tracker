import pandas as pd
import numpy as np
from database import load_all_experiments


def load_data():
    df         = load_all_experiments()
    success_df = df[df['status'] == 'success'].copy()
    failed_df  = df[df['status'] == 'failed'].copy()
    return df, success_df, failed_df


def get_overall_summary(df, success_df, failed_df):
    total        = len(df)
    successful   = len(success_df)
    failed       = len(failed_df)
    success_rate = round((successful / total) * 100, 2)
    best_auc     = round(success_df['auc_score'].max(), 4)
    worst_auc    = round(success_df['auc_score'].min(), 4)
    avg_auc      = round(success_df['auc_score'].mean(), 4)
    avg_time     = round(success_df['time_seconds'].mean(), 2)
    max_time     = round(success_df['time_seconds'].max(), 2)
    min_time     = round(success_df['time_seconds'].min(), 2)
    return {
        "total_experiments" : total,
        "successful"        : successful,
        "failed"            : failed,
        "success_rate"      : success_rate,
        "best_auc"          : best_auc,
        "worst_auc"         : worst_auc,
        "avg_auc"           : avg_auc,
        "avg_time_seconds"  : avg_time,
        "max_time_seconds"  : max_time,
        "min_time_seconds"  : min_time
    }


def get_leaderboard(success_df):
    leaderboard = success_df[[
        'exp_id','n_estimators','max_depth',
        'learning_rate','accuracy','auc_score','time_seconds'
    ]].sort_values('auc_score', ascending=False).reset_index(drop=True)
    leaderboard.insert(0, 'rank', leaderboard.index + 1)
    return leaderboard


def get_hyperparameter_impact(success_df):
    trees_impact = success_df.groupby('n_estimators').agg(
        avg_auc  = ('auc_score','mean'),
        avg_time = ('time_seconds','mean'),
        count    = ('exp_id','count')
    ).round(4).reset_index()

    depth_impact = success_df.groupby('max_depth').agg(
        avg_auc  = ('auc_score','mean'),
        avg_time = ('time_seconds','mean'),
        count    = ('exp_id','count')
    ).round(4).reset_index()

    lr_impact = success_df.groupby('learning_rate').agg(
        avg_auc  = ('auc_score','mean'),
        avg_time = ('time_seconds','mean'),
        count    = ('exp_id','count')
    ).round(4).reset_index()

    return trees_impact, depth_impact, lr_impact


def get_time_vs_performance(success_df):
    success_df = success_df.copy()
    success_df['speed_category'] = pd.cut(
        success_df['time_seconds'],
        bins   = [0, 1.0, 2.0, float('inf')],
        labels = ['Fast (0-1s)','Medium (1-2s)','Slow (2s+)']
    )
    time_analysis = success_df.groupby(
        'speed_category', observed=True
    ).agg(
        avg_auc          = ('auc_score','mean'),
        avg_time         = ('time_seconds','mean'),
        experiment_count = ('exp_id','count')
    ).round(4).reset_index()
    return time_analysis


def get_best_and_worst(success_df, n=3):
    sorted_df = success_df.sort_values(
        'auc_score', ascending=False
    ).reset_index(drop=True)
    top_n    = sorted_df.head(n)
    bottom_n = sorted_df.tail(n).reset_index(drop=True)
    return top_n, bottom_n


def get_lr_comparison(success_df):
    lr_comparison = success_df.groupby('learning_rate').agg(
        avg_auc   = ('auc_score','mean'),
        best_auc  = ('auc_score','max'),
        worst_auc = ('auc_score','min'),
        std_auc   = ('auc_score','std'),
        count     = ('exp_id','count')
    ).round(4).reset_index()
    return lr_comparison


def run_all_analytics():
    print("Loading data from database...")
    df, success_df, failed_df = load_data()
    print(f"Loaded {len(df)} experiments\n")

    print("=" * 50)
    print("INSIGHT 1 - OVERALL SUMMARY")
    print("=" * 50)
    summary = get_overall_summary(df, success_df, failed_df)
    for key, value in summary.items():
        print(f"  {key:25} : {value}")

    print("\n" + "=" * 50)
    print("INSIGHT 2 - LEADERBOARD (Top 5)")
    print("=" * 50)
    leaderboard = get_leaderboard(success_df)
    print(leaderboard.head(5).to_string(index=False))

    print("\n" + "=" * 50)
    print("INSIGHT 3 - HYPERPARAMETER IMPACT")
    print("=" * 50)
    trees_impact, depth_impact, lr_impact = get_hyperparameter_impact(success_df)
    print("\nTrees impact:")
    print(trees_impact.to_string(index=False))
    print("\nDepth impact:")
    print(depth_impact.to_string(index=False))
    print("\nLearning Rate impact:")
    print(lr_impact.to_string(index=False))

    print("\n" + "=" * 50)
    print("INSIGHT 4 - TIME VS PERFORMANCE")
    print("=" * 50)
    time_analysis = get_time_vs_performance(success_df)
    print(time_analysis.to_string(index=False))

    print("\n" + "=" * 50)
    print("INSIGHT 5 - TOP 3 AND BOTTOM 3")
    print("=" * 50)
    top_n, bottom_n = get_best_and_worst(success_df)
    print("\nTop 3:")
    print(top_n[['exp_id','n_estimators','max_depth',
                 'learning_rate','auc_score']].to_string(index=False))
    print("\nBottom 3:")
    print(bottom_n[['exp_id','n_estimators','max_depth',
                    'learning_rate','auc_score']].to_string(index=False))

    print("\n" + "=" * 50)
    print("INSIGHT 6 - LEARNING RATE COMPARISON")
    print("=" * 50)
    lr_comparison = get_lr_comparison(success_df)
    print(lr_comparison.to_string(index=False))

    print("\n✅ All analytics complete!")

    return {
        "summary"      : summary,
        "leaderboard"  : leaderboard,
        "trees_impact" : trees_impact,
        "depth_impact" : depth_impact,
        "lr_impact"    : lr_impact,
        "time_analysis": time_analysis,
        "lr_comparison": lr_comparison
    }


if __name__ == "__main__":
    run_all_analytics()