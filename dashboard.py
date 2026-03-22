import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from analytics import (
    load_data,
    get_overall_summary,
    get_leaderboard,
    get_hyperparameter_impact,
    get_time_vs_performance,
    get_best_and_worst,
    get_lr_comparison
)

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title = "ML Experiment Tracker",
    page_icon  = "🧪",
    layout     = "wide"
)

# ── COLORS ─────────────────────────────────────────────────────
AMEX_BLUE  = "#006FCF"
AMEX_DARK  = "#003087"
FAIL_RED   = "#dc3545"

# ── STYLING ────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0a0a0a; }
    .block-container { padding-top: 1rem; }
    .header-box {
        background: linear-gradient(135deg, #003087, #006FCF);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .header-title {
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
    }
    .header-subtitle {
        color: #a8d0ff;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #006FCF;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.3rem;
    }
    .metric-value {
        color: #006FCF;
        font-size: 2rem;
        font-weight: 800;
    }
    .metric-label {
        color: #aaaaaa;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    .section-title {
        color: #006FCF;
        font-size: 1.3rem;
        font-weight: 700;
        border-left: 4px solid #006FCF;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #cccccc;
        font-size: 0.9rem;
    }
    .best-tag {
        background: #006FCF;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)


# ── LOAD DATA ──────────────────────────────────────────────────
@st.cache_data
def get_all_data():
    df, success_df, failed_df = load_data()
    summary      = get_overall_summary(df, success_df, failed_df)
    leaderboard  = get_leaderboard(success_df)
    trees_impact, depth_impact, lr_impact = get_hyperparameter_impact(success_df)
    time_analysis= get_time_vs_performance(success_df)
    top_n, bottom_n = get_best_and_worst(success_df, n=3)
    lr_comparison= get_lr_comparison(success_df)
    return (df, success_df, summary, leaderboard,
            trees_impact, depth_impact, lr_impact,
            time_analysis, top_n, bottom_n, lr_comparison)

(df, success_df, summary, leaderboard,
 trees_impact, depth_impact, lr_impact,
 time_analysis, top_n, bottom_n, lr_comparison) = get_all_data()


# ── HEADER ─────────────────────────────────────────────────────
st.markdown("""
    <div class="header-box">
        <p class="header-title">🧪 ML Experiment Analytics Platform</p>
        <p class="header-subtitle">
            Inspired by AiDA Develop — American Express Enterprise ML Platform
            &nbsp;|&nbsp; Credit Card Fraud Detection Dataset
            &nbsp;|&nbsp; XGBoost Hyperparameter Search
        </p>
    </div>
""", unsafe_allow_html=True)


# ── SECTION 1: METRIC CARDS ────────────────────────────────────
st.markdown('<p class="section-title">📊 Overview</p>',
            unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary['total_experiments']}</div>
            <div class="metric-label">Total Experiments</div>
        </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary['success_rate']}%</div>
            <div class="metric-label">Success Rate</div>
        </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary['best_auc']}</div>
            <div class="metric-label">Best AUC Score</div>
        </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary['avg_auc']}</div>
            <div class="metric-label">Average AUC</div>
        </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary['avg_time_seconds']}s</div>
            <div class="metric-label">Avg Runtime</div>
        </div>""", unsafe_allow_html=True)


# ── SECTION 2: LEADERBOARD ─────────────────────────────────────
st.markdown('<p class="section-title">🏆 Performance Leaderboard</p>',
            unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2])

with col_left:
    def highlight_top(row):
        if row['rank'] == 1:
            return ['background-color: #003087; color: #FFD700'] * len(row)
        elif row['rank'] <= 3:
            return ['background-color: #1a1a2e; color: #006FCF'] * len(row)
        return [''] * len(row)
    styled = leaderboard.style.apply(highlight_top, axis=1)
    st.dataframe(styled, use_container_width=True, height=400)

with col_right:
    top10 = leaderboard.head(10)
    fig = px.bar(
        top10,
        x='exp_id',
        y='auc_score',
        color='auc_score',
        color_continuous_scale=[[0,'#003087'],[1,'#006FCF']],
        title='Top 10 Experiments by AUC Score',
        labels={'auc_score':'AUC Score','exp_id':'Experiment'}
    )
    fig.update_layout(
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        font_color='white',
        title_font_color=AMEX_BLUE,
        showlegend=False,
        height=380
    )
    fig.update_yaxes(range=[0.88, 1.0])
    st.plotly_chart(fig, use_container_width=True)


# ── SECTION 3: HYPERPARAMETER IMPACT ──────────────────────────
st.markdown('<p class="section-title">🔬 Hyperparameter Impact Analysis</p>',
            unsafe_allow_html=True)

h1, h2, h3 = st.columns(3)

with h1:
    fig = px.bar(
        trees_impact, x='n_estimators', y='avg_auc',
        title='Trees vs Avg AUC',
        color='avg_auc',
        color_continuous_scale=[[0,'#003087'],[1,'#006FCF']],
        labels={'n_estimators':'Number of Trees','avg_auc':'Avg AUC'}
    )
    fig.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font_color='white', title_font_color=AMEX_BLUE,
        showlegend=False, height=300
    )
    fig.update_yaxes(range=[0.92, 0.96])
    st.plotly_chart(fig, use_container_width=True)

with h2:
    fig = px.bar(
        depth_impact, x='max_depth', y='avg_auc',
        title='Depth vs Avg AUC',
        color='avg_auc',
        color_continuous_scale=[[0,'#003087'],[1,'#006FCF']],
        labels={'max_depth':'Tree Depth','avg_auc':'Avg AUC'}
    )
    fig.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font_color='white', title_font_color=AMEX_BLUE,
        showlegend=False, height=300
    )
    fig.update_yaxes(range=[0.94, 0.952])
    st.plotly_chart(fig, use_container_width=True)

with h3:
    fig = px.bar(
        lr_impact, x='learning_rate', y='avg_auc',
        title='Learning Rate vs Avg AUC',
        color='avg_auc',
        color_continuous_scale=[[0,'#003087'],[1,'#006FCF']],
        labels={'learning_rate':'Learning Rate','avg_auc':'Avg AUC'}
    )
    fig.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font_color='white', title_font_color=AMEX_BLUE,
        showlegend=False, height=300
    )
    fig.update_yaxes(range=[0.90, 0.98])
    st.plotly_chart(fig, use_container_width=True)


# ── SECTION 4: TIME VS PERFORMANCE ────────────────────────────
st.markdown('<p class="section-title">⏱️ Runtime vs Performance</p>',
            unsafe_allow_html=True)

sc1, sc2 = st.columns([3, 2])

with sc1:
    fig = px.scatter(
        success_df,
        x='time_seconds', y='auc_score',
        color='learning_rate', size='n_estimators',
        hover_data=['exp_id','max_depth'],
        title='Is Slower Always Better?',
        labels={
            'time_seconds' :'Time Taken (seconds)',
            'auc_score'    :'AUC Score',
            'learning_rate':'Learning Rate'
        },
        color_continuous_scale=[[0,'#006FCF'],[1,'#FFD700']]
    )
    fig.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font_color='white', title_font_color=AMEX_BLUE,
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)

with sc2:
    fig = px.bar(
        time_analysis,
        x='speed_category', y='avg_auc',
        title='Speed Category vs Avg AUC',
        color='avg_auc',
        color_continuous_scale=[[0,'#003087'],[1,'#006FCF']],
        labels={'speed_category':'Speed','avg_auc':'Avg AUC'},
        text='experiment_count'
    )
    fig.update_traces(texttemplate='%{text} exps', textposition='outside')
    fig.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font_color='white', title_font_color=AMEX_BLUE,
        showlegend=False, height=380
    )
    fig.update_yaxes(range=[0.93, 0.96])
    st.plotly_chart(fig, use_container_width=True)


# ── SECTION 5: LEARNING RATE DEEP DIVE ────────────────────────
st.markdown('<p class="section-title">📉 Learning Rate Deep Dive</p>',
            unsafe_allow_html=True)

lr1, lr2 = st.columns(2)

with lr1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Best AUC',
        x=lr_comparison['learning_rate'].astype(str),
        y=lr_comparison['best_auc'],
        marker_color='#006FCF'
    ))
    fig.add_trace(go.Bar(
        name='Avg AUC',
        x=lr_comparison['learning_rate'].astype(str),
        y=lr_comparison['avg_auc'],
        marker_color='#4a90d9'
    ))
    fig.add_trace(go.Bar(
        name='Worst AUC',
        x=lr_comparison['learning_rate'].astype(str),
        y=lr_comparison['worst_auc'],
        marker_color='#003087'
    ))
    fig.update_layout(
        barmode='group',
        title='Best / Avg / Worst AUC by Learning Rate',
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font_color='white', title_font_color=AMEX_BLUE,
        height=350,
        xaxis_title='Learning Rate',
        yaxis_title='AUC Score'
    )
    fig.update_yaxes(range=[0.88, 1.0])
    st.plotly_chart(fig, use_container_width=True)

with lr2:
    fig = px.bar(
        lr_comparison,
        x='learning_rate', y='std_auc',
        title='Consistency (Lower = More Consistent)',
        color='std_auc',
        color_continuous_scale=[[0,'#006FCF'],[1,FAIL_RED]],
        labels={'learning_rate':'Learning Rate','std_auc':'Std Deviation'}
    )
    fig.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font_color='white', title_font_color=AMEX_BLUE,
        showlegend=False, height=350
    )
    st.plotly_chart(fig, use_container_width=True)


# ── SECTION 6: BEST AND WORST ──────────────────────────────────
st.markdown('<p class="section-title">🥇 Best & Worst Experiments</p>',
            unsafe_allow_html=True)

t1, t2 = st.columns(2)

with t1:
    st.markdown("**🟢 Top 3 Experiments**")
    for idx, row in top_n.iterrows():
        st.markdown(f"""
            <div class="insight-box">
                <span class="best-tag">#{idx+1}</span>
                &nbsp; <strong>{row['exp_id']}</strong>
                &nbsp;|&nbsp; AUC: <strong style="color:#006FCF">
                {row['auc_score']}</strong>
                &nbsp;|&nbsp; Trees: {int(row['n_estimators'])}
                &nbsp;|&nbsp; Depth: {int(row['max_depth'])}
                &nbsp;|&nbsp; LR: {row['learning_rate']}
            </div>
        """, unsafe_allow_html=True)

with t2:
    st.markdown("**🔴 Bottom 3 Experiments**")
    for idx, row in bottom_n.iterrows():
        st.markdown(f"""
            <div class="insight-box">
                <span style="background:#dc3545;color:white;
                padding:2px 8px;border-radius:4px;
                font-size:0.75rem;font-weight:700;">
                #{25+idx}</span>
                &nbsp; <strong>{row['exp_id']}</strong>
                &nbsp;|&nbsp; AUC: <strong style="color:#dc3545">
                {row['auc_score']}</strong>
                &nbsp;|&nbsp; Trees: {int(row['n_estimators'])}
                &nbsp;|&nbsp; Depth: {int(row['max_depth'])}
                &nbsp;|&nbsp; LR: {row['learning_rate']}
            </div>
        """, unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
    <div style="text-align:center;color:#555;
    font-size:0.8rem;padding:1rem">
        ML Experiment Analytics Platform &nbsp;|&nbsp;
        Inspired by AiDA Develop @ American Express &nbsp;|&nbsp;
        Built with Python · SQLite · XGBoost · Streamlit
    </div>
""", unsafe_allow_html=True)