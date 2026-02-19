import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import joblib, os, io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="India Traffic Accident Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — Dark industrial dashboard
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0a0c10;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1318;
    border-right: 1px solid #1e2530;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ff4b4b;
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
}

/* ── Header ── */
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #ff4b4b 0%, #ff8c42 50%, #ffd166 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin: 0;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    color: #5a6478;
    font-size: 0.85rem;
    letter-spacing: 3px;
    margin-top: 6px;
    text-transform: uppercase;
}
.hero-bar {
    height: 3px;
    background: linear-gradient(90deg, #ff4b4b, #ff8c42, #ffd166, transparent);
    margin: 18px 0 30px 0;
    border-radius: 2px;
}

/* ── Metric Cards ── */
.metric-card {
    background: #111620;
    border: 1px solid #1e2530;
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: #ff4b4b55; }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #ff4b4b);
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #5a6478;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: #e8eaf0;
    letter-spacing: 2px;
    line-height: 1;
}
.metric-sub {
    font-size: 0.72rem;
    color: #5a6478;
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section Headers ── */
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.9rem;
    letter-spacing: 3px;
    color: #e8eaf0;
    border-left: 4px solid #ff4b4b;
    padding-left: 14px;
    margin: 32px 0 18px 0;
}

/* ── Tab Style ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111620;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2530;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 1.5px;
    color: #5a6478;
    border-radius: 7px;
    padding: 8px 18px;
    text-transform: uppercase;
}
.stTabs [aria-selected="true"] {
    background: #ff4b4b !important;
    color: #fff !important;
}

/* ── Selectbox / Sliders ── */
.stSelectbox > div > div {
    background: #111620;
    border-color: #1e2530;
    color: #e8eaf0;
    border-radius: 8px;
}
.stSlider [data-baseweb="slider"] { margin-top: 4px; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
thead tr th {
    background: #0f1318 !important;
    color: #ff8c42 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px;
}

/* ── Alert / Info boxes ── */
.st-emotion-cache-1wbqy5l { background: #111620; border-color: #ff4b4b44; }

/* ── Prediction result ── */
.predict-box {
    background: linear-gradient(135deg, #111620 0%, #1a2030 100%);
    border: 1px solid #ff4b4b44;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.predict-box::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 0%, #ff4b4b11 0%, transparent 65%);
}
.predict-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #ff4b4b, #ffd166);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.predict-label {
    font-family: 'JetBrains Mono', monospace;
    color: #5a6478;
    font-size: 0.78rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 10px;
}

/* ── Model badge ── */
.badge {
    display: inline-block;
    background: #ff4b4b22;
    color: #ff8c42;
    border: 1px solid #ff4b4b44;
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
}

/* ── Divider ── */
hr { border-color: #1e2530; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #ff4b4b, #ff8c42);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 10px 22px;
    transition: opacity 0.2s, transform 0.2s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0a0c10; }
::-webkit-scrollbar-thumb { background: #1e2530; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#111620',
    plot_bgcolor='#111620',
    font=dict(family='DM Sans', color='#e8eaf0', size=12),
    title_font=dict(family='Bebas Neue', size=20, color='#e8eaf0'),
    colorway=['#ff4b4b','#ff8c42','#ffd166','#06d6a0','#118ab2','#a8dadc','#e63946','#457b9d'],
    xaxis=dict(gridcolor='#1e2530', zerolinecolor='#1e2530', linecolor='#1e2530'),
    yaxis=dict(gridcolor='#1e2530', zerolinecolor='#1e2530', linecolor='#1e2530'),
    legend=dict(bgcolor='#0f1318', bordercolor='#1e2530', borderwidth=1),
    margin=dict(l=20, r=20, t=50, b=20),
)

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_clean(path):
    df_raw = pd.read_csv(path)
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    df = df[~df['State/UT'].str.contains('Total', na=False)]

    road_acc_cols = [c for c in df.columns if c.startswith('Road Accidents -')]
    target_col    = 'Total (Traffic Accidents)'
    railway_total = 'Railway Accidents - Total'
    rc_total      = 'Railway Crossing Accidents - Total'

    keep_cols = ['State/UT'] + road_acc_cols + [railway_total, rc_total, target_col]
    df = df[keep_cols].copy()

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=[target_col], inplace=True)
    df = df[df[target_col] > 0].reset_index(drop=True)
    df.fillna(0, inplace=True)
    return df

@st.cache_resource
def train_all_models(df):
    target_col   = 'Total (Traffic Accidents)'
    feature_cols = [c for c in df.columns if c not in ['State/UT', target_col]]

    le = LabelEncoder()
    state_encoded = le.fit_transform(df['State/UT'].values).reshape(-1, 1)
    X = np.hstack([state_encoded, df[feature_cols].values])
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    models = {
        'Linear Regression'  : LinearRegression(),
        'Ridge Regression'   : Ridge(alpha=1.0),
        'Lasso Regression'   : Lasso(alpha=0.1),
        'Decision Tree'      : DecisionTreeRegressor(random_state=42),
        'Random Forest'      : RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting'  : GradientBoostingRegressor(n_estimators=100, random_state=42),
        'AdaBoost'           : AdaBoostRegressor(n_estimators=100, random_state=42),
        'Extra Trees'        : ExtraTreesRegressor(n_estimators=100, random_state=42),
        'SVR'                : SVR(kernel='rbf', C=100, gamma=0.1),
        'KNN Regressor'      : KNeighborsRegressor(n_neighbors=5),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)
        cv     = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
        results[name] = {'MAE': round(mae,2), 'RMSE': round(rmse,2),
                         'R2': round(r2,4), 'CV_R2': round(cv,4),
                         'model': model, 'y_pred': y_pred}

    return results, X_train, X_test, y_train, y_test, scaler, le, feature_cols

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚦 NCRB 2023")
    st.markdown("**Traffic Accident Intelligence**")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload CSV", type=['csv'],
                                help="Upload NCRB_ADSI_2023_Table_1A_6.csv")
    st.markdown("---")

    st.markdown("### ⚙️ SETTINGS")
    test_size = st.slider("Test Set Size", 0.15, 0.40, 0.25, 0.05,
                          help="Fraction of data used for testing")
    selected_models = st.multiselect(
        "Models to Compare",
        ['Linear Regression','Ridge Regression','Lasso Regression',
         'Decision Tree','Random Forest','Gradient Boosting',
         'AdaBoost','Extra Trees','SVR','KNN Regressor'],
        default=['Linear Regression','Random Forest','Gradient Boosting',
                 'Extra Trees','Ridge Regression'],
        help="Choose models for the comparison view"
    )

    st.markdown("---")
    st.markdown("### 📊 CHART THEME")
    color_palette = st.selectbox("Color Palette", ['Warm (Red-Orange)','Cool (Blue-Teal)','Neon (Green-Purple)'])
    palette_map = {
        'Warm (Red-Orange)': px.colors.sequential.Reds,
        'Cool (Blue-Teal)'  : px.colors.sequential.Blues,
        'Neon (Green-Purple)': px.colors.sequential.Purpor,
    }
    chosen_palette = palette_map[color_palette]

    st.markdown("---")
    st.markdown('<div style="font-family:JetBrains Mono;font-size:0.65rem;color:#2a3040;text-align:center">NCRB · ADSI 2023 · TABLE 1A-6</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
default_path = 'NCRB_ADSI_2023_Table_1A_6.csv'

if uploaded:
    df = load_and_clean(uploaded)
elif os.path.exists(default_path):
    df = load_and_clean(default_path)
else:
    st.error("⚠️ Please upload the CSV file using the sidebar.")
    st.stop()

results, X_train, X_test, y_train, y_test, scaler, le, feature_cols = train_all_models(df)
target_col = 'Total (Traffic Accidents)'
road_acc_cols = [c for c in feature_cols if c.startswith('Road Accidents -') and c != 'Road Accidents - Total']

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 10px 0 0 0">
    <p class="hero-title">TRAFFIC ACCIDENT<br>INTELLIGENCE</p>
    <p class="hero-sub">NCRB · ADSI 2023 · TABLE 1A-6 · ALL INDIA ANALYSIS</p>
    <div class="hero-bar"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────
total_accidents    = int(df[target_col].sum())
highest_state      = df.loc[df[target_col].idxmax(), 'State/UT']
highest_val        = int(df[target_col].max())
night_total        = int(df['Road Accidents - 0000 hrs'].sum() + df['Road Accidents - 2100 hrs. to 2400hrs(Night)'].sum() + df['Road Accidents - 1800 hrs to 2100hrs (Night)'].sum())
best_model_name    = max(results, key=lambda k: results[k]['R2'])
best_r2            = results[best_model_name]['R2']

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""<div class="metric-card" style="--accent:#ff4b4b">
        <div class="metric-label">Total Traffic Accidents</div>
        <div class="metric-value">{total_accidents:,}</div>
        <div class="metric-sub">All States + UTs · 2023</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="metric-card" style="--accent:#ff8c42">
        <div class="metric-label">Highest Accident State</div>
        <div class="metric-value" style="font-size:1.5rem">{highest_state}</div>
        <div class="metric-sub">{highest_val:,} total accidents</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class="metric-card" style="--accent:#ffd166">
        <div class="metric-label">Night-Time Accidents</div>
        <div class="metric-value">{night_total:,}</div>
        <div class="metric-sub">18:00 – 06:00 hrs combined</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class="metric-card" style="--accent:#06d6a0">
        <div class="metric-label">Best Model R² Score</div>
        <div class="metric-value">{best_r2}</div>
        <div class="metric-sub">{best_model_name}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA & Insights",
    "🗺️ State Explorer",
    "⏰ Time Analysis",
    "🤖 Model Comparison",
    "🔮 Predict"
])

# ═══════════════════════════════════════════
# TAB 1 — EDA & INSIGHTS
# ═══════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">EXPLORATORY ANALYSIS</div>', unsafe_allow_html=True)

    # Top N states bar chart
    col_a, col_b = st.columns([3,1])
    with col_b:
        top_n = st.slider("Top N States", 5, len(df), 15, 1, key='topn')
        chart_type = st.radio("Chart Type", ["Bar", "Treemap"], key='ctype')

    top_df = df[['State/UT', target_col]].sort_values(target_col, ascending=False).head(top_n)

    with col_a:
        if chart_type == "Bar":
            fig = px.bar(top_df, x='State/UT', y=target_col,
                         color=target_col, color_continuous_scale=chosen_palette,
                         title=f"TOP {top_n} STATES — TOTAL TRAFFIC ACCIDENTS (2023)",
                         labels={target_col: 'Accidents', 'State/UT': ''})
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_traces(marker_line_width=0, hovertemplate='<b>%{x}</b><br>%{y:,} accidents<extra></extra>')
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.treemap(top_df, path=['State/UT'], values=target_col,
                             color=target_col, color_continuous_scale=chosen_palette,
                             title=f"TOP {top_n} STATES — TREEMAP VIEW")
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_traces(hovertemplate='<b>%{label}</b><br>%{value:,} accidents<extra></extra>')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Distribution + Boxplot
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=target_col, nbins=30, marginal='box',
                           color_discrete_sequence=['#ff4b4b'],
                           title="DISTRIBUTION OF TOTAL TRAFFIC ACCIDENTS",
                           labels={target_col: 'Total Accidents'})
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        acc_types = {
            'Road Accidents'            : 'Road Accidents - Total',
            'Railway Accidents'         : 'Railway Accidents - Total',
            'Railway Crossing Accidents': 'Railway Crossing Accidents - Total',
        }
        pie_vals = [df[v].sum() for v in acc_types.values()]
        fig = go.Figure(go.Pie(
            labels=list(acc_types.keys()),
            values=pie_vals,
            hole=0.55,
            marker=dict(colors=['#ff4b4b','#ff8c42','#ffd166'],
                        line=dict(color='#0a0c10', width=2)),
            textfont=dict(family='JetBrains Mono', size=11),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="ACCIDENT TYPE BREAKDOWN (ALL INDIA)")
        fig.add_annotation(text=f"<b>{sum(pie_vals):,}</b>", x=0.5, y=0.5,
                           font=dict(size=18, color='#e8eaf0', family='Bebas Neue'),
                           showarrow=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Scatter matrix
    st.markdown('<div class="section-title">CORRELATION EXPLORER</div>', unsafe_allow_html=True)
    slot_map = {
        '00–03 (Night)': 'Road Accidents - 0000 hrs',
        '03–06 (Night)': 'Road Accidents - 0300 hrs to 0600 hrs. (Night)',
        '06–09 (Day)'  : 'Road Accidents - 0600 hrs to 0900 hrs (Day)',
        '09–12 (Day)'  : 'Road Accidents - 0900 hrs to 1200 hrs (Day)',
        '12–15 (Day)'  : 'Road Accidents - 1200 hrs to 1500 hrs (Day)',
        '15–18 (Day)'  : 'Road Accidents - 1500 hrs to 1800 hrs (Day)',
        '18–21 (Night)': 'Road Accidents - 1800 hrs to 2100hrs (Night)',
        '21–24 (Night)': 'Road Accidents - 2100 hrs. to 2400hrs(Night)',
    }

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X Axis", list(slot_map.keys()), index=2)
    with col2:
        y_axis = st.selectbox("Y Axis", list(slot_map.keys()), index=6)

    scatter_df = df[['State/UT', slot_map[x_axis], slot_map[y_axis], target_col]].copy()
    scatter_df.columns = ['State/UT', 'X', 'Y', 'Total']
    fig = px.scatter(scatter_df, x='X', y='Y', color='Total',
                     size='Total', hover_name='State/UT',
                     color_continuous_scale=chosen_palette,
                     title=f"{x_axis}  vs  {y_axis}",
                     labels={'X': x_axis, 'Y': y_axis},
                     size_max=40)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_traces(marker_line_width=0.5, marker_line_color='#0a0c10',
                      hovertemplate='<b>%{hovertext}</b><br>X: %{x:,}<br>Y: %{y:,}<br>Total: %{marker.color:,}<extra></extra>')
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 2 — STATE EXPLORER
# ═══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">STATE / UT DEEP DIVE</div>', unsafe_allow_html=True)

    states_list = sorted(df['State/UT'].unique())
    col1, col2 = st.columns([2, 1])

    with col2:
        selected_states = st.multiselect(
            "Select States / UTs",
            states_list,
            default=states_list[:6],
            key='state_sel'
        )
        compare_metric = st.radio("Compare By", ["Total Accidents", "Road Only", "Railway Only"], key='comp_met')
        show_rank = st.checkbox("Show Rank Labels", value=True)

    metric_map = {
        "Total Accidents": target_col,
        "Road Only"      : 'Road Accidents - Total',
        "Railway Only"   : 'Railway Accidents - Total',
    }

    filtered = df[df['State/UT'].isin(selected_states)].sort_values(metric_map[compare_metric], ascending=False)

    with col1:
        fig = px.bar(
            filtered, x=metric_map[compare_metric], y='State/UT',
            orientation='h', color=metric_map[compare_metric],
            color_continuous_scale=chosen_palette,
            title=f"STATE COMPARISON — {compare_metric.upper()}",
            text=metric_map[compare_metric] if show_rank else None,
            labels={metric_map[compare_metric]: 'Accidents', 'State/UT': ''}
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside',
                          hovertemplate='<b>%{y}</b><br>%{x:,} accidents<extra></extra>')
        fig.update_layout(**PLOTLY_LAYOUT, height=max(350, len(selected_states)*45))
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart for selected states (up to 5)
    st.markdown("---")
    st.markdown('<div class="section-title">TIME-SLOT RADAR — STATE PROFILE</div>', unsafe_allow_html=True)

    radar_states = st.multiselect("States for Radar", states_list,
                                  default=states_list[:4], max_selections=6, key='radar_sel')

    short_labels = ['00–03','03–06','06–09','09–12','12–15','15–18','18–21','21–24']
    road_cols_list = [
        'Road Accidents - 0000 hrs',
        'Road Accidents - 0300 hrs to 0600 hrs. (Night)',
        'Road Accidents - 0600 hrs to 0900 hrs (Day)',
        'Road Accidents - 0900 hrs to 1200 hrs (Day)',
        'Road Accidents - 1200 hrs to 1500 hrs (Day)',
        'Road Accidents - 1500 hrs to 1800 hrs (Day)',
        'Road Accidents - 1800 hrs to 2100hrs (Night)',
        'Road Accidents - 2100 hrs. to 2400hrs(Night)',
    ]
    colors_radar = ['#ff4b4b','#ff8c42','#ffd166','#06d6a0','#118ab2','#a8dadc']

    fig = go.Figure()
    for i, state in enumerate(radar_states):
        row = df[df['State/UT'] == state].iloc[0]
        vals = [row[c] for c in road_cols_list]
        vals_closed = vals + [vals[0]]
        labels_closed = short_labels + [short_labels[0]]
        col_r = colors_radar[i % len(colors_radar)]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=labels_closed, name=state,
            fill='toself', line=dict(color=col_r, width=2),
            fillcolor=col_r + '22',
        ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      polar=dict(
                          bgcolor='#111620',
                          radialaxis=dict(visible=True, gridcolor='#1e2530',
                                          color='#5a6478', tickfont=dict(size=9)),
                          angularaxis=dict(gridcolor='#1e2530', color='#5a6478')
                      ),
                      title="TIME-SLOT ROAD ACCIDENT PROFILE PER STATE",
                      height=480)
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.markdown("---")
    st.markdown('<div class="section-title">RAW DATA TABLE</div>', unsafe_allow_html=True)
    disp_cols = ['State/UT', 'Road Accidents - Total', 'Railway Accidents - Total',
                 'Railway Crossing Accidents - Total', target_col]
    st.dataframe(
        df[df['State/UT'].isin(selected_states)][disp_cols]
          .sort_values(target_col, ascending=False)
          .reset_index(drop=True)
          .style.background_gradient(subset=[target_col,'Road Accidents - Total'], cmap='Reds'),
        use_container_width=True, height=300
    )

# ═══════════════════════════════════════════
# TAB 3 — TIME ANALYSIS
# ═══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">TIME-OF-DAY ACCIDENT PATTERNS</div>', unsafe_allow_html=True)

    time_labels = ['00–03','03–06','06–09','09–12','12–15','15–18','18–21','21–24']
    is_night    = [True, True, False, False, False, False, True, True]
    slot_totals = [df[c].sum() for c in road_cols_list]

    # Summary bar
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=time_labels, y=slot_totals,
        marker_color=['#ff4b4b' if n else '#ff8c42' for n in is_night],
        hovertemplate='<b>%{x} hrs</b><br>%{y:,} accidents<extra></extra>',
        name='Accidents'
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="TOTAL ROAD ACCIDENTS BY TIME SLOT — ALL INDIA 2023",
                      xaxis_title="Time Slot (hrs)",
                      yaxis_title="Number of Accidents",
                      height=380)
    shapes = [dict(type='rect', xref='paper', yref='paper',
                   x0=0, x1=0.26, y0=0, y1=1, fillcolor='#ff4b4b', opacity=0.04, layer='below', line_width=0),
              dict(type='rect', xref='paper', yref='paper',
                   x0=0.74, x1=1, y0=0, y1=1, fillcolor='#ff4b4b', opacity=0.04, layer='below', line_width=0)]
    fig.update_layout(shapes=shapes)
    fig.add_annotation(text="🌙 NIGHT", x=0.12, y=1.05, xref='paper', yref='paper',
                       showarrow=False, font=dict(color='#ff4b4b', size=10, family='JetBrains Mono'))
    fig.add_annotation(text="🌙 NIGHT", x=0.87, y=1.05, xref='paper', yref='paper',
                       showarrow=False, font=dict(color='#ff4b4b', size=10, family='JetBrains Mono'))
    st.plotly_chart(fig, use_container_width=True)

    # Night vs Day Donut
    col1, col2 = st.columns(2)

    with col1:
        night_sum = sum(slot_totals[i] for i in range(8) if is_night[i])
        day_sum   = sum(slot_totals[i] for i in range(8) if not is_night[i])
        fig = go.Figure(go.Pie(
            labels=['Night (18–06)', 'Day (06–18)'],
            values=[night_sum, day_sum], hole=0.6,
            marker=dict(colors=['#ff4b4b','#ff8c42'], line=dict(color='#0a0c10', width=3)),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="DAY vs NIGHT SPLIT")
        fig.add_annotation(text=f"<b>{(night_sum/(night_sum+day_sum)*100):.1f}%</b><br>Night",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color='#ff4b4b', family='Bebas Neue'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Peak hour heatmap by state (top 15)
        top15 = df.nlargest(15, target_col)
        heat_df = top15[road_cols_list].copy()
        heat_df.columns = time_labels
        heat_df.index = top15['State/UT']

        fig = px.imshow(heat_df, color_continuous_scale='Reds',
                        title="TIME-SLOT HEATMAP — TOP 15 STATES",
                        labels=dict(x='Time Slot', y='State/UT', color='Accidents'),
                        aspect='auto')
        fig.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # State-level time line chart
    st.markdown("---")
    st.markdown('<div class="section-title">STATE TIME PROFILE — LINE CHART</div>', unsafe_allow_html=True)

    line_states = st.multiselect("Select States", states_list,
                                 default=['Tamil Nadu','Madhya Pradesh','Kerala','Karnataka','Uttar Pradesh'],
                                 key='line_sel')

    fig = go.Figure()
    lcolors = ['#ff4b4b','#ff8c42','#ffd166','#06d6a0','#118ab2','#a8dadc','#e63946']
    for i, state in enumerate(line_states):
        row  = df[df['State/UT'] == state].iloc[0]
        vals = [row[c] for c in road_cols_list]
        fig.add_trace(go.Scatter(
            x=time_labels, y=vals, name=state, mode='lines+markers',
            line=dict(color=lcolors[i % len(lcolors)], width=2.5),
            marker=dict(size=7),
            hovertemplate=f'<b>{state}</b><br>%{{x}} hrs: %{{y:,}}<extra></extra>'
        ))
    fig.update_layout(**PLOTLY_LAYOUT, title="ROAD ACCIDENTS ACROSS TIME SLOTS BY STATE",
                      xaxis_title="Time Slot", yaxis_title="Road Accidents", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ═══════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">MODEL PERFORMANCE COMPARISON</div>', unsafe_allow_html=True)

    # Filter to sidebar-selected models
    filtered_results = {k: v for k, v in results.items() if k in (selected_models or list(results.keys()))}

    comp_df = pd.DataFrame([
        {'Model': k, 'R²': v['R2'], 'CV R²': v['CV_R2'], 'MAE': v['MAE'], 'RMSE': v['RMSE']}
        for k, v in filtered_results.items()
    ]).sort_values('R²', ascending=False).reset_index(drop=True)

    # Styled table
    st.dataframe(
        comp_df.style
            .background_gradient(subset=['R²','CV R²'], cmap='Greens')
            .background_gradient(subset=['MAE','RMSE'], cmap='Reds_r')
            .highlight_max(subset=['R²','CV R²'], color='#1a3a2a')
            .highlight_min(subset=['MAE','RMSE'], color='#1a3a2a')
            .format({'R²':'{:.4f}','CV R²':'{:.4f}','MAE':'{:,.2f}','RMSE':'{:,.2f}'}),
        use_container_width=True, height=320
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Metric tabs
    mt1, mt2, mt3, mt4 = st.tabs(["R² Score","MAE","RMSE","CV R²"])

    def model_bar(df_plot, col, title, color_scale, ascending_color=False):
        sorted_df = df_plot.sort_values(col, ascending=ascending_color)
        fig = px.bar(sorted_df, x='Model', y=col, color=col,
                     color_continuous_scale=color_scale,
                     title=title, text=col)
        fig.update_traces(texttemplate='%{text:.4f}' if col in ['R²','CV R²'] else '%{text:,.0f}',
                          textposition='outside',
                          hovertemplate='<b>%{x}</b><br>Score: %{y}<extra></extra>')
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_coloraxes(showscale=False)
        return fig

    with mt1:
        st.plotly_chart(model_bar(comp_df,'R²','R² SCORE (HIGHER IS BETTER)',
                                  'Greens', ascending_color=True), use_container_width=True)
    with mt2:
        st.plotly_chart(model_bar(comp_df,'MAE','MEAN ABSOLUTE ERROR (LOWER IS BETTER)',
                                  'Reds_r', ascending_color=False), use_container_width=True)
    with mt3:
        st.plotly_chart(model_bar(comp_df,'RMSE','ROOT MEAN SQUARED ERROR (LOWER IS BETTER)',
                                  'Reds_r', ascending_color=False), use_container_width=True)
    with mt4:
        st.plotly_chart(model_bar(comp_df,'CV R²','5-FOLD CROSS VALIDATED R² (HIGHER IS BETTER)',
                                  'Blues', ascending_color=True), use_container_width=True)

    # Actual vs Predicted grid
    st.markdown("---")
    st.markdown('<div class="section-title">ACTUAL vs PREDICTED</div>', unsafe_allow_html=True)

    avp_models = st.multiselect("Models to Display", list(filtered_results.keys()),
                                default=list(filtered_results.keys())[:4], max_selections=4, key='avp_sel')

    if avp_models:
        cols = st.columns(min(len(avp_models), 2))
        for i, name in enumerate(avp_models):
            y_pred = filtered_results[name]['y_pred']
            col_fig = cols[i % 2]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_pred, mode='markers',
                marker=dict(color=np.abs(y_test - y_pred), colorscale='RdYlGn_r',
                            size=9, line=dict(width=0.5, color='#0a0c10'), opacity=0.85),
                hovertemplate='Actual: %{x:,}<br>Predicted: %{y:,}<extra></extra>',
                name='Predictions'
            ))
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines',
                                     line=dict(color='#ff4b4b', dash='dash', width=2),
                                     name='Perfect Fit', showlegend=False))
            fig.update_layout(**PLOTLY_LAYOUT,
                              title=f"{name.upper()}  ·  R²={filtered_results[name]['R2']}",
                              xaxis_title='Actual', yaxis_title='Predicted', height=340)
            col_fig.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.markdown('<div class="section-title">FEATURE IMPORTANCE</div>', unsafe_allow_html=True)

    tree_available = [m for m in filtered_results
                      if hasattr(filtered_results[m]['model'], 'feature_importances_')]
    if tree_available:
        fi_model = st.selectbox("Select Tree Model", tree_available, key='fi_sel')
        feat_names = ['State_Encoded'] + feature_cols
        importances = filtered_results[fi_model]['model'].feature_importances_
        fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=True).tail(12)
        short_feats = [f.replace('Road Accidents - ','RA ').replace('Railway Accidents - Total','Rail Total')
                       .replace('Railway Crossing Accidents - Total','RC Total') for f in fi_df['Feature']]

        fig = px.bar(fi_df, x='Importance', y=short_feats, orientation='h',
                     color='Importance', color_continuous_scale=chosen_palette,
                     title=f"TOP FEATURE IMPORTANCES — {fi_model.upper()}")
        fig.update_layout(**PLOTLY_LAYOUT, height=400)
        fig.update_coloraxes(showscale=False)
        fig.update_traces(hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)

    # Save model button
    st.markdown("---")
    col_s1, col_s2, col_s3 = st.columns([1,2,1])
    with col_s2:
        best_overall = max(results, key=lambda k: results[k]['R2'])
        if st.button(f"💾 Save Best Model ({best_overall})", use_container_width=True):
            os.makedirs('saved_model', exist_ok=True)
            joblib.dump(results[best_overall]['model'], 'saved_model/best_model.pkl')
            joblib.dump(scaler, 'saved_model/scaler.pkl')
            joblib.dump(le, 'saved_model/label_encoder.pkl')
            st.success(f"✅ Model saved to saved_model/ — R² = {results[best_overall]['R2']}")

# ═══════════════════════════════════════════
# TAB 5 — PREDICT
# ═══════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">PREDICT TOTAL TRAFFIC ACCIDENTS</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1.3, 1])

    with col_l:
        st.markdown("#### Configure Accident Inputs")

        pred_model_name = st.selectbox("Prediction Model", list(results.keys()),
                                       index=list(results.keys()).index(best_model_name), key='pred_model')

        state_choice = st.selectbox("State / UT", sorted(df['State/UT'].unique()), key='pred_state')

        st.markdown("**Road Accidents by Time Slot**")
        col_i1, col_i2 = st.columns(2)
        slot_names_short = ['00–03 (Night)','03–06 (Night)','06–09 (Day)','09–12 (Day)',
                             '12–15 (Day)','15–18 (Day)','18–21 (Night)','21–24 (Night)']

        # Pre-fill from actual state data
        actual_row = df[df['State/UT'] == state_choice].iloc[0]
        slot_defaults = [int(actual_row[c]) for c in road_cols_list]

        slot_inputs = []
        for j, (short, col_c) in enumerate(zip(slot_names_short, road_cols_list)):
            container = col_i1 if j % 2 == 0 else col_i2
            val = container.number_input(short, min_value=0, max_value=100000,
                                         value=slot_defaults[j], step=50, key=f'slot_{j}')
            slot_inputs.append(val)

        road_total  = sum(slot_inputs)
        rail_total  = st.number_input("Railway Accidents Total", 0, 50000,
                                      int(actual_row.get('Railway Accidents - Total', 0)), step=10)
        rc_total_in = st.number_input("Railway Crossing Accidents Total", 0, 5000,
                                      int(actual_row.get('Railway Crossing Accidents - Total', 0)), step=5)

        st.info(f"🛣️ Auto-computed Road Total: **{road_total:,}**")

        predict_btn = st.button("🔮 Predict Traffic Accidents", use_container_width=True)

    with col_r:
        st.markdown("#### Prediction Result")
        if predict_btn or True:  # Show placeholder on load
            # Build input vector
            state_enc = le.transform([state_choice])[0]
            raw_input = np.array([[state_enc] + slot_inputs + [road_total, rail_total, rc_total_in]])
            scaled_input = scaler.transform(raw_input)
            pred_val = results[pred_model_name]['model'].predict(scaled_input)[0]

            # Find actual for comparison
            actual_val = int(actual_row[target_col])
            diff       = pred_val - actual_val
            diff_pct   = (diff / actual_val * 100) if actual_val > 0 else 0

            st.markdown(f"""
            <div class="predict-box">
                <div class="predict-label">Predicted Total Traffic Accidents</div>
                <div class="predict-number">{int(pred_val):,}</div>
                <div class="predict-label" style="margin-top:16px">
                    Model: <span style="color:#ff8c42">{pred_model_name}</span>
                    &nbsp;·&nbsp; R² = {results[pred_model_name]['R2']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Comparison card
            arrow = "▲" if diff > 0 else "▼"
            arrow_color = "#ff4b4b" if diff > 0 else "#06d6a0"
            st.markdown(f"""
            <div class="metric-card" style="--accent:#118ab2">
                <div class="metric-label">vs Actual 2023 Data ({state_choice})</div>
                <div class="metric-value" style="font-size:1.6rem">{actual_val:,}</div>
                <div class="metric-sub" style="color:{arrow_color}">
                    {arrow} {abs(diff):,.0f} &nbsp;|&nbsp; {diff_pct:+.1f}% from actual
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Mini gauge chart
            max_val = df[target_col].max()
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=float(pred_val),
                delta={'reference': float(actual_val), 'relative': False,
                       'font': {'size': 13, 'family': 'JetBrains Mono'}},
                number={'font': {'size': 28, 'family': 'Bebas Neue', 'color': '#e8eaf0'},
                        'valueformat': ',.0f'},
                gauge={
                    'axis': {'range': [0, max_val], 'tickcolor': '#5a6478',
                             'tickfont': {'size': 9, 'family': 'JetBrains Mono'}},
                    'bar': {'color': '#ff4b4b', 'thickness': 0.25},
                    'bgcolor': '#111620',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, max_val*0.33], 'color': '#1a2535'},
                        {'range': [max_val*0.33, max_val*0.66], 'color': '#1e2e40'},
                        {'range': [max_val*0.66, max_val], 'color': '#22344a'},
                    ],
                    'threshold': {'line': {'color': '#ffd166', 'width': 3},
                                  'thickness': 0.8, 'value': actual_val}
                },
                title={'text': 'PREDICTED vs MAX RANGE', 'font': {'size': 11,
                       'family': 'JetBrains Mono', 'color': '#5a6478'}}
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=270, margin=dict(l=20,r=20,t=60,b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Time slot breakdown bar
            st.markdown("---")
            fig2 = go.Figure(go.Bar(
                x=slot_names_short, y=slot_inputs,
                marker_color=['#ff4b4b' if 'Night' in s else '#ff8c42' for s in slot_names_short],
                hovertemplate='%{x}<br>%{y:,} accidents<extra></extra>'
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, title="INPUT BREAKDOWN BY TIME SLOT",
                               xaxis_title='Slot', yaxis_title='Accidents', height=280)
            st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'JetBrains Mono'; font-size:0.65rem; color:#2a3040; padding: 10px 0 20px 0; letter-spacing:2px">
    NCRB · ACCIDENTAL DEATHS & SUICIDES IN INDIA 2023 · TABLE 1A-6 · BUILT WITH STREAMLIT & SCIKIT-LEARN
</div>
""", unsafe_allow_html=True)
