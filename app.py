"""
NCRB 2023 Traffic Accident Analysis — Streamlit Dashboard
Run with: streamlit run app.py

Make sure the CSV file is in the same folder as this script,
named: NCRB_ADSI_2023_Table_1A_6.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import os

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NCRB 2023 Traffic Accident ML Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #1A237E;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; color: #1A237E; font-size: 0.85rem; }
    .metric-card p  { margin: 0; font-size: 1.6rem; font-weight: 700; color: #212121; }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1A237E;
        border-bottom: 2px solid #E3F2FD;
        padding-bottom: 6px;
        margin-bottom: 1rem;
    }
    .risk-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

CLUSTER_COLORS = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']
TIME_COLORS    = ['#2196F3','#FF5722','#4CAF50','#9C27B0','#FF9800','#009688','#E91E63','#795548']

# ─────────────────────────────────────────────────────────
# DATA LOADING & ML (cached)
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_train(filepath):
    df_raw = pd.read_csv(filepath)

    exclude = ['Total (States)', 'Total (UTs)', 'Total (All India)', 'Total (Cities)']
    df = df_raw[~df_raw['State/UT'].isin(exclude)].copy()
    df = df.dropna(subset=['Sl. No.'])
    df = df[df['Sl. No.'].astype(str).str.strip().str.match(r'^\d+$')]
    df['Sl. No.'] = df['Sl. No.'].astype(int)

    df_states = df[df['Sl. No.'] <= 36].copy()
    df_cities = df[df['Sl. No.'] > 36].copy()

    time_slots = ['0000-0300','0300-0600','0600-0900','0900-1200',
                  '1200-1500','1500-1800','1800-2100','2100-2400']
    road_cols_raw = [c for c in df.columns if c.startswith('Road Accidents -') and 'Total' not in c]
    road_total_col = 'Road Accidents - Total'
    slot_map = {col: time_slots[i] for i, col in enumerate(road_cols_raw)}

    df_states = df_states.rename(columns=slot_map)
    df_cities = df_cities.rename(columns=slot_map)

    for df_ in [df_states, df_cities]:
        for ts in time_slots:
            df_[ts] = pd.to_numeric(df_[ts], errors='coerce').fillna(0)
        df_[road_total_col] = pd.to_numeric(df_[road_total_col], errors='coerce').fillna(0)
        night_slots = ['0000-0300','0300-0600','1800-2100','2100-2400']
        day_slots   = ['0600-0900','0900-1200','1200-1500','1500-1800']
        df_['Night_Accidents'] = df_[night_slots].sum(axis=1)
        df_['Day_Accidents']   = df_[day_slots].sum(axis=1)
        df_['Night_Pct']       = df_['Night_Accidents'] / (df_[road_total_col] + 1) * 100
        df_['Peak_Slot']       = df_[time_slots].idxmax(axis=1)
        df_['Peak_Value']      = df_[time_slots].max(axis=1)
        df_['Peak_Ratio']      = df_['Peak_Value'] / (df_[road_total_col] + 1)

    # ── Clustering ──
    X_cluster = df_states[time_slots].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    inertias = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_states['Cluster'] = km_final.fit_predict(X_scaled)

    centroids = scaler.inverse_transform(km_final.cluster_centers_)
    cent_df = pd.DataFrame(centroids, columns=time_slots)
    night_mean = cent_df[['0000-0300','0300-0600','1800-2100','2100-2400']].mean(axis=1)
    label_map = {}
    sorted_clusters = night_mean.argsort()
    label_map[int(sorted_clusters.iloc[0])] = 'Low Night Risk'
    label_map[int(sorted_clusters.iloc[1])] = 'Moderate Risk'
    label_map[int(sorted_clusters.iloc[2])] = 'High Night Risk'
    label_map[int(sorted_clusters.iloc[3])] = 'Extreme Night Risk'
    df_states['Cluster_Label'] = df_states['Cluster'].map(label_map)

    # ── Regression ──
    X_reg = df_cities[time_slots[:-1]].values
    y_reg = df_cities[road_total_col].values
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)

    models_dict = {
        'Linear Regression':    LinearRegression(),
        'Ridge Regression':     Ridge(alpha=1.0),
        'Random Forest':        RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting':    GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    results = {}
    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        cv_r2 = cross_val_score(model, X_reg, y_reg, cv=5, scoring='r2').mean()
        results[name] = {
            'R2': r2_score(y_test, preds),
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
            'CV_R2': cv_r2,
            'predictions': preds,
            'model': model
        }

    best_model_name = max(results, key=lambda k: results[k]['CV_R2'])
    rf_model = results['Random Forest']['model']
    feat_imp = pd.Series(rf_model.feature_importances_, index=time_slots[:-1]).sort_values(ascending=False)

    return (df_states, df_cities, time_slots, road_total_col,
            inertias, label_map, results, best_model_name,
            feat_imp, y_test, X_test, scaler, km_final)

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/120px-Emblem_of_India.svg.png", width=80)
    st.title("🚦 NCRB 2023\nTraffic ML Dashboard")
    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader("📂 Upload NCRB CSV", type=["csv"],
                                      help="Upload NCRB_ADSI_2023_Table_1A_6.csv")

    # Default path fallback
    DEFAULT_PATH = "NCRB_ADSI_2023_Table_1A_6.csv"
    use_default  = os.path.exists(DEFAULT_PATH)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    n_clusters = st.slider("Number of Clusters (K-Means)", 2, 7, 4)
    show_cities = st.checkbox("Include Cities in Charts", value=True)
    selected_model = st.selectbox("Prediction Model", 
                                   ['Linear Regression','Ridge Regression',
                                    'Random Forest','Gradient Boosting'])
    st.markdown("---")
    st.caption("Data Source: NCRB ADSI 2023\nTable 1A.6 – Accidents by Time")

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
if uploaded_file is not None:
    import tempfile, shutil
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(uploaded_file, tmp)
        data_path = tmp.name
elif use_default:
    data_path = DEFAULT_PATH
else:
    st.warning("⚠️ Please upload the NCRB CSV file using the sidebar uploader.")
    st.stop()

with st.spinner("🔄 Loading data and training models..."):
    (df_states, df_cities, time_slots, road_total_col,
     inertias, label_map, results, best_model_name,
     feat_imp, y_test, X_test, scaler, km_final) = load_and_train(data_path)

# Re-run clustering with sidebar k if changed from 4
if n_clusters != 4:
    X_c = StandardScaler().fit_transform(df_states[time_slots].values)
    km2 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_states['Cluster'] = km2.fit_predict(X_c)
    centroids2 = StandardScaler().fit(df_states[time_slots].values).inverse_transform(km2.cluster_centers_)
    cent2 = pd.DataFrame(centroids2, columns=time_slots)
    nm2 = cent2[['0000-0300','0300-0600','1800-2100','2100-2400']].mean(axis=1)
    lm2 = {}
    sc2 = nm2.argsort()
    risk_labels = ['Very Low Risk','Low Risk','Moderate Risk','High Risk',
                   'Very High Risk','Extreme Risk','Critical Risk']
    for i, idx in enumerate(sc2):
        lm2[int(idx)] = risk_labels[i]
    df_states['Cluster_Label'] = df_states['Cluster'].map(lm2)

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg,#1A237E,#283593); 
            padding:1.5rem 2rem; border-radius:14px; margin-bottom:1.5rem;'>
  <h1 style='color:white;margin:0;font-size:1.8rem;'>
    🚦 NCRB 2023 — Traffic Accident ML Analysis
  </h1>
  <p style='color:#B3C5FF;margin:4px 0 0;font-size:0.95rem;'>
    Machine Learning Dashboard · Accidental Deaths & Suicides in India
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────────────────
total_accidents = int(df_states[road_total_col].sum())
total_by_slot   = df_states[time_slots].sum()
peak_slot       = total_by_slot.idxmax()
night_pct       = (df_states['Night_Accidents'].sum() / total_accidents * 100)
top_state       = df_states.loc[df_states[road_total_col].idxmax(), 'State/UT']
best_r2         = results[best_model_name]['CV_R2']

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, "🗺️ States/UTs", f"{len(df_states)}", "#1A237E"),
    (c2, "🏙️ Cities",      f"{len(df_cities)}", "#1565C0"),
    (c3, "💥 Total Accidents", f"{total_accidents:,}", "#B71C1C"),
    (c4, "🌙 Night Share",  f"{night_pct:.1f}%",  "#E65100"),
    (c5, "🤖 Best Model R²", f"{best_r2:.3f}",   "#1B5E20"),
]
for col, label, val, color in metrics:
    col.markdown(f"""
    <div class='metric-card' style='border-color:{color}'>
      <h3>{label}</h3><p style='color:{color}'>{val}</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🗺️ Clustering", "🤖 ML Models",
    "🔮 Predictor", "📋 Data Explorer"
])

# ═══════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>Time-of-Day Accident Distribution</div>", unsafe_allow_html=True)

    # Time slot bar chart
    fig_time = go.Figure()
    fig_time.add_trace(go.Bar(
        x=time_slots,
        y=total_by_slot.values,
        marker_color=TIME_COLORS,
        text=[f"{int(v):,}" for v in total_by_slot.values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Accidents: %{y:,}<extra></extra>'
    ))
    # Night shading
    fig_time.add_vrect(x0=-0.5, x1=1.5, fillcolor="navy", opacity=0.07,
                       annotation_text="Night", annotation_position="top left")
    fig_time.add_vrect(x0=5.5, x1=7.5, fillcolor="navy", opacity=0.07,
                       annotation_text="Night", annotation_position="top right")
    fig_time.update_layout(
        title="Total Road Accidents by Time Slot — All States/UTs (2023)",
        xaxis_title="Time Slot", yaxis_title="Number of Accidents",
        plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False, height=400
    )
    st.plotly_chart(fig_time, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Top 15 States by Total Accidents</div>", unsafe_allow_html=True)
        top15 = df_states.nlargest(15, road_total_col)
        color_map = {'Low Night Risk':'#1976D2','Moderate Risk':'#388E3C',
                     'High Night Risk':'#F57C00','Extreme Night Risk':'#7B1FA2'}
        fig_top = px.bar(
            top15, x=road_total_col, y='State/UT',
            color='Cluster_Label',
            color_discrete_map=color_map,
            orientation='h', text=road_total_col,
            labels={road_total_col: 'Total Accidents', 'State/UT': ''}
        )
        fig_top.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_top.update_layout(height=450, plot_bgcolor='white',
                              paper_bgcolor='white', showlegend=True,
                              legend_title='Risk Cluster')
        fig_top.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Night vs Day Accidents (States)</div>", unsafe_allow_html=True)
        fig_nd = px.scatter(
            df_states, x='Day_Accidents', y='Night_Accidents',
            color='Cluster_Label', hover_name='State/UT',
            color_discrete_map=color_map,
            size=road_total_col,
            labels={'Day_Accidents':'Daytime Accidents','Night_Accidents':'Nighttime Accidents'},
            size_max=40
        )
        fig_nd.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_nd, use_container_width=True)

    if show_cities:
        st.markdown("<div class='section-title'>Top 15 Cities by Total Accidents</div>", unsafe_allow_html=True)
        top15c = df_cities.nlargest(15, road_total_col)
        fig_city = px.bar(
            top15c, x='State/UT', y=road_total_col,
            color=road_total_col, color_continuous_scale='Reds',
            labels={road_total_col:'Total Accidents','State/UT':'City'},
            text=road_total_col
        )
        fig_city.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_city.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                               xaxis_tickangle=-40)
        st.plotly_chart(fig_city, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2: CLUSTERING
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>K-Means Clustering of States by Accident Time Pattern</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Elbow curve
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(range(2, 8)), y=inertias,
            mode='lines+markers',
            line=dict(color='#E91E63', width=2),
            marker=dict(size=10)
        ))
        fig_elbow.add_vline(x=n_clusters, line_dash='dash', line_color='#FF9800',
                            annotation_text=f"k={n_clusters}", annotation_position="top right")
        fig_elbow.update_layout(
            title="Elbow Curve — Optimal K",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia",
            plot_bgcolor='white', paper_bgcolor='white', height=350
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col2:
        # Pie distribution
        cluster_counts = df_states['Cluster_Label'].value_counts()
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index,
            color=cluster_counts.index,
            color_discrete_map=color_map,
            hole=0.4
        )
        fig_pie.update_layout(title="State Distribution Across Clusters",
                              height=350, paper_bgcolor='white')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Cluster heatmap
    st.markdown("<div class='section-title'>Cluster Profiles — Average Accidents per Time Slot</div>",
                unsafe_allow_html=True)
    cluster_profiles = df_states.groupby('Cluster_Label')[time_slots].mean().round(0)
    fig_heat = px.imshow(
        cluster_profiles,
        text_auto=True, color_continuous_scale='YlOrRd',
        labels=dict(x='Time Slot', y='Cluster', color='Avg Accidents'),
        aspect='auto'
    )
    fig_heat.update_layout(height=300, paper_bgcolor='white')
    st.plotly_chart(fig_heat, use_container_width=True)

    # Cluster member table
    st.markdown("<div class='section-title'>States in Each Cluster</div>", unsafe_allow_html=True)
    risk_colors_ui = {
        'Low Night Risk': '#E3F2FD',
        'Moderate Risk': '#E8F5E9',
        'High Night Risk': '#FFF3E0',
        'Extreme Night Risk': '#FCE4EC'
    }
    cluster_groups = df_states.groupby('Cluster_Label')['State/UT'].apply(list).reset_index()
    for _, row in cluster_groups.iterrows():
        lbl = row['Cluster_Label']
        bg  = risk_colors_ui.get(lbl, '#F5F5F5')
        states_str = ' · '.join(row['State/UT'])
        st.markdown(f"""
        <div style='background:{bg};border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.5rem;'>
          <strong>{lbl}</strong> ({len(row['State/UT'])} states)<br>
          <span style='font-size:0.88rem;color:#444'>{states_str}</span>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TAB 3: ML MODELS
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Model Comparison (5-Fold Cross-Validated R²)</div>",
                unsafe_allow_html=True)

    model_names = list(results.keys())
    cv_scores   = [results[m]['CV_R2']  for m in model_names]
    test_r2s    = [results[m]['R2']     for m in model_names]
    maes        = [results[m]['MAE']    for m in model_names]
    rmses       = [results[m]['RMSE']   for m in model_names]

    col1, col2 = st.columns(2)

    with col1:
        bar_colors = ['#4CAF50' if n == selected_model else '#90CAF9' for n in model_names]
        fig_models = go.Figure(go.Bar(
            x=cv_scores, y=model_names, orientation='h',
            marker_color=bar_colors,
            text=[f"{v:.4f}" for v in cv_scores],
            textposition='outside'
        ))
        fig_models.update_layout(
            title="Cross-Validated R² Score",
            xaxis=dict(range=[0, 1.05]),
            plot_bgcolor='white', paper_bgcolor='white', height=300
        )
        st.plotly_chart(fig_models, use_container_width=True)

    with col2:
        # Metrics table
        metrics_df = pd.DataFrame({
            'Model': model_names,
            'CV R²': [f"{v:.4f}" for v in cv_scores],
            'Test R²': [f"{v:.4f}" for v in test_r2s],
            'MAE': [f"{v:.1f}" for v in maes],
            'RMSE': [f"{v:.1f}" for v in rmses]
        })
        st.dataframe(metrics_df.set_index('Model'), use_container_width=True, height=200)
        best_res = results[selected_model]
        st.info(f"**{selected_model}** — CV R²: {best_res['CV_R2']:.4f} | "
                f"MAE: {best_res['MAE']:.1f} | RMSE: {best_res['RMSE']:.1f}")

    # Actual vs Predicted
    st.markdown("<div class='section-title'>Actual vs Predicted — Selected Model</div>",
                unsafe_allow_html=True)
    sel_preds = results[selected_model]['predictions']
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        x=y_test, y=sel_preds, mode='markers',
        marker=dict(color='#5C6BC0', size=9, opacity=0.75,
                    line=dict(width=1, color='white')),
        name='Predictions',
        hovertemplate='Actual: %{x:,}<br>Predicted: %{y:,}<extra></extra>'
    ))
    max_val = max(float(y_test.max()), float(sel_preds.max())) * 1.05
    fig_avp.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines', line=dict(color='red', dash='dash', width=1.5),
        name='Perfect Fit'
    ))
    fig_avp.update_layout(
        xaxis_title='Actual Accidents', yaxis_title='Predicted Accidents',
        plot_bgcolor='white', paper_bgcolor='white', height=400,
        annotations=[dict(
            x=0.05, y=0.92, xref='paper', yref='paper',
            text=f"R² = {results[selected_model]['R2']:.4f}",
            showarrow=False, font=dict(size=14, color='#1A237E'),
            bgcolor='white', bordercolor='#1A237E', borderwidth=1
        )]
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    # Feature Importance
    st.markdown("<div class='section-title'>Random Forest Feature Importance</div>",
                unsafe_allow_html=True)
    fig_fi = go.Figure(go.Bar(
        x=feat_imp.values, y=feat_imp.index, orientation='h',
        marker_color='#FF7043',
        text=[f"{v:.3f}" for v in feat_imp.values],
        textposition='outside'
    ))
    fig_fi.update_layout(
        xaxis_title='Importance Score',
        yaxis_title='Time Slot',
        plot_bgcolor='white', paper_bgcolor='white', height=320,
        yaxis={'autorange': 'reversed'}
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 4: PREDICTOR
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>🔮 Predict Total Road Accidents from Time Slot Inputs</div>",
                unsafe_allow_html=True)
    st.info("Enter the number of accidents in each time slot (except 2100–2400) "
            "to predict the total accidents using the selected model.")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        input_vals = {}
        for ts in time_slots[:-1]:
            input_vals[ts] = st.number_input(
                f"Accidents during {ts}", min_value=0, max_value=50000,
                value=500, step=10, key=f"input_{ts}"
            )

    with col_right:
        X_input = np.array([[input_vals[ts] for ts in time_slots[:-1]]])
        model_obj = results[selected_model]['model']
        prediction = float(model_obj.predict(X_input)[0])

        # Night/day split from inputs
        night_input = sum(input_vals[s] for s in ['0000-0300','0300-0600','1800-2100'])
        day_input   = sum(input_vals[s] for s in ['0600-0900','0900-1200','1200-1500','1500-1800'])
        peak_input  = max(input_vals, key=input_vals.get)

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1A237E,#283593);
                    border-radius:14px;padding:2rem;color:white;margin-top:1rem;'>
          <h3 style='margin:0;font-size:1.1rem;opacity:0.85'>Predicted Total Accidents</h3>
          <p style='font-size:3rem;font-weight:800;margin:8px 0;'>{prediction:,.0f}</p>
          <hr style='border-color:rgba(255,255,255,0.2);'>
          <p>📊 Model: <b>{selected_model}</b></p>
          <p>🌙 Nighttime Input: <b>{night_input:,}</b></p>
          <p>☀️ Daytime Input: <b>{day_input:,}</b></p>
          <p>⚡ Peak Slot: <b>{peak_input}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Radar chart of inputs
        fig_radar = go.Figure(go.Scatterpolar(
            r=[input_vals[ts] for ts in time_slots[:-1]] + [input_vals[time_slots[0]]],
            theta=time_slots[:-1] + [time_slots[0]],
            fill='toself',
            line_color='#FF5722',
            fillcolor='rgba(255,87,34,0.2)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=False, height=320,
            title="Input Distribution (Radar)",
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Compare against real states
    st.markdown("<div class='section-title'>How Does Your Input Compare to Real States?</div>",
                unsafe_allow_html=True)
    df_compare = df_states[['State/UT', road_total_col]].copy()
    df_compare['Difference'] = (df_compare[road_total_col] - prediction).abs()
    closest = df_compare.nsmallest(5, 'Difference')[['State/UT', road_total_col]]
    closest.columns = ['State/UT', 'Actual Accidents']
    closest['Your Prediction'] = int(prediction)
    st.dataframe(closest.set_index('State/UT'), use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 5: DATA EXPLORER
# ═══════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-title'>📋 Raw Data Explorer</div>", unsafe_allow_html=True)

    dataset_choice = st.radio("View Dataset", ["States/UTs", "Cities"], horizontal=True)
    df_view = df_states if dataset_choice == "States/UTs" else df_cities

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        min_acc = st.number_input("Min Total Accidents", 0, int(df_view[road_total_col].max()), 0)
    with col2:
        if dataset_choice == "States/UTs":
            cluster_filter = st.multiselect("Filter by Cluster", df_states['Cluster_Label'].unique(),
                                             default=list(df_states['Cluster_Label'].unique()))
        else:
            cluster_filter = None

    df_filtered = df_view[df_view[road_total_col] >= min_acc]
    if dataset_choice == "States/UTs" and cluster_filter:
        df_filtered = df_filtered[df_filtered['Cluster_Label'].isin(cluster_filter)]

    display_cols = ['State/UT'] + time_slots + [road_total_col, 'Night_Pct', 'Peak_Slot']
    if dataset_choice == "States/UTs":
        display_cols += ['Cluster_Label']
    display_cols = [c for c in display_cols if c in df_filtered.columns]

    st.dataframe(
        df_filtered[display_cols].set_index('State/UT').round(1),
        use_container_width=True, height=400
    )
    st.caption(f"Showing {len(df_filtered)} rows")

    # Download button
    csv_out = df_filtered[display_cols].to_csv(index=False)
    st.download_button("⬇️ Download Filtered Data as CSV", csv_out,
                       file_name="ncrb_filtered.csv", mime="text/csv")

    # Correlation heatmap
    st.markdown("<div class='section-title'>Time Slot Correlation Matrix</div>", unsafe_allow_html=True)
    corr = df_states[time_slots].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r',
                         zmin=-1, zmax=1, aspect='auto')
    fig_corr.update_layout(height=400, paper_bgcolor='white')
    st.plotly_chart(fig_corr, use_container_width=True)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;font-size:0.85rem;padding:1rem 0;'>
  🚦 NCRB 2023 Traffic Accident ML Dashboard &nbsp;|&nbsp; 
  Data: Accidental Deaths & Suicides in India (ADSI) 2023 &nbsp;|&nbsp;
  Built with Streamlit + scikit-learn + Plotly
</div>
""", unsafe_allow_html=True)
