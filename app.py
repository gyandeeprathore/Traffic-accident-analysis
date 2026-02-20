import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import joblib, os

# ── Page Config ──
st.set_page_config(page_title="Traffic Accident Analysis", page_icon="🚦", layout="wide")

# ── Load & Clean Data ──
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[~df['State/UT'].str.contains('Total', na=False)]
    road_cols = [c for c in df.columns if c.startswith('Road Accidents -')]
    target    = 'Total (Traffic Accidents)'
    keep      = ['State/UT'] + road_cols + ['Railway Accidents - Total',
                 'Railway Crossing Accidents - Total', target]
    df = df[keep].copy()
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=[target], inplace=True)
    df = df[df[target] > 0].reset_index(drop=True)
    df.fillna(0, inplace=True)
    return df

@st.cache_resource
def train_models(df):
    target    = 'Total (Traffic Accidents)'
    feat_cols = [c for c in df.columns if c not in ['State/UT', target]]
    le        = LabelEncoder()
    state_enc = le.fit_transform(df['State/UT']).reshape(-1, 1)
    X         = np.hstack([state_enc, df[feat_cols].values])
    y         = df[target].values
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    models = {
        'Linear Regression' : LinearRegression(),
        'Ridge Regression'  : Ridge(),
        'Lasso Regression'  : Lasso(),
        'Decision Tree'     : DecisionTreeRegressor(random_state=42),
        'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting' : GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Extra Trees'       : ExtraTreesRegressor(n_estimators=100, random_state=42),
        'SVR'               : SVR(kernel='rbf', C=100, gamma=0.1),
        'KNN'               : KNeighborsRegressor(n_neighbors=5),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'model'  : model,
            'y_pred' : y_pred,
            'R2'     : round(r2_score(y_test, y_pred), 4),
            'MAE'    : round(mean_absolute_error(y_test, y_pred), 2),
            'RMSE'   : round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            'CV_R2'  : round(cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean(), 4),
        }
    return results, X_train, X_test, y_train, y_test, scaler, le, feat_cols

# ── Load ──
csv_path = 'NCRB_ADSI_2023_Table_1A_6.csv'
uploaded = st.sidebar.file_uploader("Upload CSV", type='csv')
if uploaded:
    df = load_data(uploaded)
elif os.path.exists(csv_path):
    df = load_data(csv_path)
else:
    st.error("Please upload the CSV file.")
    st.stop()

TARGET = 'Total (Traffic Accidents)'
results, X_train, X_test, y_train, y_test, scaler, le, feat_cols = train_models(df)

ROAD_COLS = [c for c in feat_cols if c.startswith('Road Accidents -') and c != 'Road Accidents - Total']
TIME_LABELS = ['00-03','03-06','06-09','09-12','12-15','15-18','18-21','21-24']

# ── Sidebar Nav ──
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Model Comparison", "Predict"])

# ════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════
if page == "Overview":
    st.title("🚦 India Traffic Accident Analysis")
    st.caption("NCRB · ADSI 2023 · Table 1A-6")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Accidents",  f"{int(df[TARGET].sum()):,}")
    c2.metric("States / UTs",     len(df))
    c3.metric("Highest State",    df.loc[df[TARGET].idxmax(), 'State/UT'],
                                  f"{int(df[TARGET].max()):,}")
    c4.metric("Best Model R2",    max(v['R2'] for v in results.values()))

    st.divider()
    st.subheader("Top 10 States by Total Accidents")
    top10 = df[['State/UT', TARGET]].sort_values(TARGET, ascending=False).head(10)
    fig = px.bar(top10, x='State/UT', y=TARGET, color=TARGET,
                 color_continuous_scale='Reds',
                 labels={TARGET: 'Accidents', 'State/UT': ''})
    fig.update_coloraxes(showscale=False)
    fig.update_layout(margin=dict(t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Accident Type Breakdown")
    pie_data = {
        'Road'             : df['Road Accidents - Total'].sum(),
        'Railway'          : df['Railway Accidents - Total'].sum(),
        'Railway Crossing' : df['Railway Crossing Accidents - Total'].sum(),
    }
    fig2 = px.pie(values=list(pie_data.values()), names=list(pie_data.keys()),
                  hole=0.5, color_discrete_sequence=['#e63946','#457b9d','#2a9d8f'])
    fig2.update_layout(margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════
# PAGE: EDA
# ════════════════════════════
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.divider()

    st.subheader("Road Accidents by Time Slot (All India)")
    slot_totals = [df[c].sum() for c in ROAD_COLS]
    fig = px.bar(x=TIME_LABELS, y=slot_totals,
                 labels={'x': 'Time Slot', 'y': 'Accidents'},
                 color=slot_totals, color_continuous_scale='Oranges')
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("State-wise Time Slot Breakdown")
    states   = sorted(df['State/UT'].unique())
    selected = st.multiselect("Select States", states,
                              default=['Tamil Nadu', 'Uttar Pradesh', 'Maharashtra'])
    if selected:
        fig2   = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, state in enumerate(selected):
            row  = df[df['State/UT'] == state].iloc[0]
            vals = [row[c] for c in ROAD_COLS]
            fig2.add_trace(go.Scatter(x=TIME_LABELS, y=vals, name=state,
                                      mode='lines+markers',
                                      line=dict(color=colors[i % len(colors)], width=2.5),
                                      marker=dict(size=7)))
        fig2.update_layout(xaxis_title='Time Slot', yaxis_title='Road Accidents',
                           legend=dict(orientation='h', y=-0.2), margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Correlation Explorer")
    col1, col2 = st.columns(2)
    num_cols = [c for c in df.columns if c != 'State/UT']
    x_col = col1.selectbox("X Axis", num_cols, index=num_cols.index('Road Accidents - Total'))
    y_col = col2.selectbox("Y Axis", num_cols, index=num_cols.index(TARGET))
    fig3  = px.scatter(df, x=x_col, y=y_col, hover_name='State/UT',
                       color=TARGET, color_continuous_scale='Blues',
                       size=TARGET, size_max=30)
    fig3.update_coloraxes(showscale=False)
    fig3.update_layout(margin=dict(t=10))
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════
# PAGE: MODEL COMPARISON
# ════════════════════════════
elif page == "Model Comparison":
    st.title("🤖 Model Comparison")
    st.divider()

    comp_df = pd.DataFrame([
        {'Model': k, 'R2': v['R2'], 'CV R2': v['CV_R2'], 'MAE': v['MAE'], 'RMSE': v['RMSE']}
        for k, v in results.items()
    ]).sort_values('R2', ascending=False).reset_index(drop=True)

    st.subheader("Performance Table")
    st.dataframe(
        comp_df.style
            .highlight_max(subset=['R2', 'CV R2'], color='#d4edda')
            .highlight_min(subset=['MAE', 'RMSE'], color='#d4edda')
            .format({'R2': '{:.4f}', 'CV R2': '{:.4f}', 'MAE': '{:,.2f}', 'RMSE': '{:,.2f}'}),
        use_container_width=True, hide_index=True
    )

    st.divider()
    st.subheader("Visual Comparison")
    metric    = st.selectbox("Metric", ['R2', 'MAE', 'RMSE', 'CV R2'])
    ascending = metric in ['MAE', 'RMSE']
    sorted_df = comp_df.sort_values(metric, ascending=ascending)
    fig = px.bar(sorted_df, x='Model', y=metric, color=metric,
                 color_continuous_scale='Blues' if not ascending else 'Reds_r', text=metric)
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_coloraxes(showscale=False)
    fig.update_layout(margin=dict(t=10, b=0), xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Actual vs Predicted")
    model_pick = st.selectbox("Select Model", list(results.keys()))
    y_pred     = results[model_pick]['y_pred']
    fig2       = go.Figure()
    fig2.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                              marker=dict(size=9, opacity=0.75, color='#457b9d'),
                              hovertemplate='Actual: %{x:,}<br>Predicted: %{y:,}<extra></extra>',
                              name='Predictions'))
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    fig2.add_trace(go.Scatter(x=lims, y=lims, mode='lines',
                              line=dict(color='#e63946', dash='dash', width=2), name='Perfect Fit'))
    fig2.update_layout(xaxis_title='Actual', yaxis_title='Predicted',
                       title=f'{model_pick} — R2 = {results[model_pick]["R2"]}',
                       margin=dict(t=40))
    st.plotly_chart(fig2, use_container_width=True)

    if hasattr(results[model_pick]['model'], 'feature_importances_'):
        st.divider()
        st.subheader("Feature Importance")
        feat_names = ['State'] + feat_cols
        fi = pd.DataFrame({'Feature': feat_names,
                           'Importance': results[model_pick]['model'].feature_importances_})
        fi = fi.sort_values('Importance', ascending=True).tail(10)
        fig3 = px.bar(fi, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale='Blues')
        fig3.update_coloraxes(showscale=False)
        fig3.update_layout(margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    best_name = comp_df.iloc[0]['Model']
    if st.button(f"Save Best Model ({best_name})"):
        os.makedirs('saved_model', exist_ok=True)
        joblib.dump(results[best_name]['model'], 'saved_model/best_model.pkl')
        joblib.dump(scaler, 'saved_model/scaler.pkl')
        joblib.dump(le, 'saved_model/label_encoder.pkl')
        st.success(f"Saved to saved_model/ — R2 = {results[best_name]['R2']}")

# ════════════════════════════
# PAGE: PREDICT
# ════════════════════════════
elif page == "Predict":
    st.title("🔮 Predict Traffic Accidents")
    st.caption("Select a state, adjust time-slot values, and hit Predict.")
    st.divider()

    col_l, col_r = st.columns([1, 1])

    with col_l:
        state      = st.selectbox("State / UT", sorted(df['State/UT'].unique()))
        model_name = st.selectbox("Model", list(results.keys()),
                                  index=list(results.keys()).index(
                                      max(results, key=lambda k: results[k]['R2'])))

        st.markdown("**Road Accidents by Time Slot**")
        row      = df[df['State/UT'] == state].iloc[0]
        defaults = [int(row[c]) for c in ROAD_COLS]
        slot_vals = []
        for label, default in zip(TIME_LABELS, defaults):
            v = st.number_input(label, 0, 100000, default, step=50)
            slot_vals.append(v)

        road_total = sum(slot_vals)
        rail_total = st.number_input("Railway Accidents", 0, 50000,
                                     int(row.get('Railway Accidents - Total', 0)), step=10)
        rc_total   = st.number_input("Railway Crossing Accidents", 0, 5000,
                                     int(row.get('Railway Crossing Accidents - Total', 0)), step=5)

    with col_r:
        st.markdown("#### Prediction")
        if st.button("Predict", use_container_width=True):
            enc  = le.transform([state])[0]
            x_in = np.array([[enc] + slot_vals + [road_total, rail_total, rc_total]])
            x_sc = scaler.transform(x_in)
            pred = int(results[model_name]['model'].predict(x_sc)[0])
            actual = int(row[TARGET])
            delta  = pred - actual

            st.metric("Predicted Accidents", f"{pred:,}", delta=f"{delta:+,} vs actual")
            st.caption(f"Actual 2023 value for {state}: {actual:,}")

            st.divider()
            fig = px.bar(x=TIME_LABELS, y=slot_vals,
                         labels={'x': 'Time Slot', 'y': 'Accidents'},
                         color=slot_vals, color_continuous_scale='Reds',
                         title="Input Time Slot Breakdown")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"**{model_name}** — R2 = {results[model_name]['R2']} · "
                    f"MAE = {results[model_name]['MAE']:,.0f}")
