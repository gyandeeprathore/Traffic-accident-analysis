"""
NCRB 2023 — Traffic Accident Explorer & Predictor
Data is embedded directly. No CSV upload needed.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings, io
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NCRB 2023 — Traffic Accident Explorer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .main { background: #F0F4FF; }
  .block-container { padding-top: 1rem; padding-bottom: 2rem; }
  .kpi-box {
    background: white; border-radius: 14px; padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 10px rgba(26,35,126,0.10);
    border-left: 5px solid #3949AB; text-align: center;
  }
  .kpi-box .val { font-size: 2rem; font-weight: 800; color: #1A237E; margin: 0; }
  .kpi-box .lbl { font-size: 0.78rem; color: #666; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; }
  .risk-pill {
    display: inline-block; padding: 4px 16px; border-radius: 30px;
    font-weight: 700; font-size: 0.95rem;
  }
  .pred-card {
    background: linear-gradient(135deg, #1A237E 0%, #3949AB 100%);
    border-radius: 16px; padding: 1.5rem 2rem; color: white;
    box-shadow: 0 4px 20px rgba(26,35,126,0.3);
  }
  .pred-card .big { font-size: 2.8rem; font-weight: 900; margin: 6px 0; }
  .pred-card .sub { font-size: 0.88rem; opacity: 0.8; }
  .section-head {
    font-size: 1.05rem; font-weight: 700; color: #1A237E;
    border-bottom: 2px solid #C5CAE9; padding-bottom: 5px; margin: 1.2rem 0 0.8rem;
  }
  div[data-testid="stSidebar"] { background: #1A237E !important; }
  div[data-testid="stSidebar"] label,
  div[data-testid="stSidebar"] p,
  div[data-testid="stSidebar"] span,
  div[data-testid="stSidebar"] div { color: white !important; }
  div[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# EMBEDDED DATA
# ─────────────────────────────────────────────────────────
RAW_DATA = """Sl. No.,State/UT,0000-0300,0300-0600,0600-0900,0900-1200,1200-1500,1500-1800,1800-2100,2100-2400,Total
1,Andhra Pradesh,991,1143,2256,2704,3058,3403,4255,2139,19949
2,Arunachal Pradesh,18,13,23,22,36,55,46,20,233
3,Assam,346,389,889,1025,1051,1327,1294,1100,7421
4,Bihar,701,937,1674,1606,1525,1818,1803,950,11014
5,Chhattisgarh,464,406,1067,1990,2211,2534,3279,1518,13469
6,Goa,213,152,255,426,428,482,542,349,2847
7,Gujarat,910,700,1597,2373,2527,2719,3505,2018,16349
8,Haryana,943,953,1243,1529,1447,1534,2007,1427,11083
9,Himachal Pradesh,133,82,182,273,321,430,511,323,2255
10,Jharkhand,304,314,617,700,824,982,1111,464,5316
11,Karnataka,2312,2370,4005,5908,6706,7545,9265,5328,43439
12,Kerala,1103,1495,5411,8282,7063,8937,9542,4148,45981
13,Madhya Pradesh,3330,2714,4468,7476,9217,10227,10613,6718,54763
14,Maharashtra,2065,2000,3005,4288,4612,4892,6368,4117,31347
15,Manipur,12,5,27,50,58,90,113,43,398
16,Meghalaya,11,10,20,39,32,48,46,26,232
17,Mizoram,4,3,4,6,4,12,9,5,47
18,Nagaland,2,5,3,5,5,6,4,5,35
19,Odisha,735,916,1391,1773,1782,2054,2162,1176,11989
20,Punjab,337,382,651,857,780,921,1461,887,6276
21,Rajasthan,980,1195,2605,3503,3947,4764,5229,2638,24861
22,Sikkim,9,9,15,28,31,34,28,20,174
23,Tamil Nadu,1700,2792,7535,9629,9770,11723,17159,6905,67213
24,Telangana,1668,1296,2129,2930,3437,4198,4577,2668,22903
25,Tripura,22,12,40,98,101,123,114,67,577
26,Uttar Pradesh,3592,3767,4309,4222,4521,5374,6864,5115,37764
27,Uttarakhand,142,108,235,223,196,249,320,218,1691
28,West Bengal,1016,989,1245,1414,1502,1432,1506,1296,10400
29,Andaman and Nicobar Islands,0,1,1,4,8,1,6,4,25
30,Chandigarh,18,8,17,25,25,27,34,28,182
31,Dadra and Nagar Haveli and Daman and Diu,8,4,16,16,16,37,46,40,183
32,Delhi,618,555,652,737,621,840,839,853,5715
33,Jammu and Kashmir,223,252,611,1141,1255,1370,1004,444,6300
34,Ladakh,0,1,21,60,90,79,27,11,289
35,Lakshadweep,0,0,0,0,0,0,0,1,1
36,Puducherry,39,42,157,188,190,215,295,182,1308
37,Agra,56,72,72,53,54,65,116,65,553
38,Ahmedabad,90,53,159,299,255,247,370,294,1767
39,Amritsar,12,10,17,19,9,25,26,29,147
40,Asansol,52,43,29,52,43,47,66,69,401
41,Aurangabad,51,34,50,65,78,79,113,90,560
42,Bengaluru,383,270,591,751,725,738,810,712,4980
43,Bhopal,291,56,201,290,394,555,506,613,2906
44,Chandigarh (City),18,8,17,25,25,27,34,28,182
45,Chennai,110,249,400,586,733,469,530,576,3653
46,Coimbatore,44,42,174,179,156,155,223,188,1161
47,Delhi (City),618,555,652,737,621,840,839,853,5715
48,Dhanbad,20,23,27,32,28,36,51,40,257
49,Durg Bhilainagar,53,26,60,122,136,172,212,93,874
50,Faridabad,60,66,79,100,66,97,108,103,679
51,Ghaziabad,69,103,84,88,84,84,107,120,739
52,Gwalior,59,31,50,79,101,138,151,97,706
53,Hyderabad,154,111,192,263,303,386,411,265,2085
54,Indore,192,49,145,235,298,384,332,351,1986
55,Jabalpur,102,23,66,98,140,183,197,182,991
56,Jaipur,131,113,228,295,293,382,413,282,2137
57,Jamshedpur,45,35,57,64,65,72,88,52,478
58,Jodhpur,48,49,91,96,97,118,133,104,736
59,Kannur,51,63,268,330,267,310,367,152,1808
60,Kanpur,125,133,132,135,119,136,201,177,1158
61,Kochi,85,105,381,599,564,695,713,395,3537
62,Kolkata,141,149,128,137,133,141,127,158,1114
63,Kollam,69,90,287,397,321,405,471,200,2240
64,Kota,63,46,98,128,128,178,205,125,971
65,Kozhikode,89,104,418,620,516,636,720,298,3401
66,Lucknow,192,189,222,202,190,225,346,265,1831
67,Ludhiana,33,27,49,64,44,57,86,70,430
68,Madurai,39,42,127,162,152,152,181,124,979
69,Malappuram,84,116,382,575,477,573,665,252,3124
70,Meerut,72,98,90,90,94,107,141,133,825
71,Mumbai,54,52,36,56,65,43,52,56,414
72,Nagpur,93,40,109,195,195,160,202,219,1213
73,Nasik,31,28,33,35,29,27,41,37,261
74,Patna,19,30,47,39,33,49,43,62,322
75,Prayagraj,87,172,155,158,144,156,226,199,1297
76,Pune,56,47,48,66,67,79,69,70,502
77,Raipur,71,21,60,150,202,176,203,194,1077
78,Rajkot,25,21,42,70,88,72,117,59,494
79,Ranchi,12,23,26,41,42,47,43,29,263
80,Srinagar,0,0,2,180,190,71,24,9,476
81,Surat,27,18,79,94,88,122,131,91,650
82,Thiruvananthapuram,54,73,255,410,321,377,403,213,2106
83,Thrissur,63,80,324,415,343,432,559,228,2444
84,Tiruchirappalli,28,29,71,93,92,102,125,79,619
85,Vadodara,44,28,43,101,78,83,106,83,566
86,Varanasi,21,25,27,23,27,23,23,27,196
87,Vasai Virar,30,24,45,53,58,36,66,53,365
88,Vijayawada,86,74,159,218,236,228,295,226,1522
89,Vishakhapatnam,61,56,179,190,194,217,245,154,1297
"""

TIME_SLOTS  = ['0000-0300','0300-0600','0600-0900','0900-1200',
               '1200-1500','1500-1800','1800-2100','2100-2400']
TIME_LABELS = ['12AM–3AM','3AM–6AM','6AM–9AM','9AM–12PM',
               '12PM–3PM','3PM–6PM','6PM–9PM','9PM–12AM']
TIME_LABEL_MAP = dict(zip(TIME_SLOTS, TIME_LABELS))
NIGHT_SLOTS = ['0000-0300','0300-0600','1800-2100','2100-2400']
DAY_SLOTS   = ['0600-0900','0900-1200','1200-1500','1500-1800']
SLOT_COLORS = ['#3F51B5','#5C6BC0','#66BB6A','#FFA726',
               '#FF7043','#EF5350','#E53935','#880E4F']
RISK_META = {
    'Low Night Risk':     ('#E3F2FD','#1565C0','🟢'),
    'Moderate Risk':      ('#E8F5E9','#2E7D32','🟡'),
    'High Night Risk':    ('#FFF3E0','#E65100','🟠'),
    'Extreme Night Risk': ('#FCE4EC','#B71C1C','🔴'),
}

# ─────────────────────────────────────────────────────────
# LOAD & TRAIN (cached)
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv(io.StringIO(RAW_DATA))
    df['Sl. No.'] = df['Sl. No.'].astype(int)
    for ts in TIME_SLOTS:
        df[ts] = pd.to_numeric(df[ts], errors='coerce').fillna(0).astype(int)
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0).astype(int)
    df['Night_Total'] = df[NIGHT_SLOTS].sum(axis=1)
    df['Day_Total']   = df[DAY_SLOTS].sum(axis=1)
    df['Night_Pct']   = (df['Night_Total'] / df['Total'].replace(0, 1) * 100).round(1)
    df['Peak_Slot']   = df[TIME_SLOTS].idxmax(axis=1)
    df['Peak_Count']  = df[TIME_SLOTS].max(axis=1)
    df['Type']        = df['Sl. No.'].apply(lambda x: 'State/UT' if x <= 36 else 'City')

    # Clustering on states
    df_s = df[df['Type'] == 'State/UT'].copy()
    sc   = StandardScaler()
    Xs   = sc.fit_transform(df_s[TIME_SLOTS].values)
    km   = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_s['Cluster'] = km.fit_predict(Xs)
    cent = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=TIME_SLOTS)
    nm   = cent[NIGHT_SLOTS].mean(axis=1)
    order = nm.argsort()
    lmap  = {int(order.iloc[0]): 'Low Night Risk',
             int(order.iloc[1]): 'Moderate Risk',
             int(order.iloc[2]): 'High Night Risk',
             int(order.iloc[3]): 'Extreme Night Risk'}
    df_s['Risk_Cluster'] = df_s['Cluster'].map(lmap)
    df = df.merge(df_s[['State/UT','Risk_Cluster']], on='State/UT', how='left')

    # Train regression models on full dataset
    X_reg = df[TIME_SLOTS[:-1]].values
    y_reg = df['Total'].values
    models = {
        'Ridge Regression':  Ridge(alpha=1.0),
        'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
    }
    for m in models.values():
        m.fit(X_reg, y_reg)

    feat_imp = pd.Series(
        models['Random Forest'].feature_importances_, index=TIME_SLOTS[:-1]
    ).sort_values(ascending=False)

    return df, models, feat_imp

df, ml_models, feat_imp = load_and_train()
state_list = df[df['Type']=='State/UT']['State/UT'].tolist()
city_list  = df[df['Type']=='City']['State/UT'].tolist()

# ─────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────
def get_row(name):
    return df[df['State/UT'] == name].iloc[0]

def predict(vals_7, model_name):
    return float(ml_models[model_name].predict(np.array(vals_7).reshape(1,-1))[0])

def risk_badge(label):
    if not label or str(label) == 'nan':
        return ""
    bg, txt, dot = RISK_META.get(label, ('#F5F5F5','#333','⚪'))
    return f"<span class='risk-pill' style='background:{bg};color:{txt};'>{dot} {label}</span>"

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.8rem 0 1rem;'>
      <span style='font-size:2.8rem;'>🚦</span>
      <h2 style='margin:4px 0 2px;font-size:1.15rem;'>NCRB 2023</h2>
      <p style='font-size:0.8rem;opacity:0.75;margin:0;'>Traffic Accident Explorer</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    view_mode = st.radio("📍 View Mode", ["State / UT", "City", "Compare Two"], index=0)
    st.markdown("---")

    if view_mode == "State / UT":
        selected     = st.selectbox("🗺️ Select State / UT", state_list,
                                     index=state_list.index("Tamil Nadu"))
        compare_with = None
    elif view_mode == "City":
        selected     = st.selectbox("🏙️ Select City", city_list,
                                     index=city_list.index("Chennai"))
        compare_with = None
    else:
        selected     = st.selectbox("🗺️ First State / UT", state_list,
                                     index=state_list.index("Tamil Nadu"))
        compare_with = st.selectbox("🗺️ Second State / UT", state_list,
                                     index=state_list.index("Karnataka"))

    st.markdown("---")
    selected_slot = st.selectbox(
        "⏰ Focus Time Slot", TIME_SLOTS, index=6,
        format_func=lambda x: TIME_LABEL_MAP[x]
    )
    st.markdown("---")
    model_choice = st.selectbox("🤖 Prediction Model", list(ml_models.keys()))

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(120deg,#1A237E,#3949AB,#1565C0);
            padding:1.3rem 2rem;border-radius:16px;margin-bottom:1.2rem;'>
  <h1 style='color:white;margin:0;font-size:1.65rem;'>
    🚦 NCRB 2023 — Traffic Accident Explorer & Predictor
  </h1>
  <p style='color:#C5CAE9;margin:4px 0 0;font-size:0.88rem;'>
    Select a State, UT or City · Choose a time slot · See ML-powered predictions
  </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# COMPARE MODE
# ═══════════════════════════════════════════════════════
if view_mode == "Compare Two" and compare_with:
    row1 = get_row(selected)
    row2 = get_row(compare_with)
    st.markdown(f"## ⚖️ {selected}  vs  {compare_with}")

    # KPI cards
    c1, c2 = st.columns(2)
    for col, row, name in [(c1,row1,selected),(c2,row2,compare_with)]:
        clbl = str(row.get('Risk_Cluster',''))
        bg,txt,dot = RISK_META.get(clbl, ('#F5F5F5','#444','⚪'))
        pred_v = predict([int(row[ts]) for ts in TIME_SLOTS[:-1]], model_choice)
        with col:
            st.markdown(f"""
            <div style='background:{bg};border-radius:14px;padding:1.2rem 1.6rem;
                        border-left:6px solid {txt};margin-bottom:0.5rem;'>
              <h3 style='margin:0 0 4px;color:{txt};'>{name}</h3>
              <div style='display:flex;gap:2rem;flex-wrap:wrap;'>
                <div><p style='margin:0;font-size:1.9rem;font-weight:800;color:{txt};'>{int(row['Total']):,}</p>
                     <p style='margin:0;font-size:0.75rem;color:#666;'>Total Accidents</p></div>
                <div><p style='margin:0;font-size:1.9rem;font-weight:800;color:#1A237E;'>{pred_v:,.0f}</p>
                     <p style='margin:0;font-size:0.75rem;color:#666;'>ML Predicted</p></div>
                <div><p style='margin:0;font-size:1.9rem;font-weight:800;color:#E65100;'>{row['Night_Pct']}%</p>
                     <p style='margin:0;font-size:0.75rem;color:#666;'>Night Share</p></div>
              </div>
              <p style='margin:6px 0 0;font-size:0.85rem;'>{dot} {clbl if clbl and clbl!='nan' else 'City'}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-head'>Time Slot Comparison</div>", unsafe_allow_html=True)
    fig = go.Figure()
    for row, name, color in [(row1,selected,'#3949AB'),(row2,compare_with,'#E53935')]:
        fig.add_trace(go.Bar(
            name=name, x=TIME_LABELS,
            y=[int(row[ts]) for ts in TIME_SLOTS],
            marker_color=color,
            text=[f"{int(row[ts]):,}" for ts in TIME_SLOTS],
            textposition='outside'
        ))
    fig.update_layout(barmode='group', plot_bgcolor='white', paper_bgcolor='white',
                      height=380, legend=dict(orientation='h',yanchor='bottom',y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-head'>Radar — Accident Profile</div>", unsafe_allow_html=True)
    fig_r = go.Figure()
    for row, name, color in [(row1,selected,'#3949AB'),(row2,compare_with,'#E53935')]:
        vals = [int(row[ts]) for ts in TIME_SLOTS] + [int(row[TIME_SLOTS[0]])]
        fig_r.add_trace(go.Scatterpolar(
            r=vals, theta=TIME_LABELS+[TIME_LABELS[0]],
            fill='toself', name=name, line_color=color,
            fillcolor=color+'30'
        ))
    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True)),
                        height=420, paper_bgcolor='white')
    st.plotly_chart(fig_r, use_container_width=True)
    st.stop()

# ═══════════════════════════════════════════════════════
# SINGLE PLACE VIEW
# ═══════════════════════════════════════════════════════
row         = get_row(selected)
slot_vals   = [int(row[ts]) for ts in TIME_SLOTS]
max_sv      = max(slot_vals) if max(slot_vals) > 0 else 1
cluster_lbl = str(row.get('Risk_Cluster',''))
pred_val    = predict([int(row[ts]) for ts in TIME_SLOTS[:-1]], model_choice)
pred_err    = abs(pred_val - row['Total']) / (row['Total'] + 1) * 100
nat_avgs    = {ts: int(df[df['Type']=='State/UT'][ts].mean()) for ts in TIME_SLOTS}

# ── KPI Row ──
k1,k2,k3,k4,k5 = st.columns(5)
kpi_data = [
    (k1, f"{int(row['Total']):,}",          "Total Accidents",           "#B71C1C"),
    (k2, TIME_LABEL_MAP[row['Peak_Slot']],  "Deadliest Time Slot",       "#E65100"),
    (k3, f"{int(row['Peak_Count']):,}",     "Accidents at Peak",         "#1565C0"),
    (k4, f"{row['Night_Pct']}%",            "Night-time Share",          "#4A148C"),
    (k5, f"{pred_val:,.0f}",                f"ML Predicted Total",       "#1B5E20"),
]
for col, val, lbl, color in kpi_data:
    col.markdown(f"""
    <div class='kpi-box' style='border-color:{color};'>
      <p class='val' style='color:{color};'>{val}</p>
      <p class='lbl'>{lbl}</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Title + risk badge
badge = risk_badge(cluster_lbl) if cluster_lbl and cluster_lbl != 'nan' else ""
st.markdown(f"## {selected} &nbsp; {badge}", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# ROW 1: BAR CHART + SLOT BREAKDOWN
# ─────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.markdown("<div class='section-head'>Accidents by Time of Day</div>", unsafe_allow_html=True)
    bar_colors = [
        '#E53935' if ts == selected_slot else
        ('#3F51B5' if ts in ['0000-0300','0300-0600'] else
         ('#FF7043' if ts in ['1800-2100','2100-2400'] else '#66BB6A'))
        for ts in TIME_SLOTS
    ]
    fig_bar = go.Figure(go.Bar(
        x=TIME_LABELS, y=slot_vals,
        marker_color=bar_colors,
        text=[f"{v:,}" for v in slot_vals],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Accidents: %{y:,}<extra></extra>'
    ))
    sel_idx = TIME_SLOTS.index(selected_slot)
    fig_bar.add_annotation(
        x=TIME_LABELS[sel_idx], y=slot_vals[sel_idx],
        text="◀ Focus Slot", showarrow=False, yshift=28,
        font=dict(color='#E53935', size=10, family='Arial Black')
    )
    fig_bar.update_layout(
        plot_bgcolor='white', paper_bgcolor='white', height=340,
        showlegend=False, yaxis_title='Road Accidents',
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # vs National Average line
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=TIME_LABELS, y=[nat_avgs[ts] for ts in TIME_SLOTS],
        name='National Avg (States)', mode='lines+markers',
        line=dict(color='#90A4AE', dash='dash', width=2), marker_size=7
    ))
    fig_line.add_trace(go.Scatter(
        x=TIME_LABELS, y=slot_vals, name=selected,
        mode='lines+markers',
        line=dict(color='#3949AB', width=3), marker_size=9,
        fill='tonexty', fillcolor='rgba(57,73,171,0.08)'
    ))
    fig_line.update_layout(
        title=f"{selected} vs National Average",
        plot_bgcolor='white', paper_bgcolor='white', height=260,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(t=45, b=30)
    )
    st.plotly_chart(fig_line, use_container_width=True)

with right:
    st.markdown("<div class='section-head'>Slot-by-Slot Breakdown</div>", unsafe_allow_html=True)
    for ts, lbl, val, color in zip(TIME_SLOTS, TIME_LABELS, slot_vals, SLOT_COLORS):
        is_sel = ts == selected_slot
        border = "border:2px solid #E53935;" if is_sel else ""
        bg_row = "#FFF5F5" if is_sel else "white"
        pct    = val / max_sv * 100
        nat    = nat_avgs.get(ts, 0)
        vs_arrow = "▲" if val > nat else "▼"
        vs_color = "#C62828" if val > nat else "#2E7D32"
        st.markdown(f"""
        <div style='background:{bg_row};border-radius:10px;padding:8px 12px;
                    margin:4px 0;{border}'>
          <div style='display:flex;justify-content:space-between;align-items:center;'>
            <span style='font-size:0.8rem;font-weight:{"700" if is_sel else "500"};
                         color:{"#E53935" if is_sel else "#333"};'>{lbl}</span>
            <div style='text-align:right;'>
              <span style='font-size:0.95rem;font-weight:700;color:{color};'>{val:,}</span>
              <span style='font-size:0.72rem;color:{vs_color};margin-left:6px;'>{vs_arrow}{abs(val-nat):,}</span>
            </div>
          </div>
          <div style='background:#EEF0FB;border-radius:5px;height:7px;margin-top:5px;'>
            <div style='background:{color};width:{pct:.1f}%;height:7px;border-radius:5px;'></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Donut
    st.markdown("<div class='section-head'>Night vs Day Split</div>", unsafe_allow_html=True)
    fig_d = go.Figure(go.Pie(
        labels=['Night (6PM–6AM)', 'Day (6AM–6PM)'],
        values=[int(row['Night_Total']), int(row['Day_Total'])],
        hole=0.58, marker_colors=['#3F51B5','#FFA726'],
        textinfo='label+percent',
        hovertemplate='%{label}: %{value:,}<extra></extra>'
    ))
    fig_d.update_layout(
        height=240, paper_bgcolor='white', showlegend=False,
        margin=dict(t=10,b=10),
        annotations=[dict(text=f"<b>{int(row['Total']):,}</b><br>Total",
                          x=0.5, y=0.5, font_size=13, showarrow=False, font_color='#1A237E')]
    )
    st.plotly_chart(fig_d, use_container_width=True)

# ─────────────────────────────────────────────────────────
# ROW 2: SELECTED SLOT DEEP DIVE
# ─────────────────────────────────────────────────────────
st.markdown(f"<div class='section-head'>⏰ Focus: {TIME_LABEL_MAP[selected_slot]} Slot Deep Dive</div>",
            unsafe_allow_html=True)

s1,s2,s3,s4 = st.columns(4)
slot_count  = int(row[selected_slot])
slot_share  = slot_count / row['Total'] * 100 if row['Total'] > 0 else 0
nat_avg_slot= nat_avgs[selected_slot]
is_night    = selected_slot in NIGHT_SLOTS
rank_in_slot = int(df[selected_slot].rank(ascending=False)[df['State/UT']==selected].values[0])

for col, val, lbl, color in [
    (s1, f"{slot_count:,}",           "Accidents in Slot",                  "#1565C0"),
    (s2, f"{slot_share:.1f}%",        "Share of Total",                     "#4A148C"),
    (s3, f"{'▲' if slot_count>nat_avg_slot else '▼'} {abs(slot_count-nat_avg_slot):,}",
                                       f"vs Nat. Avg ({nat_avg_slot:,})",    "#E65100"),
    (s4, f"#{rank_in_slot}",           f"Rank in Slot (of {len(df)})",       "#1B5E20"),
]:
    col.markdown(f"""
    <div class='kpi-box' style='border-color:{color};'>
      <p class='val' style='color:{color};'>{val}</p>
      <p class='lbl'>{lbl}</p>
    </div>""", unsafe_allow_html=True)

# Top 10 leaderboard for this slot
st.markdown(f"**Top 10 places by accidents in {TIME_LABEL_MAP[selected_slot]} slot:**")
top10 = df.nlargest(10, selected_slot)[['State/UT','Type',selected_slot,'Total']].copy()
top10.columns = ['Place','Type',f'{TIME_LABEL_MAP[selected_slot]}','Total']
top10.index   = range(1,11)

# Highlight selected place if in top 10
def highlight_row(s):
    return ['background-color: #FFF3E0; font-weight: bold'
            if s['Place'] == selected else '' for _ in s]

st.dataframe(top10.style.apply(highlight_row, axis=1), use_container_width=True)

# ─────────────────────────────────────────────────────────
# ROW 3: ML PREDICTION + WHAT-IF
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-head'>🤖 ML Prediction & What-If Scenario</div>",
            unsafe_allow_html=True)

p1, p2 = st.columns([1, 2])

with p1:
    err_indicator = "✅ Accurate" if pred_err < 10 else ("⚠️ Moderate" if pred_err < 25 else "❌ High Error")
    st.markdown(f"""
    <div class='pred-card'>
      <div class='sub'>Model: <b>{model_choice}</b></div>
      <div class='sub' style='margin-top:2px;'>Predicted Total Accidents</div>
      <div class='big'>{pred_val:,.0f}</div>
      <hr style='border-color:rgba(255,255,255,0.25);margin:10px 0;'>
      <div style='display:flex;justify-content:space-between;'>
        <div><div class='sub'>Actual</div>
             <div style='font-size:1.25rem;font-weight:700;'>{int(row['Total']):,}</div></div>
        <div><div class='sub'>Error</div>
             <div style='font-size:1.25rem;font-weight:700;'>{pred_err:.1f}%</div></div>
        <div><div class='sub'>Quality</div>
             <div style='font-size:1rem;font-weight:700;'>{err_indicator}</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Radar chart
    st.markdown("")
    fig_rad = go.Figure(go.Scatterpolar(
        r=slot_vals + [slot_vals[0]],
        theta=TIME_LABELS + [TIME_LABELS[0]],
        fill='toself',
        line_color='#FF5722',
        fillcolor='rgba(255,87,34,0.15)',
        name=selected
    ))
    fig_rad.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        height=300, paper_bgcolor='white', showlegend=False,
        title=dict(text='Accident Profile Radar', font=dict(size=12,color='#1A237E')),
        margin=dict(t=40,b=10)
    )
    st.plotly_chart(fig_rad, use_container_width=True)

with p2:
    st.markdown("**🎛️ What-If Scenario** — Adjust any time slot and see how the prediction changes")

    wi_cols = st.columns(4)
    wi_vals = {}
    for i, ts in enumerate(TIME_SLOTS[:-1]):
        with wi_cols[i % 4]:
            wi_vals[ts] = st.number_input(
                TIME_LABEL_MAP[ts], min_value=0, max_value=30000,
                value=int(row[ts]), step=50, key=f"wi_{ts}"
            )

    wi_pred  = predict([wi_vals[ts] for ts in TIME_SLOTS[:-1]], model_choice)
    wi_delta = wi_pred - pred_val
    delta_col = "#B71C1C" if wi_delta > 0 else "#1B5E20"
    arrow     = "▲ +" if wi_delta > 0 else "▼ "

    st.markdown(f"""
    <div style='background:#E8EAF6;border-radius:14px;padding:1.2rem 1.6rem;
                margin-top:0.6rem;display:flex;align-items:center;justify-content:space-between;'>
      <div>
        <div style='font-size:0.8rem;color:#555;'>What-If Predicted Total</div>
        <div style='font-size:2.4rem;font-weight:900;color:#1A237E;'>{wi_pred:,.0f}</div>
        <div style='font-size:0.8rem;color:#777;'>Actual: {int(row["Total"]):,}</div>
      </div>
      <div style='text-align:right;'>
        <div style='font-size:0.8rem;color:#555;'>Change vs Baseline</div>
        <div style='font-size:1.8rem;font-weight:800;color:{delta_col};'>
          {arrow}{abs(wi_delta):,.0f}
        </div>
        <div style='font-size:0.8rem;color:#777;'>({abs(wi_delta/pred_val*100):.1f}% shift)</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Feature importance mini chart
    st.markdown("<br>**What drives the prediction? (RF Feature Importance)**")
    fig_fi = go.Figure(go.Bar(
        y=feat_imp.index, x=feat_imp.values, orientation='h',
        marker_color=['#E53935' if ts == selected_slot else '#90A4AE'
                      for ts in feat_imp.index],
        text=[f"{v:.3f}" for v in feat_imp.values],
        textposition='outside'
    ))
    fig_fi.update_layout(
        plot_bgcolor='white', paper_bgcolor='white', height=260,
        yaxis=dict(autorange='reversed'),
        xaxis_title='Importance', margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────────────────────────────────
# ROW 4: NATIONAL RANKING
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-head'>📊 Ranking Among All States/UTs & Cities</div>",
            unsafe_allow_html=True)

rank_total = int(df['Total'].rank(ascending=False)[df['State/UT']==selected].values[0])
total_rows = len(df)

r1, r2, r3 = st.columns(3)
r1.markdown(f"""
<div style='background:white;border-radius:12px;padding:1rem;text-align:center;
            box-shadow:0 2px 8px rgba(0,0,0,0.07);'>
  <div style='font-size:0.78rem;color:#666;text-transform:uppercase;'>Overall Rank</div>
  <div style='font-size:2.4rem;font-weight:900;color:#1A237E;'>
    #{rank_total} <span style='font-size:1rem;color:#999;'>/ {total_rows}</span>
  </div>
  <div style='font-size:0.8rem;color:#888;'>by total accidents</div>
</div>""", unsafe_allow_html=True)

r2.markdown(f"""
<div style='background:white;border-radius:12px;padding:1rem;text-align:center;
            box-shadow:0 2px 8px rgba(0,0,0,0.07);'>
  <div style='font-size:0.78rem;color:#666;text-transform:uppercase;'>Rank in Focus Slot</div>
  <div style='font-size:2.4rem;font-weight:900;color:#E65100;'>
    #{rank_in_slot} <span style='font-size:1rem;color:#999;'>/ {total_rows}</span>
  </div>
  <div style='font-size:0.8rem;color:#888;'>{TIME_LABEL_MAP[selected_slot]}</div>
</div>""", unsafe_allow_html=True)

peak_rank = int(df[row['Peak_Slot']].rank(ascending=False)[df['State/UT']==selected].values[0])
r3.markdown(f"""
<div style='background:white;border-radius:12px;padding:1rem;text-align:center;
            box-shadow:0 2px 8px rgba(0,0,0,0.07);'>
  <div style='font-size:0.78rem;color:#666;text-transform:uppercase;'>Rank in Peak Slot</div>
  <div style='font-size:2.4rem;font-weight:900;color:#4A148C;'>
    #{peak_rank} <span style='font-size:1rem;color:#999;'>/ {total_rows}</span>
  </div>
  <div style='font-size:0.8rem;color:#888;'>{TIME_LABEL_MAP[row["Peak_Slot"]]}</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# All-states bar (highlighted)
fig_all = go.Figure(go.Bar(
    x=df[df['Type']=='State/UT']['State/UT'],
    y=df[df['Type']=='State/UT']['Total'],
    marker_color=['#E53935' if s==selected else '#C5CAE9'
                  for s in df[df['Type']=='State/UT']['State/UT']],
    hovertemplate='<b>%{x}</b><br>%{y:,} accidents<extra></extra>'
))
fig_all.update_layout(
    title=f'All States/UTs — {selected} highlighted in red',
    plot_bgcolor='white', paper_bgcolor='white', height=310,
    xaxis_tickangle=-55, xaxis_tickfont_size=9,
    margin=dict(t=40, b=90)
)
st.plotly_chart(fig_all, use_container_width=True)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#9E9E9E;font-size:0.8rem;padding:0.4rem 0;'>
  🚦 NCRB 2023 Traffic Accident Explorer &nbsp;·&nbsp;
  Source: Accidental Deaths & Suicides in India (ADSI) 2023, Table 1A.6 &nbsp;·&nbsp;
  Built with Streamlit + scikit-learn + Plotly
</div>""", unsafe_allow_html=True)
