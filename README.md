# 🚦 NCRB 2023 Traffic Accident ML Dashboard

A Streamlit-powered machine learning dashboard for analysing India's traffic accident data from NCRB ADSI 2023 (Table 1A.6).

## 📁 Files Required

```
your-folder/
├── app.py                              ← Streamlit app (this file)
├── requirements.txt                    ← Python dependencies
└── NCRB_ADSI_2023_Table_1A_6.csv      ← Data file (place here OR upload via sidebar)
```

## 🚀 How to Run Locally

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

> 💡 If the CSV is not in the same folder, you can upload it directly via the **sidebar uploader** inside the app.

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push all files to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** ✅

> ⚠️ Upload the CSV via the sidebar uploader when using Streamlit Cloud (don't commit large data files).

---

## 📊 Dashboard Features

| Tab | Description |
|-----|-------------|
| **Overview** | Time-slot bar chart, top states, city rankings, night vs day scatter |
| **Clustering** | K-Means elbow curve, cluster heatmap, state groupings |
| **ML Models** | Model comparison (R², MAE, RMSE), actual vs predicted, feature importance |
| **Predictor** | Enter time-slot values → get total accident prediction + radar chart |
| **Data Explorer** | Filter & download data, correlation heatmap |

## ⚙️ Sidebar Controls

- **Upload CSV** — load your own NCRB data file
- **K-Means Clusters** — adjust number of clusters (2–7)
- **Include Cities** — toggle city-level charts
- **Prediction Model** — choose between Linear, Ridge, Random Forest, Gradient Boosting

---

## 🤖 ML Models Used

| Model | CV R² |
|-------|-------|
| Ridge Regression | ~0.994 |
| Linear Regression | ~0.994 |
| Gradient Boosting | ~0.929 |
| Random Forest | ~0.915 |

Models are trained on city-level data (53 cities) and predict total road accidents from 7 time-slot inputs.
