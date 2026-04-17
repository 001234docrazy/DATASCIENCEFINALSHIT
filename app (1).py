import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import gc

# ============================================================================
# 1. UI CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Malaysia Housing Price Expert", layout="wide")

st.markdown("""
    <style>
    .main-header { background-color: #1e2130; padding: 20px; color: white; text-align: center; border-radius: 8px; margin-bottom: 25px; }
    .section-title { color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 5px; margin-top: 25px; font-weight: bold; }
    .card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; margin-bottom: 10px; }
    </style>
    <div class="main-header">
        <h1>🏠 MALAYSIA HOUSING DATA SCIENCE PROJECT</h1>
        <p>Data Overview • Analysis • Machine Learning • Prediction</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 2. DATA LOADING & PROCESSING
# ============================================================================
@st.cache_resource
def load_and_process_data():
    try:
        from datasets import load_dataset
        ds = load_dataset("jienweng/housing-prices-malaysia-2025")
        df = ds['train'].to_pandas()
    except:
        st.error("Dataset could not be loaded. Please check your connection.")
        return None, None, None, None

    # Identify columns
    cols = df.columns.tolist()
    def find_col(keywords):
        return next((c for c in cols if any(k in c.lower() for k in keywords)), None)

    col_map = {
        'town': find_col(['township', 'town']),
        'area': find_col(['area', 'district']),
        'state': find_col(['state']),
        'type': find_col(['type', 'property']),
        'tenure': find_col(['tenure']),
        'psf': find_col(['psf']),
        'trans': find_col(['trans']),
        'price': find_col(['price', 'median_price'])
    }

    # Clean numeric data
    df['psf_val'] = pd.to_numeric(df[col_map['psf']], errors='coerce').fillna(0)
    df['trans_val'] = pd.to_numeric(df[col_map['trans']], errors='coerce').fillna(0)
    df['price_val'] = pd.to_numeric(df[col_map['price']], errors='coerce').fillna(0)
    
    # Encoding for ML
    encoders = {}
    df_ml = df.copy()
    for feat in ['town', 'area', 'state', 'type', 'tenure']:
        le = LabelEncoder()
        df_ml[f'{feat}_enc'] = le.fit_transform(df[col_map[feat]].astype(str))
        encoders[feat] = le

    return df, df_ml, encoders, col_map

df, df_ml, encoders, col_map = load_and_process_data()

# ============================================================================
# 3. APP NAVIGATION
# ============================================================================
menu = st.sidebar.radio("Project Sections", [
    "📋 Data Overview", 
    "📊 Exploratory Analysis", 
    "🤖 Machine Learning Results",
    "🔮 Interactive Predictor"
])

if df is not None:
    # --- SECTION 1: DATA OVERVIEW ---
    if menu == "📋 Data Overview":
        st.markdown('<p class="section-title">Dataset Snapshot</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Features", f"{len(df.columns)}")
        col3.metric("States Covered", df[col_map['state']].nunique())
        
        st.subheader("First 10 Rows")
        st.write(df.head(10))
        
        st.subheader("Data Information")
        buffer = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": [df[c].count() for c in df.columns],
            "Dtype": [df[c].dtype for c in df.columns]
        })
        st.table(buffer)

    # --- SECTION 2: EXPLORATORY ANALYSIS ---
    elif menu == "📊 Exploratory Analysis":
        st.markdown('<p class="section-title">Market Visualization</p>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Price Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['price_val'], bins=30, kde=True, ax=ax, color='teal')
            ax.set_xlim(0, df['price_val'].quantile(0.95))
            st.pyplot(fig)
            
        with c2:
            st.subheader("Top 10 States by Avg Price")
            state_avg = df.groupby(col_map['state'])['price_val'].mean().sort_values(ascending=False).head(10)
            st.bar_chart(state_avg)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df_ml[['price_val', 'psf_val', 'trans_val', 'state_enc', 'type_enc']].corr(), annot=True, cmap='RdBu', ax=ax)
        st.pyplot(fig)

    # --- SECTION 3: ML RESULTS ---
    elif menu == "🤖 Machine Learning Results":
        st.markdown('<p class="section-title">Model Comparison & Evaluation</p>', unsafe_allow_html=True)
        
        # Simple evaluation logic
        features = ['psf_val', 'trans_val', 'town_enc', 'area_enc', 'state_enc', 'type_enc', 'tenure_enc']
        X = df_ml[features]
        y = df_ml['price_val']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        m1, m2, m3 = st.columns(3)
        m1.info(f"**R² Score:** {r2_score(y_test, preds):.4f}")
        m2.warning(f"**MAE:** RM {mean_absolute_error(y_test, preds):,.0f}")
        m3.success(f"**Model:** Random Forest")

        st.subheader("Feature Importance")
        importances = pd.Series(model.feature_importances_, index=features).sort_values()
        st.bar_chart(importances)

    # --- SECTION 4: INTERACTIVE PREDICTOR ---
    elif menu == "🔮 Interactive Predictor":
        st.markdown('<p class="section-title">Real Estate Price Estimator</p>', unsafe_allow_html=True)
        
        with st.form("pred_form"):
            f1, f2, f3 = st.columns(3)
            with f1:
                st_in = st.selectbox("State", sorted(df[col_map['state']].unique()))
                type_in = st.selectbox("Property Type", sorted(df[col_map['type']].unique()))
            with f2:
                area_in = st.selectbox("Area", sorted(df[df[col_map['state']]==st_in][col_map['area']].unique()))
                tenure_in = st.selectbox("Tenure", sorted(df[col_map['tenure']].unique()))
            with f3:
                psf_in = st.number_input("Est. PSF (RM)", value=450)
                trans_in = st.number_input("Trans. Volume", value=10)
            
            submit = st.form_submit_button("PREDICT PRICE")
            
        if submit:
            # Training on whole set for prediction
            full_model = RandomForestRegressor(n_estimators=50, random_state=42)
            full_model.fit(df_ml[['psf_val', 'trans_val', 'town_enc', 'area_enc', 'state_enc', 'type_enc', 'tenure_enc']], df_ml['price_val'])
            
            # Find a representative town code
            town_code = df_ml[df_ml[col_map['area']] == area_in]['town_enc'].iloc[0]
            
            input_data = [[
                psf_in, trans_in, town_code,
                encoders['area'].transform([area_in])[0],
                encoders['state'].transform([st_in])[0],
                encoders['type'].transform([type_in])[0],
                encoders['tenure'].transform([tenure_in])[0]
            ]]
            
            res = full_model.predict(input_data)[0]
            st.success(f"### Estimated Market Price: RM {res:,.2f}")
