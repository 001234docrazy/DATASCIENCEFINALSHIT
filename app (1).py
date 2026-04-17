import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import logging
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('datasets').setLevel(logging.ERROR)

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Malaysia Housing Analysis", layout="wide")

st.markdown("""
    <style>
    .main-header {
        background-color: #1e2130;
        padding: 20px;
        color: white;
        text-align: center;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    </style>
    <div class="main-header">
        <h1>🏠 MALAYSIA HOUSING PRICE PREDICTION</h1>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
page = st.sidebar.radio("Navigation", [
    "📊 Data Overview",
    "🔍 Exploratory Analysis",
    "⚙️ Model Training",
    "📈 Results & Comparison"
])

# ============================================================================
# DATA LOADING (CACHED)
# ============================================================================
@st.cache_data
def load_data():
    try:
        from datasets import load_dataset
        ds = load_dataset("jienweng/housing-prices-malaysia-2025")
        df = ds['train'].to_pandas()
    except:
        st.warning("⚠️ Could not load dataset. Using synthetic data.")
        np.random.seed(42)
        df = pd.DataFrame({
            'township': np.random.choice(['Cheras', 'Subang', 'Shah Alam', 'Petaling Jaya', 'Klang'], 1500),
            'area': np.random.choice(['Klang Valley', 'Selangor', 'Johor', 'Penang', 'Sabah'], 1500),
            'state': np.random.choice(['Selangor', 'Johor', 'Penang', 'KL', 'Sabah'], 1500),
            'tenure': np.random.choice(['Freehold', 'Leasehold'], 1500),
            'type': np.random.choice(['Terrace', 'Condominium', 'Semi-D', 'Detached', 'Bungalow'], 1500),
            'median_price': np.random.randint(250000, 2500000, 1500),
            'median_psf': np.random.uniform(250, 1500, 1500),
            'transactions': np.random.randint(3, 250, 1500),
        })
    return df

@st.cache_resource
def prepare_data(df):
    """Data cleaning and feature engineering"""
    target_col = 'median_price' if 'median_price' in df.columns else 'Median_Price'
    
    # Clean data
    df = df.drop_duplicates()
    df = df[df[target_col].notna()].copy()
    
    Q95 = df[target_col].quantile(0.95)
    Q5 = df[target_col].quantile(0.05)
    df = df[(df[target_col] >= Q5) & (df[target_col] <= Q95)].copy()
    
    # Feature engineering
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    correlations_dict = {}
    for feat in numerical_cols:
        corr = abs(df[feat].corr(df[target_col]))
        if not np.isnan(corr):
            correlations_dict[feat] = corr
    
    top_features = sorted(correlations_dict.items(), key=lambda x: x[1], reverse=True)[:4]
    top_feature_names = [f[0] for f in top_features]
    
    for feat in top_feature_names:
        df[f'{feat}_squared'] = df[feat] ** 2
        df[f'{feat}_sqrt'] = np.sqrt(np.abs(df[feat]) + 1)
    
    interaction_count = 0
    for i in range(len(top_feature_names)):
        for j in range(i+1, len(top_feature_names)):
            feat1, feat2 = top_feature_names[i], top_feature_names[j]
            df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            interaction_count += 1
    
    df = df.drop(columns=categorical_cols)
    
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        numerical_cols_imp = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols_imp:
            numerical_cols_imp.remove(target_col)
        if len(numerical_cols_imp) > 0:
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            df[numerical_cols_imp] = imputer.fit_transform(df[numerical_cols_imp])
    
    return df, target_col

# Load data
df = load_data()
df_processed, target_col = prepare_data(df)

# ============================================================================
# PAGE 1: DATA OVERVIEW
# ============================================================================
if page == "📊 Data Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum()}")
    
    st.subheader("Column Information")
    col_info = []
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        missing_pct = (missing / len(df)) * 100
        unique = df[col].nunique()
        
        if dtype == 'object':
            col_info.append({
                'Column': col,
                'Type': str(dtype),
                'Missing': f"{missing} ({missing_pct:.2f}%)",
                'Unique': unique
            })
        else:
            col_info.append({
                'Column': col,
                'Type': str(dtype),
                'Missing': f"{missing} ({missing_pct:.2f}%)",
                'Min': f"{df[col].min():.0f}",
                'Max': f"{df[col].max():.0f}"
            })
    
    st.dataframe(pd.DataFrame(col_info), use_container_width=True)
    
    st.subheader("First 5 Rows")
    st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ============================================================================
# PAGE 2: EXPLORATORY ANALYSIS
# ============================================================================
elif page == "🔍 Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[target_col], bins=40, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(df[target_col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: RM{df[target_col].mean():,.0f}')
        ax.set_xlabel('Price (RM)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Log Target Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(np.log1p(df[target_col]), bins=40, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Log Price')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    st.subheader("Data Types Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    dtype_counts = df.dtypes.value_counts()
    ax.bar(range(len(dtype_counts)), dtype_counts.values, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(dtype_counts)))
    ax.set_xticklabels(dtype_counts.index, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_title('Data Types')
    ax.grid(alpha=0.3, axis='y')
    st.pyplot(fig)
    plt.close()
    
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr().abs(), annot=True, cmap='YlGnBu', fmt=".2f", ax=ax, cbar_kws={'label': 'Correlation'})
    st.pyplot(fig)
    plt.close()

# ============================================================================
# PAGE 3: MODEL TRAINING
# ============================================================================
elif page == "⚙️ Model Training":
    st.header("Model Training & Evaluation")
    
    st.info("🔄 Training models... This may take a moment.")
    
    # Prepare data
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    numerical_features_all = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_scaled[numerical_features_all] = scaler.fit_transform(X_train[numerical_features_all])
    X_test_scaled[numerical_features_all] = scaler.transform(X_test[numerical_features_all])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples", X_train.shape[0])
    with col2:
        st.metric("Test Samples", X_test.shape[0])
    
    st.metric("Total Features", X_train.shape[1])
    
    # Train models
    results = {}
    
    st.subheader("Model Results")
    progress_bar = st.progress(0)
    status = st.empty()
    
    # Linear Regression
    status.text("Training Linear Regression...")
    progress_bar.progress(25)
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    y_pred_lr = model_lr.predict(X_test_scaled)
    results['Linear Regression'] = {
        'R²': r2_score(y_test, y_pred_lr),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'MAE': mean_absolute_error(y_test, y_pred_lr),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred_lr),
        'predictions': y_pred_lr
    }
    
    # Ridge Regression
    status.text("Training Ridge Regression...")
    progress_bar.progress(50)
    param_grid_ridge = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge_search = GridSearchCV(Ridge(random_state=42), param_grid_ridge, cv=kfold, scoring='r2', n_jobs=-1, verbose=0)
    ridge_search.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_search.best_estimator_.predict(X_test_scaled)
    results['Ridge Regression'] = {
        'R²': r2_score(y_test, y_pred_ridge),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        'MAE': mean_absolute_error(y_test, y_pred_ridge),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred_ridge),
        'predictions': y_pred_ridge
    }
    
    # Gradient Boosting
    status.text("Training Gradient Boosting...")
    progress_bar.progress(75)
    param_grid_gb = {
        'n_estimators': [100],
        'learning_rate': [0.05],
        'max_depth': [5],
        'subsample': [0.8],
        'max_features': ['sqrt']
    }
    gb_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42, validation_fraction=0.1, n_iter_no_change=10),
        param_grid_gb, cv=kfold, scoring='r2', n_jobs=-1, verbose=0
    )
    gb_search.fit(X_train_scaled, y_train)
    y_pred_gb = gb_search.best_estimator_.predict(X_test_scaled)
    results['Gradient Boosting'] = {
        'R²': r2_score(y_test, y_pred_gb),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
        'MAE': mean_absolute_error(y_test, y_pred_gb),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred_gb),
        'predictions': y_pred_gb
    }
    
    # Random Forest
    status.text("Training Random Forest...")
    progress_bar.progress(100)
    param_grid_rf = {
        'n_estimators': [100],
        'max_depth': [20],
        'min_samples_leaf': [2],
        'max_features': ['sqrt']
    }
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid_rf, cv=kfold, scoring='r2', n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train, y_train)
    y_pred_rf = rf_search.best_estimator_.predict(X_test)
    results['Random Forest'] = {
        'R²': r2_score(y_test, y_pred_rf),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred_rf),
        'predictions': y_pred_rf
    }
    
    status.text("✅ Training Complete!")
    
    # Display results
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'R²': [v['R²'] for v in results.values()],
        'RMSE': [v['RMSE'] for v in results.values()],
        'MAE': [v['MAE'] for v in results.values()],
        'MAPE (%)': [v['MAPE'] * 100 for v in results.values()]
    })
    
    st.dataframe(results_df, use_container_width=True)
    
    best_model = results_df.loc[results_df['R²'].idxmax(), 'Model']
    best_r2 = results_df['R²'].max()
    st.success(f"🏆 Best Model: **{best_model}** (R² = {best_r2:.4f})")

# ============================================================================
# PAGE 4: RESULTS & COMPARISON
# ============================================================================
elif page == "📈 Results & Comparison":
    st.header("Model Comparison & Results")
    
    st.info("Please train models first in the 'Model Training' section.")
    
    if 'results' in locals():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("R² Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            models = list(results.keys())
            r2_scores = [results[m]['R²'] for m in models]
            ax.bar(models, r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8, edgecolor='black')
            ax.set_ylabel('R² Score')
            ax.set_title('R² Comparison')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("RMSE Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            rmse_scores = [results[m]['RMSE'] for m in models]
            ax.bar(models, rmse_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8, edgecolor='black')
            ax.set_ylabel('RMSE (RM)')
            ax.set_title('RMSE Comparison')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("MAE Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            mae_scores = [results[m]['MAE'] for m in models]
            ax.bar(models, mae_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8, edgecolor='black')
            ax.set_ylabel('MAE (RM)')
            ax.set_title('MAE Comparison')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("MAPE Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            mape_scores = [results[m]['MAPE'] * 100 for m in models]
            ax.bar(models, mape_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8, edgecolor='black')
            ax.set_ylabel('MAPE (%)')
            ax.set_title('MAPE Comparison')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with Streamlit** 🎈")
