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

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (16, 10)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Malaysia Housing Price Prediction", layout="wide")

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
page = st.sidebar.radio("Navigation", [
    "📊 Dataset Overview",
    "🔍 Initial EDA",
    "🧹 Data Cleaning",
    "🔧 Feature Engineering",
    "📋 Data Preparation",
    "⚙️ Model Training",
    "🏆 Model Evaluation"
])

# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================
@st.cache_data
def load_and_process_data():
    try:
        from datasets import load_dataset
        ds = load_dataset("jienweng/housing-prices-malaysia-2025")
        df = ds['train'].to_pandas()
    except:
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

df_original = load_and_process_data()

# Detect column names
target_col = 'median_price' if 'median_price' in df_original.columns else 'Median_Price'
trans_col = 'transactions' if 'transactions' in df_original.columns else 'Transactions'

# ============================================================================
# PAGE 1: DATASET OVERVIEW
# ============================================================================
if page == "📊 Dataset Overview":
    st.header("DATASET OVERVIEW")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Shape", f"{df_original.shape[0]} rows × {df_original.shape[1]} columns")
    with col2:
        st.metric("Data Types", f"{df_original.dtypes.nunique()} types")
    with col3:
        st.metric("Missing Values", f"{df_original.isnull().sum().sum()}")
    
    st.subheader("Column Information")
    col_info = []
    for col in df_original.columns:
        dtype = df_original[col].dtype
        missing = df_original[col].isnull().sum()
        missing_pct = (missing / len(df_original)) * 100
        unique = df_original[col].nunique()
        
        if dtype == 'object':
            col_info.append({
                'Column': col,
                'Type': str(dtype),
                'Missing': f"{missing} ({missing_pct:.2f}%)",
                'Unique': unique
            })
        else:
            try:
                min_val = f"{df_original[col].min():.0f}"
                max_val = f"{df_original[col].max():.0f}"
            except:
                min_val = "N/A"
                max_val = "N/A"
            
            col_info.append({
                'Column': col,
                'Type': str(dtype),
                'Missing': f"{missing} ({missing_pct:.2f}%)",
                'Min': min_val,
                'Max': max_val
            })
    
    st.dataframe(pd.DataFrame(col_info), use_container_width=True)
    
    st.subheader("First 5 Rows")
    st.dataframe(df_original.head(), use_container_width=True)
    
    st.subheader("Basic Statistics")
    st.dataframe(df_original.describe().round(2), use_container_width=True)

# ============================================================================
# PAGE 2: INITIAL EDA (CLEAN EXPLORATORY ANALYSIS)
# ============================================================================
elif page == "🔍 Initial EDA":
    st.header("INITIAL EXPLORATORY ANALYSIS")
    
    # Figure 1: 2x2 Grid (Target Distribution, Log, Boxplot, Correlation)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df_original[target_col], kde=True, color='#3498db', ax=ax1)
    ax1.set_title(f'Distribution of {target_col}', fontweight='bold', fontsize=13)
    ax1.set_xlabel('Price (RM)')
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(np.log1p(df_original[target_col]), kde=True, color='#2ecc71', ax=ax2)
    ax2.set_title(f'Distribution of Log({target_col})', fontweight='bold', fontsize=13)
    ax2.set_xlabel('Log Price')
    
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(x=df_original[trans_col], color='#f1c40f', ax=ax3)
    ax3.set_title(f'Market Activity: {trans_col} Spread', fontweight='bold', fontsize=13)
    ax3.set_xlabel('Number of Transactions')
    
    ax4 = fig.add_subplot(gs[1, 1])
    numeric_df = df_original.select_dtypes(include=[np.number])
    corr_matrix_positive = numeric_df.corr().abs()
    sns.heatmap(corr_matrix_positive, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax4, cbar=True)
    ax4.set_title('Strength of Feature Relationships', fontweight='bold', fontsize=13)
    
    plt.suptitle('INITIAL EXPLORATORY ANALYSIS', fontsize=15, fontweight='bold', y=0.995)
    st.pyplot(fig)
    plt.close()
    
    # Figure 2: Boxplot Analysis for Features
    st.subheader("Distribution Analysis of Features (Box Plot)")
    num_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
    features_to_plot = [c for c in num_cols if c != target_col]
    
    if len(features_to_plot) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        df_melted = df_original.melt(value_vars=features_to_plot)
        sns.boxplot(x='variable', y='value', data=df_melted, palette='Set3', ax=ax)
        ax.set_title('Distribution Analysis of Features (X Values)', fontweight='bold', fontsize=14)
        ax.set_yscale('log')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
    
    # Figure 3: Large 3x3 Grid
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    ax_types = fig.add_subplot(gs[0, 1])
    dtype_counts = df_original.dtypes.value_counts()
    ax_types.bar(range(len(dtype_counts)), dtype_counts.values, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax_types.set_xticks(range(len(dtype_counts)))
    ax_types.set_xticklabels(dtype_counts.index, fontweight='bold')
    ax_types.set_ylabel('Count', fontweight='bold')
    ax_types.set_title('Data Types', fontweight='bold', fontsize=12)
    ax_types.grid(alpha=0.3, axis='y')
    for i, v in enumerate(dtype_counts.values):
        ax_types.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    
    ax_target = fig.add_subplot(gs[1, 0])
    ax_target.hist(df_original[target_col], bins=40, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
    ax_target.axvline(df_original[target_col].mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: RM{df_original[target_col].mean():,.0f}')
    ax_target.set_xlabel('Price (RM)', fontweight='bold', fontsize=10)
    ax_target.set_ylabel('Frequency', fontweight='bold', fontsize=10)
    ax_target.set_title(f'{target_col} Distribution', fontweight='bold', fontsize=12)
    ax_target.legend(fontsize=9)
    ax_target.grid(alpha=0.3, axis='y')
    
    plt.suptitle('DATA UNDERSTANDING & EXPLORATORY ANALYSIS', fontsize=15, fontweight='bold', y=0.995)
    st.pyplot(fig)
    plt.close()
    
    # Pairplot
    st.subheader("Variable Relationships & Distributions (Pairplot)")
    numeric_subset = [col for col in num_cols if col in df_original.columns][:4]
    if len(numeric_subset) > 1:
        pairplot_fig = sns.pairplot(df_original[numeric_subset], diag_kind='kde', plot_kws={'alpha': 0.5})
        pairplot_fig.fig.suptitle('Variable Relationships & Distributions (Pairplot)', 
                                  y=1.00, fontsize=16, fontweight='bold')
        st.pyplot(pairplot_fig)
        plt.close()

# ============================================================================
# PAGE 3: DATA CLEANING
# ============================================================================
elif page == "🧹 Data Cleaning":
    st.header("DATA CLEANING PROCESS")
    
    # Calculate cleaning metrics
    initial_rows = len(df_original)
    df_clean = df_original.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean[df_clean[target_col].notna()].copy()
    
    Q95 = df_clean[target_col].quantile(0.95)
    Q5 = df_clean[target_col].quantile(0.05)
    before = len(df_clean)
    df_clean = df_clean[(df_clean[target_col] >= Q5) & (df_clean[target_col] <= Q95)].copy()
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Before/After records
    categories = ['Original', 'After Cleaning']
    counts = [initial_rows, len(df_clean)]
    colors_clean = ['#e74c3c', '#2ecc71']
    bars = axes[0, 0].bar(categories, counts, color=colors_clean, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Record Count', fontweight='bold', fontsize=11)
    axes[0, 0].set_title('Records Before & After Cleaning', fontweight='bold', fontsize=12)
    axes[0, 0].grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, counts):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 20, f'{val:,}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Removed records
    removed_count = initial_rows - len(df_clean)
    axes[0, 1].text(0.5, 0.7, f'{removed_count:,}', ha='center', va='center', 
                    fontsize=20, fontweight='bold', color='#e74c3c')
    axes[0, 1].text(0.5, 0.3, 'Records Removed', ha='center', va='center', 
                    fontsize=14, fontweight='bold')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Data Quality Improvement', fontweight='bold', fontsize=12)
    
    # Price distribution
    axes[1, 0].hist(df_clean[target_col], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1)
    axes[1, 0].axvline(Q5, color='red', linestyle='--', linewidth=2, label=f'5th: RM{Q5:,.0f}')
    axes[1, 0].axvline(Q95, color='red', linestyle='--', linewidth=2, label=f'95th: RM{Q95:,.0f}')
    axes[1, 0].set_xlabel('Price (RM)', fontweight='bold', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontweight='bold', fontsize=11)
    axes[1, 0].set_title('Price Distribution After Cleaning', fontweight='bold', fontsize=12)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Data quality metrics
    metrics = ['Duplicates', 'Missing\nTargets', 'Outliers']
    values = [initial_rows - before, 0, before - len(df_clean)]
    bars = axes[1, 1].bar(metrics, values, color=['#3498db', '#e74c3c', '#f39c12'], 
                          alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('Count', fontweight='bold', fontsize=11)
    axes[1, 1].set_title('Data Quality Issues Resolved', fontweight='bold', fontsize=12)
    axes[1, 1].grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        if val > 0:
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, val, f'{int(val)}', 
                            ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.suptitle('DATA CLEANING PROCESS', fontsize=15, fontweight='bold')
    st.pyplot(fig)
    plt.close()

# ============================================================================
# PAGE 4: FEATURE ENGINEERING
# ============================================================================
elif page == "🔧 Feature Engineering":
    st.header("FEATURE ENGINEERING PROCESS")
    
    # Prepare data for feature engineering visualization
    df_fe = df_original.copy()
    df_fe = df_fe.drop_duplicates()
    df_fe = df_fe[df_fe[target_col].notna()].copy()
    
    Q95 = df_fe[target_col].quantile(0.95)
    Q5 = df_fe[target_col].quantile(0.05)
    df_fe = df_fe[(df_fe[target_col] >= Q5) & (df_fe[target_col] <= Q95)].copy()
    
    categorical_cols = df_fe.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df_fe[col + '_encoded'] = le.fit_transform(df_fe[col].astype(str))
    
    numerical_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    correlations_dict = {}
    for feat in numerical_cols:
        try:
            corr = abs(df_fe[feat].corr(df_fe[target_col]))
            if not np.isnan(corr):
                correlations_dict[feat] = corr
        except:
            pass
    
    top_features = sorted(correlations_dict.items(), key=lambda x: x[1], reverse=True)[:4]
    top_feature_names = [f[0] for f in top_features]
    
    initial_features = len(df_fe.columns) - 1
    
    for feat in top_feature_names:
        df_fe[f'{feat}_squared'] = df_fe[feat] ** 2
        df_fe[f'{feat}_sqrt'] = np.sqrt(np.abs(df_fe[feat]) + 1)
    
    interaction_count = 0
    for i in range(len(top_feature_names)):
        for j in range(i+1, len(top_feature_names)):
            feat1, feat2 = top_feature_names[i], top_feature_names[j]
            df_fe[f'{feat1}_x_{feat2}'] = df_fe[feat1] * df_fe[feat2]
            interaction_count += 1
    
    final_features = len(df_fe.columns) - 1
    
    # Visualization
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    stages = ['Original', 'Encoded', 'After\nEngineering']
    feature_counts = [initial_features, initial_features + len(categorical_cols), final_features]
    bars = ax1.bar(stages, feature_counts, color=['#3498db', '#e74c3c', '#2ecc71'], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Feature Count', fontweight='bold', fontsize=11)
    ax1.set_title('Feature Count Progression', fontweight='bold', fontsize=12)
    ax1.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, feature_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1, f'{int(val)}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2 = fig.add_subplot(gs[0, 1])
    feature_types = ['Numerical', 'Polynomial', 'Interactions', 'Encoded Cat.']
    type_counts = [initial_features, len(top_feature_names) * 2, interaction_count, len(categorical_cols)]
    colors_types = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    wedges, texts, autotexts = ax2.pie(type_counts, labels=feature_types, autopct='%1.1f%%', 
                                        colors=colors_types, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    ax2.set_title('Feature Type Distribution', fontweight='bold', fontsize=12)
    
    ax3 = fig.add_subplot(gs[1, :])
    if len(top_feature_names) > 0:
        top_correlations = sorted([(f, correlations_dict[f]) for f in top_feature_names], 
                                  key=lambda x: x[1], reverse=True)
        features_names = [f[0] for f in top_correlations]
        features_corr = [f[1] for f in top_correlations]
        bars = ax3.barh(features_names, features_corr, color='#2ecc71', alpha=0.7, 
                        edgecolor='black', linewidth=1.5)
        ax3.set_xlabel('Absolute Correlation with Target', fontweight='bold', fontsize=11)
        ax3.set_title('Top Features Selected for Polynomial/Interaction Features', fontweight='bold', fontsize=12)
        ax3.grid(alpha=0.3, axis='x')
        for bar, val in zip(bars, features_corr):
            ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                    va='center', fontweight='bold', fontsize=10)
    
    plt.suptitle('FEATURE ENGINEERING PROCESS', fontsize=15, fontweight='bold')
    st.pyplot(fig)
    plt.close()

# ============================================================================
# PAGE 5: DATA PREPARATION
# ============================================================================
elif page == "📋 Data Preparation":
    st.header("DATA PREPARATION: SPLITTING & SCALING")
    
    # Prepare data
    df_prep = df_original.copy()
    df_prep = df_prep.drop_duplicates()
    df_prep = df_prep[df_prep[target_col].notna()].copy()
    
    Q95 = df_prep[target_col].quantile(0.95)
    Q5 = df_prep[target_col].quantile(0.05)
    df_prep = df_prep[(df_prep[target_col] >= Q5) & (df_prep[target_col] <= Q95)].copy()
    
    categorical_cols = df_prep.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df_prep[col + '_encoded'] = le.fit_transform(df_prep[col].astype(str))
    
    df_prep = df_prep.drop(columns=categorical_cols)
    
    X = df_prep.drop(target_col, axis=1)
    y = df_prep[target_col]
    
    y_log = np.log1p(y)
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.15, random_state=42, shuffle=True
    )
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    numerical_features_all = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_scaled[numerical_features_all] = scaler.fit_transform(X_train[numerical_features_all])
    X_test_scaled[numerical_features_all] = scaler.transform(X_test[numerical_features_all])
    
    # Visualization
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    split_labels = ['Training Set', 'Test Set']
    split_sizes = [X_train.shape[0], X_test.shape[0]]
    colors_split = ['#3498db', '#e74c3c']
    wedges, texts, autotexts = ax1.pie(split_sizes, labels=split_labels, autopct='%1.1f%%',
                                        colors=colors_split, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    ax1.set_title('Train-Test Split', fontweight='bold', fontsize=12)
    
    ax2 = fig.add_subplot(gs[0, 1])
    categories = ['Training', 'Test']
    sizes = [X_train.shape[0], X_test.shape[0]]
    bars = ax2.bar(categories, sizes, color=colors_split, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Samples', fontweight='bold', fontsize=11)
    ax2.set_title('Dataset Sizes', fontweight='bold', fontsize=12)
    ax2.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 5, f'{int(val):,}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.6, f'{X_train.shape[1]}', ha='center', va='center', 
            fontsize=22, fontweight='bold', color='#2ecc71')
    ax3.text(0.5, 0.2, 'Features', ha='center', va='center', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Feature Dimension', fontweight='bold', fontsize=12)
    
    ax4 = fig.add_subplot(gs[1, 0:2])
    sample_feature = numerical_features_all[0] if len(numerical_features_all) > 0 else X_train.columns[0]
    ax4.hist(X_train[sample_feature], bins=30, color='#e74c3c', alpha=0.7, label='Before Scaling', 
            edgecolor='black')
    ax4.set_xlabel(f'{sample_feature} Value', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax4.set_title(f'Feature Distribution Before Scaling (Sample: {sample_feature})', fontweight='bold', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3, axis='y')
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(X_train_scaled[sample_feature], bins=30, color='#2ecc71', alpha=0.7, label='After Scaling', 
            edgecolor='black')
    ax5.set_xlabel(f'{sample_feature} Value (Scaled)', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax5.set_title(f'Feature Distribution After Scaling', fontweight='bold', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3, axis='y')
    
    plt.suptitle('DATA PREPARATION: SPLITTING & SCALING', fontsize=15, fontweight='bold')
    st.pyplot(fig)
    plt.close()

# ============================================================================
# PAGE 6: MODEL TRAINING
# ============================================================================
elif page == "⚙️ Model Training":
    st.header("MODEL TRAINING & HYPERPARAMETER OPTIMIZATION")
    
    # Data preparation
    df_train = df_original.copy()
    df_train = df_train.drop_duplicates()
    df_train = df_train[df_train[target_col].notna()].copy()
    
    Q95 = df_train[target_col].quantile(0.95)
    Q5 = df_train[target_col].quantile(0.05)
    df_train = df_train[(df_train[target_col] >= Q5) & (df_train[target_col] <= Q95)].copy()
    
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col + '_encoded'] = le.fit_transform(df_train[col].astype(str))
    
    df_train = df_train.drop(columns=categorical_cols)
    
    X = df_train.drop(target_col, axis=1)
    y = df_train[target_col]
    
    y_log = np.log1p(y)
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.15, random_state=42, shuffle=True
    )
    
    y_train_original = np.expm1(y_train_log)
    y_test_original = np.expm1(y_test_log)
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    numerical_features_all = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_scaled[numerical_features_all] = scaler.fit_transform(X_train[numerical_features_all])
    X_test_scaled[numerical_features_all] = scaler.transform(X_test[numerical_features_all])
    
    st.info("🔄 Training models...")
    progress_bar = st.progress(0)
    
    # MODEL 1: LINEAR REGRESSION
    progress_bar.progress(25)
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train_log)
    y_pred_lr = np.expm1(model_lr.predict(X_test_scaled))
    r2_lr = r2_score(y_test_original, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test_original, y_pred_lr))
    mae_lr = mean_absolute_error(y_test_original, y_pred_lr)
    mape_lr = mean_absolute_percentage_error(y_test_original, y_pred_lr)
    
    y_train_lr = np.expm1(model_lr.predict(X_train_scaled))
    r2_train_lr = r2_score(y_train_original, y_train_lr)
    
    # MODEL 2: RIDGE REGRESSION
    progress_bar.progress(50)
    param_grid_ridge = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge_search = GridSearchCV(Ridge(random_state=42), param_grid_ridge, cv=kfold, scoring='r2', n_jobs=-1, verbose=0)
    ridge_search.fit(X_train_scaled, y_train_log)
    y_pred_ridge = np.expm1(ridge_search.best_estimator_.predict(X_test_scaled))
    r2_ridge = r2_score(y_test_original, y_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test_original, y_pred_ridge))
    mae_ridge = mean_absolute_error(y_test_original, y_pred_ridge)
    mape_ridge = mean_absolute_percentage_error(y_test_original, y_pred_ridge)
    
    y_train_ridge = np.expm1(ridge_search.best_estimator_.predict(X_train_scaled))
    r2_train_ridge = r2_score(y_train_original, y_train_ridge)
    
    # MODEL 3: GRADIENT BOOSTING
    progress_bar.progress(75)
    param_grid_gb = {
        'n_estimators': [500],
        'learning_rate': [0.05],
        'max_depth': [4, 5, 6],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt']
    }
    gb_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42, validation_fraction=0.1, n_iter_no_change=15),
        param_grid_gb, cv=kfold, scoring='r2', n_jobs=-1, verbose=0
    )
    gb_search.fit(X_train_scaled, y_train_log)
    y_pred_gb = np.expm1(gb_search.best_estimator_.predict(X_test_scaled))
    r2_gb = r2_score(y_test_original, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test_original, y_pred_gb))
    mae_gb = mean_absolute_error(y_test_original, y_pred_gb)
    mape_gb = mean_absolute_percentage_error(y_test_original, y_pred_gb)
    
    y_train_gb = np.expm1(gb_search.best_estimator_.predict(X_train_scaled))
    r2_train_gb = r2_score(y_train_original, y_train_gb)
    
    # MODEL 4: RANDOM FOREST
    progress_bar.progress(100)
    param_grid_rf = {
        'n_estimators': [200],
        'max_depth': [15, 20, 25],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt']
    }
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid_rf, cv=kfold, scoring='r2', n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train, y_train_log)
    y_pred_rf = np.expm1(rf_search.best_estimator_.predict(X_test))
    r2_rf = r2_score(y_test_original, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test_original, y_pred_rf))
    mae_rf = mean_absolute_error(y_test_original, y_pred_rf)
    mape_rf = mean_absolute_percentage_error(y_test_original, y_pred_rf)
    
    y_train_rf = np.expm1(rf_search.best_estimator_.predict(X_train))
    r2_train_rf = r2_score(y_train_original, y_train_rf)
    
    # Visualization
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    cv_results_ridge = ridge_search.cv_results_
    ax1.plot(cv_results_ridge['param_alpha'], cv_results_ridge['mean_test_score'], 
            marker='o', linewidth=2, markersize=8, color='#3498db', label='Mean CV Score')
    ax1.fill_between(cv_results_ridge['param_alpha'], 
                    cv_results_ridge['mean_test_score'] - cv_results_ridge['std_test_score'],
                    cv_results_ridge['mean_test_score'] + cv_results_ridge['std_test_score'],
                    alpha=0.2, color='#3498db')
    ax1.set_xscale('log')
    ax1.set_xlabel('Alpha (Regularization)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('CV R² Score', fontweight='bold', fontsize=11)
    ax1.set_title('Ridge Regression: Hyperparameter Tuning', fontweight='bold', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2 = fig.add_subplot(gs[0, 1])
    cv_results_gb = gb_search.cv_results_
    depths = [p for p in cv_results_gb['param_max_depth']]
    ax2.scatter(depths, cv_results_gb['mean_test_score'], s=100, alpha=0.6, color='#e74c3c', edgecolors='black')
    ax2.set_xlabel('Max Depth', fontweight='bold', fontsize=11)
    ax2.set_ylabel('CV R² Score', fontweight='bold', fontsize=11)
    ax2.set_title('Gradient Boosting: Hyperparameter Tuning', fontweight='bold', fontsize=12)
    ax2.grid(alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 0])
    cv_results_rf = rf_search.cv_results_
    depths_rf = [p for p in cv_results_rf['param_max_depth']]
    ax3.scatter(depths_rf, cv_results_rf['mean_test_score'], s=100, alpha=0.6, color='#2ecc71', edgecolors='black')
    ax3.set_xlabel('Max Depth', fontweight='bold', fontsize=11)
    ax3.set_ylabel('CV R² Score', fontweight='bold', fontsize=11)
    ax3.set_title('Random Forest: Hyperparameter Tuning', fontweight='bold', fontsize=12)
    ax3.grid(alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    params_text = f"""
BEST HYPERPARAMETERS

Ridge:
  α = {ridge_search.best_params_['alpha']}

Gradient Boosting:
  max_depth = {gb_search.best_params_['max_depth']}

Random Forest:
  max_depth = {rf_search.best_params_['max_depth']}
"""
    ax4.text(0.1, 0.5, params_text, fontsize=11, family='monospace', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax4.set_title('Optimized Parameters', fontweight='bold', fontsize=12)
    
    plt.suptitle('MODEL TRAINING & HYPERPARAMETER OPTIMIZATION', fontsize=15, fontweight='bold')
    st.pyplot(fig)
    plt.close()
    
    # Store results for later use
    st.session_state.results = {
        'Linear Regression': {'R²': r2_lr, 'RMSE': rmse_lr, 'MAE': mae_lr, 'MAPE': mape_lr, 'train_r2': r2_train_lr, 'pred': y_pred_lr},
        'Ridge Regression': {'R²': r2_ridge, 'RMSE': rmse_ridge, 'MAE': mae_ridge, 'MAPE': mape_ridge, 'train_r2': r2_train_ridge, 'pred': y_pred_ridge},
        'Gradient Boosting': {'R²': r2_gb, 'RMSE': rmse_gb, 'MAE': mae_gb, 'MAPE': mape_gb, 'train_r2': r2_train_gb, 'pred': y_pred_gb},
        'Random Forest': {'R²': r2_rf, 'RMSE': rmse_rf, 'MAE': mae_rf, 'MAPE': mape_rf, 'train_r2': r2_train_rf, 'pred': y_pred_rf}
    }
    st.session_state.y_test = y_test_original

# ============================================================================
# PAGE 7: MODEL EVALUATION
# ============================================================================
elif page == "🏆 Model Evaluation":
    st.header("MODEL EVALUATION & COMPARISON")
    
    if 'results' in st.session_state:
        results = st.session_state.results
        y_test = st.session_state.y_test
        
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Train R²': [v['train_r2'] for v in results.values()],
            'Test R²': [v['R²'] for v in results.values()],
            'RMSE': [v['RMSE'] for v in results.values()],
            'MAE': [v['MAE'] for v in results.values()],
            'MAPE (%)': [v['MAPE'] * 100 for v in results.values()]
        })
        
        st.subheader("Model Performance Summary")
        st.dataframe(results_df, use_container_width=True)
        
        # Visualization
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        models = list(results.keys())
        
        # 1. R² Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        r2_scores = [results[m]['R²'] for m in models]
        bars1 = ax1.bar(['LR', 'Ridge', 'GB', 'RF'], r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.axhline(y=0.85, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.85)')
        ax1.set_ylabel('R² Score', fontweight='bold', fontsize=11)
        ax1.set_title('R² Comparison', fontweight='bold', fontsize=12)
        ax1.set_ylim([0, 1.05])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 2. RMSE Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        rmse_scores = [results[m]['RMSE'] for m in models]
        bars2 = ax2.bar(['LR', 'Ridge', 'GB', 'RF'], rmse_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('RMSE (RM)', fontweight='bold', fontsize=11)
        ax2.set_title('RMSE Comparison', fontweight='bold', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars2, rmse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, val, f'RM{val/1000:.0f}k', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 3. MAE Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        mae_scores = [results[m]['MAE'] for m in models]
        bars3 = ax3.bar(['LR', 'Ridge', 'GB', 'RF'], mae_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('MAE (RM)', fontweight='bold', fontsize=11)
        ax3.set_title('MAE Comparison', fontweight='bold', fontsize=12)
        ax3.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars3, mae_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, val, f'RM{val/1000:.0f}k', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 4. MAPE Comparison
        ax4 = fig.add_subplot(gs[1, 0])
        mape_scores = [results[m]['MAPE'] * 100 for m in models]
        bars4 = ax4.bar(['LR', 'Ridge', 'GB', 'RF'], mape_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.7, label='< 10%')
        ax4.set_ylabel('MAPE (%)', fontweight='bold', fontsize=11)
        ax4.set_title('MAPE Comparison', fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars4, mape_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, val, f'{val:.2f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 5. Train vs Test R² (Overfitting Analysis)
        ax5 = fig.add_subplot(gs[1, 1])
        x_pos = np.arange(len(models))
        width = 0.35
        train_r2s = [results[m]['train_r2'] for m in models]
        test_r2s = r2_scores
        
        bars_train = ax5.bar(x_pos - width/2, train_r2s, width, label='Train R²', 
                            color='#2ecc71', alpha=0.7, edgecolor='black')
        bars_test = ax5.bar(x_pos + width/2, test_r2s, width, label='Test R²', 
                           color='#e74c3c', alpha=0.7, edgecolor='black')
        
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(['LR', 'Ridge', 'GB', 'RF'])
        ax5.set_ylabel('R² Score', fontweight='bold', fontsize=11)
        ax5.set_title('Overfitting Analysis', fontweight='bold', fontsize=12)
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Performance Heatmap
        ax6 = fig.add_subplot(gs[1, 2])
        heatmap_data = pd.DataFrame({
            'Train R²': train_r2s,
            'Test R²': test_r2s,
            'RMSE/1k': [results[m]['RMSE']/1000 for m in models],
            'MAE/1k': [results[m]['MAE']/1000 for m in models],
            'MAPE%': mape_scores
        }, index=['LR', 'Ridge', 'GB', 'RF'])
        
        sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='coolwarm', ax=ax6, cbar_kws={'label': 'Score'})
        ax6.set_title('Final Performance Matrix', fontweight='bold', fontsize=12)
        
        # 7. Actual vs Predicted for best model
        best_idx = np.argmax(test_r2s)
        best_model_name = models[best_idx]
        best_predictions = results[best_model_name]['pred']
        
        ax7 = fig.add_subplot(gs[2, 0:2])
        ax7.scatter(y_test, best_predictions, alpha=0.5, color='#3498db', edgecolors='black', s=30)
        min_val = min(y_test.min(), best_predictions.min())
        max_val = max(y_test.max(), best_predictions.max())
        ax7.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax7.set_xlabel('Actual Price (RM)', fontweight='bold', fontsize=11)
        ax7.set_ylabel('Predicted Price (RM)', fontweight='bold', fontsize=11)
        ax7.set_title(f'Actual vs Predicted - {best_model_name}', fontweight='bold', fontsize=12)
        ax7.legend(fontsize=10)
        ax7.grid(alpha=0.3)
        
        # 8. Residuals plot
        ax8 = fig.add_subplot(gs[2, 2])
        residuals = y_test - best_predictions
        ax8.scatter(best_predictions, residuals, alpha=0.5, color='#e74c3c', edgecolors='black', s=30)
        ax8.axhline(y=0, color='black', linestyle='--', lw=2)
        ax8.set_xlabel('Predicted Price (RM)', fontweight='bold', fontsize=11)
        ax8.set_ylabel('Residuals (RM)', fontweight='bold', fontsize=11)
        ax8.set_title('Residual Plot', fontweight='bold', fontsize=12)
        ax8.grid(alpha=0.3)
        
        plt.suptitle('MODEL EVALUATION & COMPARISON', fontsize=15, fontweight='bold', y=0.995)
        st.pyplot(fig)
        plt.close()
        
        st.success(f"🏆 Best Model: **{best_model_name}** (Test R² = {test_r2s[best_idx]:.4f})")
    else:
        st.warning("⚠️ Please train models first in the 'Model Training' section.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with Streamlit** 🎈")
