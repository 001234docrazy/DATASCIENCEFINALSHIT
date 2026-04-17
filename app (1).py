import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import gc

# ============================================================================
# UI CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(page_title="Malaysia Housing Expert", layout="wide")

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
    .section-title {
        color: #00d4ff; 
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 5px;
        margin-top: 25px;
        margin-bottom: 15px;
        font-weight: bold;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-header {
        background-color: #f1f3f9;
        padding: 15px;
        border-radius: 8px 8px 0 0;
        border-left: 10px solid #27ae60;
        font-weight: bold;
        color: #1e2130;
        margin-top: 20px;
        font-size: 1.5rem;
    }
    .result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 0 0 8px 8px;
        border: 1px solid #f1f3f9;
        border-left: 10px solid #27ae60;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    </style>
    <div class="main-header">
        <h1>🏠 MALAYSIA HOUSING PRICE EXPERT</h1>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA & MODEL LOADING
# ============================================================================
@st.cache_resource
def load_app_core():
    try:
        from datasets import load_dataset
        ds = load_dataset("jienweng/housing-prices-malaysia-2025")
        df = ds['train'].to_pandas()
    except:
        df = pd.DataFrame({
            'township': ['Default'], 'area': ['Default'], 
            'state': ['Default'], 'type': ['Default'],
            'tenure': ['Freehold'], 'median_psf': [500], 
            'transactions': [1], 'median_price': [500000]
        })

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

    df = df.dropna(subset=[col_map['state'], col_map['area'], col_map['town']])
    
    encoders = {}
    for feat in ['town', 'area', 'state', 'type', 'tenure']:
        le = LabelEncoder()
        df[f'{feat}_enc'] = le.fit_transform(df[col_map[feat]].astype(str))
        encoders[feat] = le

    df['psf_val'] = pd.to_numeric(df[col_map['psf']], errors='coerce').fillna(0)
    df['trans_val'] = pd.to_numeric(df[col_map['trans']], errors='coerce').fillna(0)
    df['price_val'] = pd.to_numeric(df[col_map['price']], errors='coerce').fillna(0)
    
    feature_cols = ['psf_val', 'trans_val', 'town_enc', 'area_enc', 'state_enc', 'type_enc', 'tenure_enc']
    model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=1, random_state=42)
    model.fit(df[feature_cols], df['price_val'])
    
    gc.collect()
    return df, model, encoders, col_map

df, model, encoders, col_map = load_app_core()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
mode = st.sidebar.radio("Navigation", ["🔮 Price Predictor", "📊 Market Rankings", "💰 Mortgage Calculator"])

# ============================================================================
# PART 1: PRICE PREDICTOR
# ============================================================================
if mode == "🔮 Price Predictor":
    st.markdown('<p class="section-title">Step 1: Location Selection</p>', unsafe_allow_html=True)
    loc_c1, loc_c2, loc_c3 = st.columns(3)

    with loc_c1:
        state_choice = st.selectbox("Select State", sorted(df[col_map['state']].unique()))
    with loc_c2:
        area_choice = st.selectbox("Select Area", sorted(df[df[col_map['state']] == state_choice][col_map['area']].unique()))
    with loc_c3:
        town_choice = st.selectbox("Select Township", sorted(df[df[col_map['area']] == area_choice][col_map['town']].unique()))

    st.markdown('<p class="section-title">Step 2: Property Details</p>', unsafe_allow_html=True)
    spec_c1, spec_c2, spec_c3, spec_c4 = st.columns(4)

    with spec_c1:
        prop_type = st.selectbox("Property Type", sorted(df[col_map['type']].unique()))
    with spec_c2:
        tenure_type = st.selectbox("Tenure", sorted(df[col_map['tenure']].unique()))
    with spec_c3:
        avg_psf_val = df[df[col_map['town']] == town_choice]['psf_val'].mean()
        psf_in = st.number_input("Median PSF (RM)", value=float(avg_psf_val) if avg_psf_val > 10 else 100.0, min_value=0.0)
    with spec_c4:
        trans_in = st.number_input("Transactions", value=10, min_value=1)

    adj = st.number_input("Manual Price Adjustment (± RM):", value=0.0, step=500.0)
    
    if st.button("CALCULATE PREDICTED PRICE", type="primary", use_container_width=True):
        input_data = [[
            psf_in, trans_in,
            encoders['town'].transform([town_choice])[0],
            encoders['area'].transform([area_choice])[0],
            encoders['state'].transform([state_choice])[0],
            encoders['type'].transform([prop_type])[0],
            encoders['tenure'].transform([tenure_type])[0]
        ]]
        raw_val = model.predict(input_data)[0]
        final_v = raw_val + adj
        
        st.markdown('<div class="result-header">📊 PREDICTED PRICE RESULTS</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="result-card">
                <span style="color: #666; font-size: 0.9em; font-weight: bold;">ESTIMATED MARKET VALUE</span>
                <h2 style="color: #27ae60; margin: 0; font-size: 2.5em;">RM {final_v:,.2f}</h2>
                <hr style="margin: 15px 0; border: 0; border-top: 1px solid #eee;">
                <p style="margin: 5px 0; color: #1e2130;"><b>Location:</b> {town_choice}, {area_choice}</p>
                <p style="margin: 5px 0; color: #1e2130;"><b>Property:</b> {prop_type} ({tenure_type})</p>
                <small style="color: #999;">Statistical Baseline: RM {raw_val:,.2f} | Adjustment: RM {adj:,.2f}</small>
            </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PART 2: MARKET RANKINGS (Big Picture Analysis)
# ============================================================================
elif mode == "📊 Market Rankings":
    st.markdown('<p class="section-title">Regional Market Analysis</p>', unsafe_allow_html=True)
    
    sel_state = st.selectbox("Select State to Analyze", sorted(df[col_map['state']].unique()))
    state_df = df[df[col_map['state']] == sel_state]
    
    # ROW 1: Expensive vs Cheap
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💎 Top 10 Most Expensive")
        top_10_price = state_df.groupby(col_map['town'])['price_val'].mean().sort_values(ascending=False).head(10)
        fig1, ax1 = plt.subplots()
        sns.barplot(x=top_10_price.values, y=top_10_price.index, palette="viridis", ax=ax1)
        ax1.set_xlabel("Avg Price (RM)")
        st.pyplot(fig1)

    with col2:
        st.subheader("🏷️ Top 10 Most Affordable")
        # Filtering for townships with at least 1 transaction to avoid outliers
        low_10_price = state_df.groupby(col_map['town'])['price_val'].mean().sort_values(ascending=True).head(10)
        fig2, ax2 = plt.subplots()
        sns.barplot(x=low_10_price.values, y=low_10_price.index, palette="crest", ax=ax2)
        ax2.set_xlabel("Avg Price (RM)")
        st.pyplot(fig2)

    # ROW 2: Activity vs Volume
    st.markdown('<p class="section-title">Transaction Trends</p>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🔥 Top 10 High Activity (Transactions)")
        top_10_trans = state_df.groupby(col_map['town'])['trans_val'].sum().sort_values(ascending=False).head(10)
        fig3, ax3 = plt.subplots()
        sns.barplot(x=top_10_trans.values, y=top_10_trans.index, palette="magma", ax=ax3)
        ax3.set_xlabel("Total Transactions")
        st.pyplot(fig3)

    with col4:
        st.subheader("📈 Price vs. Transaction Volume")
        # Scatter plot to see if high-volume areas are cheaper
        scatter_data = state_df.groupby(col_map['town']).agg({'price_val': 'mean', 'trans_val': 'sum'})
        fig4, ax4 = plt.subplots()
        sns.scatterplot(data=scatter_data, x='trans_val', y='price_val', alpha=0.6, ax=ax4, color="#00d4ff")
        ax4.set_xlabel("Total Transactions")
        ax4.set_ylabel("Avg Price (RM)")
        st.pyplot(fig4)

    st.markdown('<p class="section-title">District-wise Price Breakdown</p>', unsafe_allow_html=True)
    area_stats = state_df.groupby(col_map['area'])['price_val'].agg(['mean', 'count']).rename(columns={'mean': 'Avg Price (RM)', 'count': 'Data Points'})
    st.table(area_stats.sort_values(by='Avg Price (RM)', ascending=False))
# ============================================================================
# PART 3: MORTGAGE CALCULATOR
# ============================================================================
elif mode == "💰 Mortgage Calculator":
    st.markdown('<p class="section-title">Loan Affordability Calculator</p>', unsafe_allow_html=True)
    m_col1, m_col2 = st.columns([1, 2])
    
    with m_col1:
        price = st.number_input("Property Price (RM)", value=500000.0, step=10000.0)
        dp_pct = st.slider("Downpayment (%)", 0, 50, 10)
        interest = st.number_input("Annual Interest Rate (%)", value=4.2, step=0.1)
        tenure = st.slider("Loan Tenure (Years)", 5, 35, 30)
        
        loan = price * (1 - dp_pct/100)
        rate = (interest/100)/12
        months = tenure * 12
        monthly = (loan * rate * (1+rate)**months) / ((1+rate)**months - 1) if rate > 0 else loan/months

    with m_col2:
        st.markdown(f"""
            <div class="result-card">
                <span style="color: #666; font-size: 0.9em; font-weight: bold;">ESTIMATED MONTHLY INSTALLMENT</span>
                <h2 style="color: #1e2130; margin: 0;">RM {monthly:,.2f}</h2>
                <hr style="margin: 15px 0; border: 0; border-top: 1px solid #eee;">
                <p style="margin: 5px 0; color: #1e2130;"><b>Total Loan:</b> RM {loan:,.2f}</p>
                <p style="margin: 5px 0; color: #1e2130;"><b>Interest Only:</b> RM {(monthly*months)-loan:,.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        st.info(f"💡 Recommendation: Monthly Net Income should be at least RM {monthly/0.35:,.2f} for this loan.")
        
        fig, ax = plt.subplots()
        ax.pie([loan, (monthly*months)-loan], labels=['Principal', 'Total Interest'], autopct='%1.1f%%', colors=['#27ae60', '#1e2130'])
        st.pyplot(fig)

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
import pickle
import gzip

plt.ion()
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (16, 10)

print("="*120)
print("MALAYSIA HOUSING PRICE PREDICTION")
print("="*120)

# ============================================================================
# DATA LOADING
# ============================================================================

print("\n[DATA LOADING]")

try:
    from datasets import load_dataset
    ds = load_dataset("jienweng/housing-prices-malaysia-2025")
    df = ds['train'].to_pandas()
    print(f"✓ Dataset: {df.shape[0]:,} properties × {df.shape[1]} features")
except Exception as e:
    print(f"⚠ Using synthetic data")
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

# ============================================================================
# DATASET OVERVIEW
# ============================================================================

print("\n[DATASET OVERVIEW]")

print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn Information:")
print("-" * 120)

for col in df.columns:
    dtype = df[col].dtype
    missing = df[col].isnull().sum()
    missing_pct = (missing / len(df)) * 100
    unique = df[col].nunique()
    
    if dtype == 'object':
        print(f"  {col:20} | Type: {str(dtype):15} | Missing: {missing:>5} ({missing_pct:>5.2f}%) | Unique: {unique:>5}")
    else:
        print(f"  {col:20} | Type: {str(dtype):15} | Missing: {missing:>5} ({missing_pct:>5.2f}%) | Min: {df[col].min():>12.0f} | Max: {df[col].max():>12.0f}")

print(f"\nData Types Summary:")
print("-" * 120)
print(df.dtypes.value_counts().to_string())

print(f"\nMissing Values Summary:")
print("-" * 120)
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
if len(missing_summary) > 0:
    print(missing_summary.to_string(index=False))
else:
    print("✓ No missing values detected")

print(f"\nFirst 5 Rows:")
print("-" * 120)
print(df.head().to_string())

print(f"\nBasic Statistics:")
print("-" * 120)
print(df.describe().round(2).to_string())


# ============================================================================
# COLUMN NAME FIXER
# ============================================================================
target_col = 'Median_Price' if 'Median_Price' in df.columns else 'median_price'

trans_col = 'Transactions' if 'Transactions' in df.columns else 'transactions'

print(f"✓ Using Target Column: {target_col}")
print(f"✓ Using Transactions Column: {trans_col}")

# ============================================================================
# CLEAN EXPLORATORY ANALYSIS
# ============================================================================

print("\n[GENERATING CLEAN EXPLORATORY ANALYSIS]")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)

# 1. Target Distribution (Corrected to use detected target_col)
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(df[target_col], kde=True, color='#3498db', ax=ax1)
ax1.set_title(f'Distribution of {target_col}', fontweight='bold', fontsize=13)
ax1.set_xlabel('Price (RM)')

# 2. Log-Target Distribution
ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(np.log1p(df[target_col]), kde=True, color='#2ecc71', ax=ax2)
ax2.set_title(f'Distribution of Log({target_col})', fontweight='bold', fontsize=13)
ax2.set_xlabel('Log Price')

# 3. Market Activity (Using detected trans_col)
ax3 = fig.add_subplot(gs[1, 0])
sns.boxplot(x=df[trans_col], color='#f1c40f', ax=ax3)
ax3.set_title(f'Market Activity: {trans_col} Spread', fontweight='bold', fontsize=13)
ax3.set_xlabel('Number of Transactions')

# 4. Correlation Heatmap
ax4 = fig.add_subplot(gs[1, 1])
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix_positive = numeric_df.corr().abs() 
sns.heatmap(corr_matrix_positive, 
            annot=True, 
            cmap='YlGnBu',   # Using a sequential color map (Yellow to Blue)
            fmt=".2f", 
            ax=ax4,
            cbar=True)
ax4.set_title('Strength of Feature Relationships', fontweight='bold', fontsize=13)

# Identify target and numerical features for visualization
target_col = 'median_price'
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# B. Boxplot Analysis for Features (X Values)
# Visualizing the distribution and potential outliers of numerical predictors
plt.figure(figsize=(12, 6))
features_to_plot = [c for c in num_cols if c != target_col]
df_melted = df.melt(value_vars=features_to_plot)
sns.boxplot(x='variable', y='value', data=df_melted, palette='Set3')
plt.title('Distribution Analysis of Features (X Values)', fontweight='bold', fontsize=14)
plt.yscale('log') # Log scale if values have different magnitudes
plt.show()


fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Data types bar chart
ax_types = fig.add_subplot(gs[0, 1])
dtype_counts = df.dtypes.value_counts()
ax_types.bar(range(len(dtype_counts)), dtype_counts.values, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
ax_types.set_xticks(range(len(dtype_counts)))
ax_types.set_xticklabels(dtype_counts.index, fontweight='bold')
ax_types.set_ylabel('Count', fontweight='bold')
ax_types.set_title('Data Types', fontweight='bold', fontsize=12)
ax_types.grid(alpha=0.3, axis='y')
for i, v in enumerate(dtype_counts.values):
    ax_types.text(i, v + 0.1, str(v), ha='center', fontweight='bold')

# Target variable distribution
target_col = None
for col in df.columns:
    if 'price' in col.lower() and 'median' in col.lower():
        target_col = col
        break
if not target_col:
    for col in df.columns:
        if 'price' in col.lower():
            target_col = col
            break

ax_target = fig.add_subplot(gs[1, 0])
ax_target.hist(df[target_col], bins=40, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
ax_target.axvline(df[target_col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: RM{df[target_col].mean():,.0f}')
ax_target.set_xlabel('Price (RM)', fontweight='bold', fontsize=10)
ax_target.set_ylabel('Frequency', fontweight='bold', fontsize=10)
ax_target.set_title(f'{target_col} Distribution', fontweight='bold', fontsize=12)
ax_target.legend(fontsize=9)
ax_target.grid(alpha=0.3, axis='y')

# C. Pairplot Analysis (Subset of variables to ensure readability)
print("✓ Generating Pairplot (Target vs Key Features)...")
sns.pairplot(df[num_cols], diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Variable Relationships & Distributions (Pairplot)', y=1.02, fontsize=16, fontweight='bold')
plt.show()

plt.savefig('01_Data_Overview.png', dpi=300, bbox_inches='tight')
plt.suptitle('DATA UNDERSTANDING & EXPLORATORY ANALYSIS', fontsize=15, fontweight='bold', y=0.995)
plt.savefig('01_Data_Understanding.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_Data_Understanding.png")
plt.show()
plt.pause(1)

# ============================================================================
# TARGET IDENTIFICATION
# ============================================================================

print("\n[TARGET IDENTIFICATION]")

print(f"✓ Target: {target_col}")
print(f"  Mean: RM {df[target_col].mean():,.0f}")
print(f"  Median: RM {df[target_col].median():,.0f}")
print(f"  Range: RM {df[target_col].min():,.0f} - RM {df[target_col].max():,.0f}")

# ============================================================================
# DATA CLEANING
# ============================================================================

print("\n[DATA CLEANING]")

initial_rows = len(df)
df = df.drop_duplicates()
df = df[df[target_col].notna()].copy()

Q95 = df[target_col].quantile(0.95)
Q5 = df[target_col].quantile(0.05)
before = len(df)
df = df[(df[target_col] >= Q5) & (df[target_col] <= Q95)].copy()

print(f"✓ Removed: {initial_rows - before} records")
print(f"✓ Final: {len(df):,} records")

# DATA CLEANING VISUALIZATION
print("\n[GENERATING DATA CLEANING VISUALIZATION]")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Before/After records
categories = ['Original', 'After Cleaning']
counts = [initial_rows, len(df)]
colors_clean = ['#e74c3c', '#2ecc71']
bars = axes[0, 0].bar(categories, counts, color=colors_clean, alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0, 0].set_ylabel('Record Count', fontweight='bold', fontsize=11)
axes[0, 0].set_title('Records Before & After Cleaning', fontweight='bold', fontsize=12)
axes[0, 0].grid(alpha=0.3, axis='y')
for bar, val in zip(bars, counts):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 20, f'{val:,}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

# Removed records breakdown
removed_count = initial_rows - len(df)
axes[0, 1].text(0.5, 0.7, f'{removed_count:,}', ha='center', va='center', fontsize=20, fontweight='bold', color='#e74c3c')
axes[0, 1].text(0.5, 0.3, 'Records Removed', ha='center', va='center', fontsize=14, fontweight='bold')
axes[0, 1].set_xlim(0, 1)
axes[0, 1].set_ylim(0, 1)
axes[0, 1].axis('off')
axes[0, 1].set_title('Data Quality Improvement', fontweight='bold', fontsize=12)

# Price distribution before & after outlier removal
axes[1, 0].hist(df[target_col], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1)
axes[1, 0].axvline(Q5, color='red', linestyle='--', linewidth=2, label=f'5th: RM{Q5:,.0f}')
axes[1, 0].axvline(Q95, color='red', linestyle='--', linewidth=2, label=f'95th: RM{Q95:,.0f}')
axes[1, 0].set_xlabel('Price (RM)', fontweight='bold', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontweight='bold', fontsize=11)
axes[1, 0].set_title('Price Distribution After Cleaning', fontweight='bold', fontsize=12)
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3, axis='y')

# Data quality metrics
metrics = ['Duplicates', 'Missing\nTargets', 'Outliers']
values = [initial_rows - before, 0, before - len(df)]
bars = axes[1, 1].bar(metrics, values, color=['#3498db', '#e74c3c', '#f39c12'], alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1, 1].set_ylabel('Count', fontweight='bold', fontsize=11)
axes[1, 1].set_title('Data Quality Issues Resolved', fontweight='bold', fontsize=12)
axes[1, 1].grid(alpha=0.3, axis='y')
for bar, val in zip(bars, values):
    if val > 0:
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, val, f'{int(val)}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.suptitle('DATA CLEANING PROCESS', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('02_Data_Cleaning.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_Data_Cleaning.png")
plt.show()
plt.pause(1)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n[FEATURE ENGINEERING]")

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(f"✓ Encoded {len(categorical_cols)} categorical features")

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

initial_features = len(df.columns) - 1
for feat in top_feature_names:
    df[f'{feat}_squared'] = df[feat] ** 2
    df[f'{feat}_sqrt'] = np.sqrt(np.abs(df[feat]) + 1)

print(f"✓ Created polynomial features")

interaction_count = 0
for i in range(len(top_feature_names)):
    for j in range(i+1, len(top_feature_names)):
        feat1, feat2 = top_feature_names[i], top_feature_names[j]
        df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
        interaction_count += 1

print(f"✓ Created {interaction_count} interaction terms")

final_features = len(df.columns) - 1
df = df.drop(columns=categorical_cols)

missing_count = df.isnull().sum().sum()
if missing_count > 0:
    numerical_cols_imp = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols_imp:
        numerical_cols_imp.remove(target_col)
    
    if len(numerical_cols_imp) > 0:
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df[numerical_cols_imp] = imputer.fit_transform(df[numerical_cols_imp])
        print(f"✓ Applied KNN imputation")
else:
    print(f"✓ No missing values")

# FEATURE ENGINEERING VISUALIZATION
print("\n[GENERATING FEATURE ENGINEERING VISUALIZATION]")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Feature count
ax1 = fig.add_subplot(gs[0, 0])
stages = ['Original', 'Encoded', 'After\nEngineering']
feature_counts = [initial_features, initial_features + len(categorical_cols), final_features]
bars = ax1.bar(stages, feature_counts, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Feature Count', fontweight='bold', fontsize=11)
ax1.set_title('Feature Count Progression', fontweight='bold', fontsize=12)
ax1.grid(alpha=0.3, axis='y')
for bar, val in zip(bars, feature_counts):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1, f'{int(val)}', 
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# Feature type breakdown
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

# Top correlated features
ax3 = fig.add_subplot(gs[1, :])
top_correlations = sorted([(f, correlations_dict[f]) for f in top_feature_names], key=lambda x: x[1], reverse=True)
features_names = [f[0] for f in top_correlations]
features_corr = [f[1] for f in top_correlations]
bars = ax3.barh(features_names, features_corr, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Absolute Correlation with Target', fontweight='bold', fontsize=11)
ax3.set_title('Top Features Selected for Polynomial/Interaction Features', fontweight='bold', fontsize=12)
ax3.grid(alpha=0.3, axis='x')
for bar, val in zip(bars, features_corr):
    ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
             va='center', fontweight='bold', fontsize=10)

plt.suptitle('FEATURE ENGINEERING PROCESS', fontsize=15, fontweight='bold')
plt.savefig('03_Feature_Engineering.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_Feature_Engineering.png")
plt.show()
plt.pause(1)

# ============================================================================
# TRAIN-TEST SPLIT & SCALING
# ============================================================================

print("\n[TRAIN-TEST SPLIT & SCALING]")

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True
)

print(f"✓ Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

numerical_features_all = X_train.select_dtypes(include=[np.number]).columns.tolist()
X_train_scaled[numerical_features_all] = scaler.fit_transform(X_train[numerical_features_all])
X_test_scaled[numerical_features_all] = scaler.transform(X_test[numerical_features_all])

print(f"✓ Scaling applied")

# DATA PREPARATION VISUALIZATION
print("\n[GENERATING DATA PREPARATION VISUALIZATION]")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Train-test split
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

# Dataset sizes
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

# Features count
ax3 = fig.add_subplot(gs[0, 2])
ax3.text(0.5, 0.6, f'{X_train.shape[1]}', ha='center', va='center', 
         fontsize=22, fontweight='bold', color='#2ecc71')
ax3.text(0.5, 0.2, 'Features', ha='center', va='center', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_title('Feature Dimension', fontweight='bold', fontsize=12)

# Before scaling distribution
ax4 = fig.add_subplot(gs[1, 0:2])
sample_feature = numerical_features_all[0] if len(numerical_features_all) > 0 else X_train.columns[0]
ax4.hist(X_train[sample_feature], bins=30, color='#e74c3c', alpha=0.7, label='Before Scaling', edgecolor='black')
ax4.set_xlabel(f'{sample_feature} Value', fontweight='bold', fontsize=11)
ax4.set_ylabel('Frequency', fontweight='bold', fontsize=11)
ax4.set_title(f'Feature Distribution Before Scaling (Sample: {sample_feature})', fontweight='bold', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3, axis='y')

# After scaling distribution
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(X_train_scaled[sample_feature], bins=30, color='#2ecc71', alpha=0.7, label='After Scaling', edgecolor='black')
ax5.set_xlabel(f'{sample_feature} Value (Scaled)', fontweight='bold', fontsize=11)
ax5.set_ylabel('Frequency', fontweight='bold', fontsize=11)
ax5.set_title(f'Feature Distribution After Scaling', fontweight='bold', fontsize=12)
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3, axis='y')

plt.suptitle('DATA PREPARATION: SPLITTING & SCALING', fontsize=15, fontweight='bold')
plt.savefig('04_Data_Preparation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_Data_Preparation.png")
plt.show()
plt.pause(1)

# ============================================================================
# MODEL 1: LINEAR REGRESSION
# ============================================================================

print("\n[MODEL 1: LINEAR REGRESSION]")

model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)

print(f"✓ Linear Regression trained")

y_train_lr = model_lr.predict(X_train_scaled)
y_pred_lr = model_lr.predict(X_test_scaled)

r2_train_lr = r2_score(y_train, y_train_lr)
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)

print(f"✓ Test R²: {r2_lr:.6f} | RMSE: RM{rmse_lr:,.0f} | MAE: RM{mae_lr:,.0f} | MAPE: {mape_lr*100:.2f}%")

# ============================================================================
# MODEL 2: RIDGE REGRESSION
# ============================================================================

print("\n[MODEL 2: RIDGE REGRESSION]")

param_grid_ridge = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

ridge_search = GridSearchCV(
    Ridge(random_state=42),
    param_grid_ridge, cv=kfold, scoring='r2', n_jobs=-1, verbose=0
)
ridge_search.fit(X_train_scaled, y_train)
model_ridge = ridge_search.best_estimator_

print(f"✓ Best CV R²: {ridge_search.best_score_:.6f}")

y_train_ridge = model_ridge.predict(X_train_scaled)
y_pred_ridge = model_ridge.predict(X_test_scaled)

r2_train_ridge = r2_score(y_train, y_train_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mape_ridge = mean_absolute_percentage_error(y_test, y_pred_ridge)

print(f"✓ Test R²: {r2_ridge:.6f} | RMSE: RM{rmse_ridge:,.0f} | MAE: RM{mae_ridge:,.0f} | MAPE: {mape_ridge*100:.2f}%")

# ============================================================================
# MODEL 3: GRADIENT BOOSTING
# ============================================================================

print("\n[MODEL 3: GRADIENT BOOSTING]")

# REMOVE the 'regressor__' prefix from everything
param_grid_gb = {
    'n_estimators': [500],           
    'learning_rate': [0.05],        
    'max_depth': [4, 5, 6],       
    'subsample': [0.8, 1.0],        
    'max_features': ['sqrt']        
}

gb_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42, validation_fraction=0.1, n_iter_no_change=15),
    param_grid_gb, # Use the cleaned grid
    cv=kfold, 
    scoring='r2', 
    n_jobs=-1, 
    verbose=0
)

gb_search.fit(X_train_scaled, y_train)
model_gb = gb_search.best_estimator_

print(f"✓ Best CV R²: {gb_search.best_score_:.6f}")

y_train_gb = model_gb.predict(X_train_scaled)
y_pred_gb = model_gb.predict(X_test_scaled)

r2_train_gb = r2_score(y_train, y_train_gb)
r2_gb = r2_score(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mape_gb = mean_absolute_percentage_error(y_test, y_pred_gb)

print(f"✓ Test R²: {r2_gb:.6f} | RMSE: RM{rmse_gb:,.0f} | MAE: RM{mae_gb:,.0f} | MAPE: {mape_gb*100:.2f}%")

# ============================================================================
# MODEL 4: RANDOM FOREST
# ============================================================================

print("\n[MODEL 4: RANDOM FOREST]")

param_grid_rf = {
    'n_estimators': [200],
    'max_depth': [15, 20, 25],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt'],
}

rf_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid_rf, cv=kfold, scoring='r2', n_jobs=-1, verbose=0
)
rf_search.fit(X_train, y_train)
model_rf = rf_search.best_estimator_

print(f"✓ Best CV R²: {rf_search.best_score_:.6f}")

y_train_rf = model_rf.predict(X_train)
y_pred_rf = model_rf.predict(X_test)

r2_train_rf = r2_score(y_train, y_train_rf)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)

print(f"✓ Test R²: {r2_rf:.6f} | RMSE: RM{rmse_rf:,.0f} | MAE: RM{mae_rf:,.0f} | MAPE: {mape_rf*100:.2f}%")

# ============================================================================
# MODEL TRAINING VISUALIZATION
# ============================================================================

print("\n[GENERATING MODEL TRAINING VISUALIZATION]")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Cross-validation scores for Ridge
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

# Cross-validation scores for Gradient Boosting
ax2 = fig.add_subplot(gs[0, 1])
cv_results_gb = gb_search.cv_results_
depths = [p for p in cv_results_gb['param_max_depth']]
ax2.scatter(depths, cv_results_gb['mean_test_score'], s=100, alpha=0.6, color='#e74c3c', edgecolors='black')
ax2.set_xlabel('Max Depth', fontweight='bold', fontsize=11)
ax2.set_ylabel('CV R² Score', fontweight='bold', fontsize=11)
ax2.set_title('Gradient Boosting: Hyperparameter Tuning', fontweight='bold', fontsize=12)
ax2.grid(alpha=0.3)

# Cross-validation scores for Random Forest
ax3 = fig.add_subplot(gs[1, 0])
cv_results_rf = rf_search.cv_results_
depths_rf = [p for p in cv_results_rf['param_max_depth']]
ax3.scatter(depths_rf, cv_results_rf['mean_test_score'], s=100, alpha=0.6, color='#2ecc71', edgecolors='black')
ax3.set_xlabel('Max Depth', fontweight='bold', fontsize=11)
ax3.set_ylabel('CV R² Score', fontweight='bold', fontsize=11)
ax3.set_title('Random Forest: Hyperparameter Tuning', fontweight='bold', fontsize=12)
ax3.grid(alpha=0.3)

# Best parameters summary
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')
params_text = f"""
BEST HYPERPARAMETERS

Ridge:
  α = {model_ridge.alpha}

Gradient Boosting:
  n_estimators = {model_gb.n_estimators}
  max_depth = {model_gb.max_depth}
  learning_rate = {model_gb.learning_rate}

Random Forest:
  n_estimators = {model_rf.n_estimators}
  max_depth = {model_rf.max_depth}
"""
ax4.text(0.1, 0.5, params_text, fontsize=11, family='monospace', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax4.set_title('Optimized Parameters', fontweight='bold', fontsize=12)

plt.suptitle('MODEL TRAINING & HYPERPARAMETER OPTIMIZATION', fontsize=15, fontweight='bold')
plt.savefig('05_Model_Training.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_Model_Training.png")
plt.show()
plt.pause(1)

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("\n[MODEL COMPARISON]")

results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Gradient Boosting', 'Random Forest'],
    'Train R²': [r2_train_lr, r2_train_ridge, r2_train_gb, r2_train_rf],
    'Test R²': [r2_lr, r2_ridge, r2_gb, r2_rf],
    'RMSE': [rmse_lr, rmse_ridge, rmse_gb, rmse_rf],
    'MAE': [mae_lr, mae_ridge, mae_gb, mae_rf],
    'MAPE (%)': [mape_lr*100, mape_ridge*100, mape_gb*100, mape_rf*100]
})

print("\n" + results_df.to_string(index=False))

best_idx = results_df['Test R²'].idxmax()
best_model_name = results_df.iloc[best_idx]['Model']
best_r2 = results_df.iloc[best_idx]['Test R²']
best_rmse = results_df.iloc[best_idx]['RMSE']
best_mae = results_df.iloc[best_idx]['MAE']
best_mape = results_df.iloc[best_idx]['MAPE (%)']

print(f"\n{'='*120}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'='*120}")
print(f"R² = {best_r2:.6f} | RMSE = RM {best_rmse:,.0f} | MAE = RM {best_mae:,.0f} | MAPE = {best_mape:.2f}%")

# ============================================================================
# MODEL EVALUATION VISUALIZATION
# ============================================================================

print("\n[GENERATING MODEL EVALUATION VISUALIZATION]")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

# 1. R² Comparison
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(['LR', 'Ridge', 'GB', 'RF'], results_df['Test R²'].values, 
                 color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.axhline(y=0.85, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.85)')
ax1.set_ylabel('R² Score', fontweight='bold', fontsize=11)
ax1.set_title('R² Comparison', fontweight='bold', fontsize=12)
ax1.set_ylim([0, 1.05])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, results_df['Test R²'].values):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# 2. RMSE Comparison
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(['LR', 'Ridge', 'GB', 'RF'], results_df['RMSE'].values, 
                 color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('RMSE (RM)', fontweight='bold', fontsize=11)
ax2.set_title('RMSE Comparison', fontweight='bold', fontsize=12)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, results_df['RMSE'].values):
    ax2.text(bar.get_x() + bar.get_width()/2, val, f'RM{val/1000:.0f}k', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# 3. MAE Comparison
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.bar(['LR', 'Ridge', 'GB', 'RF'], results_df['MAE'].values, 
                 color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('MAE (RM)', fontweight='bold', fontsize=11)
ax3.set_title('MAE Comparison', fontweight='bold', fontsize=12)
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars3, results_df['MAE'].values):
    ax3.text(bar.get_x() + bar.get_width()/2, val, f'RM{val/1000:.0f}k', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# 4. MAPE Comparison
ax4 = fig.add_subplot(gs[1, 0])
bars4 = ax4.bar(['LR', 'Ridge', 'GB', 'RF'], results_df['MAPE (%)'].values, 
                 color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.7, label='< 10%')
ax4.set_ylabel('MAPE (%)', fontweight='bold', fontsize=11)
ax4.set_title('MAPE Comparison', fontweight='bold', fontsize=12)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
for bar, val in zip(bars4, results_df['MAPE (%)'].values):
    ax4.text(bar.get_x() + bar.get_width()/2, val, f'{val:.2f}%', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# 5. Train vs Test R² (Overfitting Analysis)
ax5 = fig.add_subplot(gs[1, 1])
x_pos = np.arange(len(results_df))
width = 0.35
bars_train = ax5.bar(x_pos - width/2, results_df['Train R²'].values, width, 
                      label='Train R²', color='#2ecc71', alpha=0.7, edgecolor='black')
bars_test = ax5.bar(x_pos + width/2, results_df['Test R²'].values, width, 
                     label='Test R²', color='#e74c3c', alpha=0.7, edgecolor='black')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(['LR', 'Ridge', 'GB', 'RF'])
ax5.set_ylabel('R² Score', fontweight='bold', fontsize=11)
ax5.set_title('Overfitting Analysis', fontweight='bold', fontsize=12)
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 4. Performance Heatmap
# Note: Heatmap is generated separately to avoid overlapping with GridSpec
plt.figure(figsize=(12, 6))
sns.heatmap(results_df.set_index('Model'), annot=True, fmt=".2f", cmap='coolwarm', linewidths=1)
plt.title('Final Performance Matrix', fontweight='bold')
plt.savefig('06_Detailed_Performance_Heatmap.png', dpi=300)

# 1. Sync the names and convert back from log to actual Ringgit prices
y_pred_gb = np.expm1(gb_search.best_estimator_.predict(X_test))
y_pred_ridge = np.expm1(ridge_search.best_estimator_.predict(X_test))

# 2. Now your dictionary will find 'y_pred_gb' without crashing
best_preds_dict = {
    'Linear Regression': y_pred_lr,
    'Ridge Regression': y_pred_ridge,
    'Gradient Boosting': y_pred_gb,
    'Random Forest': y_pred_rf
}
plt.suptitle('MODEL EVALUATION & COMPARISON', fontsize=15, fontweight='bold', y=0.995)
plt.savefig('06_Model_Evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_Model_Evaluation.png")
plt.show()
plt.pause(1)
