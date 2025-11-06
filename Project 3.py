import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# ---------------- CONFIG ----------------
DATA_FILE = "Housing Data.csv"  # expected filename
OUTPUT_DIR = "eda_outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summaries")

# Expected common column names (adjust if dataset differs)
# Replace these with your dataset's actual column names if necessary
CONFIG = {
    'price': 'Price',
    'area_sqft': 'Area',          # total area / square feet
    'bedrooms': 'Bedrooms',
    'bathrooms': 'Bathrooms',
    'built_year': 'YearBuilt',    # or 'Year' or 'Built'
    'date_sold': 'DateSold',      # optional
    'location': 'Location',
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    # amenity columns are auto-detected by prefix match below
    'amenity_prefixes': ['Pool', 'Garage', 'Garden', 'Fireplace', 'Balcony']
}

# ---------------- Helpers ----------------

def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)


def savefig(fig, name, dpi=150):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


# ---------------- Load Data ----------------

def load_data(path=DATA_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please place 'Housing Data.csv' in the current folder.")
    df = pd.read_csv(path)
    return df


# ---------------- Cleaning ----------------

def clean_data(df):
    # Standardize column names by stripping whitespace
    df.columns = [c.strip() for c in df.columns]

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before-after} duplicate rows.")

    # Trim whitespace in string columns
    str_cols = df.select_dtypes(include=['object']).columns
    for c in str_cols:
        df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    return df


# ---------------- Missing Values ----------------

def analyze_missing(df):
    miss_summary = df.isna().sum().sort_values(ascending=False)
    miss_pct = (miss_summary / len(df) * 100).round(2)
    miss_df = pd.concat([miss_summary, miss_pct], axis=1)
    miss_df.columns = ['missing_count', 'missing_pct']
    miss_df.to_csv(os.path.join(SUMMARY_DIR, 'missing_values.csv'))
    print('\nTop missing columns:')
    print(miss_df[miss_df['missing_count']>0].head(10))
    return miss_df


# ---------------- Univariate Analysis ----------------

def univariate_plots(df, price_col):
    # Price distribution
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df[price_col].dropna(), kde=True, ax=ax)
    ax.set_title('Price distribution')
    savefig(fig, 'price_distribution.png')

    # Log price
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(np.log1p(df[price_col].dropna()), kde=True, ax=ax)
    ax.set_title('Log-transformed price distribution')
    savefig(fig, 'price_log_distribution.png')

    # Numerical columns summary
    num = df.select_dtypes(include=[np.number])
    num.describe().to_csv(os.path.join(SUMMARY_DIR, 'numerical_describe.csv'))


# ---------------- Multivariate Analysis ----------------

def correlation_and_heatmap(df, price_col):
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    corr.to_csv(os.path.join(SUMMARY_DIR, 'correlation_matrix.csv'))

    # heatmap
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(corr, annot=False, cmap='vlag', center=0, ax=ax)
    ax.set_title('Correlation matrix heatmap')
    savefig(fig, 'correlation_heatmap.png')

    # show top correlations with price
    if price_col in corr.columns:
        top_corr = corr[price_col].drop(price_col).abs().sort_values(ascending=False)
        print('\nTop features correlated with price:')
        print(top_corr.head(10))
        top_corr.head(20).to_csv(os.path.join(SUMMARY_DIR, 'top_price_correlations.csv'))


def scatter_with_price(df, x_cols, price_col):
    for c in x_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            fig, ax = plt.subplots(figsize=(6,4))
            sns.scatterplot(x=df[c], y=df[price_col], alpha=0.6)
            ax.set_title(f'{c} vs {price_col}')
            savefig(fig, f'scatter_{c}_vs_price.png')


# ---------------- Feature Engineering ----------------

def feature_engineering(df, cfg=CONFIG):
    price_col = cfg['price']
    area_col = cfg['area_sqft']
    built_col = cfg['built_year']

    # Price per sqft
    if area_col in df.columns and price_col in df.columns:
        df['price_per_sqft'] = df[price_col] / df[area_col]
    else:
        print('Area or Price column missing; skipping price_per_sqft creation.')

    # Age of property
    if built_col in df.columns:
        # if year is full date, try to extract year
        try:
            df[built_col] = pd.to_numeric(df[built_col], errors='coerce')
            current_year = pd.Timestamp.now().year
            df['age'] = current_year - df[built_col]
            df.loc[df['age']<0, 'age'] = np.nan
        except Exception:
            print('Failed to create age column from built_year.')

    # Simplify categorical location counts
    if cfg['location'] in df.columns:
        loc_counts = df[cfg['location']].value_counts()
        df['location_count'] = df[cfg['location']].map(loc_counts)

    return df


# ---------------- Size & Feature Impact ----------------

def size_impact_plots(df, cfg=CONFIG):
    price = cfg['price']
    candidates = [cfg['area_sqft'], cfg['bedrooms'], cfg['bathrooms']]
    available = [c for c in candidates if c in df.columns]
    scatter_with_price(df, available, price)

    # Boxplots for bedrooms/bathrooms
    for f in ['Bedrooms','Bathrooms']:
        if f in df.columns and price in df.columns:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.boxplot(x=df[f], y=df[price], ax=ax)
            ax.set_title(f'{f} vs {price} (boxplot)')
            savefig(fig, f'box_{f}_vs_price.png')


# ---------------- Time-series / Market Trends ----------------

def time_series_analysis(df, cfg=CONFIG):
    date_col = cfg.get('date_sold')
    price_col = cfg['price']
    if date_col in df.columns:
        # try parse to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        ts = df.dropna(subset=[date_col, price_col]).set_index(date_col).resample('M')[price_col].median()
        if ts.dropna().empty:
            print('No valid time-series data found after parsing dates.')
            return
        fig, ax = plt.subplots(figsize=(12,5))
        ts.plot(ax=ax)
        ax.set_title('Median Price over time (monthly)')
        savefig(fig, 'median_price_timeseries.png')
    else:
        print('No date_sold column present; skipping time-series analysis.')


# ---------------- Amenities / Clustering ----------------

def detect_amenity_columns(df, cfg=CONFIG):
    amenity_cols = []
    # Detect boolean-like columns (0/1 or Yes/No) or columns with amenity prefixes
    for c in df.columns:
        if any(c.lower().startswith(pref.lower()) for pref in cfg['amenity_prefixes']):
            amenity_cols.append(c)
        elif df[c].dropna().isin([0,1]).all():
            amenity_cols.append(c)
        elif df[c].dropna().isin(['Yes','No','yes','no','Y','N']).all():
            amenity_cols.append(c)
    return list(set(amenity_cols))


def amenity_clustering(df, amenity_cols, n_clusters=4):
    if not amenity_cols:
        print('No amenity columns detected; skipping clustering.')
        return df
    sub = df[amenity_cols].copy()
    # Convert Yes/No -> 1/0
    for c in sub.columns:
        sub[c] = sub[c].replace({'Yes':1,'No':0,'Y':1,'N':0,'yes':1,'no':0}).astype(float)
    sub = sub.fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(sub)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    df['amenity_cluster'] = clusters

    # PCA for 2D plot
    pca = PCA(n_components=2)
    comp = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(comp[:,0], comp[:,1], c=clusters, cmap='tab10', alpha=0.7)
    ax.set_title('Amenity clusters (PCA 2D)')
    savefig(fig, 'amenity_clusters_pca.png')

    return df


# ---------------- Regression Baseline ----------------

def regression_baseline(df, cfg=CONFIG):
    price_col = cfg['price']
    # Select numeric features, drop NaNs in price
    df_reg = df.copy()
    df_reg = df_reg.dropna(subset=[price_col])
    num = df_reg.select_dtypes(include=[np.number])
    # drop columns that leak or are target derivatives
    drop_cols = [price_col]
    X = num.drop(columns=[c for c in drop_cols if c in num.columns], errors='ignore')
    y = df_reg[price_col]

    # simple impute
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42)

    # fit linear model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Linear Regression baseline - MSE: {mse:.2f}, RMSE: {np.sqrt(mse):.2f}, R2: {r2:.3f}')

    # Save coefficients (top positive/negative)
    coefs = pd.Series(lr.coef_, index=X_imp.columns).sort_values(ascending=False)
    coefs.head(10).to_csv(os.path.join(SUMMARY_DIR, 'top_positive_coeffs.csv'))
    coefs.tail(10).to_csv(os.path.join(SUMMARY_DIR, 'top_negative_coeffs.csv'))

    # Statsmodels OLS on a subset of top features for diagnostics
    top_feats = list(coefs.head(10).index) + list(coefs.tail(10).index)
    top_feats = [c for c in top_feats if c in X_imp.columns]
    if top_feats:
        X_ols = sm.add_constant(X_imp[top_feats])
        ols_model = sm.OLS(y, X_ols).fit()
        with open(os.path.join(SUMMARY_DIR, 'ols_summary.txt'), 'w') as f:
            f.write(ols_model.summary().as_text())
    return lr


# ---------------- Main Pipeline ----------------

def run_all():
    ensure_dirs()
    df = load_data()
    print(f'Data shape: {df.shape}')

    df = clean_data(df)
    miss = analyze_missing(df)

    # try mapping config column names if they are different (best-effort)
    # For example, common variants like 'price', 'Price', 'SalePrice'
    cols_lower = {c.lower(): c for c in df.columns}
    # update CONFIG keys if alternate names exist
    for k, v in list(CONFIG.items()):
        if isinstance(v, str) and v not in df.columns:
            alt = v.lower()
            if alt in cols_lower:
                CONFIG[k] = cols_lower[alt]

    price_col = CONFIG['price']
    if price_col not in df.columns:
        # try common alternatives
        for cand in ['Price','price','SalePrice','sale_price']:
            if cand in df.columns:
                price_col = cand
                CONFIG['price'] = cand
                break

    if price_col not in df.columns:
        raise KeyError('Price column not found in dataset. Edit CONFIG in the script to point to the price column.')

    # Ensure numeric price
    df[price_col] = pd.to_numeric(df[price_col].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')

    univariate_plots(df, price_col)
    correlation_and_heatmap(df, price_col)

    # Feature engineering
    df = feature_engineering(df, CONFIG)
    size_impact_plots(df, CONFIG)
    time_series_analysis(df, CONFIG)

    # Amenities and clustering
    amen_cols = detect_amenity_columns(df, CONFIG)
    print('\nDetected amenity columns:', amen_cols)
    df = amenity_clustering(df, amen_cols, n_clusters=4)

    # Regression baseline
    regression_baseline(df, CONFIG)

    # Save cleaned data snapshot
    df.to_csv(os.path.join(SUMMARY_DIR, 'cleaned_data_snapshot.csv'), index=False)
    print('\nEDA complete. Outputs saved to', OUTPUT_DIR)


if __name__ == '__main__':
    run_all()
