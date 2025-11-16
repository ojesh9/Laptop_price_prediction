# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


df = pd.read_csv("laptop_price.csv")
print("Shape:", df.shape)
print("Columns:\n", df.columns.tolist())
print(df.head())
print(df.info())

"""2. Data Processing"""

# missing values
print("Missing per column:\n", df.isnull().sum())

# duplicates
dups = df.duplicated().sum()
print("Duplicate rows:", dups)
if dups:
    df = df.drop_duplicates()
    print("Dropped duplicates. New shape:", df.shape)

"""3. Data Understanding"""

# EDA (examples)

num_cols = ["Inches", "CPU_Frequency", "RAM (GB)", "Weight (kg)", "Price (Euro)"]
for c in num_cols:
    if c in df.columns:
        plt.figure(figsize=(6,3))
        sns.histplot(df[c].dropna(), kde=True)
        plt.title(c)
        plt.tight_layout()
        plt.show()

#Categorical frequency examples:

cat_cols = ["Company", "TypeName", "CPU_Company", "GPU_Company", "OpSys"]
for c in cat_cols:
    if c in df.columns:
        plt.figure(figsize=(8,3))
        df[c].value_counts().head(10).plot(kind="bar")
        plt.title(f"Top values: {c}")
        plt.tight_layout()
        plt.show()

# Bivariate / relationship with price:

# scatter: Inches vs Price
if "Inches" in df.columns and "Price (Euro)" in df.columns:
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x="Inches", y="Price (Euro)")
    plt.title("Inches vs Price")
    plt.show()

# boxplot: Company vs Price (top 8 companies)

if "Company" in df.columns:
    top_comp = df["Company"].value_counts().nlargest(8).index
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df[df["Company"].isin(top_comp)], x="Company", y="Price (Euro)")
    plt.title("Company vs Price (top 8)")
    plt.xticks(rotation=45)
    plt.show()

#Heatmap for numeric correlations:

num_present = [c for c in num_cols if c in df.columns]
if len(num_present) >= 2:
    plt.figure(figsize=(6,5))
    sns.heatmap(df[num_present].corr(), annot=True, fmt=".2f")
    plt.title("Numeric features correlation")
    plt.show()

"""3. Data Cleaning & Feature Engineering"""

df_clean = df.copy()

# 1) Normalize RAM column (if it's like "8GB" or "8 GB")
if "RAM (GB)" not in df_clean.columns and "RAM" in df_clean.columns:
    # try to infer
    df_clean["RAM (GB)"] = df_clean["RAM"].astype(str).str.extract(r"(\d+)").astype(float)

# If RAM already numeric, ensure int
if "RAM (GB)" in df_clean.columns:
    df_clean["RAM (GB)"] = pd.to_numeric(df_clean["RAM (GB)"], errors="coerce")

# 2) CPU_Frequency: ensure numeric (e.g., "2.60GHz" -> 2.6)
if "CPU_Frequency" in df_clean.columns:
    df_clean["CPU_Frequency"] = df_clean["CPU_Frequency"].astype(str).str.extract(r"([\d\.]+)").astype(float)

# 3) Weight: remove 'kg' if present
if "Weight (kg)" not in df_clean.columns:
    # try some common column name
    for col in df_clean.columns:
        if "weight" in col.lower():
            df_clean["Weight (kg)"] = df_clean[col]
            break
if "Weight (kg)" in df_clean.columns:
    df_clean["Weight (kg)"] = df_clean["Weight (kg)"].astype(str).str.extract(r"([\d\.]+)").astype(float)

# 4) Parse Memory column
# Common memory field examples: "256GB SSD", "1TB HDD", "256GB SSD + 1TB HDD", "512GB SSD + 512GB SSD"
def parse_memory(mem_str):
    # returns dict: {'SSD_GB': int, 'HDD_GB': int, 'Hybrid_GB': int}
    ssd = 0
    hdd = 0
    other = 0
    if pd.isna(mem_str):
        return ssd, hdd, other
    # replace TB with GB
    mem = str(mem_str).upper().replace("TB", "000GB").replace("T", "000GB")
    # split components
    parts = [p.strip() for p in mem.split("+")]
    for p in parts:
        # find number and type
        m = p.strip()
        # number in GB
        val = 0
        num = pd.Series([x for x in re.findall(r"(\d+)", m)])
        if len(num) > 0:
            # first numeric group in the token
            try:
                val = int(num.iloc[0])
            except:
                val = 0
        # decide SSD vs HDD by keywords
        if "SSD" in m:
            ssd += val
        elif "HDD" in m:
            hdd += val
        else:
            # treat unknown as other
            other += val
    return ssd, hdd, other

import re
ssd_vals, hdd_vals, other_vals = zip(*df_clean["Memory"].apply(parse_memory))
df_clean["SSD_GB"] = ssd_vals
df_clean["HDD_GB"] = hdd_vals
df_clean["Other_Storage_GB"] = other_vals
df_clean["Total_Storage_GB"] = df_clean[["SSD_GB","HDD_GB","Other_Storage_GB"]].sum(axis=1)

# 5) Keep / drop columns you won't use
# Keep a sensible subset; adjust as needed
cols_keep = []
for c in ["Company","Product","TypeName","Inches","ScreenResolution","CPU_Company","CPU_Type",
          "CPU_Frequency","RAM (GB)","SSD_GB","HDD_GB","Total_Storage_GB","GPU_Company","GPU_Type",
          "OpSys","Weight (kg)","Price (Euro)"]:
    if c in df_clean.columns:
        cols_keep.append(c)
df_model = df_clean[cols_keep].copy()
print("Columns kept for modeling:", cols_keep)
print(df_model.head())

#Handle missing values:

# Quick missing handling: numeric -> median, categorical -> mode
num_features = df_model.select_dtypes(include=[np.number]).columns.tolist()
cat_features = df_model.select_dtypes(include=["object"]).columns.tolist()
print("Numeric features:", num_features)
print("Categorical features:", cat_features)

# Simple strategy: fill numeric with median, categorical with mode
for c in num_features:
    df_model[c] = df_model[c].fillna(df_model[c].median())
for c in cat_features:
    df_model[c] = df_model[c].fillna(df_model[c].mode().iloc[0] if not df_model[c].mode().empty else "Unknown")

"""4. Preprocessing pipeline (encoding + scaling)"""

target = "Price (Euro)"
X = df_model.drop(columns=[target])
y = df_model[target].values

# decide feature lists
numeric_feats = ["Inches","CPU_Frequency","RAM (GB)","SSD_GB","HDD_GB","Total_Storage_GB","Weight (kg)"]
numeric_feats = [c for c in numeric_feats if c in X.columns]

# choose categorical features (drop high-cardinality if needed)
categorical_feats = [c for c in X.columns if c not in numeric_feats]
# You may want to drop 'Product' because it's high-cardinality; include Company, TypeName, CPU_Company etc.
if "Product" in categorical_feats:
    categorical_feats.remove("Product")
    # If present, keep Product excluded or create different encoding strategy.

print("Numeric:", numeric_feats)
print("Categorical:", categorical_feats)

# Transformers
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])


preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_feats),
    ("cat", categorical_transformer, categorical_feats),
], remainder="drop")

"""5. Train/Test split and baseline models & Create pipelines and fit three models:"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Linear Regression pipeline
pipe_lr = Pipeline([
    ("pre", preprocessor),
    ("model", LinearRegression())
])

# Random Forest
pipe_rf = Pipeline([
    ("pre", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

# Gradient Boosting
pipe_gb = Pipeline([
    ("pre", preprocessor),
    ("model", GradientBoostingRegressor(random_state=42))
])

# Fit baseline models
for name, pipe in [("LinearRegression", pipe_lr), ("RandomForest", pipe_rf), ("GradientBoosting", pipe_gb)]:
    print("Fitting:", name)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))  # <-- FIXED
    mae = mean_absolute_error(y_test, preds)
    print(f"{name} -> R2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    print("-"*40)

"""6. Hyperparameter tuning (example for RandomForest & GradientBoosting)"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

rf_param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

gb_param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [1.0, 0.8]
}

# GridSearch for RandomForest
gs_rf = GridSearchCV(
    pipe_rf, rf_param_grid, cv=3,
    scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1
)


gs_rf.fit(X_train, y_train)
print("Best RF params:", gs_rf.best_params_)

rf_best = gs_rf.best_estimator_
pred_rf = rf_best.predict(X_test)

#  no squared=False
print(
    "RF tuned -> R2:", r2_score(y_test, pred_rf),
    "RMSE:", np.sqrt(mean_squared_error(y_test, pred_rf))
)

# GridSearch for GradientBoosting
gs_gb = GridSearchCV(
    pipe_gb, gb_param_grid, cv=3,
    scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1
)
gs_gb.fit(X_train, y_train)
print("Best GB params:", gs_gb.best_params_)

gb_best = gs_gb.best_estimator_
pred_gb = gb_best.predict(X_test)

print(
    "GB tuned -> R2:", r2_score(y_test, pred_gb),
    "RMSE:", np.sqrt(mean_squared_error(y_test, pred_gb))
)

"""7. Model evaluation function + final metrics"""

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def eval_preds(y_true, y_pred, model_name="Model"):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # ✅ no squared arg
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} -> R2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Evaluate best models
eval_preds(y_test, pred_rf, "RandomForest (tuned)")
eval_preds(y_test, pred_gb, "GradientBoosting (tuned)")
# linear baseline
eval_preds(y_test, pipe_lr.predict(X_test), "LinearRegression (baseline)")

# Confusion matrix is for classification; for regression inspect residuals:

residuals = y_test - pred_gb
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title("Residuals (GB tuned)")
plt.show()

# scatter predicted vs actual
plt.figure(figsize=(6,6))
plt.scatter(y_test, pred_gb, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (GB tuned)")
plt.show()

"""8. Feature importance (for tree-based models)"""

# extract feature names from preprocessor
def get_feature_names_from_preprocessor(preprocessor):
    # numeric
    num_names = numeric_feats
    # categorical names:
    cat_transformer = preprocessor.named_transformers_["cat"]["onehot"]
    cat_cols = categorical_feats
    cat_names = list(cat_transformer.get_feature_names_out(cat_cols))
    return num_names + cat_names

feat_names = get_feature_names_from_preprocessor(rf_best.named_steps['pre'])
importances = rf_best.named_steps["model"].feature_importances_
feat_imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(30)
plt.figure(figsize=(8,6))
sns.barplot(data=feat_imp_df, x="importance", y="feature")
plt.title("Top feature importances (RandomForest)")
plt.tight_layout()
plt.show()

"""9. Save best model"""

# Choose the best model based on CV/test metrics (example uses gb_best)
best_model = gb_best  # or rf_best, whichever performed better
joblib.dump(best_model, "best_laptop_price_model.joblib")
print("Saved model to best_laptop_price_model.joblib")
# To load later:
# loaded = joblib.load("best_laptop_price_model.joblib")
# preds_new = loaded.predict(X_new_df)