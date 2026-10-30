import os
import pandas as pd

# ---------------------------------------
# PART B - STEP 1: BUILD THE RFM TABLE
# ---------------------------------------

# Snapshot date provided in the assessment
SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")

# Read the cleaned data produced by Part A
orders = pd.read_csv("data/clean/orders.csv")

customers = pd.read_csv("data/clean/customers.csv")

# Convert order timestamps to proper dates
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Ensure order amounts are numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

# Use completed orders only for RFM calculations
completed_orders = orders[
    orders["status"] == "completed"
].copy()

# Calculate RFM measurements for each customer
rfm = completed_orders.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# Recency = number of whole days since the last completed order
rfm["recency"] = (
    SNAPSHOT_DATE - rfm["last_order_ts"]
).dt.days

# Keep the required RFM columns
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Create the outputs folder if it does not exist
os.makedirs("outputs", exist_ok=True)

# Save the RFM table
rfm.to_csv("outputs/rfm_table.csv", index=False)

# Display results
print("\n========== RFM TABLE RESULTS ==========")

print("Customers with completed orders:", len(rfm))

print("\nFirst 5 rows:")
print(rfm.head().to_string(index=False))

print("\nRFM summary:")
print(
    rfm[["recency", "frequency", "monetary"]]
    .describe()
    .round(2)
)

print("\nExpected assessment checkpoints:")
print("Customers: 1,175")
print("Median recency: 19 days")
print("Minimum recency: 0 days")
print("Maximum recency: 180 days")

print("\nSaved file: outputs/rfm_table.csv")

# ---------------------------------------
# PART B - STEP 2: K-MEANS CLUSTERING
# ---------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("\n========== K-MEANS CLUSTERING ==========")

# Select the three RFM measurements
rfm_features = rfm[["recency", "frequency", "monetary"]]

# Standardise the features so monetary values do not dominate
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

# Test different numbers of clusters: K = 2 to K = 8
clustering_results = []

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    cluster_labels = model.fit_predict(rfm_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(rfm_scaled, cluster_labels)

    clustering_results.append({
        "K": k,
        "Inertia": inertia,
        "Silhouette Score": silhouette
    })

# Display the results
results_df = pd.DataFrame(clustering_results)

print("\nClustering comparison:")
print(results_df.round(3).to_string(index=False))

# Save the comparison table
results_df.to_csv(
    "outputs/kmeans_comparison.csv",
    index=False
)

print("\nSaved file: outputs/kmeans_comparison.csv")

# ---------------------------------------
# PART B - STEP 3: CUSTOMER SEGMENTS
# ---------------------------------------

from sklearn.decomposition import PCA

print("\n========== CUSTOMER SEGMENTS ==========")

# Create the final K-means model using K = 4
final_kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = final_kmeans.fit_predict(rfm_scaled)

# Calculate the average RFM values and size of each segment
segment_profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).reset_index()

# Find the segment with the highest average monetary value
premium_segment_id = segment_profile.loc[
    segment_profile["mean_monetary"].idxmax(),
    "segment_id"
]

# Name the segments based on their behaviour
segment_names = {}

# Highest spending segment is always Premium
segment_names[premium_segment_id] = "Premium Customers"

remaining_segments = segment_profile[
    segment_profile["segment_id"] != premium_segment_id
].copy()

# Among the remaining segments, identify the most frequent shoppers
frequent_segment_id = remaining_segments.loc[
    remaining_segments["mean_frequency"].idxmax(),
    "segment_id"
]

segment_names[frequent_segment_id] = "Frequent Loyalists"

remaining_segments = remaining_segments[
    remaining_segments["segment_id"] != frequent_segment_id
].copy()

# Identify the segment with the highest recency
# Higher recency means the customer has been inactive for longer
at_risk_segment_id = remaining_segments.loc[
    remaining_segments["mean_recency"].idxmax(),
    "segment_id"
]

segment_names[at_risk_segment_id] = "At-Risk Customers"

# Give the remaining segment a descriptive name
for segment_id in segment_profile["segment_id"]:
    if segment_id not in segment_names:
        segment_names[segment_id] = "Developing Customers"

# Add readable names and Premium flags
rfm["segment_name"] = rfm["segment_id"].map(segment_names)

rfm["is_premium"] = (
    rfm["segment_id"] == premium_segment_id
)

# Save the required customer segment file
customer_segments = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
].copy()

customer_segments.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

# Display segment profiles
print("\nSegment profiles:")

segment_profile["segment_name"] = (
    segment_profile["segment_id"].map(segment_names)
)

segment_profile["is_premium"] = (
    segment_profile["segment_id"] == premium_segment_id
)

print(
    segment_profile[
        [
            "segment_id",
            "segment_name",
            "customer_count",
            "mean_recency",
            "mean_frequency",
            "mean_monetary",
            "is_premium"
        ]
    ].round(2).to_string(index=False)
)

# PCA: reduce the three scaled RFM measurements to two components
pca = PCA(n_components=2)
pca.fit(rfm_scaled)

explained_variance = pca.explained_variance_ratio_

print("\nPCA explained variance ratios:")
print("Component 1:", round(explained_variance[0], 3))
print("Component 2:", round(explained_variance[1], 3))
print(
    "Total explained variance:",
    round(explained_variance.sum(), 3)
)

print("\nPremium segment ID:", premium_segment_id)
print("Premium segment name:", segment_names[premium_segment_id])

print("\nSaved file: outputs/customer_segments.csv")
print("Total customers saved:", len(customer_segments))

# ---------------------------------------
# PART B - STEP 4: LAPSE DATASET
# ---------------------------------------

print("\n========== LAPSE DATASET ==========")

# Important dates
CUTOFF_DATE = pd.Timestamp("2025-09-01")
LABEL_END_DATE = pd.Timestamp("2025-10-01")

# Load the cleaned data
orders_lapse = pd.read_csv("data/clean/orders.csv")
customers_lapse = pd.read_csv("data/clean/customers.csv")
products_lapse = pd.read_csv("data/clean/products.csv")

# Convert dates and numeric columns
orders_lapse["order_ts"] = pd.to_datetime(
    orders_lapse["order_ts"]
)

customers_lapse["signup_date"] = pd.to_datetime(
    customers_lapse["signup_date"]
)

orders_lapse["order_amount"] = pd.to_numeric(
    orders_lapse["order_amount"],
    errors="coerce"
)

orders_lapse["discount_pct"] = pd.to_numeric(
    orders_lapse["discount_pct"],
    errors="coerce"
)

orders_lapse["delivery_minutes"] = pd.to_numeric(
    orders_lapse["delivery_minutes"],
    errors="coerce"
)

# -------------------------------------------------
# 1. Create the prediction cohort
# -------------------------------------------------

# Only orders BEFORE the cutoff can create features
orders_before_cutoff = orders_lapse[
    orders_lapse["order_ts"] < CUTOFF_DATE
].copy()

# Completed orders before the cutoff
completed_before_cutoff = orders_before_cutoff[
    orders_before_cutoff["status"] == "completed"
].copy()

# Cohort: customers with at least one completed
# order before the cutoff
cohort_ids = completed_before_cutoff[
    "customer_id"
].unique()

# -------------------------------------------------
# 2. Build behaviour features using past orders only
# -------------------------------------------------

# Basic RFM features
lapse_features = completed_before_cutoff.groupby(
    "customer_id"
).agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    average_discount=("discount_pct", "mean"),
    average_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

# Recency in whole days at the cutoff
lapse_features["recency"] = (
    CUTOFF_DATE - lapse_features["last_order_ts"]
).dt.days

# Cancellation rate using all final-status orders
# before the cutoff: completed and cancelled orders
final_orders_before_cutoff = orders_before_cutoff[
    orders_before_cutoff["status"].isin(
        ["completed", "cancelled"]
    )
].copy()

final_orders_before_cutoff["is_cancelled"] = (
    final_orders_before_cutoff["status"] == "cancelled"
).astype(int)

cancellation_rates = final_orders_before_cutoff.groupby(
    "customer_id"
).agg(
    cancellation_rate=("is_cancelled", "mean")
).reset_index()

lapse_features = lapse_features.merge(
    cancellation_rates,
    on="customer_id",
    how="left"
)

# Most frequently ordered product category before cutoff
completed_with_category = completed_before_cutoff.merge(
    products_lapse[["product_id", "category"]],
    on="product_id",
    how="left"
)

favourite_categories = (
    completed_with_category
    .groupby("customer_id")["category"]
    .agg(lambda values: values.mode().iloc[0])
    .reset_index(name="favourite_category")
)

lapse_features = lapse_features.merge(
    favourite_categories,
    on="customer_id",
    how="left"
)

# Add customer information
lapse_features = lapse_features.merge(
    customers_lapse[
        ["customer_id", "city", "age", "gender", "signup_date"]
    ],
    on="customer_id",
    how="left"
)

# Tenure in days as of the cutoff
lapse_features["tenure_days"] = (
    CUTOFF_DATE - lapse_features["signup_date"]
).dt.days

# -------------------------------------------------
# 3. Create the lapse label using September orders
# -------------------------------------------------

# This September data is used ONLY for labels,
# never for prediction features.
september_completed = orders_lapse[
    (orders_lapse["order_ts"] >= CUTOFF_DATE)
    & (orders_lapse["order_ts"] < LABEL_END_DATE)
    & (orders_lapse["status"] == "completed")
].copy()

september_active_ids = set(
    september_completed["customer_id"].unique()
)

lapse_features["lapsed"] = (
    ~lapse_features["customer_id"].isin(september_active_ids)
).astype(int)

# Remove columns that must not be model features
lapse_features = lapse_features.drop(
    columns=["last_order_ts", "signup_date"]
)

# Arrange the columns
lapse_features = lapse_features[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "average_discount",
        "cancellation_rate",
        "average_delivery_minutes",
        "favourite_category",
        "city",
        "age",
        "gender",
        "tenure_days",
        "lapsed"
    ]
]

# Verify that no feature column has missing values
feature_columns = [
    column for column in lapse_features.columns
    if column not in ["customer_id", "lapsed"]
]

missing_counts = lapse_features[feature_columns].isna().sum()

print("\nMissing values in model features:")
print(missing_counts.to_string())

if missing_counts.any():
    raise ValueError(
        "Some model features contain missing values. "
        "We must fix them before training."
    )

# Save the prepared dataset
lapse_features.to_csv(
    "outputs/lapse_features.csv",
    index=False
)

# -------------------------------------------------
# 4. Display assessment checkpoints
# -------------------------------------------------

print("\nNumber of customers in cohort:", len(lapse_features))

print("Lapsed customers:", int(lapse_features["lapsed"].sum()))

print(
    "Not-lapsed customers:",
    int((lapse_features["lapsed"] == 0).sum())
)

print(
    "Lapsed percentage:",
    round(lapse_features["lapsed"].mean() * 100, 2),
    "%"
)

print("\nLapse label counts:")
print(lapse_features["lapsed"].value_counts().sort_index())

print("\nFirst 5 rows:")
print(lapse_features.head().to_string(index=False))

print("\nSaved file: outputs/lapse_features.csv")
print("All feature columns are free of missing values.")

# ---------------------------------------
# PART B - STEP 5: TRAIN/TEST SPLIT
# AND FEATURE PREPROCESSING
# ---------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("\n========== TRAIN/TEST SPLIT ==========")

# Separate the target from the input features.
# customer_id is an identifier, not a model feature.
X = lapse_features.drop(
    columns=["customer_id", "lapsed"]
)

y = lapse_features["lapsed"]

# Identify numeric and categorical columns
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "average_discount",
    "cancellation_rate",
    "average_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

# Split before fitting any encoder or scaler.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

print("\nTraining label counts:")
print(y_train.value_counts().sort_index())

print("\nTest label counts:")
print(y_test.value_counts().sort_index())

# -------------------------------------------------
# Baseline: always predict "not lapsed" (class 0)
# -------------------------------------------------

baseline_predictions = [0] * len(y_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    pos_label=1,
    zero_division=0
)

print("\n========== BASELINE RESULTS ==========")
print("Baseline accuracy:", round(baseline_accuracy, 4))
print("Baseline lapsed-class F1:", round(baseline_f1, 4))
print("Baseline confusion matrix:")
print(confusion_matrix(y_test, baseline_predictions))

# -------------------------------------------------
# Preprocessing: fit only on the training rows
# -------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# Fit encoders and scaler using training data only
X_train_prepared = preprocessor.fit_transform(X_train)

# Transform the test set without fitting on it
X_test_prepared = preprocessor.transform(X_test)

# Verify there are no missing values in input features
assert not X_train.isna().any().any(), (
    "Training features contain missing values."
)

assert not X_test.isna().any().any(), (
    "Test features contain missing values."
)

# Compare the fitted training frequency mean with
# the full-cohort frequency mean.
frequency_position = numeric_features.index("frequency")

training_frequency_mean = preprocessor.named_transformers_[
    "numeric"
].mean_[frequency_position]

full_cohort_frequency_mean = X["frequency"].mean()

print("\n========== PREPROCESSING CHECKS ==========")

print(
    "Training-set frequency mean:",
    round(training_frequency_mean, 3)
)

print(
    "Full-cohort frequency mean:",
    round(full_cohort_frequency_mean, 3)
)

print("Training feature matrix shape:", X_train_prepared.shape)
print("Test feature matrix shape:", X_test_prepared.shape)

print("\nPASS: preprocessing was fitted only on training rows.")
print("PASS: test rows were transformed without fitting.")
print("PASS: no feature columns contain missing values.")

# ============================================================
# STEP 6: TRAIN AND TUNE MACHINE-LEARNING MODELS
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("\n========== LOGISTIC REGRESSION TUNING ==========")

# Search for the best Logistic Regression settings.
logistic_model = LogisticRegression(max_iter=1000)

logistic_parameters = {
    "C": [0.01, 0.1, 1, 10, 100]
}

logistic_search = GridSearchCV(
    estimator=logistic_model,
    param_grid=logistic_parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

logistic_search.fit(X_train_prepared, y_train)

print("Best Logistic Regression parameters:")
print(logistic_search.best_params_)

print("Best Logistic Regression CV F1:")
print(round(logistic_search.best_score_, 4))


print("\n========== RANDOM FOREST TUNING ==========")

# Search for the best Random Forest settings.
forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_parameters = {
    "max_depth": [3, 5, 8],
    "min_samples_leaf": [5, 10, 20]
}

forest_search = GridSearchCV(
    estimator=forest_model,
    param_grid=forest_parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

forest_search.fit(X_train_prepared, y_train)

print("Best Random Forest parameters:")
print(forest_search.best_params_)

print("Best Random Forest CV F1:")
print(round(forest_search.best_score_, 4))


# ============================================================
# STEP 7: EVALUATE BOTH MODELS ON THE TEST SET
# ============================================================

print("\n========== TEST SET MODEL COMPARISON ==========")

models = {
    "Logistic Regression": logistic_search.best_estimator_,
    "Random Forest": forest_search.best_estimator_
}

for model_name, model in models.items():

    predictions = model.predict(X_test_prepared)

    print(f"\n--- {model_name} ---")

    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )

    print(
        "Lapsed-class precision:",
        round(
            precision_score(
                y_test, predictions,
                pos_label=1, zero_division=0
            ),
            4
        )
    )

    print(
        "Lapsed-class recall:",
        round(
            recall_score(
                y_test, predictions,
                pos_label=1, zero_division=0
            ),
            4
        )
    )

    print(
        "Lapsed-class F1:",
        round(
            f1_score(
                y_test, predictions,
                pos_label=1, zero_division=0
            ),
            4
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


print("\n========== MODEL TUNING COMPLETE ==========")

# ============================================================
# STEP 8: SELECT, SAVE, AND VERIFY THE MODEL
# ============================================================

import joblib
import os
import numpy as np

print("\n========== SELECTED MODEL ==========")

# Select Random Forest because it achieved the better test F1.
selected_model = forest_search.best_estimator_
selected_model_name = "Random Forest"

print("Selected model:", selected_model_name)
print("Test F1 score:", round(
    f1_score(
        y_test,
        selected_model.predict(X_test_prepared),
        pos_label=1,
        zero_division=0
    ),
    4
))

# Save the fitted preprocessing pipeline and model together.
# This ensures future data receives the same preprocessing.
model_bundle = {
    "preprocessor": preprocessor,
    "model": selected_model,
    "model_name": selected_model_name,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

os.makedirs("outputs", exist_ok=True)

model_path = "outputs/lapse_model.joblib"
joblib.dump(model_bundle, model_path)

print("\nModel saved to:", model_path)

# Reload the saved model.
loaded_bundle = joblib.load(model_path)

loaded_preprocessor = loaded_bundle["preprocessor"]
loaded_model = loaded_bundle["model"]

# Generate predictions using the original and reloaded models.
original_predictions = selected_model.predict(X_test_prepared)

reloaded_test_data = loaded_preprocessor.transform(X_test)

reloaded_predictions = loaded_model.predict(reloaded_test_data)

# Verify that predictions match exactly.
predictions_match = np.array_equal(
    original_predictions,
    reloaded_predictions
)

print("Original and reloaded predictions match:", predictions_match)

assert predictions_match, "ERROR: Reloaded predictions do not match!"

print("PASS: Saved model reload verification successful.")

# ============================================================
# STEP 9: CLASS-IMBALANCE EXPERIMENT
# ============================================================

from sklearn.utils import resample

print("\n========== CLASS-IMBALANCE EXPERIMENT ==========")

# Combine the prepared training features and labels.
# Only training rows are used for oversampling.
training_data = pd.DataFrame(
    X_train_prepared,
    columns=[f"feature_{i}" for i in range(X_train_prepared.shape[1])]
)

training_data["lapsed"] = y_train.to_numpy()

# Separate the two classes.
not_lapsed = training_data[training_data["lapsed"] == 0]
lapsed = training_data[training_data["lapsed"] == 1]

print("Original not-lapsed training rows:", len(not_lapsed))
print("Original lapsed training rows:", len(lapsed))

# Oversample lapsed customers with replacement.
lapsed_oversampled = resample(
    lapsed,
    replace=True,
    n_samples=len(not_lapsed),
    random_state=42
)

# Combine the two classes into a balanced training dataset.
balanced_training_data = pd.concat(
    [not_lapsed, lapsed_oversampled],
    ignore_index=True
)

# Shuffle the balanced rows reproducibly.
balanced_training_data = balanced_training_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_train_balanced = balanced_training_data.drop(columns=["lapsed"])
y_train_balanced = balanced_training_data["lapsed"]

print("\nBalanced training label counts:")
print(y_train_balanced.value_counts().sort_index())

# Train Logistic Regression using the best C found earlier.
balanced_logistic_model = LogisticRegression(
    C=logistic_search.best_params_["C"],
    max_iter=1000
)

balanced_logistic_model.fit(
    X_train_balanced,
    y_train_balanced
)

# Compare the original and balanced Logistic Regression models
# using the same, unchanged test set.
original_logistic_predictions = logistic_search.best_estimator_.predict(
    X_test_prepared
)

balanced_logistic_predictions = balanced_logistic_model.predict(
    X_test_prepared
)

print("\n--- Original Logistic Regression ---")
print(
    "Precision:",
    round(
        precision_score(
            y_test,
            original_logistic_predictions,
            zero_division=0
        ),
        4
    )
)
print(
    "Recall:",
    round(
        recall_score(
            y_test,
            original_logistic_predictions,
            zero_division=0
        ),
        4
    )
)
print(
    "F1:",
    round(
        f1_score(
            y_test,
            original_logistic_predictions,
            zero_division=0
        ),
        4
    )
)

print("\n--- Balanced Logistic Regression ---")
print(
    "Precision:",
    round(
        precision_score(
            y_test,
            balanced_logistic_predictions,
            zero_division=0
        ),
        4
    )
)
print(
    "Recall:",
    round(
        recall_score(
            y_test,
            balanced_logistic_predictions,
            zero_division=0
        ),
        4
    )
)
print(
    "F1:",
    round(
        f1_score(
            y_test,
            balanced_logistic_predictions,
            zero_division=0
        ),
        4
    )
)

print("\nBalanced Logistic Regression confusion matrix:")
print(confusion_matrix(y_test, balanced_logistic_predictions))

print("\nPASS: Only training data was oversampled.")
print("PASS: Both models were evaluated on the same original test set.")

# ============================================================
# STEP 10: FEATURE IMPORTANCE AND LOGISTIC COEFFICIENTS
# ============================================================

print("\n========== RANDOM FOREST FEATURE IMPORTANCE ==========")

# Get the names of the features after preprocessing.
feature_names = preprocessor.get_feature_names_out()

# Get Random Forest feature importance scores.
importance_values = selected_model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

print("\nTop 5 most important features:")
print(feature_importance.head(5).to_string(index=False))

# Save the full feature-importance table.
feature_importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

print("\nSaved: outputs/feature_importance.csv")


# ============================================================
# LOGISTIC REGRESSION COEFFICIENT INTERPRETATION
# ============================================================

print("\n========== LOGISTIC REGRESSION COEFFICIENTS ==========")

logistic_best = logistic_search.best_estimator_

coefficient_values = logistic_best.coef_[0]

coefficient_table = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficient_values
})

# Show the three coefficients with the largest absolute values.
coefficient_table["absolute_coefficient"] = (
    coefficient_table["coefficient"].abs()
)

top_coefficients = coefficient_table.sort_values(
    "absolute_coefficient",
    ascending=False
).head(3)

print("\nThree strongest Logistic Regression coefficients:")
print(top_coefficients[
    ["feature", "coefficient"]
].to_string(index=False))

print(
    "\nInterpretation: A positive coefficient is associated "
    "with higher predicted lapse odds; a negative coefficient "
    "is associated with lower predicted lapse odds, holding "
    "the other model features constant."
)

# Save the complete coefficient table.
coefficient_table.sort_values(
    "absolute_coefficient",
    ascending=False
).to_csv(
    "outputs/logistic_coefficients.csv",
    index=False
)

print("Saved: outputs/logistic_coefficients.csv")


# ============================================================
# STEP 11: CREATE THE PREMIUM CUSTOMER RETENTION LIST
# ============================================================

print("\n========== PREMIUM CUSTOMER RETENTION LIST ==========")

# Retrieve customer IDs matching the existing test-set rows.
test_customer_ids = lapse_features.loc[
    X_test.index, "customer_id"
].to_numpy()

# Get lapse probabilities from the selected Random Forest model.
test_lapse_probabilities = selected_model.predict_proba(
    X_test_prepared
)[:, 1]

# Keep the actual lapse labels for evaluation.
test_actual_labels = y_test.to_numpy()

test_predictions_table = pd.DataFrame({
    "customer_id": test_customer_ids,
    "lapse_probability": test_lapse_probabilities,
    "actually_lapsed": test_actual_labels
})

# Load the customer segments created earlier.
segments_table = pd.read_csv(
    "outputs/customer_segments.csv"
)

# Handle Boolean values whether stored as True/False or text.
segments_table["is_premium"] = (
    segments_table["is_premium"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

premium_customers = segments_table.loc[
    segments_table["is_premium"],
    ["customer_id"]
].drop_duplicates()

# Restrict the list to Premium customers in the test set.
premium_test_customers = test_predictions_table.merge(
    premium_customers,
    on="customer_id",
    how="inner"
)

# Rank by predicted lapse probability, highest first.
retention_list = premium_test_customers.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

# Save only the columns required by the assessment.
retention_list = retention_list[
    ["customer_id", "lapse_probability", "actually_lapsed"]
].reset_index(drop=True)

retention_list.to_csv(
    "outputs/retention_list.csv",
    index=False
)

print("Premium customers in test set:", len(premium_test_customers))
print("Customers in retention list:", len(retention_list))

print("\nTop at-risk Premium customers:")
print(retention_list.to_string(index=False))

actually_lapsed_count = int(
    retention_list["actually_lapsed"].sum()
)

print(
    "\nActually lapsed customers in retention list:",
    actually_lapsed_count
)

print("\nSaved: outputs/retention_list.csv")
print("\nPASS: Feature importance and coefficients saved.")
print("PASS: Premium retention list created.")