import pandas as pd
from pathlib import Path

# 1. Set up project folders
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load raw orders and products
orders = pd.read_csv(RAW_DIR / "orders.csv")
products = pd.read_csv(RAW_DIR / "products.csv")

print("=" * 50)
print("STARTING ORDER DATA CLEANING")
print("=" * 50)

print("Original orders:", len(orders))

# 3. Convert order timestamps to datetime
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="coerce"
)

# 4. Remove exact duplicate rows
orders = orders.drop_duplicates().copy()

print("After removing exact duplicates:", len(orders))

# 5. Keep the latest record for each order ID
orders = (
    orders.sort_values("order_ts", kind="stable")
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)

print("After keeping latest order records:", len(orders))

# 6. Convert order amounts to numeric values
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

# 7. Join product prices and categories
product_info = products[["product_id", "price_inr", "category"]]

orders = orders.merge(
    product_info,
    on="product_id",
    how="left",
    validate="many_to_one"
)

# 8. Fill missing order amounts using the required formula
missing_amounts = orders["order_amount"].isna()

calculated_amounts = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders.loc[missing_amounts, "order_amount"] = (
    calculated_amounts[missing_amounts]
)

print("Missing amounts after cleaning:",
      orders["order_amount"].isna().sum())

# 9. Replace negative delivery times with missing values
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = float("nan")

# 10. Fill missing delivery times for completed orders
completed_mask = orders["status"].str.strip().str.lower().eq("completed")

completed_orders = orders.loc[completed_mask].copy()

category_medians = (
    completed_orders.groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"]
    .fillna(category_medians)
)

# Cancelled orders must retain missing delivery times
cancelled_mask = orders["status"].str.strip().str.lower().eq("cancelled")

orders.loc[cancelled_mask, "delivery_minutes"] = float("nan")

# 11. Remove the temporary product information columns
orders = orders.drop(columns=["price_inr", "category"])

# 12. Save cleaned orders
output_path = CLEAN_DIR / "orders.csv"

orders.to_csv(
    output_path,
    index=False,
    date_format="%Y-%m-%d %H:%M:%S"
)

# 13. Print verification results
completed_mask = orders["status"].str.strip().str.lower().eq("completed")
cancelled_mask = orders["status"].str.strip().str.lower().eq("cancelled")

print("\n" + "=" * 50)
print("CLEANED ORDER DATA RESULTS")
print("=" * 50)

print("Total cleaned orders:", len(orders))

print("\nOrder counts by status:")
print(orders["status"].value_counts())

print("\nMissing order amounts:",
      orders["order_amount"].isna().sum())

print(
    "Completed orders with missing delivery times:",
    orders.loc[completed_mask, "delivery_minutes"].isna().sum()
)

print(
    "Cancelled orders with missing delivery times:",
    orders.loc[cancelled_mask, "delivery_minutes"].isna().sum()
)

print(
    "Completed order amount total:",
    round(orders.loc[completed_mask, "order_amount"].sum(), 2)
)

print("\nCleaned data saved to:")
print(output_path)

print("\nORDER DATA CLEANING COMPLETED")