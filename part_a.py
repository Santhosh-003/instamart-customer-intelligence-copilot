import pandas as pd
from pathlib import Path

# Folder containing the original datasets
RAW = Path("data/raw")

# Load the raw datasets
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")
orders = pd.read_csv(RAW / "orders.csv")

# --------------------------------------------------
# 1. Display dataset shapes and data types
# --------------------------------------------------

datasets = {
    "Customers": customers,
    "Products": products,
    "Orders": orders
}

for name, df in datasets.items():
    print("\n" + "=" * 50)
    print(f"{name.upper()} DATA AUDIT")
    print("=" * 50)

    print("\nShape (rows, columns):")
    print(df.shape)

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing_count = df.isna().sum()
    missing_percentage = (missing_count / len(df)) * 100

    missing_report = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percentage.round(2)
    })

    print(missing_report)


# --------------------------------------------------
# 2. Audit orders
# --------------------------------------------------

print("\n" + "=" * 50)
print("ORDERS AUDIT CHECKPOINTS")
print("=" * 50)

order_amount_numeric = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

delivery_numeric = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

order_audit = {
    "Raw order rows": len(orders),
    "Exact duplicate rows": int(orders.duplicated().sum()),
    "Pending rows": int(
        (orders["status"].str.lower() == "pending").sum()
    ),
    "Distinct order IDs": int(orders["order_id"].nunique()),
    "Non-numeric order amounts": int(order_amount_numeric.isna().sum()),
    "Negative delivery times": int((delivery_numeric < 0).sum()),
    "Blank delivery times": int(
    orders["delivery_minutes"].isna().sum()
    + orders["delivery_minutes"].astype(str).str.strip().eq("").sum()
)
}

for label, value in order_audit.items():
    print(f"{label}: {value}")


# --------------------------------------------------
# 3. Audit customers
# --------------------------------------------------

print("\n" + "=" * 50)
print("CUSTOMERS AUDIT CHECKPOINTS")
print("=" * 50)

age_numeric = pd.to_numeric(customers["age"], errors="coerce")

customer_audit = {
    "Blank ages": int(
        customers["age"].isna().sum()
        + customers["age"].astype(str).str.strip().eq("").sum()
    ),
    "Ages outside 10-100": int(
        ((age_numeric < 10) | (age_numeric > 100)).sum()
    ),
    "Distinct raw city spellings": int(customers["city"].nunique()),
    "Distinct non-blank raw gender spellings": int(
        customers.loc[
            customers["gender"].astype(str).str.strip().ne("")
            & customers["gender"].notna(),
            "gender"
        ].nunique()
    ),
    "Blank genders": int(
        customers["gender"].isna().sum()
        + customers["gender"].astype(str).str.strip().eq("").sum()
    )
}

for label, value in customer_audit.items():
    print(f"{label}: {value}")

print("\nRAW DATA AUDIT COMPLETED")