import sqlite3
import pandas as pd
from pathlib import Path

# 1. Set up project folders
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"
DB_PATH = BASE_DIR / "data" / "instamart.db"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

print("Building Instamart SQLite database...")

# 2. Load cleaned customers and orders
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# 3. Load products and save a clean copy
products = pd.read_csv(RAW_DIR / "products.csv")
products.to_csv(CLEAN_DIR / "products.csv", index=False)

# 4. Standardize the date columns for database storage
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
).dt.strftime("%Y-%m-%d")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
).dt.strftime("%Y-%m-%d %H:%M:%S")

# 5. Open the SQLite database
with sqlite3.connect(DB_PATH) as connection:

    # Store all three datasets as database tables
    customers.to_sql(
        "customers", connection, if_exists="replace", index=False
    )

    products.to_sql(
        "products", connection, if_exists="replace", index=False
    )

    orders.to_sql(
        "orders", connection, if_exists="replace", index=False
    )

    # 6. Run validation checks
    checks = {}

    checks["Customer count"] = (
        pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM customers", connection
        ).iloc[0]["n"] == 1200
    )

    checks["Product count"] = (
        pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM products", connection
        ).iloc[0]["n"] == 40
    )

    checks["Order count"] = (
        pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM orders", connection
        ).iloc[0]["n"] == 12309
    )

    checks["Six cities"] = (
        pd.read_sql_query(
            "SELECT COUNT(DISTINCT city) AS n FROM customers",
            connection
        ).iloc[0]["n"] == 6
    )

    checks["No missing customer ages"] = (
        pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM customers WHERE age IS NULL",
            connection
        ).iloc[0]["n"] == 0
    )

    checks["No pending orders"] = (
        pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM orders "
            "WHERE LOWER(status) = 'pending'",
            connection
        ).iloc[0]["n"] == 0
    )

    checks["No missing order amounts"] = (
        pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM orders "
            "WHERE order_amount IS NULL",
            connection
        ).iloc[0]["n"] == 0
    )

    checks["No missing completed delivery times"] = (
        pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM orders "
            "WHERE LOWER(status) = 'completed' "
            "AND delivery_minutes IS NULL",
            connection
        ).iloc[0]["n"] == 0
    )

    checks["Cancelled orders have missing delivery times"] = (
        pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM orders "
            "WHERE LOWER(status) = 'cancelled' "
            "AND delivery_minutes IS NULL",
            connection
        ).iloc[0]["n"] == 1199
    )

    total = pd.read_sql_query(
        "SELECT ROUND(SUM(order_amount), 2) AS total "
        "FROM orders WHERE LOWER(status) = 'completed'",
        connection
    ).iloc[0]["total"]

    checks["Completed order amount total"] = (
        total is not None and abs(total - 2998668.80) < 0.01
    )

    # 7. Display validation results
    print("\nValidation results:")

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    failed = [name for name, passed in checks.items() if not passed]

    if failed:
        raise AssertionError(
            "Validation failed: " + ", ".join(failed)
        )

    print("\nALL CHECKPOINTS PASSED")

print("\nDatabase saved to:")
print(DB_PATH)