import sqlite3
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "instamart.db"

# Connect to the SQLite database
conn = sqlite3.connect(DB_PATH)

print("=" * 60)
print("PART A - PANDAS CROSS-CHECKS")
print("=" * 60)

# Load completed orders and customers
orders = pd.read_sql_query(
    "SELECT * FROM orders WHERE status = 'completed'",
    conn
)

customers = pd.read_sql_query(
    "SELECT * FROM customers",
    conn
)

print("\n1. DATA LOADED SUCCESSFULLY")
print(f"Completed orders: {len(orders)}")
print(f"Customers: {len(customers)}")

# Cross-check 1: Total completed-order revenue
sql_revenue = pd.read_sql_query(
    """
    SELECT SUM(order_amount) AS total_revenue
    FROM orders
    WHERE status = 'completed'
    """,
    conn
)["total_revenue"].iloc[0]

pandas_revenue = orders["order_amount"].sum()

print("\n2. REVENUE CROSS-CHECK")
print(f"SQL revenue:     ₹{sql_revenue:,.2f}")
print(f"Pandas revenue:  ₹{pandas_revenue:,.2f}")

if abs(sql_revenue - pandas_revenue) < 0.01:
    print("Revenue cross-check: PASS")
else:
    print("Revenue cross-check: FAIL")

# Cross-check 2: Customer counts by city
print("\n3. CUSTOMER COUNTS BY CITY")

sql_city_counts = pd.read_sql_query(
    """
    SELECT city, COUNT(*) AS customer_count
    FROM customers
    GROUP BY city
    ORDER BY city
    """,
    conn
)

pandas_city_counts = (
    customers.groupby("city")
    .size()
    .reset_index(name="customer_count")
    .sort_values("city")
    .reset_index(drop=True)
)

print("\nSQL results:")
print(sql_city_counts.to_string(index=False))

print("\nPandas results:")
print(pandas_city_counts.to_string(index=False))

if sql_city_counts.equals(pandas_city_counts):
    print("\nCity-count cross-check: PASS")
else:
    print("\nCity-count cross-check: FAIL")

# Cross-check 3: Completed-order spending statistics
print("\n4. COMPLETED-ORDER SPENDING STATISTICS")

amounts = orders["order_amount"]

print(f"Total revenue: ₹{amounts.sum():,.2f}")
print(f"Mean order amount: ₹{amounts.mean():,.2f}")
print(f"Median order amount: ₹{amounts.median():,.2f}")
print(f"Standard deviation: ₹{amounts.std():,.2f}")
print(f"Minimum order amount: ₹{amounts.min():,.2f}")
print(f"Maximum order amount: ₹{amounts.max():,.2f}")

# Quartiles and outlier detection
q1 = amounts.quantile(0.25)
q3 = amounts.quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outliers = amounts[
    (amounts < lower_fence) | (amounts > upper_fence)
]

print(f"\nFirst quartile (Q1): ₹{q1:,.2f}")
print(f"Third quartile (Q3): ₹{q3:,.2f}")
print(f"Lower outlier fence: ₹{lower_fence:,.2f}")
print(f"Upper outlier fence: ₹{upper_fence:,.2f}")
print(f"Number of outlier orders: {len(outliers)}")

# Correlation analysis
print("\n5. CORRELATION ANALYSIS")

correlation = orders[["order_amount", "discount_pct", "quantity"]].corr()

print(correlation.round(3))

amount_discount_corr = correlation.loc["order_amount", "discount_pct"]
amount_quantity_corr = correlation.loc["order_amount", "quantity"]

print(f"\nAmount vs. discount correlation: {amount_discount_corr:.3f}")
print(f"Amount vs. quantity correlation: {amount_quantity_corr:.3f}")

conn.close()

print("\n" + "=" * 60)
print("PANDAS CROSS-CHECKS AND ANALYSIS COMPLETED")
print("=" * 60)