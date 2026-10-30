import sqlite3
from pathlib import Path
import pandas as pd
import re

# Locate the project files
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "instamart.db"
SQL_PATH = BASE_DIR / "sql" / "queries.sql"

# Check that the required files exist
if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

if not SQL_PATH.exists():
    raise FileNotFoundError(f"SQL file not found: {SQL_PATH}")

# Read all seven SQL queries
sql_text = SQL_PATH.read_text(encoding="utf-8")

# Split the file into separate queries using the Q1, Q2, etc. comments
sections = re.split(r"(?=--\s*Q[1-7]\s*:)", sql_text)

queries = []

for section in sections:
    section = section.strip()

    if not re.match(r"--\s*Q[1-7]\s*:", section):
        continue

    # Remove the descriptive comment lines
    lines = section.splitlines()
    sql_lines = [
        line for line in lines
        if not line.strip().startswith("--")
    ]

    query = "\n".join(sql_lines).strip()

    if query:
        queries.append((lines[0].strip(), query))

# Run every query using Pandas
with sqlite3.connect(DB_PATH) as connection:

    print("=" * 60)
    print("INSTAMART CAPSTONE - SQL QUERY RESULTS")
    print("=" * 60)

    if len(queries) != 7:
        raise ValueError(
            f"Expected 7 SQL queries, but found {len(queries)}. "
            "Check the Q1 to Q7 headings in queries.sql."
        )

    results = {}

    for title, query in queries:
        print(f"\n{title}")

        df = pd.read_sql_query(query, connection)
        results[title.split(":")[0].replace("-- ", "")] = df

        print(f"Rows returned: {len(df)}")
        print(df.head(5).to_string(index=False))

    # Basic checkpoint checks
    expected_rows = {
        "Q1": 30,
        "Q2": 30,
        "Q3": 11,
        "Q4": 25,
        "Q5": 550,
        "Q7": 183,
    }

    print("\n" + "=" * 60)
    print("BASIC CHECKPOINTS")
    print("=" * 60)

    all_passed = True

    for query_name, expected in expected_rows.items():
        actual = len(results[query_name])
        passed = actual == expected

        print(
            f"{query_name}: expected {expected}, "
            f"got {actual} - {'PASS' if passed else 'CHECK'}"
        )

        if not passed:
            all_passed = False

    # Check total completed-order revenue
    revenue = pd.read_sql_query(
        """
        SELECT SUM(order_amount) AS total_revenue
        FROM orders
        WHERE status = 'completed'
        """,
        connection,
    ).iloc[0]["total_revenue"]

    expected_revenue = 2998668.80
    revenue_passed = abs(revenue - expected_revenue) < 0.01

    print(
        f"Completed-order revenue: ₹{revenue:,.2f} - "
        f"{'PASS' if revenue_passed else 'CHECK'}"
    )

    all_passed = all_passed and revenue_passed

    print("=" * 60)

    if all_passed:
        print("ALL BASIC SQL CHECKS PASSED")
    else:
        print("SOME CHECKS NEED ATTENTION")
        print("Review the results before continuing.")