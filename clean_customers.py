import pandas as pd
from pathlib import Path

# 1. Define the project folders
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"

# Create the clean folder if it does not exist
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load the raw customer data
customers = pd.read_csv(RAW_DIR / "customers.csv")

print("Original customer data shape:", customers.shape)

# 3. Standardize city names
customers["city"] = customers["city"].astype("string").str.strip().str.title()

# 4. Standardize gender values
def clean_gender(value):
    if pd.isna(value) or str(value).strip() == "":
        return "unknown"

    value = str(value).strip().lower()

    if value in ["m", "male", "man"]:
        return "male"
    elif value in ["f", "female", "woman"]:
        return "female"
    else:
        return "unknown"


customers["gender"] = customers["gender"].apply(clean_gender)

# 5. Convert age to numeric and mark invalid ages as missing
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[
    (customers["age"] < 10) | (customers["age"] > 100),
    "age"
] = pd.NA

# 6. Fill missing ages using the median age of each city
city_medians = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_medians)

# 7. Convert signup_date to datetime
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)

# 8. Save the cleaned customer data
output_path = CLEAN_DIR / "customers.csv"
customers.to_csv(output_path, index=False, date_format="%Y-%m-%d")

# 9. Display results for verification
print("\nCleaned customer data shape:", customers.shape)

print("\nCustomer counts by city:")
print(customers["city"].value_counts().sort_index())

print("\nCustomer counts by gender:")
print(customers["gender"].value_counts())

print("\nMissing values after cleaning:")
print(customers.isna().sum())

print("\nAge range after cleaning:")
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())

print("\nCleaned customer data saved to:")
print(output_path)