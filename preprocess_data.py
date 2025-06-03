import pandas as pd
import argparse

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser(description="Merge and clean Excel files for TCDF.")
parser.add_argument('--target', type=str, choices=['WL', 'SLA'], required=True,
                    help="Choose the target variable: WL or SLA")
args = parser.parse_args()
target = args.target

# === FILE PATHS ===
file1 = "Tanjong Pagar_UTide_Data_WP2_WL_SLA2026.xlsx"
file2 = "Singapore.xlsx"

# === LOAD EXCEL FILES ===
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# === RENAME COLUMNS ===
df1 = df1.rename(columns={"Datetime": "Time"})
df2 = df2.rename(columns={"Date and Time ": "Time"})

# === CONVERT TIME COLUMNS TO DATETIME ===
df1["Time"] = pd.to_datetime(df1["Time"])
df2["Time"] = pd.to_datetime(df2["Time"])

# === MERGE ON TIME COLUMN ===
merged_df = pd.merge(df1, df2, on="Time", how="inner")

# === DROP ROWS WITH MISSING DATA ===
merged_df = merged_df.dropna()

# === REORDER COLUMNS TO PUT TARGET LAST ===
columns = list(merged_df.columns)
if target in columns:
    columns.remove(target)
    columns.append(target)
    merged_df = merged_df[columns]
else:
    raise ValueError(f"Target column '{target}' not found in merged data.")

# === DROP NON-NUMERIC COLUMNS (like 'Time') ===
merged_df = merged_df.select_dtypes(include=['number'])

# === SAVE TO CSV ===
output_file = f"prepared_data_target_{target}.csv"
merged_df.to_csv(output_file, index=False)

# === DONE ===
print(f"✅ Merged and cleaned data saved as: {output_file}")
print(f"📌 Target column is: {target}")
print("\nPreview:")
print(merged_df.head())
