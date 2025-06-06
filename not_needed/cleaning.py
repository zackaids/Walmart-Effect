import pandas as pd
import glob
import os

ZIP_CODE = "United States"
 
files = sorted(glob.glob(f"data/{ZIP_CODE}/*.csv"))
dfs = []
for file in files:
    year = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    df['Year'] = int(year)
    df["Label (Grouping)"] = df["Label (Grouping)"].str.strip()
    dfs.append(df)
 
combined_df = pd.concat(dfs, ignore_index=True)
 
print(combined_df.shape)
 
income_metrics = [
    "Less than $10,000", 
    "    Less than $10,000",
    "$10,000 to $14,999",
    "    $10,000 to $14,999",
    "$15,000 to $24,999",
    "    $15,000 to $24,999",
    "$25,000 to $34,999",
    "    $25,000 to $34,999",
    "$35,000 to $49,999",
    "    $35,000 to $49,999",
    "$50,000 to $74,999",
    "    $50,000 to $74,999",
    "$75,000 to $99,999",
    "    $75,000 to $99,999",
    "$100,000 to $149,999",
    "    $100,000 to $149,999",
    "$150,000 to $199,999",
    "    $150,000 to $199,999",
    "$200,000 or more",
    "    $200,000 or more",
    "Median income (dollars)",
    "    Median income (dollars)",
    "Mean income (dollars)",
    "    Mean income (dollars)"
]
 
filtered_df = combined_df[combined_df["Label (Grouping)"].isin(income_metrics)].copy()
 
pivoted_df = filtered_df.pivot(
    index="Label (Grouping)", 
    columns="Year", 
    values=f"ZCTA5 {ZIP_CODE}!!Households!!Estimate"
    # values=f"{ZIP_CODE}!!Households!!Estimate"
)
 
final_df = pivoted_df.T.reset_index().rename_axis(None, axis=1)
 
print(final_df)
print(final_df.isnull().sum())
print(final_df.shape)
print(final_df.dtypes)
for col in final_df.columns:
    if "%" in str(final_df[col].iloc[0]):
        final_df[col] = final_df[col].str.replace("%", "").astype(float) / 100
 
print(final_df.dtypes)
final_df["Median income (dollars)"] = final_df["Median income (dollars)"].str.replace(",", "").astype(float)
final_df["Mean income (dollars)"] = final_df["Mean income (dollars)"].str.replace(",", "").astype(float)
 
 
print(final_df.dtypes)
 
final_df = final_df.sort_values("Year").reset_index(drop=True)
 
print(final_df)
 
final_df.to_csv(f"data/cleaned_{ZIP_CODE}.csv", index=False)