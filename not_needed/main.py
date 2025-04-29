import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 
ZIP_CODE = "29646"
WALMART_DATE = 2015
 
df = pd.read_csv(f"data/cleaned_{ZIP_CODE}.csv")

df_us = pd.read_csv("data/cleaned_United States.csv")
 
# TRENDS
 
print("\nMedian Income Growth:")
 
pre_walmart = df[df["Year"] <= WALMART_DATE]
post_walmart = df[df["Year"] >= WALMART_DATE]
 
median_pre = pre_walmart['Median income (dollars)'].pct_change().mean() * 100
median_post = post_walmart['Median income (dollars)'].pct_change().mean() * 100
 
print(f"Pre-Walmart (2011–{WALMART_DATE}) Median Income Growth: {median_pre:.2f}% per year")
print(f"Post-Walmart ({WALMART_DATE}–2023) Median Income Growth: {median_post:.2f}% per year")
 
print("\nIncome Bracket Shifts:")
brackets = [
    "Less than $10,000", 
    "$10,000 to $14,999", 
    "$15,000 to $24,999",
    "$25,000 to $34,999",
    "$35,000 to $49,999",
    "$50,000 to $74,999",
    "$75,000 to $99,999",
    "$100,000 to $149,999",
    "$150,000 to $199,999",
    "$200,000 or more"
]
 
for bracket in brackets:
    avg_pre = pre_walmart[bracket].mean()
    avg_post = post_walmart[bracket].mean()
    print(f"{bracket}: Pre-Walmart {avg_pre:.3f} → Post-Walmart {avg_post:.3f}")
 
 
# CORRELATION
print("\n")
correlation = df["Year"].corr(df["Median income (dollars)"])
print(f"Correlation (Year vs. Median Income): {correlation:.3f}")
 
# GRAPH
plt.figure(figsize=(10, 5))
plt.plot(
    df['Year'], 
    df['Median income (dollars)'], 
    marker='o', 
    label='Median Income'
)
plt.axvline(x=WALMART_DATE, color='red', linestyle='--', label=f'Walmart Built ({WALMART_DATE})')
plt.xlabel('Year')
plt.ylabel('Median Income ($)')
plt.legend()
plt.show()


df_combined = pd.merge(df, df_us, on="Year", how="left", suffixes=("_ZIP", "_US"))
plt.figure(figsize=(12, 6))
plt.plot(
    df_combined["Year"], 
    df_combined["Median income (dollars)_ZIP"], 
    marker="o", 
    linewidth=2.5, 
    label=f"ZIP {ZIP_CODE} Median"
)
plt.plot(
    df_combined["Year"], 
    df_combined["Median income (dollars)_US"], 
    marker="s", 
    linestyle="--", 
    color="gray", 
    linewidth=2, 
    label=f"U.S. Median"
)
plt.axvline(x=WALMART_DATE, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"Walmart Opens ({WALMART_DATE})")
plt.title("Median Household Income Trends (2011-2023)", fontsize=14, pad=15)
plt.xlabel("Year", labelpad=10)
plt.ylabel("Median Income ($)", labelpad=10)
plt.legend(loc="upper left", framealpha=1)
plt.grid(True, alpha=0.2)
plt.xticks(df_combined["Year"])


plt.tight_layout()
plt.show()