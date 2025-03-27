import pandas as pd
import matplotlib.pyplot as plt
 
df = pd.read_csv("cleaned_29646_2011_to_2023.csv")
 
# TRENDS
 
print("\nMedian Income Growth:")
 
pre_walmart = df[df["Year"] <= 2015]
post_walmart = df[df["Year"] >= 2015]
 
median_pre = pre_walmart['Median income (dollars)'].pct_change().mean() * 100
median_post = post_walmart['Median income (dollars)'].pct_change().mean() * 100
 
print(f"Pre-Walmart (2011–2015) Median Income Growth: {median_pre:.2f}% per year")
print(f"Post-Walmart (2015–2023) Median Income Growth: {median_post:.2f}% per year")
 
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
plt.plot(df['Year'], df['Median income (dollars)'], marker='o', label='Median Income')
plt.axvline(x=2015, color='red', linestyle='--', label='Walmart Built (2015)')
plt.xlabel('Year')
plt.ylabel('Median Income ($)')
plt.legend()
plt.show()