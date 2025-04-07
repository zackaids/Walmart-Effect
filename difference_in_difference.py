import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("data/did_data/treatedvscontrol.csv")

data = df.sort_values(['fips', 'year'])
data['ever_treated'] = data.groupby('fips')['walmart_entry'].transform('max') == 1

treat_years = data[data['walmart_entry'].diff() == 1].groupby('fips')['year'].min()
fips_to_treat_year = dict(zip(treat_years.index, treat_years.values))

data['treat_year'] = data['fips'].map(fips_to_treat_year)
data['treat_year'] = data['treat_year'].fillna(float('inf'))  # Control groups never get treated
data['post'] = data['year'] >= data['treat_year']

# Treatment x Post (DiD) interaction term
data['did'] = data['ever_treated'] * data['post']

# DiD regression
outcomes = ['avg_annual', 'all_employees', 'total_wages', 'establishments', 'avg_weekly']

results = {}
for outcome in outcomes:
    model = smf.ols(f"{outcome} ~ ever_treated + post + did", data=data)
    results[outcome] = model.fit()
    print(f"\nDiD Results for {outcome}:")
    print(results[outcome].summary().tables[1])

# visual
def plot_trends(data, outcome, title):
    trends = data.groupby(['year', 'ever_treated'])[outcome].mean().reset_index()
    trends = trends.pivot(index='year', columns='ever_treated', values=outcome)
    trends.columns = ['Control', 'Treatment']
    
    plt.figure(figsize=(10, 6))
    trends.plot(marker='o')
    plt.axvline(x=data[data['did'].eq(1)]['year'].min(), color='red', linestyle='--', 
                label='First Walmart Entry')
    plt.title(f"{title} - Treatment vs Control")
    plt.ylabel(outcome)
    plt.xlabel('Year')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt


trend_plot = plot_trends(data, 'avg_annual', 'Average Annual Wages')
trend_plot.show()
trend_plot = plot_trends(data, 'all_employees', '# of Employees')
trend_plot.show()
trend_plot = plot_trends(data, 'total_wages', 'Total Wages')
trend_plot.show()
trend_plot = plot_trends(data, 'establishments', '# of Establishments')
trend_plot.show()
trend_plot = plot_trends(data, 'avg_weekly', 'Average Weekly  Wages')
trend_plot.show()