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
def plot_trends(data, outcome, title, model=None):
    # Prepare trend data for Control and Treated
    trends = data.groupby(['year', 'ever_treated'])[outcome].mean().reset_index()
    trends = trends.pivot(index='year', columns='ever_treated', values=outcome)
    trends.columns = ['Control', 'Treatment']

    # Create base plot
    fig, ax = plt.subplots(figsize=(10, 6))
    trends.plot(marker='o', ax=ax)

    # Add vertical line for first Walmart entry
    first_treatment_year = data[data['did'] == 1]['year'].min()
    ax.axvline(x=first_treatment_year, color='red', linestyle='--', label='First Walmart Entry')

    # ➕ Add counterfactual line (if model provided)
    if model is not None:
        data_cf = data.copy()
        data_cf['predicted_actual'] = model.predict(data_cf)

        # Estimate treatment effect
        treatment_effect = model.params['did']

        # Generate counterfactual by subtracting DiD effect
        data_cf['counterfactual'] = data_cf['predicted_actual']
        data_cf.loc[data_cf['did'] == 1, 'counterfactual'] -= treatment_effect

        # Group counterfactual for treated units
        counterfactual = data_cf[data_cf['ever_treated']].groupby('year')['counterfactual'].mean()
        ax.plot(counterfactual.index, counterfactual.values, linestyle='--', color='black', label='Treated (Counterfactual)')

    # Final plot settings
    ax.set_title(f"{title} - Treatment vs Control")
    ax.set_ylabel(outcome)
    ax.set_xlabel('Year')
    ax.grid(True, alpha=0.3)
    ax.legend()

    return plt
        

plot_trends(data, 'avg_annual', 'Average Annual Wages', model=results['avg_annual']).show()
plot_trends(data, 'all_employees', '# of Employees', model=results['all_employees']).show()
plot_trends(data, 'total_wages', 'Total Wages', model=results['total_wages']).show()
plot_trends(data, 'establishments', '# of Establishments', model=results['establishments']).show()
plot_trends(data, 'avg_weekly', 'Average Weekly Wages', model=results['avg_weekly']).show()

