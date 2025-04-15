import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

df = pd.read_csv("data/did_data/walmart_study_summary_data.csv")

print(df.head())

def prepare_data(data):
    data = data.sort_values(['county_name', 'data_year'])
    data['ever_treated'] = data['is_treatment']

    treatment_years = {
        "Androscoggin_County": 2001,
        "Berkshire_County": 1995,
        "Cheshire_County": 2002,
        "Hall_County": 1994,
        "Jackson_County": 2007,
        "Madison_County": 1998,
        "Rutland_County": 1997,
        "Sangamon_County": 2001,
        "Solano_County": 1993,
        "Washington_County": 2010,
        "bennington_county": 2000,
        "clarke_county": 2005,
        "cumberland_county": 2003,
        "franklin_county_ma": 2005,
        "franklin_county_vt": 2000,
        "hood_river_county": 2005,
        "humboldt_county": 2008,
        "jackson_county": 2000,
        "los_angeles_county": 2004,
        "montgomery_county": 2000
    }
    data['treat_year'] = data['county_name'].map(
        lambda x: treatment_years.get(x, float('inf'))
    )
    data['post'] = data['data_year'] >= data['treat_year']
    data['did'] = data['ever_treated'] * data['post']
    return data

did_data = prepare_data(df)

# "county_name","is_treatment","data_year",
outcomes = [
    'retail_priv_employment',
    'retail_priv_weekly_wage', 
    'retail_priv_annual_pay',
    'retail_priv_estabs_count',
    'retail_share_of_private_employment',
    'retail_relative_wage'
]

results = {}
for outcome in outcomes:
    if outcome not in did_data.columns:
        print(f"Skipping {outcome}: not found in dataset")
        continue
        
    # Basic DiD model
    model = smf.ols(f"{outcome} ~ ever_treated + post + did", data=did_data)
    results[outcome] = model.fit()
    
    print(f"\nDiD Results for {outcome}:")
    print(results[outcome].summary().tables[1])

    # Enhanced model with county and year fixed effects
    model_fe = smf.ols(f"{outcome} ~ did + C(county_name) + C(data_year)", data=did_data)
    results[f"{outcome}_fe"] = model_fe.fit()
    print(f"\nDiD Results for {outcome} with fixed effects:")
    print(results[f"{outcome}_fe"].summary().tables[1])

def plot_trends(data, outcome, title):
    if outcome not in data.columns:
        print(f"Cannot plot {outcome}: not found in dataset")
        return None
        
    trends = data.groupby(['data_year', 'ever_treated'])[outcome].mean().reset_index()
    trends = trends.pivot(index='data_year', columns='ever_treated', values=outcome)
    trends.columns = ['Control', 'Treatment']
    
    plt.figure(figsize=(12, 7))
    trends.plot(marker='o')
    
    # Add vertical line at first treatment year
    # Find the earliest treatment year (excluding infinity)
    treatment_years = data[data['ever_treated'] == 1]['treat_year'].unique()
    treatment_years = [year for year in treatment_years if year != float('inf')]
    
    if treatment_years:
        first_treatment = min(treatment_years)
        plt.axvline(x=first_treatment, color='red', linestyle='--', 
                    label=f'First Treatment ({first_treatment})')
    
    plt.title(f"{title}", fontsize=14)
    plt.ylabel(outcome.replace('_', ' ').title(), fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer x-axis
    
    return plt

# Create trend plots for each outcome
for outcome in outcomes:
    if outcome in did_data.columns:
        title = f"{outcome.replace('_', ' ').title()} - Treatment vs Control"
        plot = plot_trends(did_data, outcome, title)
        if plot:
            plt.tight_layout()
            plt.savefig(f"{outcome}_trend.png")
            plt.close()

def run_event_study(data, outcome):
    if outcome not in data.columns:
        print(f"Cannot run event study for {outcome}: not found in dataset")
        return None
    
    # Create relative time indicators (years since treatment)
    data['rel_year'] = data['data_year'] - data['treat_year']
    
    # Create dummies for each relative year for treated units
    year_dummies = {}
    for i in range(-5, 6):
        if i == -1:
            continue
        colname = f'rel_year_{i}'
        data[colname] = ((data['rel_year'] == i) & (data['ever_treated'] == 1)).astype(int)
        year_dummies[i] = colname
    
    formula = f"{outcome} ~ " + " + ".join(year_dummies.values()) + " + C(data_year)"
    
    model = smf.ols(formula, data=data[data['rel_year'].between(-5, 5)])
    results = model.fit()
    
    print(f"\nEvent Study Results for {outcome}:")
    print(results.summary().tables[1])
    
    # Plot event study coefficients
    coeffs = results.params
    conf_int = results.conf_int()
    
    rel_years = sorted([y for y in range(-5, 6) if y != -1])
    coef_values = [coeffs.get(f'rel_year_{y}', 0) for y in rel_years]
    ci_lower = [conf_int.loc[f'rel_year_{y}', 0] for y in rel_years]
    ci_upper = [conf_int.loc[f'rel_year_{y}', 1] for y in rel_years]
    
    plt.figure(figsize=(12, 7))
    plt.errorbar(rel_years, coef_values, yerr=[
        [c - l for c, l in zip(coef_values, ci_lower)],
        [u - c for c, u in zip(coef_values, ci_upper)]
    ], fmt='o', capsize=5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Years Relative to Treatment', fontsize=12)
    plt.ylabel('Coefficient Estimate', fontsize=12)
    plt.title(f'Event Study: {outcome.replace("_", " ").title()}', fontsize=14)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    return plt, results