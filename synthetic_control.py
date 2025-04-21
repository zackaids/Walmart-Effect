import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def get_synthetic_control_weights(X_treated, X_controls):
    """
    Find optimal weights for synthetic control
    Using constrained optimization to ensure weights are non-negative and sum to 1
    
    Parameters:
    X_treated: Features of treated unit (array)
    X_controls: Features of control units (2D array)
    
    Returns:
    weights: Optimal weights for control units
    """
    # Number of control units
    n_controls = X_controls.shape[0]
    
    # Initial weights (equal)
    initial_weights = np.ones(n_controls) / n_controls
    
    # Objective function: squared difference between treated and synthetic control
    def objective(weights):
        synthetic = np.dot(weights, X_controls)
        return np.sum((X_treated - synthetic)**2)
    
    # Constraints: weights sum to 1 and are non-negative
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
    ]
    bounds = [(0, 1) for _ in range(n_controls)]  # Non-negative
    
    # Optimize
    result = minimize(
        objective, 
        initial_weights, 
        method='SLSQP', 
        constraints=constraints,
        bounds=bounds
    )
    
    return result.x

def create_synthetic_control(treated_data, control_data, matching_vars, outcome_var, 
                             treatment_year, county_name, control_counties):
    """
    Create synthetic control for a given treated county
    
    Parameters:
    treated_data: DataFrame with treated county data
    control_data: DataFrame with control counties data
    matching_vars: Variables to match in pre-treatment period
    outcome_var: Outcome variable to predict
    treatment_year: Year of treatment
    county_name: Name of treated county
    control_counties: List of control county names
    
    Returns:
    weights: Series with weights for each control county
    synth_outcomes: DataFrame with actual and synthetic outcomes
    """
    # Get pre-treatment data
    pre_treated = treated_data[treated_data['data_year'] < treatment_year]
    
    # Get data for control counties
    control_counties_data = {}
    control_features = {}
    
    # For each control county, get pre-treatment features
    for county in control_counties:
        county_data = control_data[control_data['county_name'] == county]
        if not county_data.empty:
            # Store full time series
            control_counties_data[county] = county_data
            
            # Get pre-treatment features for matching
            pre_county = county_data[county_data['data_year'] < treatment_year]
            if not pre_county.empty:
                # Average pre-treatment values of matching variables
                control_features[county] = pre_county[matching_vars].mean().values
    
    # Average pre-treatment values for treated county
    treated_features = pre_treated[matching_vars].mean().values
    
    # Convert control features to 2D array for optimization
    control_names = list(control_features.keys())
    if not control_names:
        print(f"No valid control counties found for {county_name}")
        return None, None
        
    X_controls = np.array([control_features[c] for c in control_names])
    
    # Get weights
    weights = get_synthetic_control_weights(treated_features, X_controls)
    
    # Create synthetic outcomes for all years
    years = sorted(treated_data['data_year'].unique())
    actual_outcomes = {}
    synthetic_outcomes = {}
    
    for year in years:
        # Get actual outcome for this year
        year_treated = treated_data[treated_data['data_year'] == year]
        if not year_treated.empty:
            actual_outcomes[year] = year_treated[outcome_var].values[0]
        
        # Create synthetic outcome as weighted average of controls
        synthetic_value = 0
        total_weight = 0
        
        for i, county in enumerate(control_names):
            county_data = control_counties_data[county]
            year_data = county_data[county_data['data_year'] == year]
            
            if not year_data.empty:
                synthetic_value += weights[i] * year_data[outcome_var].values[0]
                total_weight += weights[i]
        
        if total_weight > 0:
            synthetic_outcomes[year] = synthetic_value / total_weight
    
    # Create result DataFrame
    results = pd.DataFrame({
        'Year': years,
        'Actual': [actual_outcomes.get(y, np.nan) for y in years],
        'Synthetic': [synthetic_outcomes.get(y, np.nan) for y in years],
    })
    
    # Add treatment effects
    results['Treatment_Effect'] = results['Actual'] - results['Synthetic']
    results['Percent_Effect'] = (results['Actual'] / results['Synthetic'] - 1) * 100
    
    # Convert weights to Series
    weights_series = pd.Series(weights, index=control_names)
    
    return weights_series, results

def plot_synthetic_control(results, outcome_var, treatment_year, county_name):
    """
    Plot synthetic control results
    
    Parameters:
    results: DataFrame with actual and synthetic outcomes
    outcome_var: Outcome variable name
    treatment_year: Year of treatment
    county_name: Name of treated county
    
    Returns:
    fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual and synthetic outcomes
    ax.plot(results['Year'], results['Actual'], label='Actual', marker='o', markersize=4)
    ax.plot(results['Year'], results['Synthetic'], label='Synthetic Control', marker='s', markersize=4)
    
    # Add vertical line at treatment year
    ax.axvline(x=treatment_year, color='red', linestyle='--')
    ax.text(treatment_year + 0.5, ax.get_ylim()[1] * 0.95, 'Walmart Entry', 
            color='red', ha='left', va='top', rotation=90)
    
    # Add title and labels
    ax.set_title(f'Synthetic Control Analysis: {outcome_var} in {county_name}')
    ax.set_xlabel('Year')
    ax.set_ylabel(outcome_var)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def run_synthetic_control_analysis(data, treatment_years, outcome_vars, matching_vars=None):
    """
    Run synthetic control analysis for all treatment counties
    
    Parameters:
    data: DataFrame with all counties data
    treatment_years: Dictionary mapping county names to treatment years
    outcome_vars: List of outcome variables to analyze
    matching_vars: List of variables to match in pre-treatment period (optional)
    
    Returns:
    results: Dictionary with results for each county and outcome
    """
    if matching_vars is None:
        # Default matching variables if not specified
        matching_vars = [
            'retail_priv_employment', 
            'retail_priv_annual_pay',
            'all_priv_employment',
            'all_priv_annual_pay',
            'retail_share_of_private_employment',
            'retail_relative_wage'
        ]
    
    # Split data into treatment and control groups
    treatment_data = data[data['is_treatment'] == 1].copy()
    control_data = data[data['is_treatment'] == 0].copy()
    
    # Get list of treatment and control counties
    treatment_counties = treatment_data['county_name'].unique()
    control_counties = control_data['county_name'].unique()
    
    # Check for counties in treatment_years that aren't in the data
    for county in treatment_years.keys():
        if county not in treatment_counties and county not in control_counties:
            print(f"Warning: {county} found in treatment_years but not in data")
    
    # Initialize results dictionary
    results = {}
    
    # Analyze each treatment county
    for county in treatment_counties:
        if county in treatment_years:
            treatment_year = treatment_years[county]
            county_data = treatment_data[treatment_data['county_name'] == county]
            
            print(f"Analyzing {county} (treatment year: {treatment_year})")
            county_results = {}
            
            for outcome_var in outcome_vars:
                print(f"  Analyzing outcome: {outcome_var}")
                
                # Create synthetic control
                weights, synth_results = create_synthetic_control(
                    county_data, 
                    control_data, 
                    matching_vars, 
                    outcome_var, 
                    treatment_year,
                    county,
                    control_counties
                )
                
                if weights is not None and synth_results is not None:
                    # Plot results
                    fig = plot_synthetic_control(
                        synth_results, 
                        outcome_var, 
                        treatment_year, 
                        county
                    )
                    
                    # Calculate average treatment effect for 5 years post-treatment
                    post_treatment = synth_results[synth_results['Year'] >= treatment_year]
                    post_treatment_5yr = post_treatment[post_treatment['Year'] < treatment_year + 5]
                    
                    if not post_treatment_5yr.empty:
                        avg_effect = post_treatment_5yr['Treatment_Effect'].mean()
                        avg_pct_effect = post_treatment_5yr['Percent_Effect'].mean()
                    else:
                        avg_effect = np.nan
                        avg_pct_effect = np.nan
                    
                    # Store results
                    county_results[outcome_var] = {
                        'weights': weights,
                        'results': synth_results,
                        'avg_effect': avg_effect,
                        'avg_pct_effect': avg_pct_effect,
                        'plot': fig
                    }
                    
                    print(f"    Average effect (5yr): {avg_effect:.2f} ({avg_pct_effect:.2f}%)")
                    print(f"    Control weights: {weights.sort_values(ascending=False).head(3)}")
                else:
                    print(f"    Failed to create synthetic control")
            
            results[county] = county_results
        else:
            print(f"Warning: No treatment year found for {county}")
    
    return results

# Example usage (would need to be adapted to your actual data)
# Assuming 'data' is your DataFrame with all counties' data

# Define outcome variables to analyze
outcome_vars = [
    'retail_priv_employment',
    'retail_priv_annual_pay',
    'retail_share_of_private_employment',
    'retail_relative_wage'
]

# Define matching variables
matching_vars = [
    'retail_priv_employment', 
    'retail_priv_annual_pay',
    'all_priv_employment',
    'all_priv_annual_pay',
    'retail_share_of_private_employment',
    'retail_relative_wage'
]

data = pd.read_csv("data/did_data/walmart_study_summary_data.csv")

# Define treatment years
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
    "bennington_county": 2000,  # These are actually control counties (Walmart banned)
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

# Run analysis
results = run_synthetic_control_analysis(data, treatment_years, outcome_vars, matching_vars)

# Function to summarize results across all counties
def summarize_results(results, outcome_vars):
    """
    Summarize results across all counties
    
    Parameters:
    results: Dictionary with results for each county
    outcome_vars: List of outcome variables
    
    Returns:
    summary: DataFrame with summary statistics
    """
    summary = {}
    
    for outcome_var in outcome_vars:
        effects = []
        pct_effects = []
        
        for county, county_results in results.items():
            if outcome_var in county_results:
                effect = county_results[outcome_var]['avg_effect']
                pct_effect = county_results[outcome_var]['avg_pct_effect']
                
                if not np.isnan(effect) and not np.isnan(pct_effect):
                    effects.append(effect)
                    pct_effects.append(pct_effect)
        
        if effects:
            summary[outcome_var] = {
                'mean_effect': np.mean(effects),
                'median_effect': np.median(effects),
                'mean_pct_effect': np.mean(pct_effects),
                'median_pct_effect': np.median(pct_effects),
                'n_counties': len(effects)
            }
    
    return pd.DataFrame(summary).T

# Example of summarizing results
summary = summarize_results(results, outcome_vars)
print(summary)

