import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
from scipy import optimize

# Improved function to create synthetic control with proper optimization
def synthetic_control(treated_data, control_data, predictors, outcome, treatment_year):
    """Improved synthetic control implementation"""
    # Get pre-treatment data
    pre_treatment_treated = treated_data[treated_data['data_year'] < treatment_year]
    pre_treatment_controls = control_data[control_data['data_year'] < treatment_year]
    
    if len(pre_treatment_treated) == 0:
        raise ValueError(f"No pre-treatment data for treated unit")
    
    # Standardize predictors to avoid scale issues
    predictor_means = pre_treatment_controls[predictors].mean()
    predictor_stds = pre_treatment_controls[predictors].std()
    
    # Avoid division by zero
    predictor_stds = predictor_stds.replace(0, 1)
    
    # Calculate Z-scores for treated unit
    treated_means = pre_treatment_treated[predictors].mean()
    treated_z = (treated_means - predictor_means) / predictor_stds
    
    # Group control units by county and calculate Z-scores
    control_counties = pre_treatment_controls['county_name'].unique()
    control_z = {}
    
    for county in control_counties:
        county_data = pre_treatment_controls[pre_treatment_controls['county_name'] == county]
        if len(county_data) > 0:
            county_means = county_data[predictors].mean()
            county_z = (county_means - predictor_means) / predictor_stds
            control_z[county] = county_z.values
    
    # Define optimization problem
    counties = list(control_z.keys())
    control_matrix = np.vstack([control_z[county] for county in counties])
    
    # Print diagnostics
    print(f"Treated unit predictors (standardized): {treated_z.values}")
    print(f"Control matrix shape: {control_matrix.shape}")
    
    # Define objective function with explicit debug output
    def objective(weights):
        synth = np.dot(weights, control_matrix)
        loss = np.sum((treated_z.values - synth)**2)
        return loss
    
    # Constraints: weights sum to 1 and are non-negative
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(len(counties))]
    
    # Initial guess: equal weights
    initial_weights = np.ones(len(counties)) / len(counties)
    
    # Optimize with increased iterations and tolerance
    result = optimize.minimize(
        objective, 
        initial_weights, 
        bounds=bounds, 
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 10000, 'ftol': 1e-10}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge: {result.message}")
    
    # Print optimization results
    print(f"Optimization result: {result.success}, message: {result.message}")
    print(f"Final objective value: {result.fun}")
    
    # Convert to pandas Series
    weights_series = pd.Series(result.x, index=counties)
    
    # Create synthetic control time series
    treated_years = sorted(treated_data['data_year'].unique())
    synth_outcome = {}
    
    for year in treated_years:
        year_controls = control_data[control_data['data_year'] == year]
        if len(year_controls) == 0:
            continue
            
        weighted_sum = 0
        weight_used = 0
        
        for county, weight in weights_series.items():
            county_data = year_controls[year_controls['county_name'] == county]
            if len(county_data) > 0:
                county_outcome = county_data[outcome].values[0]
                weighted_sum += weight * county_outcome
                weight_used += weight
        
        # Rescale in case some counties are missing data for this year
        if weight_used > 0:
            synth_outcome[year] = weighted_sum / weight_used
    
    return weights_series, pd.Series(synth_outcome)

# Function to calculate treatment effects
def calculate_treatment_effects(treated_outcome, synthetic_outcome, treatment_year):
    """Calculate treatment effects with proper index handling"""
    # Ensure both series have the same index
    all_years = sorted(set(treated_outcome.index) | set(synthetic_outcome.index))
    
    # Create DataFrame with proper index
    combined = pd.DataFrame(index=all_years)
    combined['treated'] = treated_outcome
    combined['synthetic'] = synthetic_outcome
    
    # Calculate treatment effects
    combined['effect'] = combined['treated'] - combined['synthetic']
    
    # Calculate percent effects (avoid division by zero)
    combined['percent_effect'] = np.where(
        combined['synthetic'] != 0,
        (combined['treated'] / combined['synthetic'] - 1) * 100,
        np.nan
    )
    
    # Print diagnostics
    print("\nTreated vs Synthetic Control:")
    print(combined)
    
    return combined

# Improved function to plot results and save/display the plot
def plot_synthetic_control_results(treated_data, synth_outcome, outcome_var, treatment_year, county_name, save_path=None):
    """
    Plot synthetic control results
    
    Parameters:
    treated_data: DataFrame of treated unit with time series
    synth_outcome: Series of synthetic control outcome values
    outcome_var: Name of outcome variable
    treatment_year: Year of treatment
    county_name: Name of treated county
    save_path: Path to save the figure (if None, the figure is displayed)
    
    Returns:
    fig: Matplotlib figure
    """
    treated_outcome = treated_data.set_index('data_year')[outcome_var]
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Treated': treated_outcome,
        'Synthetic Control': synth_outcome
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_data.plot(ax=ax)
    
    # Calculate and plot difference (treatment effect)
    effect_data = plot_data['Treated'] - plot_data['Synthetic Control']
    ax2 = ax.twinx()
    effect_data.plot(ax=ax2, color='green', linestyle='--', label='Treatment Effect')
    ax2.set_ylabel('Treatment Effect', color='green')
    
    # Add vertical line for treatment year
    ax.axvline(x=treatment_year, color='red', linestyle='--')
    ax.text(treatment_year + 0.1, ax.get_ylim()[1] * 0.95, 'Walmart Entry', 
            color='red', rotation=90, va='top')
    
    # Labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel(outcome_var)
    ax.set_title(f'Synthetic Control Analysis: {outcome_var} for {county_name}')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Save or display the plot
    if save_path:
        plt.savefig(f"{save_path}/{county_name}_{outcome_var.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return fig

def validate_and_preprocess_data(data):
    """Validate and preprocess data"""
    # Check for missing values
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print("Warning: Found missing values:")
        print(missing[missing > 0])
        
        # Fill missing values with county means
        for county in data['county_name'].unique():
            county_mask = (data['county_name'] == county)
            for col in data.columns:
                if col not in ['county_name', 'data_year']:
                    data.loc[county_mask, col] = data.loc[county_mask, col].fillna(
                        data.loc[county_mask, col].mean()
                    )
    
    # Check for outliers in key variables
    for var in ['retail_priv_employment', 'retail_priv_annual_pay']:
        mean = data[var].mean()
        std = data[var].std()
        outliers = data[(data[var] > mean + 5*std) | (data[var] < mean - 5*std)]
        
        if len(outliers) > 0:
            print(f"\nWarning: Found outliers in {var}:")
            print(outliers[['county_name', 'data_year', var]])
            
            # Winsorize extreme values (cap at 5 standard deviations)
            upper_bound = mean + 5*std
            lower_bound = mean - 5*std
            data[var] = data[var].clip(lower_bound, upper_bound)
    
    return data


# Improved function to run placebo tests
def run_placebo_tests(treated_county, control_counties, all_data, outcome_var, treatment_year, predictors):
    """
    Run placebo tests by treating each control county as if it received treatment
    
    Parameters:
    treated_county: Name of actual treated county
    control_counties: List of control county names
    all_data: DataFrame with all counties' data
    outcome_var: Outcome variable to analyze
    treatment_year: Year of treatment
    predictors: List of predictor variables
    
    Returns:
    placebo_effects: DataFrame of placebo treatment effects
    p_value: Estimated p-value for the treatment effect
    """
    # Get treatment effect for actual treated county
    treated_data = all_data[all_data['county_name'] == treated_county].copy()
    control_data = all_data[all_data['county_name'].isin(control_counties)].copy()
    
    actual_weights, actual_synth = synthetic_control(
        treated_data, control_data, predictors, outcome_var, treatment_year
    )
    
    actual_outcome = treated_data.set_index('data_year')[outcome_var]
    actual_effects = calculate_treatment_effects(actual_outcome, actual_synth, treatment_year)
    
    # Calculate RMSPE pre-treatment for the actual treated unit
    pre_treatment_actual = actual_effects[actual_effects.index < treatment_year]
    actual_pre_rmspe = np.sqrt(np.mean(pre_treatment_actual['effect']**2))
    
    # Calculate RMSPE post-treatment for the actual treated unit
    post_treatment_actual = actual_effects[actual_effects.index >= treatment_year]
    actual_post_rmspe = np.sqrt(np.mean(post_treatment_actual['effect']**2))
    
    # Calculate ratio of post/pre RMSPE for the actual treated unit
    if actual_pre_rmspe > 0:
        actual_rmspe_ratio = actual_post_rmspe / actual_pre_rmspe
    else:
        actual_rmspe_ratio = float('inf')
    
    # Run placebo tests
    placebo_effects = {}
    rmspe_ratios = []
    
    for placebo_county in control_counties:
        # Treat this control county as if it received treatment
        placebo_treated = all_data[all_data['county_name'] == placebo_county].copy()
        placebo_controls = all_data[all_data['county_name'].isin([c for c in control_counties if c != placebo_county] + [treated_county])].copy()
        
        try:
            placebo_weights, placebo_synth = synthetic_control(
                placebo_treated, placebo_controls, predictors, outcome_var, treatment_year
            )
            
            placebo_outcome = placebo_treated.set_index('data_year')[outcome_var]
            placebo_effects[placebo_county] = calculate_treatment_effects(placebo_outcome, placebo_synth, treatment_year)
            
            # Calculate RMSPE pre-treatment
            pre_treatment_placebo = placebo_effects[placebo_county][placebo_effects[placebo_county].index < treatment_year]
            placebo_pre_rmspe = np.sqrt(np.mean(pre_treatment_placebo['effect']**2))
            
            # Calculate RMSPE post-treatment
            post_treatment_placebo = placebo_effects[placebo_county][placebo_effects[placebo_county].index >= treatment_year]
            placebo_post_rmspe = np.sqrt(np.mean(post_treatment_placebo['effect']**2))
            
            # Calculate ratio of post/pre RMSPE
            if placebo_pre_rmspe > 0:
                placebo_rmspe_ratio = placebo_post_rmspe / placebo_pre_rmspe
                rmspe_ratios.append(placebo_rmspe_ratio)
            
        except Exception as e:
            print(f"Error in placebo test for {placebo_county}: {e}")
    
    # Calculate p-value as proportion of placebo effects larger than actual effect
    p_value = sum(1 for ratio in rmspe_ratios if ratio >= actual_rmspe_ratio) / (len(rmspe_ratios) + 1)
    
    return placebo_effects, actual_effects, p_value

# Improved analyze_county function
def analyze_county(treated_county, treatment_year, control_counties, all_data, outcome_vars, save_plots=True, output_dir='output'):
    """
    Analyze the impact of Walmart on a specific county
    
    Parameters:
    treated_county: Name of treated county
    treatment_year: Year of treatment
    control_counties: List of control county names
    all_data: DataFrame with all counties' data
    outcome_vars: List of outcome variables to analyze
    save_plots: Whether to save plots to files
    output_dir: Directory to save output
    
    Returns:
    results: Dictionary with analysis results
    """
    # Create output directory if it doesn't exist
    import os
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Normalize county names for consistency
    # This assumes your data has a 'county_name' column
    all_data['county_name'] = all_data['county_name'].str.lower()
    treated_county = treated_county.lower()
    control_counties = [county.lower() for county in control_counties]
    
    # Filter data
    treated_data = all_data[all_data['county_name'] == treated_county].copy()
    control_data = all_data[all_data['county_name'].isin(control_counties)].copy()
    
    if len(treated_data) == 0:
        raise ValueError(f"No data found for treated county: {treated_county}")
    
    if len(control_data) == 0:
        raise ValueError(f"No data found for control counties: {control_counties}")
        
    results = {}
    
    for outcome_var in outcome_vars:
        print(f"\nAnalyzing {outcome_var}...")
        
        # Define predictors (use all outcome variables as predictors)
        predictors = outcome_vars.copy()
        
        # Run synthetic control method
        weights, synth_outcome = synthetic_control(
            treated_data, 
            control_data, 
            predictors, 
            outcome_var, 
            treatment_year
        )
        
        # Calculate treatment effects
        treated_outcome = treated_data.set_index('data_year')[outcome_var]
        effects = calculate_treatment_effects(treated_outcome, synth_outcome, treatment_year)
        
        # Run placebo tests for inference
        placebo_effects, actual_effects, p_value = run_placebo_tests(
            treated_county, 
            control_counties, 
            all_data, 
            outcome_var, 
            treatment_year, 
            predictors
        )
        
        # Store results
        results[outcome_var] = {
            'weights': weights,
            'synthetic_outcome': synth_outcome,
            'treatment_effects': effects,
            'placebo_effects': placebo_effects,
            'p_value': p_value
        }
        
        # Create and save/display plot
        save_path = output_dir if save_plots else None
        fig = plot_synthetic_control_results(
            treated_data, 
            synth_outcome, 
            outcome_var, 
            treatment_year, 
            treated_county,
            save_path
        )
        
        # Plot placebo test results
        fig_placebo = plot_placebo_test_results(
            actual_effects, 
            placebo_effects, 
            treatment_year, 
            treated_county, 
            outcome_var,
            save_path
        )
        
        results[outcome_var]['plot'] = fig
        results[outcome_var]['placebo_plot'] = fig_placebo
        
    return results

# New function to plot placebo test results
def plot_placebo_test_results(actual_effects, placebo_effects, treatment_year, county_name, outcome_var, save_path=None):
    """
    Plot placebo test results
    
    Parameters:
    actual_effects: DataFrame of actual treatment effects
    placebo_effects: Dictionary of placebo treatment effects
    treatment_year: Year of treatment
    county_name: Name of treated county
    outcome_var: Name of outcome variable
    save_path: Path to save the figure (if None, the figure is displayed)
    
    Returns:
    fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot placebo effects
    for placebo_county, placebo_effect in placebo_effects.items():
        placebo_effect['effect'].plot(ax=ax, color='lightgray', alpha=0.5)
    
    # Plot actual effect with thicker line
    actual_effects['effect'].plot(ax=ax, color='blue', linewidth=2, label=county_name)
    
    # Add vertical line for treatment year
    ax.axvline(x=treatment_year, color='red', linestyle='--')
    ax.text(treatment_year + 0.1, ax.get_ylim()[1] * 0.95, 'Treatment Year', 
            color='red', rotation=90, va='top')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Treatment Effect')
    ax.set_title(f'Placebo Tests: {outcome_var} for {county_name}')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save or display the plot
    if save_path:
        plt.savefig(f"{save_path}/{county_name}_{outcome_var.replace(' ', '_')}_placebo.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return fig

# Function to print comprehensive results
def print_results(results, treated_county, treatment_year):
    """
    Print comprehensive results
    
    Parameters:
    results: Dictionary with analysis results
    treated_county: Name of treated county
    treatment_year: Year of treatment
    """
    print(f"\n========== RESULTS FOR {treated_county.upper()} ==========")
    print(f"Treatment year: {treatment_year}")
    
    for outcome_var, result in results.items():
        print(f"\n----- {outcome_var} -----")
        
        print("Control weights:")
        sorted_weights = result['weights'].sort_values(ascending=False)
        for county, weight in sorted_weights.items():
            if weight > 0.001:  # Only show non-negligible weights
                print(f"  {county}: {weight:.4f}")
        
        print("\nTreatment effects:")
        post_treatment = result['treatment_effects'].loc[treatment_year:treatment_year+5]
        avg_effect = post_treatment['effect'].mean()
        avg_percent = post_treatment['percent_effect'].mean()
        
        print(f"  Average effect (5 years post-treatment): {avg_effect:.4f}")
        print(f"  Average percent effect: {avg_percent:.2f}%")
        print(f"  p-value: {result['p_value']:.4f}")
        
        if result['p_value'] < 0.05:
            print("  The effect is statistically significant at the 5% level.")
        elif result['p_value'] < 0.1:
            print("  The effect is statistically significant at the 10% level.")
        else:
            print("  The effect is not statistically significant.")

# Example of usage (adjust with actual data)
def main():
    # Load data
    data = pd.read_csv("data/did_data/walmart_study_summary_data.csv")
    # Call this at the beginning of your main() function
    data = validate_and_preprocess_data(data)
    
    # Define treatment counties and years (normalized to lowercase)
    treatment_counties = {
        "androscoggin_county": 2001,
        "berkshire_county": 1995,
        "cheshire_county": 2002,
        "hall_county": 1994,
        "jackson_county": 2007,
        "madison_county": 1998,
        "rutland_county": 1997,
        "sangamon_county": 2001,
        "solano_county": 1993,
        "washington_county": 2010
    }
    
    # Define control counties (all lowercase)
    control_counties = [
        "bennington_county",
        "clarke_county",
        "cumberland_county",
        "franklin_county_ma",
        "franklin_county_vt",
        "hood_river_county",
        "humboldt_county",
        "jackson_county",
        "los_angeles_county",
        "montgomery_county"
    ]
    # In your main() function, after defining control_counties:
    # Check for overlaps between treatment and control counties
    treatment_set = set(treatment_counties.keys())
    control_set = set(control_counties)
    overlap = treatment_set.intersection(control_set)

    if overlap:
        print(f"Warning: These counties appear in both treatment and control groups: {overlap}")
        # Remove overlapping counties from control list
        control_counties = [c for c in control_counties if c not in overlap]
        print(f"Updated control counties: {control_counties}")
    # Define outcomes to analyze
    outcome_vars = [
        'retail_priv_employment',
        'retail_priv_annual_pay',
        'retail_share_of_private_employment',
        'retail_relative_wage'
    ]
    # Add this at the beginning of your main() function
    # Check for duplicates
    print("Checking for duplicates in data...")
    duplicates = data.duplicated(subset=['county_name', 'data_year'], keep=False)
    if duplicates.any():
        print(f"Found {duplicates.sum()} duplicate county-year observations:")
        print(data[duplicates].sort_values(['county_name', 'data_year']))
        
        # Remove duplicates, keeping the first occurrence
        data = data.drop_duplicates(subset=['county_name', 'data_year'], keep='first')
        print(f"Removed duplicates. Data shape now: {data.shape}")
    
    # Normalize county names in data
    data['county_name'] = data['county_name'].str.lower()
    
    # Run analysis for a specific county or for all treatment counties
    treated_county = "washington_county"  # Change this to analyze different counties
    
    if treated_county == "all":
        # Run for all treatment counties
        all_results = {}
        for county, year in treatment_counties.items():
            print(f"\nAnalyzing {county}...")
            try:
                results = analyze_county(county, year, control_counties, data, outcome_vars)
                all_results[county] = results
                print_results(results, county, year)
            except Exception as e:
                print(f"Error analyzing {county}: {e}")
    else:
        # Run for a single county
        treatment_year = treatment_counties[treated_county]
        results = analyze_county(treated_county, treatment_year, control_counties, data, outcome_vars)
        print_results(results, treated_county, treatment_year)

if __name__ == "__main__":
    main()