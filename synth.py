import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
from scipy import optimize

def synthetic_control(treated_data, control_data, predictors, outcome, treatment_year):
    pre_treatment_treated = treated_data[treated_data['data_year'] < treatment_year]
    pre_treatment_controls = control_data[control_data['data_year'] < treatment_year]
    
    if len(pre_treatment_treated) == 0:
        raise ValueError(f"No pre-treatment data for treated unit")
    
    predictor_means = pre_treatment_controls[predictors].mean()
    predictor_stds = pre_treatment_controls[predictors].std()
    
    predictor_stds = predictor_stds.replace(0, 1)
    
    treated_means = pre_treatment_treated[predictors].mean()
    treated_z = (treated_means - predictor_means) / predictor_stds
    
    control_counties = pre_treatment_controls['county_name'].unique()
    control_z = {}
    
    for county in control_counties:
        county_data = pre_treatment_controls[pre_treatment_controls['county_name'] == county]
        if len(county_data) > 0:
            county_means = county_data[predictors].mean()
            county_z = (county_means - predictor_means) / predictor_stds
            control_z[county] = county_z.values
    
    counties = list(control_z.keys())
    control_matrix = np.vstack([control_z[county] for county in counties])
    
    print(f"Treated unit predictors (standardized): {treated_z.values}")
    print(f"Control matrix shape: {control_matrix.shape}")
    
    def objective(weights):
        synth = np.dot(weights, control_matrix)
        loss = np.sum((treated_z.values - synth)**2)
        return loss
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(len(counties))]
    
    initial_weights = np.ones(len(counties)) / len(counties)
    
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
    
    print(f"Optimization result: {result.success}, message: {result.message}")
    print(f"Final objective value: {result.fun}")
    
    weights_series = pd.Series(result.x, index=counties)
    
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
        
        if weight_used > 0:
            synth_outcome[year] = weighted_sum / weight_used
    
    return weights_series, pd.Series(synth_outcome)

def calculate_treatment_effects(treated_outcome, synthetic_outcome, treatment_year):
    all_years = sorted(set(treated_outcome.index) | set(synthetic_outcome.index))
    
    combined = pd.DataFrame(index=all_years)
    combined['treated'] = treated_outcome
    combined['synthetic'] = synthetic_outcome
    
    combined['effect'] = combined['treated'] - combined['synthetic']
    
    combined['percent_effect'] = np.where(
        combined['synthetic'] != 0,
        (combined['treated'] / combined['synthetic'] - 1) * 100,
        np.nan
    )
    
    print("\nTreated vs Synthetic Control:")
    print(combined)
    
    return combined

def plot_synthetic_control_results(treated_data, synth_outcome, outcome_var, treatment_year, county_name, save_path=None):
    treated_outcome = treated_data.set_index('data_year')[outcome_var]
    
    plot_data = pd.DataFrame({
        'Treated': treated_outcome,
        'Synthetic Control': synth_outcome
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_data.plot(ax=ax)
    
    effect_data = plot_data['Treated'] - plot_data['Synthetic Control']
    ax2 = ax.twinx()
    effect_data.plot(ax=ax2, color='green', linestyle='--', label='Treatment Effect')
    ax2.set_ylabel('Treatment Effect', color='green')
    
    ax.axvline(x=treatment_year, color='red', linestyle='--')
    ax.text(treatment_year + 0.1, ax.get_ylim()[1] * 0.95, 'Walmart Entry', 
            color='red', rotation=90, va='top')
    
    ax.set_xlabel('Year')
    ax.set_ylabel(outcome_var)
    ax.set_title(f'Synthetic Control Analysis: {outcome_var} for {county_name}')
    
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    if save_path:
        plt.savefig(f"{save_path}/{county_name}_{outcome_var.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return fig

def validate_and_preprocess_data(data):
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print("Warning: Found missing values:")
        print(missing[missing > 0])
        
        for county in data['county_name'].unique():
            county_mask = (data['county_name'] == county)
            for col in data.columns:
                if col not in ['county_name', 'data_year']:
                    data.loc[county_mask, col] = data.loc[county_mask, col].fillna(
                        data.loc[county_mask, col].mean()
                    )
    
    for var in ['retail_priv_employment', 'retail_priv_annual_pay']:
        mean = data[var].mean()
        std = data[var].std()
        outliers = data[(data[var] > mean + 5*std) | (data[var] < mean - 5*std)]
        
        if len(outliers) > 0:
            print(f"\nWarning: Found outliers in {var}:")
            print(outliers[['county_name', 'data_year', var]])
            
            upper_bound = mean + 5*std
            lower_bound = mean - 5*std
            data[var] = data[var].clip(lower_bound, upper_bound)
    
    return data

def run_placebo_tests(treated_county, control_counties, all_control_data, treated_data, outcome_var, treatment_year, predictors):
    control_data = all_control_data[all_control_data['county_name'].isin(control_counties)].copy()
    
    actual_weights, actual_synth = synthetic_control(
        treated_data, control_data, predictors, outcome_var, treatment_year
    )
    
    actual_outcome = treated_data.set_index('data_year')[outcome_var]
    actual_effects = calculate_treatment_effects(actual_outcome, actual_synth, treatment_year)
    
    pre_treatment_actual = actual_effects[actual_effects.index < treatment_year]
    actual_pre_rmspe = np.sqrt(np.mean(pre_treatment_actual['effect']**2))
    
    post_treatment_actual = actual_effects[actual_effects.index >= treatment_year]
    actual_post_rmspe = np.sqrt(np.mean(post_treatment_actual['effect']**2))
    
    if actual_pre_rmspe > 0:
        actual_rmspe_ratio = actual_post_rmspe / actual_pre_rmspe
    else:
        actual_rmspe_ratio = float('inf')
    
    placebo_effects = {}
    rmspe_ratios = []
    
    placebo_counties = control_counties[:10]  # Limit to 10 placebo tests for efficiency
    
    for placebo_county in placebo_counties:
        placebo_treated = all_control_data[all_control_data['county_name'] == placebo_county].copy()
        placebo_controls = all_control_data[all_control_data['county_name'].isin([c for c in control_counties if c != placebo_county])].copy()
        
        try:
            placebo_weights, placebo_synth = synthetic_control(
                placebo_treated, placebo_controls, predictors, outcome_var, treatment_year
            )
            
            placebo_outcome = placebo_treated.set_index('data_year')[outcome_var]
            placebo_effects[placebo_county] = calculate_treatment_effects(placebo_outcome, placebo_synth, treatment_year)
            
            pre_treatment_placebo = placebo_effects[placebo_county][placebo_effects[placebo_county].index < treatment_year]
            placebo_pre_rmspe = np.sqrt(np.mean(pre_treatment_placebo['effect']**2))
            
            post_treatment_placebo = placebo_effects[placebo_county][placebo_effects[placebo_county].index >= treatment_year]
            placebo_post_rmspe = np.sqrt(np.mean(post_treatment_placebo['effect']**2))
            
            if placebo_pre_rmspe > 0:
                placebo_rmspe_ratio = placebo_post_rmspe / placebo_pre_rmspe
                rmspe_ratios.append(placebo_rmspe_ratio)
            
        except Exception as e:
            print(f"Error in placebo test for {placebo_county}: {e}")
    
    p_value = sum(1 for ratio in rmspe_ratios if ratio >= actual_rmspe_ratio) / (len(rmspe_ratios) + 1)
    
    return placebo_effects, actual_effects, p_value

def plot_placebo_test_results(actual_effects, placebo_effects, treatment_year, county_name, outcome_var, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for placebo_county, placebo_effect in placebo_effects.items():
        placebo_effect['effect'].plot(ax=ax, color='lightgray', alpha=0.5)
    
    actual_effects['effect'].plot(ax=ax, color='blue', linewidth=2, label=county_name)
    
    ax.axvline(x=treatment_year, color='red', linestyle='--')
    ax.text(treatment_year + 0.1, ax.get_ylim()[1] * 0.95, 'Treatment Year', 
            color='red', rotation=90, va='top')
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Treatment Effect')
    ax.set_title(f'Placebo Tests: {outcome_var} for {county_name}')
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.savefig(f"{save_path}/{county_name}_{outcome_var.replace(' ', '_')}_placebo.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return fig

def analyze_county(treated_county_name, treatment_year, control_data, treated_data, outcome_vars, save_plots=True, output_dir='output'):
    import os
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    treated_county_data = treated_data[treated_data['county_name'] == treated_county_name].copy()
    
    if len(treated_county_data) == 0:
        raise ValueError(f"No data found for treated county: {treated_county_name}")
    
    control_counties = control_data['county_name'].unique()
    
    results = {}
    
    for outcome_var in outcome_vars:
        print(f"\nAnalyzing {outcome_var}...")
        
        predictors = outcome_vars.copy()
        
        weights, synth_outcome = synthetic_control(
            treated_county_data, 
            control_data, 
            predictors, 
            outcome_var, 
            treatment_year
        )
        
        treated_outcome = treated_county_data.set_index('data_year')[outcome_var]
        effects = calculate_treatment_effects(treated_outcome, synth_outcome, treatment_year)
        
        placebo_effects, actual_effects, p_value = run_placebo_tests(
            treated_county_name, 
            control_counties, 
            control_data,
            treated_county_data,
            outcome_var, 
            treatment_year, 
            predictors
        )
        
        results[outcome_var] = {
            'weights': weights,
            'synthetic_outcome': synth_outcome,
            'treatment_effects': effects,
            'placebo_effects': placebo_effects,
            'p_value': p_value
        }
        
        save_path = output_dir if save_plots else None
        fig = plot_synthetic_control_results(
            treated_county_data, 
            synth_outcome, 
            outcome_var, 
            treatment_year, 
            treated_county_name,
            save_path
        )
        
        fig_placebo = plot_placebo_test_results(
            actual_effects, 
            placebo_effects, 
            treatment_year, 
            treated_county_name, 
            outcome_var,
            save_path
        )
        
        results[outcome_var]['plot'] = fig
        results[outcome_var]['placebo_plot'] = fig_placebo
        
    return results

def print_results(results, treated_county, treatment_year):
    print(f"\n========== RESULTS FOR {treated_county.upper()} ==========")
    print(f"Treatment year: {treatment_year}")
    
    for outcome_var, result in results.items():
        print(f"\n----- {outcome_var} -----")
        
        print("Control weights:")
        sorted_weights = result['weights'].sort_values(ascending=False)
        for county, weight in sorted_weights.items():
            if weight > 0.001:
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

def main():
    control_data = pd.read_csv("cleaned_control_summary_data.csv")
    treated_data = pd.read_csv("treatment_data.csv")
    
    control_data = validate_and_preprocess_data(control_data)
    treated_data = validate_and_preprocess_data(treated_data)
    
    treatment_counties = {
        "Androscoggin_County": 2001,
        "Berkshire_County": 1995,
        "Cheshire_County": 2002,
        "Hall_County": 1994,
        "Jackson_County": 2007,
        "Madison_County": 1998,
        "Rutland_County": 1997,
        "Sangamon_County": 2001,
        "Solano_County": 1993,
        "Washington_County": 2010
    }
    
    outcome_vars = [
        'retail_priv_employment',
        'retail_priv_annual_pay',
        'retail_share_of_private_employment',
        'retail_relative_wage'
    ]
    
    print("Checking for duplicates in control data...")
    duplicates = control_data.duplicated(subset=['county_name', 'data_year'], keep=False)
    if duplicates.any():
        print(f"Found {duplicates.sum()} duplicate county-year observations in control data")
        control_data = control_data.drop_duplicates(subset=['county_name', 'data_year'], keep='first')
        print(f"Removed duplicates. Control data shape now: {control_data.shape}")
    
    print("Checking for duplicates in treatment data...")
    duplicates = treated_data.duplicated(subset=['county_name', 'data_year'], keep=False)
    if duplicates.any():
        print(f"Found {duplicates.sum()} duplicate county-year observations in treatment data")
        treated_data = treated_data.drop_duplicates(subset=['county_name', 'data_year'], keep='first')
        print(f"Removed duplicates. Treatment data shape now: {treated_data.shape}")
    
    treated_county = "all"  # Change this to analyze different counties
    
    if treated_county == "all":
        all_results = {}
        # Create a file to save all results
        with open("all_counties_results.txt", "w") as f:
            for county, year in treatment_counties.items():
                print(f"\nAnalyzing {county}...")
                f.write(f"\n========== RESULTS FOR {county.upper()} ==========\n")
                f.write(f"Treatment year: {year}\n\n")
                try:
                    results = analyze_county(county, year, control_data, treated_data, outcome_vars)
                    all_results[county] = results
                    
                    # Save results to file
                    for outcome_var, result in results.items():
                        f.write(f"----- {outcome_var} -----\n")
                        
                        f.write("Control weights:\n")
                        sorted_weights = result['weights'].sort_values(ascending=False)
                        for county_name, weight in sorted_weights.items():
                            if weight > 0.001:
                                f.write(f"  {county_name}: {weight:.4f}\n")
                        
                        f.write("\nTreatment effects:\n")
                        post_treatment = result['treatment_effects'].loc[year:year+5]
                        avg_effect = post_treatment['effect'].mean()
                        avg_percent = post_treatment['percent_effect'].mean()
                        
                        f.write(f"  Average effect (5 years post-treatment): {avg_effect:.4f}\n")
                        f.write(f"  Average percent effect: {avg_percent:.2f}%\n")
                        f.write(f"  p-value: {result['p_value']:.4f}\n")
                        
                        if result['p_value'] < 0.05:
                            f.write("  The effect is statistically significant at the 5% level.\n\n")
                        elif result['p_value'] < 0.1:
                            f.write("  The effect is statistically significant at the 10% level.\n\n")
                        else:
                            f.write("  The effect is not statistically significant.\n\n")
                    
                    # Also print to console
                    print_results(results, county, year)
                    
                except Exception as e:
                    print(f"Error analyzing {county}: {e}")
                    f.write(f"Error analyzing {county}: {e}\n\n")
    else:
        treatment_year = treatment_counties[treated_county]
        results = analyze_county(treated_county, treatment_year, control_data, treated_data, outcome_vars)
        print_results(results, treated_county, treatment_year)

if __name__ == "__main__":
    main()