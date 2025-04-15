import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns

# Modify the synthetic_control function to handle small sample sizes
def synthetic_control(data, treated_unit, outcome_var, treatment_year, 
                      predictor_vars=None, pre_period_start=None):
    """
    Create synthetic control for one treated unit.
    
    Parameters:
    -----------
    data : DataFrame
        Full dataset
    treated_unit : str
        Name of the treated unit (county)
    outcome_var : str
        Outcome variable to analyze
    treatment_year : int
        Year when treatment occurred
    predictor_vars : list
        Variables to use for prediction (if None, uses outcome_var pre-treatment)
    pre_period_start : int
        First year of pre-treatment period (if None, uses earliest year in data)
    
    Returns:
    --------
    results_df : DataFrame
        Data with actual and synthetic outcomes for treated unit
    weights : Series
        Weights assigned to each control unit
    """
    # Ensure the data is sorted
    data = data.sort_values(['county_name', 'data_year'])
    
    # Determine pre-period
    if pre_period_start is None:
        pre_period_start = data['data_year'].min()
    
    # Separate data for treated unit and potential controls
    treated_data = data[data['county_name'] == treated_unit].copy()
    control_data = data[(data['is_treatment'] == 0)].copy()
    
    # Define pre-treatment period
    treated_pre = treated_data[(treated_data['data_year'] < treatment_year) & 
                              (treated_data['data_year'] >= pre_period_start)].copy()
    control_pre = control_data[(control_data['data_year'] < treatment_year) & 
                              (control_data['data_year'] >= pre_period_start)].copy()
    
    # Check if we have enough pre-treatment data (at least 3 years)
    if len(treated_pre) < 3:
        print(f"Not enough pre-treatment data for {treated_unit} (only {len(treated_pre)} years)")
        return None, None
    
    # If no specific predictors, use pre-treatment outcome values
    if predictor_vars is None:
        # Create features matrix: pivot to have years as features
        treated_features = treated_pre.pivot(index='county_name', 
                                           columns='data_year', 
                                           values=outcome_var)
        
        control_features = control_pre.pivot(index='county_name', 
                                           columns='data_year', 
                                           values=outcome_var)
        
        # Remove control counties with missing data
        control_features = control_features.dropna()
        
        if len(control_features) < 5:
            print(f"Too few control units with complete data for {treated_unit}")
            return None, None
        
        # Standardize the features
        scaler = StandardScaler()
        control_features_scaled = pd.DataFrame(
            scaler.fit_transform(control_features),
            index=control_features.index,
            columns=control_features.columns
        )
        
        treated_features_scaled = pd.DataFrame(
            scaler.transform(treated_features),
            index=treated_features.index,
            columns=treated_features.columns
        )
    else:
        # Use specific predictors
        pass
    
    # FIX: Adjust the CV parameter based on number of samples
    n_years = len(treated_features.columns)
    cv = min(5, n_years)  # Use at most n_years folds
    
    # Use a simpler approach for cases with very few pre-treatment periods
    if n_years < 3:
        # Use Ridge regression instead of cross-validation
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0, positive=True, fit_intercept=False)
    else:
        model = LassoCV(cv=cv, positive=True, fit_intercept=False)
    
    # Reshape for sklearn: treated unit is target, control units are features
    X = control_features_scaled.T  # Transpose so counties are features
    y = treated_features_scaled.T.iloc[:, 0]  # First (only) column of treated
    
    # Fit the model
    model.fit(X, y)
    
    # Get the weights assigned to each control unit
    if hasattr(model, 'coef_'):
        weights = pd.Series(model.coef_, index=control_features_scaled.index)
    else:
        # For alternative models
        weights = pd.Series(model.coef_, index=control_features_scaled.index)
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum() if weights.sum() > 0 else weights
    
    # Keep only non-zero weights
    weights = weights[weights > 0.001]
    
    # Create synthetic control for all years (pre and post treatment)
    # Get all available data for each county (not just pre-treatment)
    control_wide = control_data.pivot(index='data_year', 
                                    columns='county_name', 
                                    values=outcome_var)
    
    # We can only use control counties that have weights and data
    available_controls = list(set(weights.index) & set(control_wide.columns))
    
    # Check if we have enough controls with data
    if len(available_controls) < 1:
        print(f"No control counties with complete data for {treated_unit}")
        return None, None
    
    # Recalculate weights using only available controls
    available_weights = weights[available_controls]
    
    # Renormalize weights
    if available_weights.sum() > 0:
        available_weights = available_weights / available_weights.sum()
    
    # Calculate synthetic control values using available controls
    synthetic_values = control_wide[available_controls].dot(available_weights)
    
    # Get actual values for treated unit
    actual_values = treated_data.set_index('data_year')[outcome_var]
    
    # Combine actual and synthetic into one dataframe
    results_df = pd.DataFrame({
        'actual': actual_values,
        'synthetic': synthetic_values
    }).reset_index()
    
    # Add treatment info
    results_df['period'] = 'Pre-Treatment'
    results_df.loc[results_df['data_year'] >= treatment_year, 'period'] = 'Post-Treatment'
    results_df['treated_unit'] = treated_unit
    results_df['treatment_year'] = treatment_year
    
    return results_df, available_weights

# Rest of your functions remain the same

# Function to run synthetic control for all treated units and summarize results
def run_all_synthetic_controls(data, outcome_var, treatment_years_dict, 
                               pre_period_start=None):
    """
    Run synthetic control for all treated units.
    
    Parameters:
    -----------
    data : DataFrame
        Full dataset
    outcome_var : str
        Outcome variable to analyze
    treatment_years_dict : dict
        Dictionary mapping treated units to their treatment years
    pre_period_start : int
        First year of pre-treatment period (if None, uses earliest year in data)
    
    Returns:
    --------
    all_results : DataFrame
        Combined results for all treated units
    all_weights : dict
        Dictionary of weights for each treated unit
    """
    all_results = []
    all_weights = {}
    
    for unit, year in treatment_years_dict.items():
        # Check if unit exists in the data
        if unit not in data['county_name'].unique():
            print(f"Warning: {unit} not found in dataset")
            continue
        
        print(f"Creating synthetic control for {unit} (treated in {year})")
        results, weights = synthetic_control(
            data=data,
            treated_unit=unit,
            outcome_var=outcome_var,
            treatment_year=year,
            pre_period_start=pre_period_start
        )
        
        if results is not None:
            all_results.append(results)
            all_weights[unit] = weights
    
    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        return all_results_df, all_weights
    else:
        return None, {}

# Define treatment years
treatment_years = {
    "Androscoggin_County": 2001,
    "Berkshire_County": 1995,
    "Cheshire_County": 2002,
    # Add more as needed
}

# Set earliest year to use for pre-treatment period
# This helps ensure enough years for each county
earliest_pre_year = 1990

# Choose one outcome variable to test
outcome_var = 'retail_priv_employment'

# Load your data
df = pd.read_csv("data/did_data/walmart_study_summary_data.csv")

# Print data info
print(f"Year range in data: {df['data_year'].min()} to {df['data_year'].max()}")
print(f"Number of counties: {df['county_name'].nunique()}")

# Run synthetic control for one county first as a test
test_county = "Androscoggin_County"
test_year = treatment_years[test_county]

print(f"\nTesting synthetic control for {test_county} (treated in {test_year})")
test_results, test_weights = synthetic_control(
    data=df,
    treated_unit=test_county,
    outcome_var=outcome_var,
    treatment_year=test_year,
    pre_period_start=earliest_pre_year
)

if test_results is not None:
    print("Test successful! Now running for all counties...")
    
    # Run for all counties
    results_df, weights_dict = run_all_synthetic_controls(
        data=df,
        outcome_var=outcome_var,
        treatment_years_dict=treatment_years,
        pre_period_start=earliest_pre_year
    )
    
    if results_df is not None:
        # Calculate treatment effects
        results_df['effect'] = results_df['actual'] - results_df['synthetic']
        results_df['rel_year'] = results_df['data_year'] - results_df['treatment_year']
        
        # Save results
        results_df.to_csv(f"synthetic_control_results_{outcome_var}.csv", index=False)
        
        print(f"Analysis complete for {outcome_var}")
        print(f"Results saved to synthetic_control_results_{outcome_var}.csv")
else:
    print("Test failed. Please check your data and parameters.")