import numpy as np
import pandas as pd
from scipy import optimize
from tqdm import tqdm
import os

class LargeScaleSyntheticControl:
    def __init__(self, control_data_path, chunk_size=100000):
        self.control_data_path = control_data_path
        self.chunk_size = chunk_size
        self.control_stats = None
        
    def precompute_control_stats(self):
        stats = []
        for chunk in tqdm(pd.read_csv(self.control_data_path, chunksize=self.chunk_size, dtype={'county_name': str}), desc="Processing control data"):
            chunk['county_name'] = chunk['county_name'].str.zfill(5)
            stats.append(chunk.groupby('county_name').agg({
                'retail_priv_employment': 'mean',
                'retail_priv_annual_pay': 'mean',
                'all_priv_employment': 'mean',
                'all_priv_annual_pay': 'mean'
            }))
        self.control_stats = pd.concat(stats).groupby(level=0).mean()
        return self.control_stats
    
    def process_treated_county(self, treated_data, predictors, outcome_var, treatment_year):
        pre_treatment = treated_data[treated_data['data_year'] < treatment_year]
        if len(pre_treatment) == 0:
            raise ValueError("No pre-treatment data available")
        
        if self.control_stats is None:
            self.precompute_control_stats()
            
        means = self.control_stats[predictors].mean()
        stds = self.control_stats[predictors].std().replace(0, 1)
        treated_means = pre_treatment[predictors].mean()
        treated_z = (treated_means - means) / stds
        control_z = (self.control_stats[predictors] - means) / stds
        
        def objective(weights):
            synth = np.dot(weights, control_z.values)
            return np.sum((treated_z.values - synth)**2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(len(control_z))]
        initial_weights = np.ones(len(control_z)) / len(control_z)
        
        result = optimize.minimize(
            objective, initial_weights, 
            bounds=bounds, constraints=constraints,
            method='SLSQP', options={'maxiter': 1000}
        )
        
        if not result.success:
            print(f"Optimization warning: {result.message}")
        
        weights = pd.Series(result.x, index=self.control_stats.index)
        synth_outcome = {}
        
        for year in treated_data['data_year'].unique():
            year_data = []
            for chunk in pd.read_csv(self.control_data_path, chunksize=self.chunk_size, dtype={'county_name': str}):
                chunk = chunk[chunk['data_year'] == year]
                if len(chunk) > 0:
                    chunk['county_name'] = chunk['county_name'].str.zfill(5)
                    year_data.append(chunk)
            
            if year_data:
                year_df = pd.concat(year_data)
                weighted_sum = 0
                total_weight = 0
                
                for county, weight in weights.items():
                    if weight > 0.01:
                        county_data = year_df[year_df['county_name'] == county]
                        if len(county_data) > 0:
                            weighted_sum += weight * county_data[outcome_var].values[0]
                            total_weight += weight
                
                if total_weight > 0:
                    synth_outcome[year] = weighted_sum / total_weight
        
        return weights, pd.Series(synth_outcome)

def create_results_df(synth_outcome, treated_data, outcome_var):
    synth_years = set(synth_outcome.index)
    actual_years = set(treated_data['data_year'])
    common_years = sorted(synth_years & actual_years)
    
    if not common_years:
        raise ValueError("No overlapping years between synthetic and actual data")
    
    results = []
    for year in common_years:
        row = {
            'year': year,
            'synthetic_outcome': synth_outcome[year],
            'actual_outcome': treated_data.loc[treated_data['data_year'] == year, outcome_var].values[0]
        }
        results.append(row)
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    sc = LargeScaleSyntheticControl("cleaned_control_summary_data.csv")
    treated_data = pd.read_csv("treatment_data.csv", dtype={'county_name': str})
    treated_data['county_name'] = treated_data['county_name'].str.zfill(5)
    
    predictors = ['retail_priv_employment', 'retail_priv_annual_pay', 'all_priv_employment', 'all_priv_annual_pay']
    outcome_var = 'retail_priv_employment'
    treatment_year = 2010
    
    weights, synth_outcome = sc.process_treated_county(treated_data, predictors, outcome_var, treatment_year)
    
    results = create_results_df(synth_outcome, treated_data, outcome_var)
    results.to_csv(f"results_{treated_data['county_name'].iloc[0]}.csv", index=False)