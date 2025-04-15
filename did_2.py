import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

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
        "Washington_County": 2010
    }