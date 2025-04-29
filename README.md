# Walmart Effect

Research & Report by: **Zack Aidarov & Max Suc**

This is the code for the report that follows the effects of Walmart's entry into 10 different U.S. counties between 1993 and 2010 using the synthetic control method.
It examines four key retail market indicators:
* Private Retail Employment
* Annual Retail Wages
* Retail's Share of Private Employment
* Relative Retail Wages
  
The paper can be found by clicking [here](FinalReport.pdf)

## Navigating the files
* `synth.py` contains the main synthetic control method
* `cleaned_control_summary_data.csv` contains the data for control counties
* `treatment_data.csv` contains the data for treatment counties
* `all_counties_results.txt` contains the statistical results including the weights of each control county
* `output` contains all the graphs including synthetic control and placebo tests
