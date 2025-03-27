import pandas as pd

df = pd.read_csv("https://api.census.gov/data/2013/acs/acs5/subject?get=group(S1901)&ucgid=8600000US33328")

print(df.head())