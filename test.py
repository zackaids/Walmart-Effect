import requests
import pandas as pd
import json

# df = pd.read_csv("https://api.census.gov/data/2013/acs/acs5/subject?get=group(S1901)&ucgid=8600000US33328")

# print(df.head())

# import requests
# import json
# import prettytable
# headers = {'Content-type': 'application/json'}
# data = json.dumps({"seriesid": ['CUUR0000SA0','SUUR0000SA0'],"startyear":"2011", "endyear":"2014"})
# p = requests.post('https://api.bls.gov/publicAPI/v1/timeseries/data/', data=data, headers=headers)
# json_data = json.loads(p.text)
# for series in json_data['Results']['series']:
#     x=prettytable.PrettyTable(["series id","year","period","value","footnotes"])
#     seriesId = series['seriesID']
#     for item in series['data']:
#         year = item['year']
#         period = item['period']
#         value = item['value']
#         footnotes=""
#         for footnote in item['footnotes']:
#             if footnote:
#                 footnotes = footnotes + footnote['text'] + ','
    
#         if 'M01' <= period <= 'M12':
#             x.add_row([seriesId,year,period,value,footnotes[0:-1]])
#     output = open(seriesId + '.txt','w')
#     output.write (x.get_string())
#     output.close()



# total employees (retail)
# # of establishments (total)
# average annual pay
# population changes
# consumer spending

year = "2005"
# key = '?registrationkey={}'
fips = '06037'
# fips = 45047

series_id = "ENU4210110544-45"
api_key = 'c05795b2cc90416ab359e80a52588bac'

headers = {'Content-type': 'application/json'}

data = {
    "seriesid": [series_id],
    "startyear": "2015",
    "endyear": "2022",
    "registrationkey": api_key,
    "catalog": True
}

url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
response = requests.post(url, data=json.dumps(data), headers=headers)

results = response.json()

data_pts = results["Results"]["series"][0]["data"]
# print(json.dumps(data_pts, indent=2))
# print(results)
# json_data = json.loads(results.text)
# print(json_data)

# df = pd.DataFrame(data_pts)
# print(df.head())
# print(df.head())

# 

