import os
import csv
import shutil

# ==== CONFIGURATION ====
store_csv_path = './WalmartsWithCountyState.csv'
county_folder_path = './1990.annual.by_area'
supercenter_folder = './WalmartSupercenter'
other_walmart_folder = './WalmartGeneral'
no_walmart_folder = './NoWalmart'
# ========================

# Create output folders
os.makedirs(supercenter_folder, exist_ok=True)
os.makedirs(other_walmart_folder, exist_ok=True)
os.makedirs(no_walmart_folder, exist_ok=True)

# 1. Load Walmart store data grouped by county
county_store_types = {}

with open(store_csv_path, newline='', encoding='utf-8') as store_file:
    reader = csv.DictReader(store_file)
    for row in reader:
        county = row.get("County", "").strip().lower()
        store_type = row.get("Type", "").strip()

        if county and county not in ("n/a", "error"):
            if county not in county_store_types:
                county_store_types[county] = set()
            county_store_types[county].add(store_type)

# 2. Classify and copy each county file
for filename in os.listdir(county_folder_path):
    if not filename.endswith('.csv'):
        continue

    parts = filename.split(' ', 2)
    if len(parts) < 3:
        continue

    county_part = parts[2]
    county_name = county_part.split(',')[0].strip().lower()
    source_path = os.path.join(county_folder_path, filename)

    store_types = county_store_types.get(county_name)

    if store_types:
        if "Supercenter" in store_types:
            shutil.copy(source_path, os.path.join(supercenter_folder, filename))
        else:
            shutil.copy(source_path, os.path.join(other_walmart_folder, filename))
    else:
        shutil.copy(source_path, os.path.join(no_walmart_folder, filename))

print("Files organized into folders:")
print(f" - Supercenter folder: {supercenter_folder}")
print(f" - Other Walmart folder: {other_walmart_folder}")
print(f" - No Walmart folder: {no_walmart_folder}")
