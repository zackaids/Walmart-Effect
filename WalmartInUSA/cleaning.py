import os
import csv

folder_path = './WalmartInUSA/1990.annual.by_area'
csv_file = './WalmartInUSA/WalmartsWithCountyState.csv'
output_csv = './WalmartInUSA/matchingFiles.csv'

# List all files and filter only files (exclude folders)
files = os.listdir(folder_path)
only_files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

# Read county + state + type from input CSV
entries = []
with open(csv_file, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        county = row.get("County", "").strip().lower()
        state = row.get("St.", "").strip().lower()
        store_type = row.get("Type", "").strip()
        if county and state and county not in ("n/a", "error"):
            entries.append((county, state, store_type))

# Match files and track results with type
matched_files = []
for county, state, store_type in entries:
    for filename in only_files:
        filename_lower = filename.lower()
        if county in filename_lower and state in filename_lower:
            matched_files.append([store_type, county.title(), state.upper(), filename])

# Write matched results to output CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Type', 'County', 'State', 'Filename'])
    writer.writerows(matched_files)

print(f"Found {len(matched_files)} matching files.")
