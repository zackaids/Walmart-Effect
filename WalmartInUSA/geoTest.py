import csv
from geopy.geocoders import Nominatim
import time

input_file = './WalmartInUSA/Walmarts.csv'
output_file = './WalmartInUSA/WalmartsWithCounty.csv'

geolocator = Nominatim(user_agent="county_finder")

with open(input_file, newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    reader.fieldnames = [field.strip() for field in reader.fieldnames]  # Strip header whitespace
    fieldnames = reader.fieldnames + ['County']  # Add new column

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        try:
            address = f"{row['Address']}, {row['City']}, {row['St.']}"
            location = geolocator.geocode(address)
            if location:
                reverse_location = geolocator.reverse((location.latitude, location.longitude), exactly_one=True)
                county = reverse_location.raw.get("address", {}).get("county", "N/A")
            else:
                county = 'N/A'
        except Exception as e:
            county = "N/A"

        row['County'] = county
        writer.writerow(row)
        print(f"{address} → {county}")
        time.sleep(.20)  # Avoid API rate limit
