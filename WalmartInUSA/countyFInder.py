from geopy.geocoders import Nominatim
import csv
import time

geolocator = Nominatim(user_agent="county_finder")

with open('./WalmartInUSA/Walmarts.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in range(10):#reader:
        address = f"{row['Address']}, {row['City']}, {row['St.']}"
        try:
            location = geolocator.geocode(address)
            if location:
                reverse_location = geolocator.reverse((location.latitude, location.longitude), exactly_one=True)
                county = reverse_location.raw.get("address", {}).get("county", "N/A")
            else:
                county = "Not found"
        except Exception as e:
            county = f"Error: {e}"

        print(f"{address} → County: {county}")
        time.sleep(1)
