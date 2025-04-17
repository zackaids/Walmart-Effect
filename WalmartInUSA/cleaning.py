import os

folder_path = './WalmartInUSA/1990.annual.by_area'

# List all files and directories
files = os.listdir(folder_path)

# Optional: Filter only files (exclude folders)
only_files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

print(only_files)
