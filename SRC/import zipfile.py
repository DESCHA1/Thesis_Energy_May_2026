import zipfile
import glob
import os

# 1. Define paths (Looking in the main DATA folder for the zips)
zip_search_pattern = r'C:\Users\desch\.vscode\THESIS_ENERGY_2026\DATA\zipped_WEATHER_DATA\*.zip'
extraction_folder = r'C:\Users\desch\.vscode\THESIS_ENERGY_2026\DATA\ERA5_Weather'

# Create the target folder
os.makedirs(extraction_folder, exist_ok=True)

# Find all zip files
zip_files = sorted(glob.glob(zip_search_pattern))
print(f"Found {len(zip_files)} zip files. Starting smart extraction...")

# 2. Extract and rename to prevent overwriting
for i, zip_path in enumerate(zip_files, 1):
    # Get the name of the zip file itself (e.g., 'download_2015')
    base_zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for internal_file in zip_ref.namelist():
            # Extract the generically named file
            extracted_path = zip_ref.extract(internal_file, extraction_folder)
            
            # Create a unique new name! 
            # (e.g., 'download_2015_data_stream-oper_stepType-accum.nc')
            new_name = f"{base_zip_name}_{internal_file}"
            new_path = os.path.join(extraction_folder, new_name)
            
            # Rename the file immediately
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(extracted_path, new_path)

print(f"✅ Success! All files extracted and uniquely renamed in: {extraction_folder}")