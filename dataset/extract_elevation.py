import pandas as pd
import rasterio
import glob

# Load your filtered grid
df = pd.read_csv("hp_grid.csv")  # <-- your file

# Load all SRTM tiles
tiles = glob.glob("srtm/*.hgt")

def get_elevation(lat, lon):
    for tile in tiles:
        try:
            with rasterio.open(tile) as src:
                for val in src.sample([(lon, lat)]):
                    return val[0]
        except:
            continue
    return None

# Apply elevation extraction
df["elevation"] = df.apply(lambda row: get_elevation(row["lat"], row["lon"]), axis=1)

# Save updated dataset
df.to_csv("dataset_with_elevation.csv", index=False)

print("Done extracting elevation!")