import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load your grid
df = pd.read_csv("grid.csv")

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Load India states shapefile
shapefile = gpd.read_file("gadm41_IND_1.shp")

# Filter Himachal Pradesh
hp = shapefile[shapefile["NAME_1"] == "Himachal Pradesh"]

# Keep only points inside Himachal
gdf_hp = gpd.sjoin(gdf, hp, how="inner", predicate="within")

# Save result
gdf_hp[["lat", "lon"]].to_csv("hp_grid.csv", index=False)

print("Filtered points:", gdf_hp.shape)
