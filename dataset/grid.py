import pandas as pd
import numpy as np

# Himachal Pradesh approx boundaries
lat_min, lat_max = 30.5, 33.0
lon_min, lon_max = 75.5, 79.0

# step size (controls grid density)
step = 0.1   # ~10 km (we'll refine later)

lats = np.arange(lat_min, lat_max, step)
lons = np.arange(lon_min, lon_max, step)

grid = []

for lat in lats:
    for lon in lons:
        grid.append([lat, lon])

df = pd.DataFrame(grid, columns=["lat", "lon"])

df.to_csv("grid.csv", index=False)

print("Grid created:", df.shape)