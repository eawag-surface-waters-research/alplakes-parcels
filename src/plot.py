import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from functions import rotate_velocity


particle_file = "runs/garda/particle_tracks.zarr"
input_file = "data/delft3d-flow/garda/20260201.nc"

ds = xr.open_dataset(input_file)
data_xarray = xr.open_zarr(particle_file)

x = ds["XCOR"].values
y = ds["YCOR"].values

u = ds["U1"].values
v = ds["V1"].values
alfas = ds["ALFAS"].values

print(u.shape)

import matplotlib.pyplot as plt
import numpy as np

mask = (x != 0) | (y != 0)
cell_mask = mask[:-1,:-1] & mask[1:,:-1] & mask[:-1,1:] & mask[1:,1:]
data = np.ma.masked_where(~cell_mask, np.ones_like(cell_mask, dtype=float))

# Collapse invalid coordinates to NaN won't work, but we can
# replace them with neighboring valid values so degenerate cells have zero area
x_fixed = x.copy()
y_fixed = y.copy()
x_fixed[~mask] = np.nan
y_fixed[~mask] = np.nan

# Use PolyCollection instead — it only draws valid cells
from matplotlib.collections import PolyCollection

# Build polygons only for valid cells
polys = []
for i in range(x.shape[0] - 1):
    for j in range(x.shape[1] - 1):
        if cell_mask[i, j]:
            poly = [
                [x[i, j],     y[i, j]],
                [x[i+1, j],   y[i+1, j]],
                [x[i+1, j+1], y[i+1, j+1]],
                [x[i, j+1],   y[i, j+1]],
            ]
            polys.append(poly)

fig, ax = plt.subplots()
pc = PolyCollection(polys, facecolor='white', edgecolor='#1a9bc0', linewidth=0.5)
ax.add_collection(pc)
ax.autoscale()
ax.set_aspect('equal')
ax.set_facecolor('#F5F5F5')

plt.plot(data_xarray["lon"].T, data_xarray["lat"].T, marker="o")
plt.show()
