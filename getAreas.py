#!/usr/bin/env python
import sys
import os.path
import datetime 
import numpy as np
import glob
from netCDF4 import Dataset, MFDataset
import math

# # open basin_mask.nc
meshFile = glob.glob("meshmask.nc")
nc_mesh_id = Dataset(meshFile[0], 'r' )

e1t = nc_mesh_id.variables["e1t"][0,:,:]
e2t = nc_mesh_id.variables["e2t"][0,:,:]
tmask = nc_mesh_id.variables["tmask"][0,0,:,:]
lon = nc_mesh_id.variables["nav_lon"][:]
lat = nc_mesh_id.variables["nav_lat"][:]

print(e1t.shape)
print(e2t.shape)
print(tmask.shape)
print(lon.shape)
print(lat.shape)

areas = e1t * e2t
print(areas.shape)
masked_areas = tmask * areas

# Written out in 10^8 km^2
print(np.sum(masked_areas)/1000000/1E8)

print(masked_areas.shape)

# eq_masked_areas = np.copy(lat)


N_masked_areas = np.where(lat >= 30.0, masked_areas, 0)

S_masked_areas = np.where(lat <= -30.0, masked_areas, 0)
# print(np.sum(eq_masked_areas))
print(np.sum(N_masked_areas)/1000000/1E8)
print(np.sum(S_masked_areas)/1000000/1E8)
print((np.sum(masked_areas)-np.sum(N_masked_areas)-np.sum(S_masked_areas))/1000000/1E8)

# nc_basin_id = Dataset(basinFile[0], 'r' )
# area = nc_basin_id.variables["AREA"][:]
# vol = nc_basin_id.variables["VOLUME"][:]