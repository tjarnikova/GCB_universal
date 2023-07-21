#!/usr/bin/env python
import sys
import os.path
import datetime 
import numpy as np
import glob
from netCDF4 import Dataset, MFDataset
import math
import datetime

inFile = "GCB_Simulation_A.nc"
missingVal = 1E20

# create mask
maskFile = "WOAmask.nc"
nc_mask_id = Dataset(maskFile, 'w', format='NETCDF3_CLASSIC')
nc_mask_id.createDimension("LONGITUDE", 360)
nc_mask_id.createDimension("LATITUDE", 180)
nc_mask_id.createVariable("mask", 'f', ('LATITUDE', 'LONGITUDE'))

nc_in = Dataset(inFile, 'r', format='NETCDF3_CLASSIC')

data = nc_in.variables["sst"][0,:,:]
nc_in.close()


mask = np.copy(data) 
mask[ mask < missingVal ] = 1
mask[ mask == missingVal ] = 0
mask[0:12,:] = 0

# greenwhich centered

# mask = np.roll(mask,180,axis=1)

nc_mask_id.variables["mask"][:] = mask
nc_mask_id.close()

