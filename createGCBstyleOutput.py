### script for producing global carbon budget-style output for any model version, 2023
### usage:
### python createGCBstyleOutput.py {inputNumber} {modelversion} {yrFrom} {yrTo}
### python createGCBstyleOutput.py -1 TOM12_TJ_1ASA 1959 2023
#### see /gpfs/data/greenocean/GCB/GCB_universal/inputNumberReference.txt to see which 
#######
#base directory where results go, change if necessary:
resultsbd = '/gpfs/data/greenocean/GCB/GCB_universal/GCBstyleOutput/'
#directory where model results are read from 
basedir = '/gpfs/data/greenocean/software/runs/' 
### updated by Tereza Jarnikova, based heavily on Dave Willis's /gpfs/data/greenocean/GCB/GCB_RECCAPcreateGCB_RECCAP.py

import sys
import os
import os.path
import datetime 
import numpy as np
import glob
from netCDF4 import Dataset, MFDataset
import math
from scipy.interpolate import griddata
import datetime
from scipy.ndimage import median_filter
# import arrow
import warnings
import time
import datetime
warnings.filterwarnings('ignore')

OutputNumber = int(sys.argv[1])
modelName = (sys.argv[2])
yrFrom = int(sys.argv[3])
yrTo = int(sys.argv[4])

tt = time.time()
value = datetime.datetime.fromtimestamp(tt)
timestamp = (value.strftime('%Y%m%d'))

print(f'Calculating variable number {OutputNumber} for {modelName}, years {yrFrom}-{yrTo}, timestamp = {timestamp}')
print(f'model is read from {basedir}{modelName}')
eraPressure = False # are we using era or not
#### inputs, etc
resultsdir = f'{resultsbd}{modelName}/'
try:
    os.mkdir(resultsdir) #make a directory to put our files in
    print(f'made results directory {resultsdir}')
except:
    print('results directory already exists')
#put runs in a list for looping
runs = [modelName]
horse = False #flag for debugging, keep True if not debugging


# ========= TO GENERATE OUTPUTS FOR GCB OR RECCAP ==========

# for GCB: tier 1: -1, Anc_v3, 0,23,2,10,11,8,9,24,
#          tier 2: 23, 3,4,7,22,21,12,13,14,15,16,17,18,19,20,    25
#          tier 2 (3D): 26,27,28,29,30,34,31,32,33

# We need the pressure at the surface for a conversion to fugacity
if eraPressure:
    forcedir = "/gpfs/data/greenocean/software/products/ERA5_v202303_TJ/"
else:
    forcedir = "/gpfs/data/greenocean/software/products/NCEPForcingData/" #ncep pressures, used until the 2022 submission 
 #change to ERA forcing

# set origin of the timing required 
# ---------------------------------------
origin = datetime.datetime(1950,1,1,0,0,0) # GCB
time_origin_string = "1950-JAN-01 00:00:00"
time_origin_units = "seconds since 1950-01-01 00:00:00"

############################

def regrid(var, var_lons, var_lats, target_lon, target_lat, tmask, missingVal):
    # duplicate edge pixels 
    # print(var.shape)
    for iy in range(0,var.shape[1]):
        var[:,iy,0] = (var[:,iy,1]+var[:,iy,-2])/2
        var[:,iy,-1] =  var[:,iy,0]
    for ix in range(0,var.shape[2]):
        var[:,-1,ix] = var[:,-2,ix] 

    # adjust nav_lon to WOA grid 
    var_lons[ var_lons < 0 ] = var_lons[ var_lons < 0 ] + 360
    # tmask = np.roll(tmask, tmask.shape[0]/2, axis=0) # tmask is -180 to 180 -> 0-360

    list_lons = var_lons.ravel()
    list_lats = var_lats.ravel()
    points = np.column_stack([list_lats,list_lons])

    # if( len(target_lon.shape) < 2):
    # # if(target_lat == None or target_lon == None):
    #     # create obs_lon and obs_lat
    #     # -179.5->179.5; -89.5->89.5, 1 deg res
    #     # target_lon = np.arange(-179.5,180.5,1)
    #     # target_lon = np.roll(target_lon, 180, axis=0) # convert to 0-360
    #     target_lon = np.arange(-179.5,180.5,1)
    #     target_lat = np.arange(-89.5,90.5,1)

    # re-interpolate onto WOA grid, centred on greenwich meridian - NO! WOA is 0-360
    data_out = np.zeros((var.shape[0],target_lat.shape[0],target_lon.shape[0]),dtype=float) + np.nan
    for t in range(0,12):
        # preprocesss input data to remove mask
        vals = []
        maskVals = []
        for iy in range(0,var_lons.shape[0]):
            for ix in range(0,var_lons.shape[1]):
                maskVals.append(tmask[iy,ix])
                vals.append(var[t,iy,ix])
        vals = np.array(vals)

        valsFilt = []
        pointsFilt = []
        for p in range(0,points.shape[0]):
            if maskVals[p] == 1:
                valsFilt.append(vals[p])
                pointsFilt.append( ( points[p,0], points[p,1] ) )
        pointsFilt = np.array(pointsFilt)
        valsFilt = np.array(valsFilt)

        if pointsFilt.shape[0] > 0:
            # vals = []
            # for iy in range(0,var_lons.shape[0]):
            #     for ix in range(0,var_lons.shape[1]):
            #         vals.append(var[t,iy,ix])
            # vals = np.array(vals)

            grid_lon,grid_lat = np.meshgrid(target_lon,target_lat)
            data_out[t,:,:] = griddata(pointsFilt,valsFilt,(grid_lat,grid_lon), method='linear')
            
            data_out_near = griddata(pointsFilt,valsFilt,(grid_lat[:,0:2],grid_lon[:,0:2]), method='nearest') # take these values as has errors at date line due to itnerpoloation
            data_out[t,:,0:2] = data_out_near
            data_out_near = griddata(pointsFilt,valsFilt,(grid_lat[:,-3:],grid_lon[:,-3:]), method='nearest') # take these values as has errors at date line due to itnerpoloation
            data_out[t,:,-3:] = data_out_near[:,-3:]
        else:
            data_out[t,:,:] = missingVal

    # tidy up
    data_out[ data_out > missingVal/1000. ] = missingVal


    return data_out
def subDomainORCA(lonLim, latLim, var_lons, var_lats, in_data, landMask, volMask, missingVal):

    lonStart = int(lonLim[0])
    lonEnd = int(lonLim[1])
    latStart = int(latLim[0])
    latEnd = int(latLim[1])

    if len(in_data.shape) == 3:
        mask = np.zeros(in_data[0,:,:].shape) + 1
    if len(in_data.shape) == 4:
        mask = np.zeros(in_data[0,0,:,:].shape) + 1

    # mask[var_lons < lonStart] = 0
    # mask[var_lons > lonEnd] = 0
    # mask[var_lats < latStart] = 0
    # mask[var_lats > latEnd] = 0

    mask[var_lons < lonStart] = missingVal
    mask[var_lons > lonEnd] = missingVal
    mask[var_lats < latStart] = missingVal
    mask[var_lats > latEnd] = missingVal
    
    ind_mask = np.where( mask == missingVal )
    # ind_land = np.where( landMask == missingVal )
    # ind_vol = np.where( volMask == missingVal )
    ind_land = np.isnan(landMask)
    ind_vol = np.isnan(volMask)

    if len(in_data.shape) == 3:
        for t in range(0,in_data.shape[0]):
            temparr = in_data[t,:,:]
            temparr[ ind_mask ] = missingVal
            temparr[ ind_land ] = missingVal
            in_data[t,:,:] = temparr
            # print(temparr[60,:])
            # input('------')

    if len(in_data.shape) == 4:
        for t in range(0,in_data.shape[0]):
            for z in range(0,in_data.shape[1]):
                temparr = in_data[t,z,:,:]
                temparr[ ind_mask ] = missingVal
                in_data[t,z,:,:] = temparr
            temparr = in_data[t,:,:,:]
            if ind_vol.shape[0] == in_data.shape[1]:
                temparr[ ind_vol ] = missingVal
            in_data[t,:,:,:] = temparr

    return in_data


# Include 1D data, equivalent to csv files as before
if OutputNumber == -1:
    print('Generating 1D data for fluxes')

# constants
missingVal = 1E20
Er = 6.3781E6 # meters
Ec = 2*math.pi*Er
Ea = 4*math.pi*Er*Er
peta = 1e-15
terra = 1e-12
giga = 1e-9
carbon = 12
litre = 1000
secondsInYear = 3600.*24.*365.  # seconds in a year (non leap)


area = np.zeros((180,360))
for y in range(0,180):
    ang = np.radians(y-90)
    area[y,:] = Ec*(1/360.)*math.cos(ang) * Ec*(1/360.)

# seconds per year
raass = 3600.*24.*365.

# get target mesh values
target_lon = np.arange(0.5,360.5,1) # triggers setting in regrid as < 2 dims
target_lat = np.arange(-89.5,90.5,1)

# get source co-ords
meshFile = glob.glob("/gpfs/data/greenocean/software/resources/breakdown/mesh_mask3_6_low_res.nc")
nc_mesh_id = Dataset(meshFile[0], 'r' )
depths = nc_mesh_id.variables["gdept_1d"][0,:].data
zDim = len(depths)
# print(zDim, depths)
tmask = nc_mesh_id.variables["tmask"][0,0,:,:]
tmaskDepth = nc_mesh_id.variables["tmask"][0,:,:,:]
var_lons = nc_mesh_id.variables["nav_lon"][:]
var_lats = nc_mesh_id.variables["nav_lat"][:]

# open basin_mask.nc
basinFile = glob.glob('/gpfs/data/greenocean/software/resources/breakdown/basin_mask.nc')
nc_basin_id = Dataset(basinFile[0], 'r' )
mask_area = nc_basin_id.variables["AREA"][:].data #m^2
mask_vol = nc_basin_id.variables["VOLUME"][:].data #m^3
landMask = np.copy(mask_area)
landMask[ landMask > 0 ] = 1
landMask[ landMask == 0 ] = np.nan
volMask = np.copy(mask_vol)
volMask[ volMask > 0 ] = 1
volMask[ volMask == 0 ] = np.nan

# get mask
maskFile = glob.glob('/gpfs/data/greenocean/software/resources/breakdown/WOAmask.nc')
nc_mask_id = Dataset(maskFile[0], 'r' )
mask = nc_mask_id.variables["mask"][:]
# mask = np.roll(mask, int(mask.shape[1]/2), axis = 1)

# get Ancillary Data
ancFile = glob.glob('/gpfs/data/greenocean/software/resources/breakdown/AncillaryData_v3.nc')
nc_anc_id = Dataset(ancFile[0], 'r' )
vol_mask = nc_anc_id.variables["MASK_VOL"][:]

if OutputNumber == -1:

    # loop over each simulation
    for r in range(0,len(runs)):

        # create outputfile
        #outputFile = "PlankTOM_Simulation_"+str(sim[r])+"_integrated_timelines.nc"
        outputFile = f'{resultsdir}integrated_timelines_PlankTOM_1_gr_{yrFrom}-{yrTo}_v{timestamp}.nc'
        print(f'making file: {outputFile}')
        if horse:
            nc_out_id = Dataset(outputFile, 'w', format='NETCDF4_CLASSIC')

            # create dimensions and variables of these
            nc_d_lat = nc_out_id.createDimension("REGION", 3)
            nc_d_tim = nc_out_id.createDimension("TIME", None)
            nc_v_tim = nc_out_id.createVariable("TIME", 'f', ('TIME'))
            nc_v_tim.setncattr("time_origin", time_origin_string)
            nc_v_tim.setncattr("units", time_origin_units)
            # nc_v_reg = nc_out_id.createVariable("REGION", 'f', ('REGION'))

            # create output variables
            nc_v_f = nc_out_id.createVariable("fgco2_glob", 'f', ('TIME'))
            nc_v_f.setncattr("missing_value", np.array(1E20,'f'))
            nc_v_f.setncattr("units", "PgC/yr")

            nc_v_f = nc_out_id.createVariable("fgco2_reg", 'f', ('TIME', 'REGION'))
            nc_v_f.setncattr("missing_value", np.array(1E20,'f'))
            nc_v_f.setncattr("units", "PgC/yr")
            nc_v_f.setncattr("region attributes", "0 = South, 1 = Tropics, 2 = North")

            nc_v_f = nc_out_id.createVariable("intDIC_1994_glob", 'f', () )
            nc_v_f.setncattr("units", "PgC")
            nc_v_f = nc_out_id.createVariable("intDIC_1994_reg", 'f',('REGION'))
            nc_v_f.setncattr("units", "PgC")
            nc_v_f = nc_out_id.createVariable("intDIC_2007_glob", 'f', ())
            nc_v_f.setncattr("units", "PgC")
            nc_v_f = nc_out_id.createVariable("intDIC_2007_reg", 'f',('REGION'))
            nc_v_f.setncattr("units", "PgC")

            tMaster = 0
            for y in range(yrFrom,yrTo+1):
                print("processing ",y, runs[r])
                print(basedir+runs[r]+"/ORCA2_1m_"+str(y)+"0101*diad*.nc")
                file_dia = glob.glob(basedir+runs[r]+"/ORCA2_1m_"+str(y)+"0101*diad*.nc")
                print(file_dia)
                if len(file_dia) > 0: # if a file exists
                    nc_id_dia = Dataset(file_dia[0], 'r' )
                    cflx = nc_id_dia.variables["Cflx"][:].data[:,:,:]

                    # extract the totals for the global and regional limits, using NaN to mask out
                    # unit conversion to required output units
                    units = peta*carbon*secondsInYear

                    # global
                    lonLim = [-180,180]
                    latLim = [-90,90]
                    var_in = np.copy(cflx)
                    var = subDomainORCA(lonLim, latLim, var_lons, var_lats, var_in, tmask, tmaskDepth, missingVal)
                    varNanGlob = np.copy(var)
                    varNanGlob[ varNanGlob > missingVal/10. ] = np.nan
                    varNanGlob[ varNanGlob < -missingVal/10. ] = np.nan

                    # south
                    latLim = [-90,-30]
                    var_in = np.copy(cflx)
                    var = subDomainORCA(lonLim, latLim, var_lons, var_lats, var_in, tmask, tmaskDepth, missingVal)
                    varNanS = np.copy(var)
                    varNanS[ varNanS > missingVal/10. ] = np.nan
                    varNanS[ varNanS < -missingVal/10. ] = np.nan

                    # Tropic
                    latLim = [-30,30]
                    var_in = np.copy(cflx)
                    var = subDomainORCA(lonLim, latLim, var_lons, var_lats, var_in, tmask, tmaskDepth, missingVal)
                    varNanT = np.copy(var)
                    varNanT[ varNanT > missingVal/10. ] = np.nan
                    varNanT[ varNanT < -missingVal/10. ] = np.nan

                    # North
                    latLim = [30,90]
                    var_in = np.copy(cflx)
                    var = subDomainORCA(lonLim, latLim, var_lons, var_lats, var_in, tmask, tmaskDepth, missingVal)
                    varNanN = np.copy(var)
                    varNanN[ varNanN > missingVal/10. ] = np.nan
                    varNanN[ varNanN < -missingVal/10. ] = np.nan

                    tDim = var.shape[0]
                    annualTotal = 0
                    for t in range(0,tDim):
                        monthlyTotalGlob = np.nansum(varNanGlob[t,:,:] * mask_area[:,:] * units )#/ tDim this shouldn't be divided by months (TJ, you should calculate the pGc/year for each month (the mean of the 12 months will give you the pgC/year total)
                        monthlyTotalS = np.nansum(varNanS[t,:,:] * mask_area[:,:] * units )#/ tDim)/ tDim)
                        monthlyTotalT = np.nansum(varNanT[t,:,:] * mask_area[:,:] * units )#/ tDim)/ tDim)
                        monthlyTotalN = np.nansum(varNanN[t,:,:] * mask_area[:,:] * units )#/ tDim)
                        # time dimensiion value
                        dt = datetime.datetime(y,t+1,12,0,0,0)
                        secs = (dt - origin).total_seconds()

                        nc_out_id.variables["fgco2_glob"][tMaster] = monthlyTotalGlob
                        nc_out_id.variables["fgco2_reg"][tMaster,0] = monthlyTotalS
                        nc_out_id.variables["fgco2_reg"][tMaster,1] = monthlyTotalT
                        nc_out_id.variables["fgco2_reg"][tMaster,2] = monthlyTotalN

                        nc_out_id.variables["TIME"][tMaster] = secs
                        tMaster = tMaster + 1

                if y == 1994 or y == 2007 : # for some reason GCB wants these specific years... who knows...
                    print('its a benchmark year')
                    file_trc = glob.glob(basedir+runs[r]+"/ORCA2_1m_"+str(y)+"0101*ptrc*.nc")
                    print(file_trc)
                    if len(file_trc) > 0: # if a file exists
                        nc_id_trc = Dataset(file_trc[0], 'r' )
                        dic = nc_id_trc.variables["DIC"][:].data
                        # global
                        lonLim = [-180,180]
                        latLim = [-90,90]
                        var_in = np.copy(dic)
                        var = subDomainORCA(lonLim, latLim, var_lons, var_lats, var_in, tmask, tmaskDepth, missingVal)
                        varNanGlob = np.copy(var)
                        varNanGlob[ varNanGlob > missingVal/10. ] = np.nan
                        varNanGlob[ varNanGlob < -missingVal/10. ] = np.nan

                        # south
                        latLim = [-90,-30]
                        var_in = np.copy(dic)
                        var = subDomainORCA(lonLim, latLim, var_lons, var_lats, var_in, tmask, tmaskDepth, missingVal)
                        varNanS = np.copy(var)
                        varNanS[ varNanS > missingVal/10. ] = np.nan
                        varNanS[ varNanS < -missingVal/10. ] = np.nan

                        # Tropic
                        latLim = [-30,30]
                        var_in = np.copy(dic)
                        var = subDomainORCA(lonLim, latLim, var_lons, var_lats, var_in, tmask, tmaskDepth, missingVal)
                        varNanT = np.copy(var)
                        varNanT[ varNanT > missingVal/10. ] = np.nan
                        varNanT[ varNanT < -missingVal/10. ] = np.nan

                        # North
                        latLim = [30,90]
                        var_in = np.copy(dic)
                        var = subDomainORCA(lonLim, latLim, var_lons, var_lats, var_in, tmask, tmaskDepth, missingVal)
                        varNanN = np.copy(var)
                        varNanN[ varNanN > missingVal/10. ] = np.nan
                        varNanN[ varNanN < -missingVal/10. ] = np.nan

                        units = litre*peta*carbon

                        tDim = var.shape[0]
                        totalDICGlob = 0
                        totalDICS = 0
                        totalDICT = 0
                        totalDICN = 0
                        for t in range(0,tDim):
                            totalDICGlob = totalDICGlob + np.nansum(varNanGlob[t,:,:,:] * mask_vol[:,:,:] * units / tDim)
                            totalDICS = totalDICS + np.nansum(varNanS[t,:,:,:] * mask_vol[:,:,:] * units / tDim)
                            totalDICT = totalDICT + np.nansum(varNanT[t,:,:,:] * mask_vol[:,:,:] * units / tDim)
                            totalDICN = totalDICN + np.nansum(varNanN[t,:,:,:] * mask_vol[:,:,:] * units / tDim)
                            
                        if y == 1994 :
                            nc_out_id.variables["intDIC_1994_glob"][:] = totalDICGlob
                            nc_out_id.variables["intDIC_1994_reg"][0] = totalDICS
                            nc_out_id.variables["intDIC_1994_reg"][1] = totalDICT
                            nc_out_id.variables["intDIC_1994_reg"][2] = totalDICN

                        if y == 2007 :
                            nc_out_id.variables["intDIC_2007_glob"][:] = totalDICGlob
                            nc_out_id.variables["intDIC_2007_reg"][0] = totalDICS
                            nc_out_id.variables["intDIC_2007_reg"][1] = totalDICT
                            nc_out_id.variables["intDIC_2007_reg"][2] = totalDICN

                    else:
                        print("94 or 07 FILE MISSING")
            else:
                print("FILE MISSING")

            print(nc_out_id.variables['fgco2_reg'][:,0])
            nc_out_id.close()
     


var = []

# 2d
if OutputNumber == 0: var.append( {'name':'fgco2',        'source':['Cflx'],                'file':'diad', 'units': 'mol/m2/s',       'dims': 2   , 'factor':1}  )
if OutputNumber == 1: var.append( {'name':'spco2',        'source':['pCO2'],                'file':'diad', 'units': 'uatm',       'dims': 2  , 'factor':1 }  )
if OutputNumber == 2: var.append( {'name':'fice',       'source':['ice_pres'],            'file':'icemod',  'units': '',       'dims': 2  , 'factor':1 }  )
if OutputNumber == 3: var.append( {'name':'intpp',        'source':['PPINT'],                 'file':'diad', 'units': 'mol/m2/s',       'dims': 2   , 'factor':1}  )
if OutputNumber == 4: var.append( {'name':'epc100',       'source':['EXP'],                 'file':'diad',  'units': 'mol/m2/s',       'dims': 2   , 'depth': 9 , 'factor':1}  ) 
if OutputNumber == 5: var.append( {'name':'epc1000',       'source':['EXP'],                 'file':'ptrc',  'units': 'mol/L',       'dims': 2    , 'depth': 21 }  ) 
# if OutputNumber == 6: var.append( {'name':'epc100exp',       'source':['EXP'],                 'file':'diad', 'units': 'mol/m2/s',       'dims': 2   , 'depth': 9  }  ) 
if OutputNumber == 7: var.append( {'name':'epcalc100',    'source':['ExpCO3'],               'file':'diad',  'units': 'mol/m2/s',       'dims': 2  , 'depth': 9  , 'factor':1}  )
if OutputNumber == 8: var.append( {'name':'tos',          'source':['tos'],            'file':'grid_T',  'units': 'degC',       'dims': 2   , 'factor':1}  )
if OutputNumber == 9: var.append( {'name':'sos',          'source':['sos'],            'file':'grid_T',  'units': '1e-3',       'dims': 2   , 'factor':1}  )
if OutputNumber == 10: var.append( {'name':'dissicos',     'source':['DIC'],                 'file':'ptrc',  'units': 'mol/m3',       'dims': 2   , 'depth': 0  , 'factor':1000}  )
if OutputNumber == 11: var.append( {'name':'talkos',       'source':['Alkalini'],            'file':'ptrc',  'units': 'eq/m3',       'dims': 2   , 'depth': 0  , 'factor':1000}  )
if OutputNumber == 12: var.append( {'name':'sios',         'source':['Si'],                'file':'ptrc', 'units': 'mol/m3',       'dims': 2   , 'depth': 0 , 'factor':1000}  )
if OutputNumber == 13: var.append( {'name':'dfeos',        'source':['Fer'],                  'file':'ptrc', 'units': 'mol/m3',       'dims': 2  , 'depth': 0  , 'factor':1000}  )
if OutputNumber == 14: var.append( {'name':'o2os',         'source':['O2'],                  'file':'ptrc',  'units': 'mol/m3',       'dims': 2  , 'depth': 0   , 'factor':1000}  )
if OutputNumber == 15: var.append( {'name':'intphyc',      'source':['DIA', 'COC', 'BAC', 'PTE', 'PIC', 'MIX', 'PHA', 'FIX'],    'file':'ptrc', 'units': 'mol/m2',       'dims': 2 , 'depth': 'all'  , 'factor':1000}  )
if OutputNumber == 16: var.append( {'name':'intphynd',     'source':['COC', 'BAC', 'PTE', 'PIC', 'MIX', 'PHA', 'FIX'],           'file':'ptrc', 'units': 'mol/m2',       'dims': 2 , 'depth': 'all'  , 'factor':1000}  )
if OutputNumber == 17: var.append( {'name':'intdiac',      'source':['DIA'],                                                     'file':'ptrc', 'units': 'mol/m2',       'dims': 2 , 'depth': 'all'  , 'factor':1000}  )
if OutputNumber == 18: var.append( {'name':'intzooc',      'source':['PRO', 'MES', 'GEL', 'MAC'],                                'file':'ptrc',  'units': 'mol/m2',       'dims': 2 , 'depth': 'all'  , 'factor':1000}  )
if OutputNumber == 19: var.append( {'name':'chlos',        'source':['TChl'],        'file':'diad', 'units': 'kg Chl/m3',       'dims': 2 ,  'depth': 0   , 'factor':1}  )
if OutputNumber == 20: var.append( {'name':'mld',          'source':['mldr10_1'],            'file':'grid_T', 'units': 'm',       'dims': 2   , 'factor':1}  )
if OutputNumber == 21: var.append( {'name':'po4os',          'source':['PO4'],                 'file':'ptrc',  'units': 'mol/m3',       'dims': 2,  'depth': 0   , 'factor':1000}  )
if OutputNumber == 22: var.append( {'name':'no3os',          'source':['NO3'],                 'file':'ptrc',  'units': 'mol/m3',       'dims': 2,  'depth': 0   , 'factor':1000}  )

# included addition processing for fugacity
if OutputNumber == 23: var.append( {'name':'sfco2',          'source':['pCO2'],                 'file':'diad',  'units': 'uatm',       'dims': 2, 'factor':1 }  )

if OutputNumber == 24: var.append( {'name':'intdic',          'source':['DIC'],                 'file':'ptrc',  'units': 'mol/m2',       'dims': 2,  'depth': 'all'   , 'factor':1000}  )
if OutputNumber == 25: var.append( {'name':'fgo2',          'source':['Oflx'],                 'file':'diad',  'units': 'mol/m2/s',       'dims': 2   , 'factor':1}  )


# # 3d
if OutputNumber == 26: var.append( {'name':'dissic',       'source':['DIC'],                 'file':'ptrc',  'units': 'mol/m3',       'dims': 3   , 'factor':1000}  )
if OutputNumber == 27: var.append( {'name':'talk',         'source':['Alkalini'],            'file':'ptrc',  'units': 'eq/m3',       'dims': 3   , 'factor':1000}  )
if OutputNumber == 28: var.append( {'name':'thetao',       'source':['votemper'],            'file':'grid_T',  'units': 'degC',       'dims': 3   , 'factor':1}  )
if OutputNumber == 29: var.append( {'name':'so',           'source':['vosaline'],            'file':'grid_T',  'units': '1e-3',       'dims': 3   , 'factor':1}  )
if OutputNumber == 30: var.append( {'name':'epc',          'source':['EXP'],                 'file':'diad',  'units': 'mol/m2/s',       'dims': 3   , 'factor':1}  )
if OutputNumber == 31: var.append( {'name':'po4',          'source':['PO4'],                 'file':'ptrc',  'units': 'mol/m3',       'dims': 3   , 'factor':1000}  )
if OutputNumber == 32: var.append( {'name':'si',           'source':['Si'],                  'file':'ptrc',  'units': 'mol/m3',       'dims': 3   , 'factor':1000}  )
if OutputNumber == 33: var.append( {'name':'o2',           'source':['O2'],                  'file':'ptrc',  'units': 'mol/m3',       'dims': 3   , 'factor':1000}  )
if OutputNumber == 34: var.append( {'name':'no3',          'source':['NO3'],                 'file':'ptrc',  'units': 'mol/m3',       'dims': 3   , 'factor':1000}  )






if OutputNumber > -1:

    for v in var: 
        print("Processing: ",v['name'])
        parm = v['name']
        for r in range(0,len(runs)):

            # create outputfile
            outputFile = "PlankTOM_Sim_"+str(sim[r])+"_"+str(v['name'])+".nc"
            outputFile = f'{resultsdir}{parm}_PlankTOM_1_gr_{yrFrom}-{yrTo}_v{timestamp}.nc'
            print(f'making {outputFile}')
            if horse:
                nc_out_id = Dataset(outputFile, 'w', format='NETCDF4_CLASSIC')

                # Set the available dimensions
                nc_d_lon = nc_out_id.createDimension("LONGITUDE", 360)
                nc_d_lat = nc_out_id.createDimension("LATITUDE", 180)
                nc_d_dep = nc_out_id.createDimension("DEPTH", zDim)
                nc_d_tim = nc_out_id.createDimension("TIME", None)

                nc_v_lon = nc_out_id.createVariable("LONGITUDE", 'f', ('LONGITUDE'))
                nc_v_lat = nc_out_id.createVariable("LATITUDE", 'f', ('LATITUDE'))
                nc_v_tim = nc_out_id.createVariable("TIME", 'f', ('TIME'))

                if v['dims'] == 2:
                    nc_v_v = nc_out_id.createVariable(v['name'], 'f', ("TIME", "LATITUDE", "LONGITUDE"), fill_value=missingVal)
                else:
                    nc_v_dep = nc_out_id.createVariable("DEPTH", 'f', ('DEPTH'))
                    nc_v_v = nc_out_id.createVariable(v['name'], 'f', ("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), fill_value=missingVal)

                nc_v_v.setncattr("missing_value", np.array(1E20,'f'))
                nc_v_v.setncattr("units", v['units'])

                t = 0
                for y in range(yrFrom,yrTo+1):
                    print("processing ",y, runs[r])
                    
                    #  open file depending on name
                    file = glob.glob(basedir+runs[r]+"/ORCA2_1m_"+str(y)+"0101*"+v['file']+"*.nc")
                    print(file)

                    if len(file) > 0:
                        print("opening ",file[0])
                        nc_id = Dataset(file[0], 'r' )

                        if t == 0:
                            for ncattr in nc_id.variables["time_counter"].ncattrs():
                                if ncattr != 'calendar':
                                    nc_v_tim.setncattr(ncattr, nc_id.variables["time_counter"].getncattr(ncattr))                              
                            nc_v_tim.setncattr("time_origin", time_origin_string) 
                            nc_v_tim.setncattr("units", time_origin_units) 

                            s = v['source']
                            longName = v['name'] #+" source: "
                            # for j in range(0,len(s)):
                            #     longName = longName + s[j]+"+"
                            # longName = longName[0:-1]

                            for ncattr in nc_id.variables[s[0]].ncattrs():
                                if ncattr != '_FillValue':  
                                    nc_v_v.setncattr(ncattr, nc_id.variables[s[0]].getncattr(ncattr))
                            nc_v_v.setncattr('long_name', longName)
                            nc_v_v.setncattr("units", v['units'])
                            nc_out_id.variables["LONGITUDE"][:] = target_lon
                            nc_out_id.variables["LATITUDE"][:]  = target_lat

                            if v['dims'] == 3:
                                for ncattr in nc_id.variables["deptht"].ncattrs():
                                    nc_v_dep.setncattr(ncattr, nc_id.variables["deptht"].getncattr(ncattr))
                                nc_out_id.variables["DEPTH"][:] = nc_id.variables["deptht"][0:zDim]
                        
                        # process the data
                        s = v['source']
                        atDepth = 'depth' in v.keys()

                        #  get data and sum if need be
                        if v['dims'] == 2 and atDepth == False :
                            data = nc_id.variables[s[0]][:,:,:].data
                            for j in range(1,len(s)):
                                data = data + nc_id.variables[s[j]][:,:,:].data
                        else:
                            data = nc_id.variables[s[0]][:,:,:,:].data
                            for j in range(1,len(s)):
                                data = data + nc_id.variables[s[j]][:,:,:,:].data

                        # add forcing data and preprocess if needed
                        if OutputNumber == 23:
                            file_force = glob.glob(forcedir+"/ncep_kelvin_"+str(y)+".nc")
                            file_force = glob.glob(f'{forcedir}/bulk_{y}_9_era5_daily.nc') #era5
                            print("forcing data for conversion: ",y, file_force)
                            file_phys = glob.glob(basedir+runs[r]+"/ORCA2_1m_"+str(y)+"0101*grid_T*.nc")

                            nc_id_force = Dataset(file_force[0], 'r' )
                            nc_id_phys  = Dataset(file_phys[0], 'r' )

                            pres = nc_id_force.variables["pres"][:]
                            sst = nc_id_phys.variables["tos"][:]

                            spco2 = np.copy(data)
                            #  this is adapted from a R routine 
                            days = [0,31,59,90,120,151,181,212,243,273,304,334,365]
                            Ptot = np.zeros((12,pres.shape[1],pres.shape[2]))
                            for m in range(0,12):
                                Ptot[m,:,:] = np.mean(pres[days[m]:days[m+1],:,:], axis=0 )
                            Ptot = Ptot / 101324.9966 # conv to atmospheres for this calc 
                            # Ptot = pres[:,:,:]  # Patm + Phydro_atm # total pressure (atm + hydrostatic) at surface = atm only

                            TK = sst[:,:,:] + 273.15    # temperature kelvin
            
                            B =  -1636.75 + 12.0408*TK - 0.0327957*np.power(TK,2.0) + 0.0000316528*np.power(TK,3.0) #  <- -1636.75 + 12.0408*TK - 0.0327957*TK^2 + 0.0000316528*TK^3
                            Del = 57.7-0.118*TK   # <- 57.7-0.118*TK
                            xc2 = np.power( (1-spco2*1E-6), 2.0 )

                            fugcoeff =  np.exp( Ptot*(B + 2*xc2*Del)/(82.057*TK) )          # exp( Ptot*(B + 2*xc2*Del)/(82.057*TK) )
                            # sfco2 = spco2 * fugcoeff
                            data = spco2 * fugcoeff

                        var_lons = nc_id.variables['nav_lon'][:].data
                        var_lats = nc_id.variables['nav_lat'][:].data

                        # regrid it
                        if v['dims'] == 2 and atDepth == False :
                            v_regrid = regrid(data, var_lons, var_lats, target_lon, target_lat, tmask, missingVal)
                        else:
                            if atDepth == True:
                                print("Depth: ",v['depth'])
                                if v['depth'] == 'all':
                                    data = np.sum(data, axis=1)
                                    v_regrid = regrid(data, var_lons, var_lats, target_lon, target_lat, tmask, missingVal)
                                else:
                                    print("Depth: ",v['depth'])
                                    data = data[:,v['depth'],:,:]
                                    v_regrid = regrid(data, var_lons, var_lats, target_lon, target_lat, tmaskDepth[v['depth'],:,:], missingVal)
                            else:
                                v_regrid = np.zeros((12,zDim,180,360))
                                for z in range(0,zDim):
                                    print('a: ',z, zDim, data.shape)
                                    v_regrid[:,z,:,:] = regrid(data[:,z,:,:], var_lons, var_lats, target_lon, target_lat, tmaskDepth[z,:,:], missingVal)

                        if len(v_regrid.shape) == 3:

                            # fill in the odd values where the regrid hasn't filled in N/S with lat averages
                            lat_v_regrid = np.nanmean(v_regrid,axis=2)
                            # get first val (at south pole)
                            for i in range(0,12):
                                southLatVal = np.nan
                                j = 0
                                while np.isnan(southLatVal):
                                    southLatVal = lat_v_regrid[i,j]
                                    j = j + 1
                                ind = np.isnan( lat_v_regrid[i,:] )
                                lat_v_regrid[i, ind ] = southLatVal
                            lat_grid_v = np.zeros((12,180,360))
                            for j in range(0,360):
                                lat_grid_v[:,:,j] = lat_v_regrid
                            ind = np.isnan( v_regrid )
                            v_regrid[ ind ] = lat_grid_v[ ind ]
                            # mask the land with missingVal
                            ind = np.where( mask == 0 )
                            for i in range(0,12):
                                map = v_regrid[i,:,:]
                                map[ ind ] = missingVal
                                v_regrid[i,:,:] = map
                        else:
                            for z in range(0,zDim):
                                print('b: ',z, zDim)
                                # fill in the odd values where the regrid hasn't filled in N/S with lat averages
                                d_regrid = v_regrid[:,z,:,:]
                                lat_v_regrid = np.nanmean(d_regrid,axis=2)
                                # get first val (at south pole)
                                for i in range(0,12):
                                    southLatVal = np.nan
                                    j = 0
                                    while np.isnan(southLatVal):
                                        southLatVal = lat_v_regrid[i,j]
                                        j = j + 1
                                    ind = np.isnan( lat_v_regrid[i,:] )
                                    lat_v_regrid[i, ind ] = southLatVal
                                lat_grid_v = np.zeros((12,180,360))
                                for j in range(0,360):
                                    lat_grid_v[:,:,j] = lat_v_regrid
                                ind = np.isnan( d_regrid )
                                d_regrid[ ind ] = lat_grid_v[ ind ]
                                # mask the land with missingVal
                                ind = np.where( vol_mask[z,:,:] == 0 )
                                for i in range(0,12):
                                    map = d_regrid[i,:,:]
                                    map[ ind ] = missingVal
                                    d_regrid[i,:,:] = map
                                v_regrid[:,z,:,:] = d_regrid

                        # output data 
                        v_regrid = v_regrid * v['factor']

                        for i in range(0,12):
                            if v['dims'] == 2:

                                v_regrid[ v_regrid > missingVal/10. ] =  missingVal
                                nc_out_id.variables[v['name']][i+t,:,:] = v_regrid[i,:,:] 

                                # time dimensiion value
                                dt = datetime.datetime(y,i+1,12,0,0,0)
                                secs = (dt - origin).total_seconds()
                                days = int(secs / (24*60*60))
                                nc_out_id.variables["TIME"][i+t] = secs


                            if v['dims'] == 3:
                                v_regrid[ v_regrid > missingVal/10. ] =  missingVal
                                nc_out_id.variables[v['name']][i+t,:,:,:] = v_regrid[i,:,:,:] 

                                # time dimensiion value
                                dt = datetime.datetime(y,i+1,12,0,0,0)
                                secs = (dt - origin).total_seconds()
                                days = int(secs / (24*60*60))
                                nc_out_id.variables["TIME"][i+t] = secs


                            # # v_regrid = v_regrid * v['factor']
                            # v_regrid[ v_regrid > missingVal/10. ] =  missingVal
                            # annual_avg = np.nanmean( v_regrid, axis=0 ) 
                            # annual_avg[ annual_avg > missingVal/10. ] =  missingVal
                            # nc_out_id.variables[v['name']][ int(t/12) ,:,:,:] = annual_avg 
                            # daysSince1980 = (datetime.datetime(y,1,1) - datetime.datetime(1980,1,1)).days
                            # nc_out_id.variables["TIME"][int(t/12)] = daysSince1980 

                        t = t + 12

                        # nc_out_id.variables["TIME"][:] = nc_id.variables["time"][:]
                
