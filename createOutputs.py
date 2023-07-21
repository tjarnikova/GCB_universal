#!/usr/bin/env python
import sys
import os.path
import datetime 
import numpy as np
import glob
from netCDF4 import Dataset, MFDataset
import math
from scipy.interpolate import griddata
import datetime
from scipy.ndimage import median_filter


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

# GCB
# CO2flux Pg C/y positive into the ocean
# Cflx --> fgco2
# pCO2 --> fCO2 --> spco2


# years 1958-2019

# basedir = "/gpfs/data/greenocean/software/runs/GCB/2021/"
basedir="/gpfs/data/greenocean/software/runs/GCB/RECCAP/2021/" #RECCAP
forcedir = "/gpfs/data/greenocean/software/products/NCEPForcingData/"
# runs = ["TOM53_DW_NCR1", "TOM53_DW_NCR2", "TOM53_DW_NCR3", "TOM53_DW_NCR4"]
# runs = ["TOM53_DW_80LA", "TOM53_DW_80LB", "TOM53_DW_80LC", "TOM53_DW_80LD"] # 2020 submission
# runs = ["TOM12_DW_WD30", "TOM12_DW_WD31", "TOM12_DW_WD32"] # 2021 submission
# runs = ["TOM12_DW_WD30", "TOM12_DW_WD31", "TOM12_DW_WD32", "TOM12_DW_WD46"] # 2021 submission for RECCAP as THEY want 4, the princesses.
runs = ["TOM12_DW_WD48", "TOM12_DW_WD49", "TOM12_DW_WD50", "TOM12_DW_WD51"] # 2021 submission for RECCAP as THEY want 4, the princesses.

sim = ["2021_A","2021_B","2021_C","2021_D"]

GCB = False
RECCAP = True

RECCAPnumber = int(sys.argv[1])
print('RECCAP number is: ', RECCAPnumber)

missingVal = 1E20
Er = 6.3781E6 # meters
Ec = 2*math.pi*Er
Ea = 4*math.pi*Er*Er

area = np.zeros((180,360))
for y in range(0,180):
    ang = np.radians(y-90)
    area[y,:] = Ec*(1/360.)*math.cos(ang) * Ec*(1/360.)

print(np.sum(area),Ea)

# # open basin_mask.nc
# basinFile = glob.glob("basin_mask.nc")
# nc_basin_id = Dataset(basinFile[0], 'r' )
# area = nc_basin_id.variables["AREA"][:]
# vol = nc_basin_id.variables["VOLUME"][:]
raass = 3600.*24.*365.

# get target mesh values
target_lon = np.arange(0.5,360.5,1)# triggers setting in regrid as < 2 dims
# target_lat = np.arange(-89.5,90.5,1)
# target_lon = np.arange(-179.5,180.5,1)
target_lat = np.arange(-89.5,90.5,1)

# get source co-ords
meshFile = glob.glob("meshmask.nc")
nc_mesh_id = Dataset(meshFile[0], 'r' )
# var_lons = nc_mesh_id.variables["nav_lon"][:]
# var_lats = nc_mesh_id.variables["nav_lat"][:]
depths = nc_mesh_id.variables["gdept_1d"][0,:].data
zDim = len(depths)
print(zDim, depths)
tmask = nc_mesh_id.variables["tmask"][0,0,:,:]
tmaskDepth = nc_mesh_id.variables["tmask"][0,:,:,:]

# get mask
maskFile = glob.glob('WOAmask.nc')
nc_mask_id = Dataset(maskFile[0], 'r' )
mask = nc_mask_id.variables["mask"][:]
# mask = np.roll(mask, int(mask.shape[1]/2), axis = 1)


# get Ancillary Data
ancFile = glob.glob('/gpfs/home/yzh17dvu/scratch/ModelRuns/GCB_RECCAP/MakeAncillary/AncillaryData_v2.nc')
nc_anc_id = Dataset(ancFile[0], 'r' )
vol_mask = nc_anc_id.variables["MASK_VOL"][:]


if GCB == True:
    for r in range(0,len(runs)):

        # create outputfile
        outputFile = "GCB_Simulation_"+str(sim[r])+".nc"
        nc_out_id = Dataset(outputFile, 'w', format='NETCDF3_CLASSIC')

        # Set the available dimensions
        nc_d_lon = nc_out_id.createDimension("LONGITUDE", 360)
        nc_d_lat = nc_out_id.createDimension("LATITUDE", 180)
        nc_d_tim = nc_out_id.createDimension("TIME", None)
        nc_v_lon = nc_out_id.createVariable("LONGITUDE", 'f', ('LONGITUDE'))
        nc_v_lat = nc_out_id.createVariable("LATITUDE", 'f', ('LATITUDE'))
        nc_v_tim = nc_out_id.createVariable("TIME", 'f', ('TIME'))

        nc_v_f = nc_out_id.createVariable("fgco2", 'f', ('TIME','LATITUDE', 'LONGITUDE'))
        nc_v_f.setncattr("missing_value", np.array(1E20,'f'))
        nc_v_f.setncattr("units", "mol/m2/s")

        nc_v_f = nc_out_id.createVariable("spco2", 'f', ('TIME','LATITUDE', 'LONGITUDE'))
        nc_v_f.setncattr("missing_value", np.array(1E20,'f'))
        nc_v_f.setncattr("units", "ppm")

        nc_v_s = nc_out_id.createVariable("sfco2", 'f', ('TIME','LATITUDE', 'LONGITUDE'))
        nc_v_s.setncattr("missing_value", np.array(1E20,'f'))
        nc_v_s.setncattr("units", "uatm")


        tMaster = 0
        origin = datetime.datetime(1959,1,1,0,0,0)
        
        for y in range(1959,2021):
        # for y in range(1990,1991):

            print("processing ",y, runs[r])
            print(basedir+runs[r]+"/ORCA2_1m_"+str(y)+"0101*diad*.nc")
            file_dia = glob.glob(basedir+runs[r]+"/ORCA2_1m_"+str(y)+"0101*diad*.nc")

            file_force = glob.glob(forcedir+"/ncep_kelvin_"+str(y)+".nc")
            print("forcing data for conversion: ",y, file_force)

            file_phys = glob.glob(basedir+runs[r]+"/ORCA2_1m_"+str(y)+"0101*grid*.nc")

            print(file_dia)
            if len(file_dia) > 0:
                # open 
                nc_id_dia2d = Dataset(file_dia[0], 'r' )
                nc_id_force = Dataset(file_force[0], 'r' )
                nc_id_phys  = Dataset(file_phys[0], 'r' )

                # regrid and convert input data
                # regrid(var, var_lons, var_lats, target_lon, target_lat, missingVal, intType)

                fgco2 = nc_id_dia2d.variables["Cflx"][:].data[:,:,:]
                spco2 = nc_id_dia2d.variables["pCO2"][:].data[:,:,:]

                # forcing data
                pres = nc_id_force.variables["pres"][:]
                sst = nc_id_phys.variables["tos"][:]

                # --------------------------------------------------------------
                # convert data from pCO2 to fCO2
                # get data (from forcing data or grid_T)

                # get monthly averages of forcing data
                daysInMonth = [31,28,31,30,31,30,31,31,30,31,30,31]
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
                sfco2 = spco2 * fugcoeff

                print('p: ',Ptot[1,100,30])
                print('tk: ',TK[1,100,30])
                print('B: ',B[1,100,30])
                print('Del: ',Del[1,100,30])
                print('xc2: ',xc2[1,100,30])
                print('fugcoeff: ',fugcoeff[1,100,30])
                print('sp: ',spco2[1,100,30])
                print('sf: ',sfco2[1,100,30])               
                # --------------------------------------------------------------

                # replace land with lateral averages not missing or nan as breaks the linear interpolation in regrid
                # take a large median filter and replace land
    
                # testFile = "test.nc"
                # nc_test_id = Dataset(testFile, 'w', format='NETCDF3_CLASSIC')
                # nc_test_id.createDimension("x", 182)
                # nc_test_id.createDimension("y", 149)
                # nc_test_id.createVariable("test", 'f', ('y', 'x'))
                # nc_test_id.createVariable("test2", 'f', ('y', 'x'))

                ind = np.where( tmask == 0 )
                for t in range(0,12):
                    map = fgco2[t, :, : ]
                    map_nozero = np.copy(map)
                    map_nozero[ ind ] = np.nan
                    lat_map = np.nanmean(map_nozero,axis=1)
                    lat_grid = np.zeros((149,182))
                    for j in range(0,182):  
                        lat_grid[:,j] = lat_map
                    map[ ind ] = lat_grid[ ind ]
                    fgco2[t, :,:] = map

                    map = spco2[t, :, : ]
                    map_nozero = np.copy(map)
                    map_nozero[ ind ] = np.nan
                    lat_map = np.nanmean(map_nozero,axis=1)
                    lat_grid = np.zeros((149,182))
                    for j in range(0,182):  
                        lat_grid[:,j] = lat_map
                    map[ ind ] = lat_grid[ ind ]
                    spco2[t, :,:] = map

                    map = sfco2[t, :, : ]
                    map_nozero = np.copy(map)
                    map_nozero[ ind ] = np.nan
                    lat_map = np.nanmean(map_nozero,axis=1)
                    lat_grid = np.zeros((149,182))
                    for j in range(0,182):  
                        lat_grid[:,j] = lat_map
                    map[ ind ] = lat_grid[ ind ]
                    sfco2[t, :,:] = map

                # median filter to remove outliers on land
                for t in range(0,12):
                    map = fgco2[t, :, : ]
                    land_map = median_filter(map, size=5)
                    map[ ind ] =  land_map [ ind ]
                    fgco2[t,:,:] = map

                    map = spco2[t, :, : ]
                    land_map = median_filter(map, size=5)
                    map[ ind ] =  land_map [ ind ]
                    spco2[t,:,:] = map
                    
                    map = sfco2[t, :, : ]
                    land_map = median_filter(map, size=5)
                    map[ ind ] =  land_map [ ind ]
                    sfco2[t,:,:] = map


                # nc_test_id.variables["test"][:] = spco2[0,:,:]
                # nc_test_id.variables["test2"][:] =  tmask

                fgco2_regrid = regrid(fgco2, var_lons, var_lats, target_lon, target_lat, missingVal)
                spco2_regrid = regrid(spco2, var_lons, var_lats, target_lon, target_lat, missingVal)
                sfco2_regrid = regrid(sfco2, var_lons, var_lats, target_lon, target_lat, missingVal)

                # fill in the odd values where the regrid hasn't filled in N/S with lat averages
                lat_fgco2 = np.nanmean(fgco2_regrid,axis=2)
                lat_spco2 = np.nanmean(spco2_regrid,axis=2)
                lat_sfco2 = np.nanmean(sfco2_regrid,axis=2)
                # get first val (at south pole)
                for t in range(0,12):
                    southLatVal_fgco2 = np.nan
                    southLatVal_spco2 = np.nan
                    southLatVal_sfco2 = np.nan

                    j = 0
                    while np.isnan(southLatVal_fgco2):
                        southLatVal_fgco2 = lat_fgco2[t,j]
                        southLatVal_spco2 = lat_spco2[t,j]
                        southLatVal_sfco2 = lat_sfco2[t,j]
                        j = j + 1

                    ind = np.isnan( lat_fgco2[t,:] )
                    lat_fgco2[t, ind ] = southLatVal_fgco2
                    lat_spco2[t, ind ] = southLatVal_spco2
                    lat_sfco2[t, ind ] = southLatVal_sfco2

                lat_grid_fgco2 = np.zeros((12,180,360))
                lat_grid_spco2 = np.zeros((12,180,360))
                lat_grid_sfco2 = np.zeros((12,180,360))

                for j in range(0,360):
                    lat_grid_fgco2[:,:,j] = lat_fgco2
                    lat_grid_spco2[:,:,j] = lat_spco2
                    lat_grid_sfco2[:,:,j] = lat_sfco2

                ind = np.isnan( fgco2_regrid )
                fgco2_regrid[ ind ] = lat_grid_fgco2[ ind ]
                spco2_regrid[ ind ] = lat_grid_spco2[ ind ]
                sfco2_regrid[ ind ] = lat_grid_sfco2[ ind ]

                # mask the land with missingVal
                ind = np.where( mask == 0 )
                for t in range(0,12):
                    map = fgco2_regrid[t, :, : ]
                    map[ ind ] = missingVal
                    fgco2_regrid[t, :,:] = map
                    map = spco2_regrid[t, :, : ]
                    map[ ind ] = missingVal
                    spco2_regrid[t, :,:] = map
                    map = sfco2_regrid[t, :, : ]
                    map[ ind ] = missingVal
                    sfco2_regrid[t, :,:] = map

                # input('-------------')
                print('processed year: ',y, datetime.datetime.now())

                for t in range(0,12):

                    # time dimensiion value
                    dt = datetime.datetime(y,t+1,12,0,0,0)
                    secs = (dt - origin).total_seconds()

                    if tMaster == 0:
                        # copy parameters and data for the dimensions
                        # for ncattr in nc_id_dia2d.variables["x"].ncattrs():
                        #     nc_v_lon.setncattr(ncattr, nc_id_dia2d.variables["x"].getncattr(ncattr))
                        # for ncattr in nc_id_dia2d.variables["y"].ncattrs():
                        #     nc_v_lat.setncattr(ncattr, nc_id_dia2d.variables["y"].getncattr(ncattr))
                        # for ncattr in nc_id.variables["time"].ncattrs():
                        #     nc_v_tim.setncattr(ncattr, nc_id.variables["time"].getncattr(ncattr))
                        nc_out_id.variables["LONGITUDE"][:] = target_lon #nc_id_dia2d.variables["x"][:]
                        nc_out_id.variables["LATITUDE"][:] = target_lat #nc_id_dia2d.variables["y"][:]
                        # correct time variable starting point
                        nc_v_tim.setncattr("time_origin", "1950-JAN-01 00:00:00")
                        nc_v_tim.setncattr("units", "seconds since 1950-01-01 00:00:00")
                    
                        # nc_v_tim.setncattr("calendar", "leap")
                        nc_v_tim.setncattr("long_name", "Time axis")
                        nc_v_tim.setncattr("missing_value", missingVal)
                        nc_v_tim.setncattr("_Fillvalue", missingVal)


                    nc_out_id.variables["fgco2"][tMaster,:,:] = fgco2_regrid[t,:,:]   #fgco2[t,:,:] #* raass * 12 * 1E-15 / 12
                    nc_out_id.variables["spco2"][tMaster,:,:] = spco2_regrid[t,:,:]   #spco2[t,:,:]
                    nc_out_id.variables["sfco2"][tMaster,:,:] = sfco2_regrid[t,:,:]  

                    nc_out_id.variables["TIME"][tMaster] = secs
                    tMaster = tMaster + 1

            else:
                print(r,y,"missing")

        nc_out_id.close()


# COPY OF R code to do it
# p2fCO2 <- function(T=25, Patm=1, P=0, pCO2){
# tk <- 273.15;           # [K] (for conversion [deg C] <-> [K])
# TK <- T + tk;           # TK [K]; T[C]
# Phydro_atm = P / 1.01325  # convert hydrostatic pressure from bar to atm (1.01325 bar / atm)
# Ptot = Patm + Phydro_atm  # total pressure (in atm) = atmospheric pressure + hydrostatic pressure

# # Original "seacarb" f2pCO2 calculation:
# # B <- (-1636.75+12.0408*TK-0.0327957*(TK*TK)+0.0000316528*(TK*TK*TK))*1e-6
# # fCO2 <-  pCO2*(1/exp((1*100000)*(B+2*(57.7-0.118*TK)*1e-6)/(8.314*TK)))^(-1)
# # Above calculation:
# # - uses incorrect R (wrong units, incompatible with pressure in atm)
# # - neglects a term "x2" (see below)
# # - assumes pressure is always 1 atm (wrong for subsurface)

# # To compute fugcoeff, we need 3 other terms (B, Del, xc2) in addition to 3 others above (TK, Ptot, R)
#   B   <- -1636.75 + 12.0408*TK - 0.0327957*TK^2 + 0.0000316528*TK^3
#   Del <- 57.7-0.118*TK

# # "x2" term often neglected (assumed = 1) in applications of Weiss's (1974) equation 9
# # x2 = 1 - x1 = 1 - xCO2 (it is close to 1, but not quite)
# # Let's assume that xCO2 = pCO2. Resulting fugcoeff is identical to at least 8th digit after the decimal.
#   xCO2approx <- pCO2
#   xc2 <- (1 - xCO2approx*1e-6)^2 

#   fugcoeff = exp( Ptot*(B + 2*xc2*Del)/(82.057*TK) )
#   fCO2 <- pCO2 * fugcoeff






    print("-----DONE-------------")



    
        # for y in range(1959,2020):
        #     print("processing ",y, runs[r])
        #     print(basedir+runs[r]+"/ORCA2_1m_"+str(y)+"0101*dia2d*regrid*.nc")
        #     file_dia2d = glob.glob(basedir+runs[r]+"/ORCA2_1m_"+str(y)+"0101*dia2d*regrid*.nc")
        #     file_grid = glob.glob(basedir+runs[r]+"/ORCA2_1m_"+str(y)+"0101*grid*regrid*.nc")
        #     print(file_dia2d,file_grid)
        #     if len(file_dia2d) > 0:
        #         # open 
        #         nc_id_dia2d = Dataset(file_dia2d[0], 'r' )
        #         nc_id_grid = Dataset(file_grid[0], 'r' )

        #         if tMaster == 0:
        #             # copy parameters and data for the dimensions
        #             for ncattr in nc_id_dia2d.variables["x"].ncattrs():
        #                 nc_v_lon.setncattr(ncattr, nc_id_dia2d.variables["x"].getncattr(ncattr))
        #             for ncattr in nc_id_dia2d.variables["y"].ncattrs():
        #                 nc_v_lat.setncattr(ncattr, nc_id_dia2d.variables["y"].getncattr(ncattr))
        #             # for ncattr in nc_id.variables["time"].ncattrs():
        #             #     nc_v_tim.setncattr(ncattr, nc_id.variables["time"].getncattr(ncattr))
        #             nc_out_id.variables["LONGITUDE"][:] = nc_id_dia2d.variables["x"][:]
        #             nc_out_id.variables["LATITUDE"][:] = nc_id_dia2d.variables["y"][:]
        #             # correct time variable starting point
        #             nc_v_tim.setncattr("time_origin", "1959-JAN-01 00:00:00")
        #             nc_v_tim.setncattr("units", "seconds since 1959-01-01 00:00:00")
        #             # nc_v_tim.setncattr("calendar", "leap")
        #             nc_v_tim.setncattr("long_name", "Time axis")
        #             nc_v_tim.setncattr("missing_value", missingVal)
        #             nc_v_tim.setncattr("_Fillvalue", missingVal)

        #             # print("=====")
        #             # print(nc_id.variables["y"][:][0:60])
        #             # print("=====")
        #             # print(nc_id.variables["y"][:][60:120])
        #             # print("=====")
        #             # print(nc_id.variables["y"][:][120:])
        #             # print("=====")
        #             # input("  ")

        #         fgco2 = nc_id_dia2d.variables["Cflx"][:].data[:,0,:,:]
        #         spco2 = nc_id_dia2d.variables["pCO2"][:].data[:,0,:,:]
        #         sfluxo2 = nc_id_dia2d.variables["Oflx"][:].data[:,0,:,:] / 1000
        #         # sfluxn2 = nc_id_dia2d.variables["N2Oflux"][:].data[:,0,:,:] / 1000 

        #         shflux = nc_id_grid.variables["sohefldo"][:].data[:,0,:,:] 
        #         sst = nc_id_grid.variables["sosstsst"][:].data[:,0,:,:] 
        #         sss = nc_id_grid.variables["sosaline"][:].data[:,0,:,:] 

        #         tDim = fgco2.shape[0]

        #         # factor out 1000 to set min
        #         sfluxo2[ sfluxo2 >= missingVal/1000 ]  = missingVal
        #         # sfluxn2[ sfluxn2 >= missingVal/1000 ]  = missingVal
        #         print(sfluxo2.shape)

        #         print(fgco2.shape, spco2.shape,y,file_dia2d[0], tDim)
        #         print(np.min(sfluxo2),np.max(sfluxo2),np.median(sfluxo2))

        #         fgco2_c = np.copy(fgco2)
        #         fgco2_c[ fgco2_c > missingVal/10.] = np.nan
        #         sumGlo = 0
        #         sumS = 0
        #         sumEq = 0
        #         sumN = 0

        #         # surfaceArea = np.copy(fgco2[0,:,:])
        #         # surfaceArea = np.where( surfaceArea > missingVal/10., 0, area )
 
        #         # surfaceArea[ surfaceArea < missingVal/10. ] = area
        #         # surfaceArea[ surfaceArea > missingVal/10. ] = 0
        #         print( surfaceArea.shape, np.sum(surfaceArea) )
        #         nc_test_id.variables["test"][:] = surfaceArea
        #         input(" -- ")
        #         # temp = 0
        #         # dt = datetime.datetime(y,t,12,0,0,0)
        #         for t in range(0,tDim):

        #             # print(t,tDim)
        #             dt = datetime.datetime(y,t+1,12,0,0,0)
                    
        #             secs = (dt - origin).total_seconds()
        #             nc_out_id.variables["fgco2"][tMaster,:,:] = fgco2[t,:,:] #* raass * 12 * 1E-15 / 12
        #             nc_out_id.variables["spco2"][tMaster,:,:] = spco2[t,:,:]
        #             nc_out_id.variables["sfluxo2"][tMaster,:,:] = sfluxo2[t,:,:]
        #             # nc_out_id.variables["sfluxn2"][tMaster,:,:] = sfluxn2[t,:,:]
        #             nc_out_id.variables["shflux"][tMaster,:,:] = shflux[t,:,:]
        #             nc_out_id.variables["sst"][tMaster,:,:] = sst[t,:,:]
        #             nc_out_id.variables["sss"][tMaster,:,:] = sss[t,:,:]

        #             # nc_out_id.variables["TIME"][tMaster] = nc_id.variables["time"][t]
        #             nc_out_id.variables["TIME"][tMaster] = secs
        #             tMaster = tMaster + 1
        #             # print(dt,origin,secs)
        #             # input("")

        #             fgco2_c[t,:,:] = fgco2_c[t,:,:] * area

        #         #  mol/m^2/s
        #         sumGlo = np.nansum(fgco2_c) * raass * 12 * 1E-15 / 12
        #         sumS = np.nansum(fgco2_c[:,0:60,:]) * raass * 12 * 1E-15 / 12
        #         sumEq = np.nansum(fgco2_c[:,60:120,:]) * raass * 12 * 1E-15 / 12
        #         sumN = np.nansum(fgco2_c[:,120:,:]) * raass * 12 * 1E-15 / 12
        #         print(y,sumGlo,sumS,sumEq,sumN)
        #         textFile.write(str(round(y,3))+","+str(round(sumGlo,3))+","+str(round(sumS,3))+","+str(round(sumEq,3))+","+str(round(sumN,3))+"\n")

        #     else:
        #         print(r,y,"missing")

        # nc_out_id.close()













# GCB remaining
# 1. Please submit your river carbon runoff (if any) and your C-burial (i.e. net flux into
# the sediment (PgC yr)
# 2. Please specify how you convert the provided xCO2 (ppm) forcing to pCO2 (uatm)
# Do you use atmospheric pressure and water vapour for this conversion?
# 3. Please submit the ocean area covered globally, and also in the three latitudinal
# bands (north of 30N, 30S-30N, south of 30S).
# 3. Ocean surface area covered by your native model grid (unit: km2)
# Globe, South (<30S), Tropics (30S-30N), North (>30N)
# 4. River carbon inflow (PgC/yr) and net flux into the sediment (PgC/yr)

# RECCAP 2

# dictionary
var = []

# 2d
if RECCAPnumber == 0: var.append( {'name':'fgco2',        'source':['Cflx'],                'file':'diad', 'units': 'mol/m2/s',       'dims': 2   , 'factor':1}  )
if RECCAPnumber == 1: var.append( {'name':'spco2',        'source':['pCO2'],                'file':'diad', 'units': 'uatm',       'dims': 2  , 'factor':1 }  )
if RECCAPnumber == 2: var.append( {'name':'fice',       'source':['ice_pres'],            'file':'icemod',  'units': '',       'dims': 2  , 'factor':1 }  )
if RECCAPnumber == 3: var.append( {'name':'intpp',        'source':['PPINT'],                 'file':'diad', 'units': 'mol/m2/s',       'dims': 2   , 'factor':1}  )
if RECCAPnumber == 4: var.append( {'name':'epc100',       'source':['EXP'],                 'file':'diad',  'units': 'mol/m2/s',       'dims': 2   , 'depth': 9 , 'factor':1}  ) 
# if RECCAPnumber == 4: var.append( {'name':'epc100',       'source':['POC'],                 'file':'ptrc',  'units': 'mol/L',       'dims': 2   , 'depth': 9 }  ) 
# if RECCAPnumber == 5: var.append( {'name':'epc1000',       'source':['POC'],                 'file':'ptrc',  'units': 'mol/L',       'dims': 2    , 'depth': 21 }  ) 
# if RECCAPnumber == 6: var.append( {'name':'epc100exp',       'source':['EXP'],                 'file':'diad', 'units': 'mol/m2/s',       'dims': 2   , 'depth': 9  }  ) 
if RECCAPnumber == 7: var.append( {'name':'epcalc100',    'source':['ExpCO3'],               'file':'diad',  'units': 'mol/m2/s',       'dims': 2  , 'depth': 9  , 'factor':1}  )
if RECCAPnumber == 8: var.append( {'name':'tos',          'source':['tos'],            'file':'grid',  'units': 'degC',       'dims': 2   , 'factor':1}  )
if RECCAPnumber == 9: var.append( {'name':'sos',          'source':['sos'],            'file':'grid',  'units': '1e-3',       'dims': 2   , 'factor':1}  )
if RECCAPnumber == 10: var.append( {'name':'dissicos',     'source':['DIC'],                 'file':'ptrc',  'units': 'mol/m3',       'dims': 2   , 'depth': 0  , 'factor':1000}  )
if RECCAPnumber == 11: var.append( {'name':'talkos',       'source':['Alkalini'],            'file':'ptrc',  'units': 'eq/m3',       'dims': 2   , 'depth': 0  , 'factor':1000}  )
# if RECCAPnumber == 12: # var.append( {'name':'sios',         'source':['SiO3'],                'file':'ptrc', 'units': 'mol/L',       'dims': 2   , 'depth': 0     }  )
if RECCAPnumber == 13: var.append( {'name':'dfeos',        'source':['Fer'],                  'file':'ptrc', 'units': 'mol/m3',       'dims': 2  , 'depth': 0  , 'factor':1000}  )
if RECCAPnumber == 14: var.append( {'name':'o2os',         'source':['O2'],                  'file':'ptrc',  'units': 'mol/m3',       'dims': 2  , 'depth': 0   , 'factor':1000}  )
if RECCAPnumber == 15: var.append( {'name':'intphyc',      'source':['DIA', 'COC', 'BAC', 'PTE', 'PIC', 'MIX', 'PHA', 'FIX'],    'file':'ptrc', 'units': 'mol/m2',       'dims': 2 , 'depth': 'all'  , 'factor':1000}  )
if RECCAPnumber == 16: var.append( {'name':'intphynd',     'source':['COC', 'BAC', 'PTE', 'PIC', 'MIX', 'PHA', 'FIX'],           'file':'ptrc', 'units': 'mol/m2',       'dims': 2 , 'depth': 'all'  , 'factor':1000}  )
if RECCAPnumber == 17: var.append( {'name':'intdiac',      'source':['DIA'],                                                     'file':'ptrc', 'units': 'mol/m2',       'dims': 2 , 'depth': 'all'  , 'factor':1000}  )
if RECCAPnumber == 18: var.append( {'name':'intzooc',      'source':['PRO', 'MES', 'GEL', 'MAC'],                                'file':'ptrc',  'units': 'mol/m2',       'dims': 2 , 'depth': 'all'  , 'factor':1000}  )
if RECCAPnumber == 19: var.append( {'name':'chlos',        'source':['TChl'],        'file':'diad', 'units': 'kg Chl/m3',       'dims': 2 ,  'depth': 0   , 'factor':1}  )
if RECCAPnumber == 20: var.append( {'name':'mld',          'source':['mldr10_1'],            'file':'grid', 'units': 'm',       'dims': 2   , 'factor':1}  )
if RECCAPnumber == 21: var.append( {'name':'po4os',          'source':['PO4'],                 'file':'ptrc',  'units': 'mol/m3',       'dims': 2,  'depth': 0   , 'factor':1000}  )
if RECCAPnumber == 22: var.append( {'name':'no3os',          'source':['NO3'],                 'file':'ptrc',  'units': 'mol/m3',       'dims': 2,  'depth': 0   , 'factor':1000}  )


# # 3d
if RECCAPnumber == 23: var.append( {'name':'dissic',       'source':['DIC'],                 'file':'ptrc',  'units': 'mol/m3',       'dims': 3   , 'factor':1000}  )
if RECCAPnumber == 24: var.append( {'name':'talk',         'source':['Alkalini'],            'file':'ptrc',  'units': 'eq/m3',       'dims': 3   , 'factor':1000}  )
if RECCAPnumber == 25: var.append( {'name':'thetao',       'source':['votemper'],            'file':'grid',  'units': 'degC',       'dims': 3   , 'factor':1}  )
if RECCAPnumber == 26: var.append( {'name':'so',           'source':['vosaline'],            'file':'grid',  'units': '1e-3',       'dims': 3   , 'factor':1}  )
if RECCAPnumber == 27: var.append( {'name':'epc',          'source':['POC'],                 'file':'ptrc',  'units': 'mol/m3',       'dims': 3   , 'factor':1000}  )
if RECCAPnumber == 28: var.append( {'name':'po4',          'source':['PO4'],                 'file':'ptrc',  'units': 'mol/m3',       'dims': 3   , 'factor':1000}  )
if RECCAPnumber == 29: var.append( {'name':'si',           'source':['Si'],                  'file':'ptrc',  'units': 'mol/m3',       'dims': 3   , 'factor':1000}  )
if RECCAPnumber == 30: var.append( {'name':'o2',           'source':['O2'],                  'file':'ptrc',  'units': 'mol/m3',       'dims': 3   , 'factor':1000}  )

if RECCAP == True:

    for v in var:

        parm = v['name']
        print(parm)

        for r in range(0,len(runs)):
            # need to re get data from regridded files
            # create outputfile

            outputFile = "RECCAP_Sim_"+str(sim[r])+"_"+str(v['name'])+".nc"

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
            # origin = datetime.datetime(1959,1,1,0,0,0)

            for y in range(1980,2019): #2019): # RECCAP
            # for y in range(1948,2021): # BODC

                print("processing ",y, runs[r])
                
                #  open file depending on name
                file = glob.glob(basedir+runs[r]+"/ORCA2_1m_"+str(y)+"0101*"+v['file']+"*.nc")
                print(file)
                if len(file) > 0:
                    # open 
                    print("opening ",file[0])
                    nc_id = Dataset(file[0], 'r' )

                    if t == 0:
                        # copy parameters and data for the dimensions
                        # for ncattr in nc_id.variables["x"].ncattrs():
                        #     nc_v_lon.setncattr(ncattr, nc_id.variables["x"].getncattr(ncattr))

                        # for ncattr in nc_id.variables["y"].ncattrs():
                        #     nc_v_lat.setncattr(ncattr, nc_id.variables["y"].getncattr(ncattr))

                        for ncattr in nc_id.variables["time_counter"].ncattrs():
                            if ncattr != 'calendar':
                                nc_v_tim.setncattr(ncattr, nc_id.variables["time_counter"].getncattr(ncattr))
                        nc_v_tim.setncattr("time_origin", "1980-JAN-01 00:00:00") #RECCAP
                        nc_v_tim.setncattr("units", "days since 1980-01-01 00:00:00") #RECCAP

                        # nc_v_tim.setncattr("time_origin", "1948-JAN-01 00:00:00") # BODC
                        # nc_v_tim.setncattr("units", "days since 1948-01-01 00:00:00") # BODC

                        s = v['source']

                        longName = ""
                        for j in range(0,len(s)):
                            longName = longName + s[j]+"+"
                        longName = longName[0:-1]
                        print(s,longName)

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

                    print(v.keys())
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

                    var_lons = nc_id.variables['nav_lon'][:].data
                    var_lats = nc_id.variables['nav_lat'][:].data
                    print(np.min(var_lons), np.max(var_lons))

                    # PREPROCESS to get land to mimic coastal values
                    # list_lons = var_lons.ravel()
                    # list_lats = var_lats.ravel()
                    # points = np.column_stack([list_lats,list_lons])
                    # print(points.shape)
                    # for t in range(0,1):
                    #     print(t)
                    #     if len(data.shape) == 4:
                    #         for z in range(0,zDim):
                    #             print("holder")
                    #             # data_filles = np.zeros((var.shape[0],target_lat.shape[0],target_lon.shape[0]),dtype=float) + np.nan

                    #     else:
                    #         var = data[t,:,:]
                    #         vals = []
                    #         maskVals = []
                    #         for iy in range(0,var_lons.shape[0]):
                    #             for ix in range(0,var_lons.shape[1]):
                    #                 maskVals.append(tmask[iy,ix])
                    #                 vals.append(var[iy,ix])
                    #         vals = np.array(vals)

                    #         print(vals.shape)

                    #         valsFilt = []
                    #         pointsFilt = []
                    #         for p in range(0,points.shape[0]):
                    #             if maskVals[p] == 1:
                    #                 valsFilt.append(vals[p])
                    #                 pointsFilt.append( ( points[p,0], points[p,1] ) )
                    #         points = np.array(pointsFilt)
                    #         vals = np.array(valsFilt)

                    #         print(points.shape, vals.shape)

                    #         grid_lon,grid_lat = np.meshgrid(target_lon,target_lat)
                    #         ind = np.where( tmask == 0 )
                   

                    #         data_near = np.zeros((target_lat.shape[0],target_lon.shape[0]),dtype=float) + np.nan
                    #         print(data_near.shape, grid_lat.shape)

                    #         data_near = griddata(points,vals, ( grid_lat, grid_lon ) , method='nearest')



                    #         print(data_near.shape)
                    #         testFile = "test.nc"
                    #         nc_test_id = Dataset(testFile, 'w', format='NETCDF4_CLASSIC')
                    #         nc_test_id.createDimension("x", 182)
                    #         nc_test_id.createDimension("y", 149)
                    #         nc_test_id.createDimension("x", 182)
                    #         nc_test_id.createDimension("y", 149)
                    #         nc_test_id.createVariable("test", 'f', ('y', 'x'))
                    #         nc_test_id.createVariable("test2", 'f', ('y', 'x'))
                    #         nc_test_id['test'][:] = var
                    #         nc_test_id['test2'][:] = data_near
                    #         nc_test_id.close()

                    # input('-+-')

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


                    print(v_regrid.shape)

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

                    print(v_regrid.shape)


                    # output data 
                    v_regrid = v_regrid * v['factor']

                    for i in range(0,12):
                        if v['dims'] == 2:

                            v_regrid[ v_regrid > missingVal/10. ] =  missingVal

                            nc_out_id.variables[v['name']][i+t,:,:] = v_regrid[i,:,:] 

                        # else:
                        #     v_regrid[ v_regrid > missingVal/10. ] =  missingVal
                        #     nc_out_id.variables[v['name']][i+t,:,:,:] = v_regrid[i,:,:,:]

                            # nc_out_id.variables["TIME"][i+t] =  nc_id.variables["time_counter"][i]
                            daysSince1980 = (datetime.datetime(y,i+1,15) - datetime.datetime(1980,1,1)).days
                            daysSince1948 = (datetime.datetime(y,i+1,15) - datetime.datetime(1948,1,1)).days

                            nc_out_id.variables["TIME"][i+t] = daysSince1980 # RECCAP
                            # nc_out_id.variables["TIME"][i+t] = daysSince1948 # BODC

                    if v['dims'] == 3:
                        # v_regrid = v_regrid * v['factor']
                        v_regrid[ v_regrid > missingVal/10. ] =  missingVal
                        annual_avg = np.nanmean( v_regrid, axis=0 ) 
                        annual_avg[ annual_avg > missingVal/10. ] =  missingVal
                        nc_out_id.variables[v['name']][ int(t/12) ,:,:,:] = annual_avg 
                        daysSince1980 = (datetime.datetime(y,1,1) - datetime.datetime(1980,1,1)).days
                        nc_out_id.variables["TIME"][int(t/12)] = daysSince1980 
                    t = t + 12

                        # nc_out_id.variables["TIME"][:] = nc_id.variables["time"][:]
                