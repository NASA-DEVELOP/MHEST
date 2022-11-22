#!/usr/bin/env python
# coding: utf-8

# # Automated Satellite Mixing height Observations and Known Remote Estimations (A-SMOKRE)
# 
# __Fall 2020 Authors/Collaborators:__ Ashwini Badgujar, Sean Cusick, Patrick Giltz, Ella Griffith, Brandy Nisbet-Wilcox, Dr. Kenton Ross, Dr. Travis Toth, Keith Weber 
# 
# __Spring 2021 Authors/Collaborators:__ Jukes Liu, Lauren Mock, Dean Berkowitz, Chris Wright, Brandy Nisbet-Wilcox, Dr. Kenton Ross, Dr. Travis Toth, Keith Weber 
# 
# __Description:__ Mixing height is critical to decision making regarding air quality forecasting as it indicates the altitude at which smoke disperses. This code allows the user to input hdf files containing vertical feature masked data from CALIPSO and receive a numeric output of the observed mixing heights. This code extracts features of relevance from the hdf file to find continuous aerosols relative to the earth’s surface. The altitude at which the aerosol ends is recorded as the mixing height, along with a matching latitude and longitude. The numeric output will include a mixing height observation at a particular location. These values can be applied as per the end user’s individual needs. 
# 
# __Functions:__ 
#  - Computes mixing height values for a given CALIPSO hdf file.
#  - Plots graphs to visualize the location of aerosol, surface and attenuated data
#  - Saves the calculated mixing height values to a CSV file which gets stored at the specified location
#  
# __Parameters:__
# 
# In:
# 
#  - desired dates of analysis (matching_CALIPSO_transect_names.csv)
#  - latitudes of desired transects with dates (calipso_transects.csv)
#  - CALIPSO files for the corresponding dates. 
#  
# Out:
# 
#  - CSV files with the mixing height values with respective latitude and longitude (CSV file stored in specified directory, see MH_calc)
#  - Plot showing the location of the features considered (aerosol, surface/subsurface, attenuated data, clear air)

# In[128]:


# Importing all the required packages
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import splev,splrep
import sys
import pandas as pd
from pyhdf.SD import SD, SDC

# Initializing variables
os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"


# ### Reclassify, Mesh, Calculate Mixing Heights

# In[129]:


def MH_calc(feature_type2, aerosol_class2,filedate,roi_profiles, lat2, lon2, automated):

        # 1st dimension (rows) in the feature_type2 array : columns in figure  (-1 = last column)
        # 2nd dimension  (columns) in the feature_type2 array : rows in figure (-1 = last row)
        


        # Assigning numeric values to the features 
        feature_type2[feature_type2 < 3], aerosol_class2[aerosol_class2 < 2] = 0,0 # other feature_type, not determined (AND MARINE)
        feature_type2[feature_type2 == 3], aerosol_class2[aerosol_class2 == 6] = 1,6 # tropospheric aerosol, smoke
        feature_type2[feature_type2 == 4], aerosol_class2[aerosol_class2 == 5] = 0,5 # other feature_type, polluted dust
        feature_type2[feature_type2 == 5], aerosol_class2[aerosol_class2 ==2] = 2,2 # surface, dust
        feature_type2[feature_type2 == 6], aerosol_class2[(aerosol_class2 > 2) & (aerosol_class2 < 5)]  = 2, 3 # subsurface, continental
        feature_type2[feature_type2 == 7], aerosol_class2[(aerosol_class2 == 7)] = 3,7 # bad feature_type, other
        
        print("number of chunks identified as smoke aerosol")
        print(len(aerosol_class2[(feature_type2 == 1)& (aerosol_class2 == 6)]))




        # Generate altitude feature_type according to file specification [1].
        alt = np.zeros(290) # change to altitude_levels variable
        type(alt)


        # Generating altitude data
        
        for i in range (0, 290):
            alt[i] = -0.5 + i*0.03



        # Contouring the feature_type on a grid of latitude vs. pressure
        latitude, altitude = np.meshgrid(lat2, alt)
        #print(altitude.shape)
        #print(latitude.shape)
        
        # if not automated return values for visualization
        #print (lat2)
        #print(alt)
        if not automated: return feature_type2, aerosol_class2, lat2, alt, lon2


        # Reversing the altitude data for calculation 
        altitude_reversed = alt[::-1]
        altitude_reversed
        
        
        # --------------------------------------------------------------------------------------------------
        # Calculating the mixing heights and copying them to csv
        # Main idea here: if transition from ground to air has aerosol immediately, we keep. 
        # --------------------------------------------------------------------------------------------------

        latitude_output = []
        longitude_output = []
        altitude_output = []
        aerosol_class_output = []
        
        for profile in range(roi_profiles):
            if feature_type2[profile][-1] != 2: # checking if last value is surface/subsurface
                continue
            # loop through the vertical profile
            first_aerosol = True
            second_aerosol = True
            for alt_index in range((len(altitude_reversed)-1), 0, -1):
                if feature_type2[profile][alt_index] == 2: # check if it is surface/subsurface
                    last_value = 2
                    continue
                if feature_type2[profile][alt_index] == 3:
                    last_value = 3
                    continue
                if feature_type2[profile][alt_index] == 1: # checking if it is aerosol
                    if first_aerosol:
                        aerosol_bottom = altitude_reversed[alt_index] # record the aerosol bottom
                    first_aerosol = False #we are no longer looking for the first aerosol
                    temp_alt = altitude_reversed[alt_index]
                    temp_class = aerosol_class2[profile][alt_index]
                    last_value = 1
                    continue
                # when we arrive at non-aerosol, non-surface, non-attenuated feature_type...
                elif (feature_type2[profile][alt_index] == 0) & (not first_aerosol):
                    # if we just emerged from an aerosol cloud, store
                    # if we haven't hit aerosol yet, keep going!
                    
                    # If this is the second aerosol cloud, delete the last aerosol and break!
                    if not second_aerosol:
                        del latitude_output[-1]
                        del longitude_output[-1]
                        del altitude_output[-1]
                        del aerosol_class_output[-1]
                        break
                    if last_value == 1:
                        aerosol_top = altitude_reversed[alt_index] # record the aerosol top
                        if ((aerosol_bottom > 0.3) & (aerosol_top > 6)) | (aerosol_bottom > 4.7) | (aerosol_top-aerosol_bottom <0.15): # if lofted or < 150m thickness
                            break
                        latitude_output.append(lat2[profile])
                        longitude_output.append(lon2[profile])
                        altitude_output.append(temp_alt)
                        aerosol_class_output.append(temp_class)
                        value = 'latitude :' + str(lat2[profile]) + ' longitude :' + str(lon2[profile]) + ' altitude :' + str(temp_alt) + ' aerosol_class :' + str(temp_class) # add to csv
                        print(value)
                        first_plume_top = aerosol_top
                    second_aerosol = False
                    first_aerosol = True
                    
                    # if we've reached a cloud, whether there was aerosol or not, break. 
                    # this means that contiguity with the ground is ensured.


        # Converting the feature_type to Dataframe for copying to csv    
        df = pd.DataFrame(
            {'Latitude': latitude_output,
             'Longitude': longitude_output,
             'Altitude': altitude_output,
             'Aerosol_Type': aerosol_class_output,
             'Date': [filedate]*len(aerosol_class_output)
            })
        df.head()
        
        print()
        if df.shape[0] == 0:
            print("No MH calculated for this transect. Causes may include: data attenuation, lack of aerosol data, other")
            
            return

        # ---------------------------
        # CHANGE OUTPUT FILENAME HERE
        # ---------------------------
        
        # Copying the Dataframe to csv
        # check if a csv already exists, extend that file if so.
        # MAKE SURE THE DIRECTORY IS EMPTY PRIOR TO RUNNING!
        if os.path.exists('Z:/SIHAQII/Data/Mixing_Heights/MH_'+filedate+'.csv'):
            df.to_csv('Z:/SIHAQII/Data/ASMOKRE_MH_Old/MH_'+filedate+'.csv', mode = 'a', index = False, header = False)
        else:
            df.to_csv('Z:/SIHAQII/Data/ASMOKRE_MH_Old/MH_'+filedate+'.csv', index = False, header=True)
        


# ### Process CALIPSO file

# In[130]:


# ---------------------------
# CHANGE: Remove all files in directory prior to executing. 
# ---------------------------
def Process_CALIPSO(FILE_PATH, lat_min, lat_max,filedate, automated):
    vfm_hdf = SD(FILE_PATH, SDC.READ)

    # Getting the feature type from hdf file
    DATAFIELD_NAME = 'Feature_Classification_Flags' # Datafield from hdf file which has features
    feature_type2D = vfm_hdf.select(DATAFIELD_NAME)
    feature_type = np.array(feature_type2D[:,:])
    #print(feature_type)
    #print(feature_type.shape)

    # Reading geolocation datasets.
    latitude = vfm_hdf.select('Latitude')
    lat = np.array(latitude[:])

    roiNDX_initial = np.where(lat >= lat_min)[0][0]       
    # We want the last of the values which is less than the max lat. 
    roiNDX_final = np.where(lat <= lat_max)[0][-1]
    #print(roiNDX_initial)
    #print(roiNDX_final)
    if roiNDX_initial >= roiNDX_final: 
        print("Error: Lat range is too small. This transect will be thrown out")
        return

    # Reading geolocation feature_typesets.
    longitude = vfm_hdf.select('Longitude')
    lon = np.array(longitude[:])
    #print(lon)

    # # Assigning the granule blocks and profile values 
    granule_blocks = lat.shape[0]
    granule_profiles = profile5km * granule_blocks
    profNDX = np.array(range(granule_profiles))
    prof2blockNDX = np.array(range(7,granule_profiles,profile5km))
    bigNDX_initial = 15 * roiNDX_initial
    bigNDX_final = 15 * roiNDX_final

    # # Assigning spline latitude value
    spline_latitude = splrep(prof2blockNDX, lat)
    lat2 = splev(profNDX, spline_latitude)
    #print(lat2)

    # #Assigning spline longitude value
    spline_longitude = splrep(prof2blockNDX, lon)
    lon2 = splev(profNDX, spline_longitude)
    #print(lon2)

    # # Extracting Feature Type only (1-3 bits) through bitmask.
    aerosol_class = (feature_type & 0b111000000000) >> 9 # downshift by 9 places. 
    feature_type = feature_type & 0b111 # store 7 as binary in variable

    # # Considering latitude and longitude of interest
    lat = lat[roiNDX_initial:roiNDX_final]
    lat2 = lat2[bigNDX_initial:bigNDX_final]
    lon2 = lon2[bigNDX_initial:bigNDX_final]
    roi_blocks = lat.shape[0]
    roi_profiles = lat2.shape[0]

    # # Extracting the feature_type from the area of interest
    feature_type2d = feature_type[roiNDX_initial:roiNDX_final, 1165:] 
    aerosol_class2d = aerosol_class[roiNDX_initial:roiNDX_final, 1165:]
    
    # converting it from 3d to 2d
    # reshape b/c feature_type needs to be oriented correctly
    
    # initialize array with row for each profile and column for each altitude
    ftr = np.empty((roi_profiles, altitude_levels), int)
    atr = np.empty((roi_profiles, altitude_levels), int)
    granule_blocks = roi_blocks
    
    # step through each row (CALIPSO 5km block) of VFM data
    for block in range(granule_blocks):
        # step across each row one profile at a time
        # each row has 15 profiles of 290 altitude elements each
        for profile in range(profile5km):
            # create a running index of which profile you're on
            # for the entire region of interest
            bpNDX = block*profile5km + profile
            # calculate initial and final elements to slice out elements
            # corresponding to an individual profile
            pa_i = profile*altitude_levels
            pa_f = profile*altitude_levels + altitude_levels
            # transfer data from array organized by block
            # to array organized by profile
            ftr[bpNDX,:] = feature_type2d[block,pa_i:pa_f]
            atr[bpNDX,:] = aerosol_class2d[block,pa_i:pa_f]
         
    feature_type2 = ftr
    aerosol_class2 = atr

# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
    
    # -----------------------------------------------------------------------------------------------------------
    # Move on to reclassify data, mesh grid, loop through profiles and extract MH 
    # -----------------------------------------------------------------------------------------------------------
    if not automated: 
        feature_type2, aerosol_class2, lat, alt, lon = MH_calc(feature_type2, aerosol_class2,filedate,roi_profiles, lat2, lon2, automated) # see function above
        return feature_type2, aerosol_class2, lat, alt, lon
    
    MH_calc(feature_type2, aerosol_class2,filedate,roi_profiles, lat2, lon2, automated) # see function above


# ### Automated calculation of all MH plumes:

# ##### Get list of matching filenames for identified CALIPSO passes

# In[132]:


# Assigning some universal values to profiles and altitudes
profile5km = 15
altitude_levels = 290

csvpath = 'Z:\SIHAQII\Data\input_CSVs/' # path to the folder with the csv files
overlap_transects_df = pd.read_csv(csvpath+'calipso_transects.csv') # read in the manually identified overlap info 
transect_dates = overlap_transects_df['Date'] # grab the dates column

# Read in the matches
matches_df = pd.read_csv(csvpath+'matching_CALIPSO_transect_names.csv', usecols=[1], names = ['filenames'], header = 0)
matches = list(matches_df['filenames']) # grab the filenames as a list
len(matches) # number of matches


# ##### Read in files, call functions above

# In[126]:


FILE_DIR = 'Z:/SIHAQII/Data/CALIPSO_HDF_FILES/CALIPSO_HDFs/xfr139.larc.nasa.gov/'
counter = 0
for file in matches:
        #print(file) # print the filename
        filedate = file[30:40] # slice the string to get the date
        print(filedate)
        lat_mins = overlap_transects_df.loc[overlap_transects_df['Date'] == filedate, ['Latitude_min']] # grab the latitude mins
        lat_maxs = overlap_transects_df.loc[overlap_transects_df['Date'] == filedate, ['Latitude_max']] # grab the latitude maxs
        if len(lat_mins) > 0: # if there are matches found
            for plumen in range(0, len(lat_mins)): # loop through the latmins and latmaxs identified
                lat_minmax = [lat_mins.values[plumen][0],lat_maxs.values[plumen][0]]
                lat_min = min(lat_minmax)
                lat_max = max(lat_minmax)
                print(lat_min, lat_max)
                
                # Create filepath, call functions above
                FILE_PATH = FILE_DIR + file
                automated = True
                Process_CALIPSO(FILE_PATH, lat_min, lat_max,filedate, automated)
                counter = counter+1
print("Final number of plumes analyzed:")
print(counter)


# ### For examining just one plume:

# In[139]:


# ---------------------------------------------
# Specify file location AND min / max latitudes
# ---------------------------------------------

FILE_DIR = 'Z:/SIHAQII/Data/Testfiles_softwarerelease/CALIPSO_test_files'
FILE_WILD = 'CAL_LID_L2_VFM-Standard-V4-20.2006-08-16T20-29-26ZD.hdf'
FILE_PATH= FILE_DIR + '/' + FILE_WILD
filedate = FILE_WILD[30:40]

# # Assigning initial and final values to the latitudes
lat_min = 48.3066
lat_max = 48.9223

# Assigning some universal values to profiles and altitudes
profile5km = 15
altitude_levels = 290

# letting it know to return just the one plume
automated = False
feature_type2, aerosol_class2, lat2, alt, longitude = Process_CALIPSO(FILE_PATH, lat_min, lat_max, filedate, automated)
latitude, altitude = np.meshgrid(lat2, alt)

# only use below for diagnostics on a single plumes classifications
#automated = True
#Process_CALIPSO(FILE_PATH, lat_min, lat_max, filedate, automated)


# In[140]:


import matplotlib
cgfont = {'fontname': 'Century Gothic'}
fig, ax = plt.subplots(figsize = (8,6))
cmap = plt.get_cmap('Dark2')
bounds = [0.,2,3.,5,6,7,8]
ticks = [1, 2.5,4,5.5,6.5,7.5]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
a = ax.pcolor(lat2, alt, np.rot90(aerosol_class2),cmap=cmap, norm = norm, vmin=0, vmax=7)
cbar = fig.colorbar(a, norm = norm, ticks = ticks,
           aspect = 20)
plt.ylabel('Altitude (km AGL)',fontsize = 18, **cgfont)
plt.xlabel('Latitude',fontsize = 18, **cgfont)
cbar.set_ticklabels(['Und','Dust','Continental','Polluted Dust','Smoke','Other'])
#cbar.set_title('Aerosol Type')


# In[ ]:



