#!/usr/bin/env python
# coding: utf-8

# # MOD07 processing for comparison with CALIPSO smoke plume overpasses
# 
# ### This script processes the MOD07 (MODIS Atmospheric Profiles) files over wildfire smoke events that the team identified. Functions used in the code are listed at the bottom of the notebook. Run those prior to running the script.
# 
# _Last modified Mar.17 2021._

# In[2]:


import pandas as pd
import numpy as np
import datetime
import math
import os
import shutil
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt


# In[17]:


basepath = "Testfiles_softwarerelease/MOD07/" # set main working directory - CHANGE 
csvpath = "Testfiles_softwarerelease/input_csvs/" # path to the folder with the csv file containing the smoke plume event info - CHANGE
outputpath = 'Testfiles_softwarerelease/outputs/'
if not os.path.exists(outputpath):
    os.mkdir(outputpath)
overlaps_df = pd.read_csv(csvpath+'calipso_transects.csv') # read in the info into a pandas dataframe
overlap_dates = overlaps_df['Date']; overlap_times = overlaps_df['Time_UTC'] # grab the dates and times
overlaps_df.head() # show the top of the dataframe

# FUNCTIONS:
def filter_lat(lat):
    import numpy as np
    # Replaces all latitude values below -90 and above 90 with Nans.
    # INPUTS:
    # - lat = array of latitude values
    # OUTPUTS:
    # - lat_filtered = array of latitude values with unphysical values replaced with NaNs
    # SYNTAX: lat_filtered = filter_lat(lat)
    lat[lat < -90] = np.NaN
    lat[lat > 90] = np.NaN
    return lat

def filter_lon(lon):
    import numpy as np
    # Replaces all longitude values below -180 and above 180 with Nans.
    # INPUTS:
    # - lon = array of longitude values
    # OUTPUTS:
    # - lon_filtered = array of longitude values with unphysical values replaced with NaNs
    # SYNTAX: lon_filtered = filter_lon(lon)
    lon[lon < -180] = np.NaN
    lon[lon > 180] = np.NaN
    return lon


def inflection_point(array):
    import numpy as np
    # Detects where there is an inflection point for a 1D array. Used for determination
    # of the inflection point in the gradient of the vertical water vapor mixing ratio profiles.
    # INPUTS:
    # - array = 1D array
    # OUTPUTS:
    # - idx = index of the inflection point
    # SYNTAX: idx = inflection_point(array)
    a2 = np.diff(array) # take the differences of the array
    idx = np.NaN # set idx to nan to start
    for i in range(1, len(a2)-1): # for all points except the first and the last
        if a2[i-1] < 0 and a2[i] > 0: # if difference changes from negative to positive at the point
            idx = i # grab the inflection point index
            break # stop the loop
    return idx # will be NaN if no idx is identified


def MH_calc(WV_MR, lat, lon, lat_ubound, lat_lbound, lon_ubound, lon_lbound, plotfs, plotspath):
    # Calculates mixing heights from MODIS vertical Water Vapor Mixing Ratio profiles
    # within an area bounded by the starting lat, lon and ending lat, lon 
    # of a portion of a CALIPSO transect that overlapped a smoke plume
    # INPUTS:
    # - WV_MR = Water Vapor mixing ratio field retrieved from HDF file
    # - lat = latitudes from HDF file
    # - lon = longitudes from HDF file
    # - lat_ubound = upper bound on latitude to subset the MODIS scene
    # - lat_lbound = lower bound on latitude to subset the MODIS scene
    # - lon_ubound = upper bound on longitude
    # - lon_lbound = lower bound on longitude
    # - plotfs = standard font size for the plots
    # - plotspath = path to the directory where the plots will be output, end with a /
    # OUTPUTS:
    # - list of lats
    # - list of lons
    # - list of calculated Mixing Heights
    # SYNTAX: lats, lons, MHs = MH_calc(WV_MR, lat, lon, lat_ubound, lat_lbound, lon_ubound, lon_lbound, plotfs, plotspath)
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # 0) RESOLVE THE Y-AXIS
    # 20 vertical pressure levels that the atmospheric profiles are resolved at in HPa or mbar:
    P = [1000, 950, 920, 850, 780, 700, 620, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5]; P.reverse()
    P = np.array(P)   
    # convert pressure to altitude using NWS formula:
    hft = (1 - (P/1013.24)**0.190284)*145366.45 # result in units of feet
    hm = 0.3048*hft; hkm = hm/1000;  # convert to meters and kilometers

    MHs = [] # store mixing heights
    lats = [] # store lats
    lons = [] # store lons
    cell_counter = 0 # counts all cells within the bounds set
    for i in range (0, lat.shape[0]):
        for j in range(0,lat.shape[1]):
            gridlat = lat[i,j]; gridlon = lon[i,j] # grab the lat/lon corresponding to the grid cell
            if not np.isnan(gridlat) and not np.isnan(gridlon): # if the lat, lon aren't nans
                if gridlat >= lat_lbound and gridlat <= lat_ubound and gridlon >= lon_lbound and gridlon <= lon_ubound: # search within bounds
                    # 1) Grab the Water Vapor mixing ratios from the data file
                    WV_MR = [] # to hold the WV mixing ratios
                    for h in range(0,20): # for each pressure layer
                        MR = mo_WV_MR[h, i, j] # grab the WV mixing ratio
                        if MR < 0 or MR > 20000:
                            MR = np.NaN # replace Nan values
                        WV_MR.append(MR)

                    # 2) Calculate mixing height for the profiles that exist
                    if np.count_nonzero(np.isnan(WV_MR)) < 20: # if there is at least one non-NaN in the array iwth 20 elements
                        MR_grad = -np.gradient(WV_MR) # calculate the gradient of the WV mixing ratios
                        MR_grad = np.append([np.nan], MR_grad[:-1]) # shift the gradients to associated height

                        # slice all below 10km since MH will be lower than 10km altitude (for plotting)
                        idxslice = 9; h_slice = hkm[idxslice:]; # slice height
                        WV_MR_slice = WV_MR[idxslice:]; MR_grad_slice = MR_grad[idxslice:] # slice water vapor mixing ratio and gradient

                        # Find MH boundary:
                        idxslice2 = 5 # slice the gradient even further to avoid the small gradients near the top  of the 10km
                        # find where the inflection point exists:
                        idx = inflection_point(MR_grad_slice[idxslice2:])

                        if not np.isnan(idx): # if the index is not a Nan:
                            idxMR = idx + idxslice2 # correct for the second slice
                            cell_counter = cell_counter + 1 # count the cell we successfully pulled MH from

                            # 3) Create folder to hold the plotted results if it doesn't already exist:
                            if not os.path.exists(plotspath):
                                os.mkdir(plotspath)

                            # 4) Plot the results:
                            plt.figure(figsize=(5,5))
                            plt.plot(WV_MR_slice, h_slice, 'o-')
                            plt.plot(MR_grad_slice, h_slice, 'ko-', alpha=0.5)
                            H_MR = h_slice[idxMR] # grab the altitude where the change in WV MR occurs
                            plt.plot([np.nanmin(MR_grad)-100,np.nanmax(WV_MR)+100], [H_MR,H_MR], 'm--') # plot a straight line at MH estimate
                            plt.ylabel('Altitude (km)', fontsize=plotfs); plt.xlabel('Water Vapor Mixing Ratio (g/kg)', fontsize=plotfs) # axis labels
                            plt.ylim(0,10); plt.xlim(np.nanmin(MR_grad)-100, np.nanmax(WV_MR)+100) # axis limits
                            plt.legend(['MR', 'MR gradient', 'MH estimate'], fontsize=plotfs) # legend entries
                            plt.xticks(fontsize=plotfs); plt.yticks(fontsize=plotfs)
                            plt.grid(); plt.tight_layout()
                            plt.savefig(plotspath+'WV_MR_'+str(time)+'_'+str(i).zfill(3)+'_'+str(j).zfill(3)+'.jpg', dpi=200)
    #                         plt.show()
                              plt.close() 
        
                            # 4) Store results in a table
                            lats.append(gridlat)
                            lons.append(gridlon)
                            MHs.append(H_MR) # store estimated mixing height

                    else:
                        print('all NaNs')  
                    
    print(cell_counter)
    return lats, lons, MHs


# # 1) Convert dates into year and julian days
# 
# This section of code takes the Date and Time_UTC from the smoke plume events we identified (held in a pandas dataframe) and translates them into Year and Julian Day, which is how the MOD07 files are downloaded and stored. This relies the datetime python package.

# In[4]:


datetimes = []; years = []; juliandays = []; times = [] # initialize lists to store the data
for i in range (0, len(overlaps_df)):
    date = overlap_dates[i]
    time = overlap_times[i]
    year = int(date[:4])
    dt = datetime.datetime.strptime(date+' '+time, '%Y-%m-%d %H:%M') # read in date and time strings as datetime objects
    datetimes.append(dt)
    # convert to julian day of the year
    firstday = datetime.date(year, 1, 1 ) # first of the year in julian days
    julianday = dt.toordinal() - firstday.toordinal() # subtract the date from the first of the year to get the Julian day of year
    juliandays.append(julianday); years.append(year); times.append(time)

# append julian day and year to overlaps_df
overlaps_df['Year'] = years
overlaps_df['julian_day'] = juliandays
overlaps_df = overlaps_df.sort_values(by ='Date')
overlaps_df.head(10)


# # 2) Identify MOD07 data file closest to the overlap time that overlaps the smoke plume geographically
# 
# Not all the MOD07 atmospheric profile files collected throughout the day will overlap our smoke plume geographically. This section of code sifts through each of the files for the day (1 file every 5 minutes) and identifies which overlap spatially with the smoke plume. The UTC time associated with each overlapping file is collected and differenced from the UTC time of the smoke plume overlap identified. The file with the least time offset is identified as the match for the smoke event and added back into the original data frame (overlaps_df). 
# 
# This code was modified to pull all overlapping data files for the day of the smoke event (overlaps_df_new) instead of just the one closest to the time of the CALIPSO pass. Those sections of the code are commented out currently.

# In[15]:


file_matches = [] # initialize list to store the filename for each smoke plume overpass that geographically overlaps
# on the same day and is closest in time!
time_diffs = [] # tracks the time difference between the CALIPSO pass and this MODIS file

# # for calculation of all overlapping files for the day:
# all_lat1 = []; all_lat2 = []
# all_lon1 = []; all_lon2 = []
# all_yrs = []; all_times = []; all_dates = []; all_jdays = []

progress_counter = 0
for idx, row in overlaps_df.iterrows(): # for each row in the dataframe
# for idx, row in subset_df.iterrows(): # use this for specific entries or groups of entries
    lat1 = row['Latitude_min']; lat2 = row['Latitude_max'] # grab latitude bounds
    lon1 = row['Longitude_min']; lon2 = row['Longitude_max'] # grab longitude bounds
    date = row['Date']
    year = row['Year']; day = row['julian_day']; timeUTC = row['Time_UTC'] # grab the year, day, time
    time = timeUTC.replace(":","") # remove the colon from UTC Time to match MOD07 filenames
    progress_counter = progress_counter + 1
    if time.startswith('2'): # if time associated with smoke plume overlap is 20:00 UTC or later, proceed
        filepath = basepath+str(year)+'/'+str(day)+'/' # find the right folder
        print(year, day, time)
        
        # Smoke plume overlap bounds:    
        latmin = np.min([lat1,lat2])
        latmax = np.max([lat1,lat2])
        lonmin = np.min([lon1,lon2])
        lonmax = np.max([lon1,lon2])
        print(latmin, latmax)
        print(lonmin, lonmax)
               
        overlap_files = []; time_offsets = [] # intiialize lists to hold the overlapping file names and time offsets
        if os.path.exists(filepath):
            for file in os.listdir(filepath): # loop through all the MOD07 files for the day:
                filetime = file[18:22]; # slice string to grab time associated with the file

                # process the HDF file to grab the geographic bounds:
                hdf_file = SD(filepath+file, SDC.READ) # read file
                # grab location coordinates:
                lat = filter_lat(hdf_file.select('Latitude').get()) # latitude with nonphysical values filtered out
                lon = filter_lon(hdf_file.select('Longitude').get()) # longtiude with nonphysical values filtered out
 
                # Find the files that overlap geographically with our smoke plumes:
                if latmin >= np.nanmin(lat) and latmax <= np.nanmax(lat): # within lat bounds
                    if lonmin >= np.nanmin(lon) and lonmax <= np.nanmax(lon): # within lon bounds
                        if np.nanmin(lon) > -178 and np.nanmax(lon) < 178: # remove those files that go across the 180/-180 transition
                            print('Overlap found at ', filetime)
                            print('File lat range:', np.nanmin(lat), np.nanmax(lat))
                            print('File lon range:', np.nanmin(lon), np.nanmax(lon))
                            overlap_files.append(file)
                            dt = abs(int(time) - int(filetime)) # calculate time difference
                            if len(str(dt))>2: # if there are more than two digits, the first digits are in hours, not minutes
                                dt = int(str(dt)[:-2])*60 + int(str(dt)[-2:]) # grab the hours, convert to minutes, and add to the minutes
                            time_offsets.append(dt) # time difference between the file and our overlap

                            # for calculation of all overlapping files:
    #                         all_lat1.append(lat1)
    #                         all_lat2.append(lat2)
    #                         all_lon1.append(lon1)
    #                         all_lon2.append(lon2)
    #                         all_yrs.append(year)
    #                         all_times.append(timeUTC)
    #                         all_dates.append(date)
    #                         all_jdays.append(day)
    #                         file_matches.append(file)
    #                         time_diffs.append(dt)
        
        if len(overlap_files) > 0:
            idxmin = np.nanargmin(time_offsets) # returns index of the minimum value in time_offsets
            time_diffs.append(np.min(time_offsets)) # store the time difference
            file_matches.append(overlap_files[idxmin]) # grabs the filename associated with the least time offset and appends to the list
        else:
            print('No overlapping file found.')
            time_diffs.append(np.NaN); file_matches.append(np.NaN) # append Nans
            
    else: # if the smoke plume overlap time doesn't start with 2, it is a row that must be removed
        time_diffs.append(np.NaN) # append NaN to the time diff list
        file_matches.append(np.NaN) # append NaN to the file match list
    print("PROGRESS:", progress_counter/len(overlaps_df)*100, "% done.")
        
overlaps_df['MODIS_filename'] = file_matches # add new column with the matching filenames
overlaps_df['time_offset'] = time_diffs # add new column with the time differences
overlaps_df = overlaps_df.dropna() # drop all the rows with NaNs in them
overlaps_df # show the updated dataframe


# In[125]:


# TOGGLE FOR ALL OVERLAPPING FILES
# overlaps_df_new = pd.DataFrame(list(zip(all_lat1, all_lon1, all_lat2, all_lon2, all_dates, all_times, all_yrs, all_jdays, file_matches, time_diffs)),
#                                columns = ['Latitude_min', 'Longitude_min', 'Latitude_max', 'Longitude_max','Date', 'Time_UTC','Year','julian_day',
#                                          'MODIS_filename', 'time_offset'])
# overlaps_df_new


# In[22]:


#### write the dataframe to a csv file so that it only needs to be run once:
overlaps_df.to_csv(path_or_buf = csvpath+'CALIPSO_MODIS_plume_overlaps.csv') # closest overlapping file
# overlaps_df_new.to_csv(path_or_buf = csvpath+'CALIPSO_MODIS_plume_overlaps_ALL.csv') # all overlapping files


# # 3) Calculate and output mixing heights from the identified MOD07 files


# In[18]:


# read in csv file generated in the previous step if already run
overlaps_df = pd.read_csv(csvpath+'CALIPSO_MODIS_plume_overlaps.csv', usecols=[1,2,3,4,5,6,7,8,9,10]) # read in the info into a pandas dataframe
overlaps_df = overlaps_df.dropna()


# In[24]:


for idx, row in overlaps_df.iterrows(): 
# for idx, row in subset_df.iterrows(): # toggle this line to use a subset of the whole dataframe
    lat1 = row['Latitude_min']; lat2 = row['Latitude_max'] # grab latitude bounds
    lon1 = row['Longitude_min']; lon2 = row['Longitude_max'] # grab longitude bounds
    year = row['Year']; day = row['julian_day']; timeUTC = row['Time_UTC'] # grab the year, day, time
    date = row['Date']
    time = timeUTC.replace(":","") # remove the colon from UTC Time to match MOD07 filenames
    filename = row['MODIS_filename'] # grab MODIS filename
    filetime = filename[18:22] # grab MODIS file time
    filepath = basepath+str(year)+'/'+str(day)+'/'+filename # path to the MODIS file
    print(date, time, filename)
    
    # Smoke plume overlap bounds:    
    latmin = math.floor(np.min([lat1,lat2])) # round min latitude down
    latmax = math.ceil(np.max([lat1,lat2])) # round max latitude up
    lonmin = math.floor(np.min([lon1,lon2]))-1 # round min lon down and widen search range 
    lonmax = math.ceil(np.max([lon1,lon2]))+1 # round max lon up and widen search range
    print(latmin, latmax)
    print(lonmin, lonmax)
    
    if os.path.exists(filepath): # if the file exists
        # process the HDF file:
        hdf_file = SD(filepath, SDC.READ) # read file
        lat = filter_lat(hdf_file.select('Latitude').get()) # grab latitudes of the grid cells and filter for nonphysical values
        lon = filter_lon(hdf_file.select('Longitude').get()) # grab longitudes of the grid cells and filter for nonphysical values
        mo_WV_MR = hdf_file.select('Retrieved_WV_Mixing_Ratio_Profile').get() # grab Water Vapor Mixing Ratio field from file

        # set plot output path: CHANGE TO REFLECT THE FOLDER STRUCTURE YOU DESIRE (folders auto generated by MH_calc function)
        plotspath = outputpath+str(date)+'_J'+str(day)+'_'+str(filetime)+'/'

         # Run the MH calculation function:
        [lats, lons, MHs] = MH_calc(mo_WV_MR, lat, lon, latmax, latmin, lonmax, lonmin, 14, plotspath)

        if len(MHs) > 0: # if MHs were calculated over the plume
            # save output to csv files:
            MH_df = pd.DataFrame(list(zip(lats, lons, MHs)), columns=['lat', 'lon', 'mixing_height'])
            MH_df.to_csv(path_or_buf=outputpath+str(date)+'_J'+str(day)+'_'+str(filetime)+'.csv')
            MH_df.head()
        else:
            print('No MHs calculated for',filename)
    else:
        print(filepath, 'does not exist')


# ## 4) Stitch all generated csv files together for an external GIS operation

# In[130]:


MHcsvpath = outputpath # set path to the output csv

all_lats = []; all_lons = []; all_MHs = []; all_dates = []; all_times = []; # lists to store all of the lat, lon, MH data
for csvfile in os.listdir(MHcsvpath): # loop through all the csv files produced
    if csvfile.startswith('2') and csvfile.endswith('csv'): # do not account for other files, all should start with 2 because of the year
        csv_df = pd.read_csv(MHcsvpath+csvfile) # read in the csv file as a dataframe
        datestring = csvfile[:10] # grab the date from the filename
        filetime = csvfile[-8:-4] # grab the time from the filename

        # concatenate entries from each file into the overall list
        all_lats = all_lats + list(csv_df.lat)
        all_lons = all_lons + list(csv_df.lon)
        all_MHs = all_MHs + list(csv_df.mixing_height)

        for l in range(0, len(csv_df.lat)): # for each list entry
            all_dates.append(datestring)
            all_times.append(str(filetime))


# In[131]:


# stitch into an overall dataframe and export
total_df = pd.DataFrame(list(zip(all_lats, all_lons, all_MHs, all_dates,all_times)), columns=['Lat', 'Lon', 'mixing_height_km', 'Date','Time_UTC'])
total_df.shape 


# In[133]:


total_df.to_csv(MHcsvpath+'all_MHs_asof20200315.csv') # save as a csv file


# In[ ]:


#jukes

