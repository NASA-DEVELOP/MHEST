#!/usr/bin/env python
# coding: utf-8

# # National Weather Service Archive Scraper and Processing (Scrape_NWS)
# 
# __Spring 2021 Authors/Collaborators:__ Chris Wright, Jukes Liu, Lauren Mock, Dean Berkowitz, Brandy Nisbet-Wilcox, Dr. Kenton Ross, Dr. Travis Toth, Keith Weber 
# 
# __Functions:__ 
#  - Scrapes mixing height data off iowa state mesonet
#  
# __Parameters:__
# 
# In:
#  - CSV files with all the ASMOKRE mixing height values with respective latitude and longitude (entire folder: here = ASMOKRE_Mixing_Heights/)
#  - CSV file with desired dates, PILS, FWZs of interest (Plume_overlaps_FWZ_with_PILs.csv)
#  
# Out:
#  - NWS mixing height data

# In[6]:


# import packages
import pandas as pd # for storing data
import numpy as np # ^
import matplotlib.pyplot as plt # plotting
import os
import bs4 # digesting html
import urllib.request as urllib # opening the url
import re
#import requests


# In[7]:


# Read in the FWF zone overlaps with plumes identified by Worldview search
fwf_filenames_csv = pd.read_csv('Tutorial_files/input_csvs/Plume_overlaps_FWZ_with_PILs.csv') # read
# assign useful variables
PILS = fwf_filenames_csv.WFO
state_zone = fwf_filenames_csv.STATE_ZONE
Date = fwf_filenames_csv.Date
Date = pd.to_datetime(Date)#, format = '%m/%d/%y')
time = fwf_filenames_csv.Time_UTC


# In[8]:


# -------------------------------------------------------------------------------------
# define a function to scrape MH from NWS text products with the url, PILS id as input
# -------------------------------------------------------------------------------------
def read_NWS(urls, PILS):
    # open the page
    html_page = urllib.urlopen(urls)
    soup = bs4.BeautifulSoup(html_page) # turn it into a beautifulsoup object
    links = []
    for link in soup.findAll('a', attrs = {'href': re.compile("p.php")}):
        if link.get('href')[10:13] == 'FWS': # CHANGE TO FWF IFF FWF DESIRED
            links.append(link.get('href')) # get all the links to the FWS or FWF data

    # to be filled below with data from each link on the website
    Mixing_heights = []
    Meta_Data = []
    Date_time = []
    PILS_long = []
    # --------------------------------------------------------------------
    # Loop through each FWF text product available for that day and PILS
    # --------------------------------------------------------------------
    for link in links:
        to_scrape = 'https://mesonet.agron.iastate.edu/wx/afos/'+link
        page = urllib.urlopen(to_scrape)
        html_bytes = page.read()
        html = html_bytes.decode("utf-8")
  
        # # Find the mixing height
        noMH = True
        MH = ''
        html_all = html
        moreMH = False
        # -------------------------------------------------------------------------------------
        # figure out which terms to use and if the file mentions mixing height
        # -------------------------------------------------------------------------------------
        if ((html.find('TODAY...') >0) | (html.find('Today...')>0)) & ((html.find('MIXING HEIGHT')>0) | (html.find('Mixing height')>0) | (html.find('Mixing Height')>0)):
            moreMH = True
            if html.find('Mixing height')>0: 
                MH_str = 'Mixing height'
            elif html.find('Mixing Height')>0:
                MH_str = 'Mixing Height'
            else: MH_str = 'MIXING HEIGHT'
            if html.find('TODAY...')>0:
                day_str = 'TODAY...'
            else:
                day_str = 'Today...'
                print('Lowercase today')
        # -------------------------------------------------------------------------------------
        # loop through the text product and store the mixing height, zone data
        # -------------------------------------------------------------------------------------
        while moreMH == True:
            subhtml_1 = html[html.find(day_str):]
            subhtml_2 = subhtml_1[:subhtml_1.find('TONIGHT...')] # replace with dayofweek +1 or 'tonight'
            subhtml = subhtml_2
            if subhtml.find(MH_str)>0: # in case not 
                mhi = subhtml.find(MH_str)
                i = 0
                while subhtml[mhi+len(MH_str)+i] != '\n':
                    # just grab everything in the line
                    MH = MH + subhtml[mhi+len(MH_str)+i]      
                    i = i +1 
                # now add that whole line to the MH column
                Mixing_heights.append(MH)
                MH = ''

                # now go above where you are to find the meta data, stop when you reach '\n\n'
                # if there is a mixing height within that chunk
                metadata = html[:html.find(day_str)]
                last_new_double = metadata.rindex('\n\n')
                penult_new_double = metadata[:last_new_double].rindex('\n\n')
                meta_data_gold = metadata[penult_new_double:last_new_double]
                
                # if the metadata has a fire weather warning (ie isn't what we expect)
                # then keep looking
                if meta_data_gold[:10].find('...')>0:
                    metadata = html[:penult_new_double-1]
                    penpenult = metadata.rindex('\n\n')
                    meta_data_gold = metadata[penpenult:penult_new_double]

                # add the meta data
                Meta_Data.append(meta_data_gold)
                    

                #now store the date and PILS
                Date_time.append(to_scrape[-12:])
                PILS_long.append(PILS)

                # change for the next loop so we search the parts after this
                html = subhtml_1[subhtml_1.find('TONIGHT...'):]
                if (html.find(day_str) <0) | (html.find(MH_str)<0):
                    moreMH = False
            else: moreMH = False
    
    return Mixing_heights, Meta_Data, Date_time, PILS_long


# In[9]:


# ----------------------------------------
# call function for every url. Get output
# ----------------------------------------

Mixing_heights = []
Meta_Data, Date_time, PILS_long = [],[],[]
i = 0
for i in range(len(PILS)):
    print(i)
    urls = 'https://mesonet.agron.iastate.edu/wx/afos/list.phtml?source='+str(PILS[i])+'&year='+str(Date.dt.year[i])+'&month='+ str(Date.dt.month[i]) + '&day='+ str(Date.dt.day[i]) + '&view=grid&order=asc'
    print(urls)
    
    Mixing_heights_t, Meta_Data_t, Date_time_t, PILS_t = read_NWS(urls, PILS[i])
    Mixing_heights.extend(Mixing_heights_t)
    Meta_Data.extend(Meta_Data_t)
    Date_time.extend(Date_time_t)
    PILS_long.extend(PILS_t)


# In[10]:


# ----------------------------------------
# Clean up the data and pack it into a df
# ----------------------------------------

Meta_data_clean = [Meta_Datum[2:] for Meta_Datum in Meta_Data] # get rid of \n
DateNWS = [Date_1[:8] for Date_1 in Date_time]
timeNWS = [time[8:] for time in Date_time]
Zones = [zone[:2] + zone[3:6] for zone in Meta_data_clean]
Dates = pd.to_datetime(DateNWS)
#times = pd.to_datetime(timeNWS, utc = True)
df_FWF = pd.DataFrame(
            {'STATE_ZONE': Zones,
             'Mixing_heights': Mixing_heights,
             'Date': Dates,
             'Time': timeNWS,
             'PILS': PILS_long
            })
df_FWF.head()

#remove duplicates
df_FWF = df_FWF.drop_duplicates()


# In[13]:


# Save to a csv
FWF_path = # CHANGE ME
df_FWF.to_csv(FWF_path) # CHANGE FILE PATH


# ## For probing one day / url

# In[948]:


urls = 'https://mesonet.agron.iastate.edu/wx/afos/list.phtml?source=OTX&year=2018&month=8&day=19&year2=2021&month2=3&day2=11&view=grid&order=asc'
print(urls)
    
Mixing_heights_t, Meta_Data_t, Date_time_t, PILS_t = read_NWS(urls, 'OTX') # make sure PILS corresponds to URL
Meta_Data_t

