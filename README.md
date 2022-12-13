# Mixing Height Estimation Toolbox (MHEST)
*Code for the Southern Idaho Health and Air Quality II NASA DEVELOP project*
---

*Node*: Pocatello, Idaho <br />
*Term*: Spring 2021

## *Description*
The MHEST tool takes CALIPSO and MODIS data, calculates mixing heights, and stages them for comparison with NWS Fire Weather Forecasts (and /or Spot Forecasts). The Fire Weather Forecasts are scrapes from an online archive, while CALIPSO and MODIS data for desired dates must be downloaded.

## *POC Contact Info*
`Chris Wright, 510-387-6338, chrisw97@uw.edu`

## *Authors/Collaborators*
*Spring 2021:* Chris Wright, Julia Liu, Dean Berkowitz, Lauren Mock <br />
*Fall 2020:* Ashwini Badgujar, Sean Cusick, Patrick Giltz, Ella Griffith, Brandy Nisbet-Wilcox <br />
*Advisors:* Dr. Kenton Ross, Dr. Travis Toth, Keith Weber

## Requirements for running code
CALIPSO LiDAR Level 2 Vertical Feature Mask HDF files, MODIS data files, python editor. 

### In `ASMOKRE_02.22.2021`
We loop through CALIPSO HDF-format files on days of interest and extract data at relevant latitudes. We use bit-wise extraction to reveal:
1. feature classification and
2. feature sub-type

across the different transects. We then save the data to a specified directory. 

*Inputs* <br />
List of dates, lat / lon of interest (called 'matches'); CALIPSO HDF files

### In `Scrape_NWS` 
We loop through webpages corresponding to dates and fire weather zones of interest. From these webpages, we extract mixing height for “Today”, the day that the data was collected. We then process the data to stage it for comparison

*Inputs* <br />
List of dates and PILS desired

### In `MOD07_processing`
We loop through MODIS HDF-format files on days of interest, select the data within a geographic area of interest, plot the vertical Water Vapor Mixing Ratio profiles, calculate the gradients of the vertical profiles, and use the profiles to identify the mixing height altitudes. 

*Inputs* <br />
List of desired dates, lat / lon; MODIS files

## To run
1. Run `ASMOKRE`. Correct output from MSL to AGL in Arc
2. Run `Scrape_NWS`. Follow instructions in script for staging data for comparison (merging).
3. Run `MOD07_processing`

## Recommended file structure
1. A folder for CALIPSO files of interest
2. A folder for MODIS files of interest
3. An empty folder for ```ASMOKRE``` output
4. A folder for all the inputs (desired days, latitudes, intermediate outputs, etc.)

```Anything marked "CHANGE" can / must be changed to include your file path, set of files, personal preference on analysis type, etc.```
