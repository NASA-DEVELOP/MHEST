#!/usr/bin/env python
# coding: utf-8

# # Compare_mixing_heights
# 
# __Spring 2021 Authors/Collaborators:__ Chris Wright, Jukes Liu, Lauren Mock, Dean Berkowitz, Brandy Nisbet-Wilcox, Dr. Kenton Ross, Dr. Travis Toth, Keith Weber 
# 
# __Description:__ 
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
#  - FWF data, cleaned up, merged with ASMOKRE output (CALIPSO_NWS_Overlap_Dirty1.csv)
#  - FWS data, cleaned up, merged with ASMOKRE output (FWS_CALIPSO_Overlap_AGL.csv)
#  - MODIS data, cleaned up, staged for comparison (ALL_MODIS_MHs_asof20200315_FWZs.csv, CIMMS_PBLH_points_withFWZ_and_PILs.csv, Copy_Clean_NWS_MH.csv, FWS_data.csv)
#  - Case study data (calipso_MH_20150827_DEM_corrcted.csv)
# 
# Out:
# 
#  - Plots, statistical tests

# In[23]:


# import packages
import pandas as pd                   # for storing data
import numpy as np                    # ^
import matplotlib.pyplot as plt       # plotting
import os                             # navigating files
import bs4                            # digesting html
import urllib.request as urllib       # opening the url
import scipy.stats as sts
#import requests


# ## Systematic Bias / Data Vizualization

# ### Read in clean FWF

# In[82]:


FWF_CALIPSO = pd.read_csv('Z:/SIHAQII/Data/NWS_MH_Data/CALIPSO_NWS_Overlap_Dirty_04012021.csv') # read # CHANGE FILEPATH
FWF_CALIPSO = FWF_CALIPSO.drop(FWF_CALIPSO[FWF_CALIPSO.NWS_km != FWF_CALIPSO.NWS_km].index) # drop NaNs


# In[83]:


# Mask out low MH
#FWF_CALIPSO = FWF_CALIPSO[(FWF_CALIPSO.NWS_km >.75) & (FWF_CALIPSO.MH_AGL >.75)] # mask out low MH
FWF_CALIPSO = FWF_CALIPSO[(FWF_CALIPSO.NWS_km >1) & (FWF_CALIPSO.Altitude_AGL >1)] # mask out low MH


# In[84]:


NWS_FWF = FWF_CALIPSO.NWS_km # assign variables
CALIPSO = FWF_CALIPSO.Altitude_AGL
#CALIPSO = FWF_CALIPSO.MH_AGL
CALIPSO_std = FWF_CALIPSO['std']
aerosol = FWF_CALIPSO.Aerosol_Ty


# ### Read in clean FWS

# In[54]:


merge_FWS = pd.read_csv('Z:/SIHAQII/Data/NWS_MH_Data/FWS_data_04012021.csv') # read


# In[48]:


merge_FWS = merge_FWS[merge_FWS.NWS_km != 0]
merge_FWS = merge_FWS[merge_FWS.NWS_km == merge_FWS.NWS_km]


# In[55]:


merge_FWS


# In[77]:


# mask out low MH
merge_FWS = merge_FWS[(merge_FWS.NWS_km >1) & (merge_FWS.Altitude_AGL >1)] # mask out low MH
NWS_FWS = merge_FWS.NWS_km # assign variables
#CALIPSO_FWS = merge_FWS.Altitude_AGL
CALIPSO_FWS = merge_FWS.MH_AGL


# ## Regression

# In[94]:


x = CALIPSO
y = NWS_FWF

# -------------------------------
# If wanting to compare FWS data:
# -------------------------------
# x1, x2 = x, CALIPSO_FWS
# y1,y2 = y, NWS_FWS # for plotting
# x = x.append(CALIPSO_FWS) # if regression through FWF and FWS desired
# y = y.append(NWS_FWS) # if regression through FWF and FWS desired


# In[95]:


# -------------------------------
# Orthogonal Distance Regression
# -------------------------------

import scipy.odr as odr
def f(B, x):
    # Linear function y = m*x + b
    # B is the beta parameters
    return B[0]*x + B[1]

# variance of x
sx = np.power(np.std(x),2)+np.zeros(x.size)
#sx = np.power(CALIPSO_std,2)

# variance in y should be the standard deviation of the entire NWS data set
sy = np.power(np.std(y),2)+np.zeros(y.size)
linear = odr.Model(f)
mydata = odr.Data(x, y, wd=1./sx, we=1./sy)
myodr = odr.ODR(mydata, linear, beta0=[0., 1.])

#Run the fit.:
myoutput = myodr.run()

#Examine output:
z =myoutput.beta
#myoutput.pprint()

x_res = myoutput.delta
y_res = myoutput.eps
#m_sigma = myoutput.beta_std
m_sigma = myoutput.sd_beta[0]


# 
# ### Run regression and find regression confidence limits

# In[96]:


# fit a curve to the data using a least squares 1st order polynomial fit

#z = np.polyfit(x,y,1) # z already determined from ODR
p = np.poly1d(z)
fit = p(x)

# CHANGE: use these outputs if looking to obtain statistics about ODR regression. These outputs are not good for plotting
# fit = myoutput.y
# xhat = myoutput.xplus

# calculate r^2
#pearson R (below) shows little covariance. It does not indicate how good our fit is. 
r = sts.pearsonr(x,y)
r2 = r[0]
pval = r[1]

# CHANGE: UNCOMMENT BELOW IF YOU WANT R^2
# calculate r^2 using ODR output. Do this if fit = myoutput.y
# yhat = fit                         
# ybar = np.sum(y)/len(y)
# xbar = np.sum(x)/len(x)
# sum(x)/len(x)

# # get sum of squares of residuals and total SoS for r2
# ssres = np.sum(np.power((y_res**2) +(x_res**2),.5))
# sstot = np.sum(np.power((xhat-xbar)**2+((yhat-ybar)**2),.5)) #+ np.sum(np.power(x-xbar))
# r2 = 1 - ssres/sstot

# get the coordinates for the fit curve
c_y = [np.min(fit),np.max(fit)]
c_x = [np.min(x),np.max(x)]

 
# predict y values of original data using the fit
p_y = z[0] * x + z[1]
 
# calculate the y-error (residuals)
y_err = y -p_y
 
# create series of new test x-values to predict for
p_x = np.arange(np.min(x),np.max(x)+1,1)
 
# ---------------------------------------------------------
# now calculate confidence intervals for new test x-series
# ---------------------------------------------------------

mean_x = np.mean(x)         # mean of x
n = len(x)              # number of samples in origional fit
t = 1.970                # appropriate t value (where n=230, two tailed 95%)
s_err = np.sum(np.power(y_err,2))   # sum of the squares of the residuals

confs = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((p_x-mean_x),2)/
            ((np.sum(np.power(x,2)))-n*(np.power(mean_x,2))))))


# now predict y based on test x-values
p_y = z[0]*p_x+z[1]
 
# get lower and upper confidence limits based on predicted y and confidence intervals
lower = p_y - abs(confs)
upper = p_y + abs(confs)

# set font type
cgfont = {'fontname': 'Century Gothic'}

# set-up the plot
fig, ax = plt.subplots(figsize= (8,8))
plt.xlabel('CALIPSO Mixing height (km AGL)',fontsize = 18, **cgfont)
plt.ylabel('NWS Forecasted Mixing Height (km AGL)',fontsize = 18, **cgfont)
plt.title('N = ' + str(len(x)),fontsize = 18, **cgfont)
 

 #plot line of best fit
plt.plot(c_x,c_y,'r-',label='Regression line')
 
# plot confidence limits
plt.plot(p_x,lower,'b--',label='Lower confidence limit (95%)')
plt.plot(p_x,upper,'b--',label='Upper confidence limit (95%)')

#plot 1-1 line
unbiased_x = [0,1,2,3,5,6,7]
unbiased_y = unbiased_x
plt.plot(unbiased_x,unbiased_y,'black',  label = 'Unbiased')
 
# set coordinate limits
plt.xlim(0,5.5)
plt.ylim(0,5.5)

# Scatter data
plt.scatter(x1,y1,label='FWF Forecasts', s = 80)
plt.scatter(x2,y2, label = 'Spot Forecasts', s =80)

# configure legend
plt.legend(loc=0)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext,fontsize = 18, **cgfont)
# show the plot

import matplotlib
cmap = plt.get_cmap('Dark2')
bounds = [0.,2,3.,5,6,7,8]
ticks = [1, 2.5,4,5.5,6.5,7.5]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# # # CHANGE: Use below code to plot sample data if interested in only FWF
# a = plt.scatter(x,y,label='FWF Sample Data', c = aerosol, cmap = cmap, norm = norm, s = 80)
# cbar = fig.colorbar(a, norm = norm, ticks = ticks,
#            aspect = 20)
# cbar.set_ticklabels(['Und','Dust','Continental','Polluted Dust','Smoke','Other'])

print('r^2 = ',r2)
print('r^2 p value', pval)
print('b[0] = ', z[0])
print('b[1] = ', z[1])


# In[227]:


# t test - find p value for slope

df = len(x) - 2  # equivalently, df = odr_out.iwork[10]
t_stat = myoutput.beta[0] / myoutput.sd_beta[0]  # t statistic for the slope parameter
p_val = sts.t.sf(np.abs(t_stat), df) * 2 # obtain p value for probability that slope = 0 is a better fit. 

#display
print('Recovered equation: y={:3.2f}x + {:3.2f}, t={:3.2f}, p={:.2e}'.format(myoutput.beta[0], myoutput.beta[1], t_stat, p_val))


# ### Root mean square error

# In[391]:


# Of the regression
RMSE_reg_x = (np.sum((x-xhat)**2)/len(x))**.5
RMSE_reg_y = (np.sum((y-yhat)**2)/len(y))**.5

# of the data
RMSE_data = (np.sum((y-x)**2)/len(x))**.5
RMSEs = {'reg_x':RMSE_reg_x,'reg_y':RMSE_reg_y,
                     'data':RMSE_data}


# In[392]:


RMSEs['data']


# ## Plotting relative error

# In[17]:


# figure, title
fig, ax = plt.subplots(figsize = (6,6))
#plt.title('Relative Error (FWF-CALIOP)/FWF vs. FWF',fontsize = 18, **cgfont)
plt.ylabel('Relative Error (% above NWS)',fontsize = 18, **cgfont)
plt.xlabel('NWS Mixing Height (km AGL)',fontsize = 18, **cgfont)
plt.axhline(y = 0, color = 'black', linestyle = 'dashdot', label = 'No bias')
ax.set_ylim(-100, 350)

# define the colormap
cmap = plt.get_cmap('Dark2')
bounds = [0.,2,3.,5,6,7,8]
ticks = [1, 2.5,4,5.5,6.5,7.5]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# scatter
# a = plt.scatter(y, (x-y)/y*100, c= aerosol,  cmap = cmap, norm = norm, s=80)
a = plt.scatter(y, (x-y)/y*100, s=80)
# plt.axhline(y = 0, color = 'black', linestyle = 'dashdot', label = 'No bias')
#a = plt.scatter(x,y,label='FWS Sample Data')
# cbar = fig.colorbar(a, norm = norm, ticks = ticks,
#            aspect = 20)
# cbar.set_ticklabels(['Undetermined','Dust','Continental','Polluted Dust','Smoke','Other']) # label cbar


# same process as above
fig2 = plt.figure(figsize = (6,6))
#plt.title('Relative Error (FWF - CALIOP)/CALIOP vs. CALIOP',fontsize = 18, **cgfont)
plt.ylabel('Relative Error (% above ASMOKRE)',fontsize = 18, **cgfont)
plt.xlabel('ASMOKRE Mixing Height (km AGL)',fontsize = 18, **cgfont)
# b = plt.scatter(x, (y-x)/x*100,c= aerosol, cmap = cmap, norm = norm, s=80)
b = plt.scatter(x, (y-x)/x*100, s=80)

plt.axhline(y = 0, color = 'black', linestyle = 'dashdot', label = 'No bias')
# cbar = fig2.colorbar(b, norm = norm, ticks = ticks,
#            aspect = 20, fontsize = 18, **cgfont)
# cbar.set_ticklabels(['Undetermined','Dust','Continental','Polluted Dust','Smoke','Other'])


# ## Case Study Plotting

# In[1074]:


CALIPSO_MH_case = pd.read_csv('Z:/SIHAQII/Data/Elk_Fire_Case_Study/calipso_MH_20150827_DEM_corrcted.csv')
from pylab import *
cgfont = {'fontname': 'Century Gothic'}
fig, ax = plt.subplots(figsize = (12,6))
min_lat = 46.9
#plt.scatter(CALIPSO_MH.Latitude,CALIPSO_MH.Altitude, c = CALIPSO_MH.Aerosol_Type)
plt.plot(CALIPSO_MH_case.Latitude,CALIPSO_MH_case.Altitude_km_AGL)
ax.set_xlim(47, max(CALIPSO_MH_case.Latitude))
ax.set_ylim(0, max(CALIPSO_MH_case.Altitude_km_AGL))
ax.set_title('ID101 Mixing Heights 08/27/2015', fontsize = 16, **cgfont)
ax.set_ylabel('Altitude (km AGL)',fontsize = 14, **cgfont)
ax.set_xlabel('Latitude',fontsize = 14, **cgfont)
cdict = {0: 'red', 2: 'pink', 3: 'purple',  5: 'green',6: 'blue', 7:'white'}
adict = {0: 'Other', 2: 'Dust', 3: 'Polluted Continental', 5: 'Polluted Dust', 6: 'Smoke', 7: 'Other' }

for g in np.unique(CALIPSO_MH.Aerosol_Type):
    ix = np.where(CALIPSO_MH.Aerosol_Type == g)
    #print(ix[0])
    ax.scatter(CALIPSO_MH.Latitude[ix[0]], CALIPSO_MH.Altitude_km_AGL[ix[0]], c = cdict[g], label = adict[g], s = 10)

plt.axhline(y=1.828, color = 'red', linestyle = 'dashdot', label = 'FWF Estimate')
plt.axhline(y = 2.151682, color = 'black', linestyle = 'dashdot', label = 'MODIS Estimate')
ax.legend(title = 'CALIPSO Aerosol Type')
plt.show()
#plt.colorbar()


# # MODIS Comparison

# In[ ]:


import pandas as pd
import numpy as np
import datetime
import math
import os
import shutil
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt


# In[ ]:


# set paths to all the data
basepath = "Z:/SIHAQII/data/" # set main working directory
CALIPSO_path = "input_CSVs/ALL_CALIPSO_MH_0401_joined_FWZ.csv"
MODIS_path = "input_CSVs/ALL_MODIS_MHs_asof20200315_FWZs.csv"
CIMSS_path = "CIMMS/CIMMS_PBLH_points_withFWZ_and_PILs.csv"
FWF_path = 'NWS_MH_Data/Copy_Clean_NWS_MH.csv'
FWS_path = 'NWS_MH_Data/FWS_data.csv'


# In[ ]:


# read in MODIS data
MODIS_df = pd.read_csv(basepath+MODIS_path,usecols=[2,3,4,5,6,7,8],
                       names=['Lat', 'Lon', 'mixing_height', 'Date','Time_UTC','STATE_ZONE','CWA'], header=0)
MODIS_df = MODIS_df[MODIS_df.Time_UTC > 1900] # select only those determined after noon PST
MODIS_df


# In[ ]:


# read in CIMMS data
CIMSS_df = pd.read_csv(basepath+CIMSS_path, usecols=[2,3,4,5,6,7,8])
CIMSS_df.head()


# In[ ]:


# read in ASMOKRE data
CALIPSO_df = pd.read_csv(basepath+CALIPSO_path)#, usecols=[3,4,5,6,7,8,9,10,11])
CALIPSO_df.head()


# In[ ]:


# read in NWS FWF mixing heights
FWF_df = pd.read_csv(basepath+FWF_path, usecols=[1,2,3,4,5,6,7])
FWF_df.head()


# In[ ]:


# read in NWS FWS mixing heights
FWS_df = pd.read_csv(basepath+FWS_path,usecols=[2,3,4,5,6,7,8,9])
FWS_df.head()


# In[ ]:


output_fig_path = 'Z:/SIHAQII/Data/Figures/'


# ### 1) CIMSS vs. FWF

# In[ ]:


# CIMMS vs. FWF
CIMSS_FWF_df = CIMSS_df.merge(FWF_df, on=['Date','STATE_ZONE'])
CIMSS_FWS_df = CIMSS_df.merge(FWS_df, on=['Date','CWA'])
# plot:
plt.scatter(CIMSS_FWF_df.groupby(['STATE_ZONE','Date'])['PBLH_km'].median().reset_index().PBLH_km,
            CIMSS_FWF_df.groupby(['STATE_ZONE','Date'])['NWS_km'].median().reset_index().NWS_km)
plt.scatter(CIMSS_FWS_df.groupby(['Date'])['PBLH_km'].median().reset_index().PBLH_km,
            CIMSS_FWS_df.groupby(['Date'])['NWS_km'].median().reset_index().NWS_km)
plt.plot([0, 5], [0, 5], 'k-')
plt.xlabel('CIMMS mixing heights (km)')
plt.ylabel('NWS Forecasted mixing heights (km)')
# cbar = plt.colorbar(); cbar.set_label('Aerosol Type')
n = int(len(CIMSS_FWF_df.groupby(['STATE_ZONE','Date'])['PBLH_km'].median().reset_index()))+int(len(CIMSS_FWS_df.groupby(['Date'])['PBLH_km'].median().reset_index().PBLH_km))
plt.title('n = '+str(n))
plt.legend(['1 to 1','FWF','FWS'],loc='lower right')
plt.axes().set_aspect('equal'); plt.tight_layout()
plt.savefig(output_fig_path+'CIMMS_vs_NWS.jpg', dpi=350)
plt.show()


# In[ ]:


CIMSS_FWF_df.to_csv(basepath+'MH_comparison/CIMSS_FWF.csv')
CIMSS_FWS_df.to_csv(basepath+'MH_comparison/CIMSS_FWS.csv')


# ### 2) ASMOKRE vs. CIMSS

# In[ ]:


# ASMOKRE vs. CIMSS
CALIPSO_CIMMS_df = CIMSS_df.merge(CALIPSO_df, on=['Date', 'STATE_ZONE']) # merge on Date, State Zone, and PIL
CIMMS_zonalmed_df =  CALIPSO_CIMMS_df.groupby(['STATE_ZONE','Date'])['PBLH_km'].median().reset_index() # zonal median of CIMMS MHs
ASMOKRE_zonalmed_df = CALIPSO_CIMMS_df.groupby(['STATE_ZONE','Date'])['Altitude_AGL'].median().reset_index() # take zonal median of ASMOKRE MHs
modal_aerosol_type = CALIPSO_CIMMS_df.groupby(['STATE_ZONE','Date'])['Aerosol_Type'].agg(lambda x: pd.Series.mode(x)[0])
# Plot:
plt.scatter(CIMMS_zonalmed_df.PBLH_km, ASMOKRE_zonalmed_df.Altitude_AGL, c=modal_aerosol_type)
plt.plot([0, 6], [0, 6], 'k-')
plt.xlabel('CIMSS mixing heights (km)')
plt.ylabel('ASMOKRE mixing heights (km)')
cbar = plt.colorbar(); cbar.set_label('Aerosol Type')
n = int(len(CIMMS_zonalmed_df))
plt.title('n = '+str(n))
plt.axes().set_aspect('equal'); 
plt.savefig(output_fig_path+'ASMOKRE_vs_CIMSS.jpg', dpi=350)
plt.show()


# In[ ]:


CALIPSO_CIMMS_df.to_csv(basepath+'MH_comparison/ASMOKRE_CIMSS.csv')


# ### 3) ASMOKRE vs. MODIS

# In[ ]:


# ASMOKRE vs. MODIS
ASMOKRE_MODIS_df = MODIS_df.merge(CALIPSO_df, on=['Date', 'STATE_ZONE']) # merge on Date, State Zone, and PIL
MODIS_zonalmed_df =  ASMOKRE_MODIS_df.groupby(['STATE_ZONE','Date'])['mixing_height'].median().reset_index() # zonal median of CIMMS MHs
ASMOKRE_zonalmed_df = ASMOKRE_MODIS_df.groupby(['STATE_ZONE','Date'])['Altitude_AGL'].median().reset_index() # take zonal median of ASMOKRE MHs
modal_aerosol_type = ASMOKRE_MODIS_df.groupby(['STATE_ZONE','Date'])['Aerosol_Type'].agg(lambda x: pd.Series.mode(x)[0])

# Plot:
plt.scatter(MODIS_zonalmed_df.mixing_height, ASMOKRE_zonalmed_df.Altitude_AGL, c=modal_aerosol_type)
plt.plot([0, 7], [0, 7], 'k-')
plt.xlabel('MODIS mixing heights (km)')
plt.ylabel('ASMOKRE mixing heights (km)')
cbar = plt.colorbar(); cbar.set_label('Aerosol Type')
n = int(len(ASMOKRE_zonalmed_df))
plt.title('n = '+str(n))
plt.axes().set_aspect('equal'); 
plt.savefig(output_fig_path+'ASMOKRE_vs_MODIS.jpg', dpi=350)
plt.show()


# In[ ]:


ASMOKRE_MODIS_df.to_csv(basepath+'MH_comparison/ASMOKRE_MODIS.csv')


# ### 4) MODIS vs. NWS

# In[ ]:


# MODIS vs. FWF
MODIS_FWF_df = MODIS_df.merge(FWF_df, on=['Date','STATE_ZONE'])
MODIS_FWS_df = MODIS_df.merge(FWS_df, on=['Date','CWA'])
# plot:
plt.scatter(MODIS_FWF_df.groupby(['STATE_ZONE','Date'])['mixing_height'].median().reset_index().mixing_height,
            MODIS_FWF_df.groupby(['STATE_ZONE','Date'])['NWS_km'].median().reset_index().NWS_km)
plt.scatter(MODIS_FWS_df.groupby(['Date'])['mixing_height'].median().reset_index().mixing_height,
            MODIS_FWS_df.groupby(['Date'])['NWS_km'].median().reset_index().NWS_km)
plt.plot([0, 6], [0, 6], 'k-')
plt.xlabel('MODIS mixing heights (km)')
plt.ylabel('NWS Forecasted mixing heights (km)')
n = int(len(MODIS_FWF_df.groupby(['STATE_ZONE','Date'])['mixing_height'].median()))+int(len(MODIS_FWS_df.groupby(['Date'])['mixing_height'].median()))
plt.title('n = '+str(n))
# plt.legend(['1 to 1','FWF','FWS'],loc='lower right')
plt.axes().set_aspect('equal'); 
plt.savefig(output_fig_path+'MODIS_vs_NWS.jpg', dpi=350)
plt.show()


# In[ ]:


MODIS_FWF_df.to_csv(basepath+'MH_comparison/MODIS_FWF.csv')
MODIS_FWS_df.to_csv(basepath+'MH_comparison/MODIS_FWS.csv')


# ### 5) MODIS vs. CIMSS

# In[ ]:


# # MODIS vs. CIMSS
# MODIS_CIMMS_df = CIMSS_df.merge(MODIS_df, on=['Date', 'STATE_ZONE']) # merge on Date, State Zone
# CIMMS_zonalmed_df =  MODIS_CIMMS_df.groupby(['STATE_ZONE','Date'])['PBLH_km'].median().reset_index() # zonal median of CIMMS MHs
# MODIS_zonalmed_df = MODIS_CIMMS_df.groupby(['STATE_ZONE','Date'])['mixing_height'].median().reset_index() # take zonal median of ASMOKRE MHs

# # Plot:
# plt.scatter(MODIS_zonalmed_df.mixing_height,CIMMS_zonalmed_df.PBLH_km)
# plt.plot([0, 5], [0, 5], 'k-')
# plt.ylabel('CIMSS mixing heights (km)')
# plt.xlabel('MODIS mixing heights (km)')
# n = int(len(CIMMS_zonalmed_df))
# plt.title('n = '+str(n))
# plt.axes().set_aspect('equal'); 
# plt.savefig(output_fig_path+'MODIS_vs_CIMSS.jpg', dpi=350)
# plt.show()


# ### 6) ASMOKRE vs. NWS

# In[ ]:


# ASMOKRE vs. FWF
ASMOKRE_FWF_df = CALIPSO_df.merge(FWF_df, on=['Date','STATE_ZONE'])
ASMOKRE_FWS_df = CALIPSO_df.merge(FWS_df, on=['Date','CWA'])
# plot:
plt.scatter(ASMOKRE_FWF_df.groupby(['STATE_ZONE','Date'])['MH_AGL'].median().reset_index().MH_AGL,
            ASMOKRE_FWF_df.groupby(['STATE_ZONE','Date'])['NWS_km'].median().reset_index().NWS_km)
plt.scatter(ASMOKRE_FWS_df.groupby(['Date'])['MH_AGL'].median().reset_index().MH_AGL,
            ASMOKRE_FWS_df.groupby(['Date'])['NWS_km'].median().reset_index().NWS_km)
plt.plot([0, 6], [0, 6], 'k-')
plt.xlabel('ASMOKRE mixing heights (km)')
plt.ylabel('NWS Forecasted mixing heights (km)')
# cbar = plt.colorbar(); cbar.set_label('Aerosol Type')
n = int(len(ASMOKRE_FWF_df.groupby(['STATE_ZONE','Date'])['MH_AGL'].median().reset_index()))+int(len(ASMOKRE_FWS_df.groupby(['Date'])['MH_AGL'].median().reset_index()))
plt.title('n = '+str(n))
# plt.legend(['1 to 1','FWF','FWS'],loc='upper right')
plt.axes().set_aspect('equal'); 
plt.savefig(output_fig_path+'ASMOKRE_vs_NWS.jpg', dpi=350)
plt.show()

