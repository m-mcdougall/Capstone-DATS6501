# -*- coding: utf-8 -*-

#Non-Specific Imports
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import concurrent.futures as cf
from tqdm import tqdm
import math
import nltk
import seaborn as sns

#Flatten list utility function
def flatten_list(list_in):
    return [item for sublist in list_in for item in sublist]


#Set the working directory
wd=os.path.abspath('C://Users//Mariko//Documents//GitHub//Capstone-DATS6501')
os.chdir(wd)

#%%

#Loop through all cities - Collect all Hotels
all_files = os.listdir(wd+'\\Data\\Cleaned\\')
all_cities = [file for file in all_files if '.pkl' in file and 'Hotels' in file]

hotels = []
for city in all_cities:
    hotels_file =  pd.read_pickle(wd+'\\Data\\Cleaned\\'+city)
    hotels.append(hotels_file)


hotels_df = pd.concat(hotels)

#%%


#Split the address by commas
new = hotels_df.hotel_address.str.split(',')

#State and Zip Code in the final comma set (total number varies)
new = new.apply(lambda x: x[-1])
#Remove zip code
new = new.str.replace(r'[ ][\d-]+', '', regex=True)
new = new.str.replace(r'[ ]', '', regex=True)

#Rejoin to the dataframe
hotels_df['State'] = new


#%%

hotels_state_unique=hotels_df.groupby('State').nunique()
hotels_state_sum=hotels_df.groupby('State').sum()
hotels_state_mean=hotels_df.groupby('State').mean()


sns.barplot(x=hotels_state_unique.index, y=hotels_state_unique.hotel_ID, )
plt.title('Unique Hotels')
plt.show()

sns.barplot(x=hotels_state_sum.index, y=hotels_state_sum.Number_reviews, )
plt.title('Total number reviews')
plt.show()

sns.barplot(x=hotels_state_mean.index, y=hotels_state_mean.Review_rating_mean, )
plt.title('Mean review Value')
plt.show()

sns.barplot(x=hotels_state_mean.index, y=hotels_state_mean.Review_rating_std, )
plt.title('Mean std review Value')
plt.show()

#%%

"""
Cartopy imports
"""


import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.patheffects as PathEffects
from matplotlib.colors import Normalize
import matplotlib
from matplotlib import cm


def check_color(hotel_count):
    color_cutoffs = {0:'#581845', 50:'#900C3F', 100:'#C70039',150:'#FF5733'} # Else: #CACBC1
    hotel_color='grey'
    for key in color_cutoffs.keys():
        if hotel_count>key:
            hotel_color=color_cutoffs[key]
    return hotel_color

#Initialize the figure
fig = plt.figure(figsize=(11, 8))

#Center the main map on the US
ax_us = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal())
ax_us.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())


## Add in Alaska Subplot
ax_ak = fig.add_axes([0.01, 0.15, 0.28, 0.20], projection=ccrs.PlateCarree())
ax_ak.set_extent([-169, -130, 53, 71],  crs=ccrs.PlateCarree()) #Set lat and logitiude to display


## Add in Hawaii Subplot      
ax_hi = fig.add_axes([0.01, 0.35, 0.15, 0.15], projection=ccrs.PlateCarree())
ax_hi.set_extent([-161, -154, 23, 18],  crs=ccrs.PlateCarree())#Set lat and logitiude


#Load the shapefile, and the correct borders
shapename = 'admin_1_states_provinces_lakes_shp'
states_shp = shpreader.natural_earth(resolution='110m',
                                     category='cultural', name=shapename)


#Set the background colors/visibility
for ax in [ax_us, ax_ak, ax_hi]:
    ax.outline_patch.set_visible(False)  
    ax.background_patch.set_visible(True)
    #ax.background_patch.set_facecolor('blue')


#Add Title
title_str='Number of Unique Hotels Scraped per State\nMost Populous City in each State Scraped'
ax_us.set_title(title_str)


#Loop through each state and paint the borders and facecolor according to the RGBA values we calculated
for astate in shpreader.Reader(states_shp).records():
    try:
        #Get hotel info and colour for the state
        state_abbrev = astate.attributes['postal']
        hotel_count = hotels_state_unique.loc[state_abbrev,'hotel_ID']
        hotel_color=check_color(hotel_count)
        
        if state_abbrev == "AK":
            ax_ak.add_geometries([astate.geometry], ccrs.PlateCarree(),
                      facecolor=hotel_color, edgecolor='white')
        elif state_abbrev == "HI":
            ax_hi.add_geometries([astate.geometry], ccrs.PlateCarree(),
                      facecolor='grey', edgecolor='white')
        else:
            ax_us.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor=hotel_color, edgecolor='white')
    except:
        #This may be a territory, or a state which has not stations(eg, RI)
        ax_us.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor='grey', edgecolor='white')
        print(f'{state_abbrev}: SKIPPED')
        pass
    



##NOTE: This is just needed until HI data comes in
## DELETE AFTER DONE
for astate in shpreader.Reader(states_shp).records():
    if astate.attributes['postal'] == 'HI':
        ax_hi.add_geometries([astate.geometry], ccrs.PlateCarree(),
                      facecolor='grey', edgecolor='white')


#Add stand-alone colourbar to show the direction of the gradient
#c_map_ax = fig.add_axes([0.91, 0.33, 0.01, 0.36])
#c_map_ax.axes.get_xaxis().set_visible(False)
#c_map_ax.axes.get_yaxis().set_visible(False)
#matplotlib.colorbar.ColorbarBase(c_map_ax, cmap='coolwarm', orientation = 'vertical')


plt.show()



#%%


## This one's for number of reviews

def check_color(hotel_count):
    color_cutoffs = {0:'#581845', 50000:'#900C3F', 100000:'#C70039',150000:'#FF5733'} # Else: #CACBC1
    hotel_color='grey'
    for key in color_cutoffs.keys():
        if hotel_count>key:
            hotel_color=color_cutoffs[key]
    return hotel_color

#Initialize the figure
fig = plt.figure(figsize=(11, 8))

#Center the main map on the US
ax_us = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal())
ax_us.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())


## Add in Alaska Subplot
ax_ak = fig.add_axes([0.01, 0.15, 0.28, 0.20], projection=ccrs.PlateCarree())
ax_ak.set_extent([-169, -130, 53, 71],  crs=ccrs.PlateCarree()) #Set lat and logitiude to display


## Add in Hawaii Subplot      
ax_hi = fig.add_axes([0.01, 0.35, 0.15, 0.15], projection=ccrs.PlateCarree())
ax_hi.set_extent([-161, -154, 23, 18],  crs=ccrs.PlateCarree())#Set lat and logitiude


#Load the shapefile, and the correct borders
shapename = 'admin_1_states_provinces_lakes_shp'
states_shp = shpreader.natural_earth(resolution='110m',
                                     category='cultural', name=shapename)


#Set the background colors/visibility
for ax in [ax_us, ax_ak, ax_hi]:
    ax.outline_patch.set_visible(False)  
    ax.background_patch.set_visible(True)
    #ax.background_patch.set_facecolor('blue')


#Add Title
title_str='Total Reviews Scraped per State\nMost Populous City in each State Scraped'
ax_us.set_title(title_str)


#Loop through each state and paint the borders and facecolor according to the RGBA values we calculated
for astate in shpreader.Reader(states_shp).records():
    try:
        #Get hotel info and colour for the state
        state_abbrev = astate.attributes['postal']
        hotel_count = hotels_state_sum.loc[state_abbrev,'Number_reviews']
        hotel_color=check_color(hotel_count)
        
        if state_abbrev == "AK":
            ax_ak.add_geometries([astate.geometry], ccrs.PlateCarree(),
                      facecolor=hotel_color, edgecolor='white')
        elif state_abbrev == "HI":
            ax_hi.add_geometries([astate.geometry], ccrs.PlateCarree(),
                      facecolor='grey', edgecolor='white')
        else:
            ax_us.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor=hotel_color, edgecolor='white')
    except:
        #This may be a territory, or a state which has not stations(eg, RI)
        ax_us.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor='grey', edgecolor='white')
        print(f'{state_abbrev}: SKIPPED')
        pass
    



##NOTE: This is just needed until HI data comes in
## DELETE AFTER DONE
for astate in shpreader.Reader(states_shp).records():
    if astate.attributes['postal'] == 'HI':
        ax_hi.add_geometries([astate.geometry], ccrs.PlateCarree(),
                      facecolor='grey', edgecolor='white')


#Add stand-alone colourbar to show the direction of the gradient
#c_map_ax = fig.add_axes([0.91, 0.33, 0.01, 0.36])
#c_map_ax.axes.get_xaxis().set_visible(False)
#c_map_ax.axes.get_yaxis().set_visible(False)
#matplotlib.colorbar.ColorbarBase(c_map_ax, cmap='coolwarm', orientation = 'vertical')


plt.show()


