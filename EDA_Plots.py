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

###
# Load in all Hotels
###

#Loop through all cities - Collect all Hotels
all_files = os.listdir(wd+'\\Data\\Cleaned\\')
all_cities = [file for file in all_files if '.pkl' in file and 'Hotels' in file]

hotels = []
for city in all_cities:
    hotels_file =  pd.read_pickle(wd+'\\Data\\Cleaned\\'+city)
    
    #Drop some columns to make the dataframe a bit lighter
    hotels_file.drop(['hotel_city','hotel_blurb', 'hotel_prop_amenity',
                      'hotel_room_amenity','hotel_location_food', 'hotel_location_attract'], 
                     axis=1, inplace=True)
    
    hotels.append(hotels_file)

hotels_df = pd.concat(hotels)

#Delete intermediaries to free up memory
del all_files, all_cities, city, hotels, hotels_file



## Engineer the state from the address

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

###
# Load in all Reviews
###

#Loop through all cities - Collect all Hotels
all_files = os.listdir(wd+'\\Data\\Cleaned\\')
all_cities = [file for file in all_files if '.pkl' in file and 'Reviews' in file]

hotels = []
for city in tqdm(all_cities):
    hotels_file =  pd.read_pickle(wd+'\\Data\\Cleaned\\'+city)
    
    #Drop some columns to make the dataframe a bit lighter
    hotels_file.drop(['Review_date', 'Reviewer_loc','Review_stay_date', 'Review_title', 'Review_text',], 
                     axis=1, inplace=True)
        
    hotels.append(hotels_file)

#Merge into one dataframe
reviews_df = pd.concat(hotels)
reviews_df.rename({'Hotel_ID':'hotel_ID'}, axis=1, inplace=True)

#Delete intermediaries to free up memory
del all_files, all_cities, city, hotels, hotels_file


## Merge the staes to the Reviews


#Get the states 
state_df = hotels_df.filter(['hotel_ID', 'State']).copy()

#Merge the states into each review, for grouping.
reviews_df = reviews_df.merge(state_df, on = 'hotel_ID')

#Get a sample, because you canot open the full dataset
sample = reviews_df.sample(n=100)

#%%

#Start looking at basic EDA

hotels_state_unique=hotels_df.groupby('State').nunique()
hotels_state_sum=hotels_df.groupby('State').sum()
hotels_state_mean=hotels_df.groupby('State').mean()


fig = plt.figure(figsize=(14, 6))
sns.barplot(x=hotels_state_unique.index, y=hotels_state_unique.hotel_ID, )
plt.title('Unique Hotels')
plt.show()


fig = plt.figure(figsize=(14, 6))
sns.barplot(x=hotels_state_sum.index, y=hotels_state_sum.Number_reviews, )
plt.title('Total number reviews')
plt.show()


fig = plt.figure(figsize=(14, 6))
sns.barplot(x=hotels_state_mean.index, y=hotels_state_mean.Review_rating_mean, )
plt.title('Mean review Value')
plt.show()

sns.barplot(x=hotels_state_mean.index, y=hotels_state_mean.Review_rating_std, )
plt.title('Mean std review Value')
plt.show()

#%%

#
#  A small readout of basic stats on the data
#  Useful for populating the data statistics table
#


print(f'\nNumber of unique hotels: {hotels_df.hotel_ID.nunique()}')
print()
print(f'Number of Reviews - Total: {reviews_df.shape[0]}')
print(f'Number of Reviews - Pre: {reviews_df[reviews_df.Stay_PrePandemic==True].shape[0]}')
print(f'Number of Reviews - Post: {reviews_df[reviews_df.Stay_PrePandemic==False].shape[0]}')
print()
print(f'Mean of Reviews - Total: {reviews_df.Review_rating.mean()}')
print(f'Mean of Reviews - Pre: {reviews_df[reviews_df.Stay_PrePandemic==True].Review_rating.mean()}')
print(f'Mean of Reviews - Post: {reviews_df[reviews_df.Stay_PrePandemic==False].Review_rating.mean()}')
print()
print(f'STD of Reviews - Total: {reviews_df.Review_rating.std()}')
print(f'STD of Reviews - Pre: {reviews_df[reviews_df.Stay_PrePandemic==True].Review_rating.std()}')
print(f'STD of Reviews - Post: {reviews_df[reviews_df.Stay_PrePandemic==False].Review_rating.std()}')

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
from matplotlib.lines import Line2D


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
ax_us.set_title(title_str, fontsize=15)


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
                      facecolor=hotel_color, edgecolor='white')
        else:
            ax_us.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor=hotel_color, edgecolor='white')
    except:
        #This may be a territory, or a state which has not stations(eg, RI)
        ax_us.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor='grey', edgecolor='white')
        print(f'{state_abbrev}: SKIPPED')
        pass
    





## Create a custom legend for the colours
#Put the colours here
custom_lines = [Line2D([0], [0], color='#581845', lw=0, marker='s', markersize=10),
                Line2D([0], [0], color='#900C3F', lw=0, marker='s', markersize=10),
                Line2D([0], [0], color='#C70039', lw=0, marker='s', markersize=10),
                Line2D([0], [0], color='#FF5733', lw=0, marker='s', markersize=10),
                Line2D([0], [0], color='Grey', lw=0, marker='s', markersize=10),]
#Display legend
ax_us.legend(custom_lines, ['0-50', '50-100', '100-150', '150+', 'Unknown'],
             loc='right', fontsize=12, frameon=False)

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
ax_us.set_title(title_str, fontsize=15)


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
                      facecolor=hotel_color, edgecolor='white')
        else:
            ax_us.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor=hotel_color, edgecolor='white')
    except:
        #This may be a territory, or a state which has not stations(eg, RI)
        ax_us.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor='grey', edgecolor='white')
        print(f'{state_abbrev}: SKIPPED')
        pass
    




## Create a custom legend for the colours
#Put the colours here
custom_lines = [Line2D([0], [0], color='#581845', lw=0, marker='s', markersize=10),
                Line2D([0], [0], color='#900C3F', lw=0, marker='s', markersize=10),
                Line2D([0], [0], color='#C70039', lw=0, marker='s', markersize=10),
                Line2D([0], [0], color='#FF5733', lw=0, marker='s', markersize=10),
                Line2D([0], [0], color='Grey', lw=0, marker='s', markersize=10),]
#Display legend
ax_us.legend(custom_lines, ['0-50k', '50-100k', '100-150k', '150k+', 'Unknown'],
             loc='right', fontsize=12, frameon=False)

plt.show()


#0:'#581845', 50000:'#900C3F', 100000:'#C70039',150000:'#FF5733'



#%%


x=reviews_df.groupby('State').mean()
state_mean_review = x.Review_rating.rename('Review_rating_mean')


x=reviews_df.groupby('State').std()
state_std_review = x.Review_rating.rename('Review_rating_std')


x=reviews_df.groupby('State').sem()
state_sem_review = x.Review_rating.rename('Review_rating_sem')




#%%

# Plot the mean and STD review rating by state


fig = plt.figure(figsize=(14, 6))
state_order =list(reviews_df.State.unique());state_order.sort()
sns.barplot(data = reviews_df, x='State', y='Review_rating',order=state_order, ci='sd')
plt.ylim(0,5)

plt.title('Mean Review Rating by State', fontsize=15)
plt.ylabel('Review Rating')
plt.show()


#%%

#Plot both pre and post pandemic, but looks pretty bad. Going with a delta instead.
fig = plt.figure(figsize=(14,6))
state_order =list(reviews_df.State.unique());state_order.sort()
sns.catplot(data = reviews_df, x='State', y='Review_rating', hue='Stay_PrePandemic',
            order=state_order, ci='sd', kind='bar',
            height = 5, aspect=3, palette=('muted'))
plt.ylim(0,5)
plt.title('Mean Review Rating by State', fontsize=15)
plt.show()
#%%

#Calculate the Delta in Reviews pre-and post pandemic
pandemic_review_mean=reviews_df.groupby(['State', 'Stay_PrePandemic']).mean().Review_rating
pandemic_review_mean=pandemic_review_mean.reset_index()
pandemic_review_mean=pandemic_review_mean.pivot(index='State', columns='Stay_PrePandemic')['Review_rating']
pandemic_review_mean["Change_in_Review"] = pandemic_review_mean[False] - pandemic_review_mean[True]



#Create the normalized gradient centered on the zero between the max and negative max
#Find the max distance from 0 

norm = Normalize(vmin=-1, vmax=1)
color_vals=[cm.jet(norm(val),) for val in pandemic_review_mean.Change_in_Review ]


#Add a column to the dataframe for the RGBA values we calculated
pandemic_review_mean['color'] = color_vals





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
title_str='Change in Average Review Score During the Pandemic\nMost Populous City in each State Scraped'
ax_us.set_title(title_str, fontsize=15)


#Loop through each state and paint the borders and facecolor according to the RGBA values we calculated
for astate in shpreader.Reader(states_shp).records():
    try:
        #Get hotel info and colour for the state
        state_abbrev = astate.attributes['postal']
        
        
        if state_abbrev == "AK":
            hotel_color = pandemic_review_mean.loc['AK','color']
            ax_ak.add_geometries([astate.geometry], ccrs.PlateCarree(),
                      facecolor=hotel_color, edgecolor='white')
        elif state_abbrev == "HI":
            hotel_color = pandemic_review_mean.loc['HI','color']
            ax_hi.add_geometries([astate.geometry], ccrs.PlateCarree(),
                      facecolor=hotel_color, edgecolor='white')
        else:
            hotel_color = pandemic_review_mean.loc[state_abbrev,'color']
            ax_us.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor=hotel_color, edgecolor='white')
    except:
        #This may be a territory, or a state which has not stations(eg, RI)
        ax_us.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor='grey', edgecolor='white')
        print(f'{state_abbrev}: SKIPPED')
        pass
    






#Add stand-alone colourbar to show the direction of the gradient
c_map_ax = fig.add_axes([0.91, 0.33, 0.01, 0.36])
c_map_ax.axes.get_xaxis().set_visible(False)
#c_map_ax.axes.get_yaxis().set_visible(False)
cbar = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=cm.jet, orientation = 'vertical', ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['-1 Star', '0', '+1 Star'])

plt.show()

