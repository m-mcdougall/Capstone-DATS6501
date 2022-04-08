# -*- coding: utf-8 -*-

#Non-Specific Imports
import os
import re
import pandas as pd
import numpy as np
import time
import concurrent.futures as cf
from tqdm import tqdm
import math


#Plotting Imports
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns


#NLP Imports
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


#Modeling Imports
import torch
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW


#Cartopy Imports
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.patheffects as PathEffects
from matplotlib.colors import Normalize
import matplotlib
from matplotlib import cm
from matplotlib.lines import Line2D

#Flatten list utility function
def flatten_list(list_in):
    return [item for sublist in list_in for item in sublist]


#Set the working directory
wd=os.path.abspath('C://Users//Mariko//Documents//GitHub//Capstone-DATS6501')
os.chdir(wd)

#%%
##############
#
#  Load in the features and predictions
#
##############


features = pd.read_csv(wd+'\\Data\\Features\\Sentences\\feature_sentence_matrix.csv', index_col=0)
predictions = pd.read_csv(wd+'\\Data\\Features\\Results\\ensemble_predictions.csv',index_col=0)


#Join features and predictions
features = features.join(predictions)
features = features.rename({'Prediction_Ensemble':"Prediction"}, axis =1)
features.head(10)

features.Prediction = features.Prediction+1

del predictions
#%%


# Now time to groupby for analysis
feature_val_overall=features.groupby(['Feature', 'Sentiment'])['Prediction'].mean()

feature_val_state = features.groupby(['Feature','State', 'Sentiment'])['Prediction'].mean()
feature_val_pand = features.groupby([ 'Feature','Pandemic_Timing','Sentiment'])['Prediction'].mean()


#%%

#Changes in Value

#Show the change from the mean for each state
delta_state = feature_val_state - feature_val_overall
#delta_pand = feature_val_pand - feature_val_overall



feature_val_pand_before = features[features.Pandemic_Timing == 'before']
feature_val_pand_after = features[features.Pandemic_Timing == 'after']
feature_val_pand_after = feature_val_pand_after.groupby([ 'Feature','Sentiment'])['Prediction'].mean()
feature_val_pand_before = feature_val_pand_before.groupby([ 'Feature','Sentiment'])['Prediction'].mean()

feature_val_pand = feature_val_pand_before - feature_val_pand_after


#%%

#Changes during the pandemic, by state

feature_val_pand_state_after = features[features.Pandemic_Timing == 'after'].groupby([ 'Feature','State','Sentiment'])['Prediction'].mean()
feature_val_pand_state_before = features[features.Pandemic_Timing == 'before'].groupby([ 'Feature','State','Sentiment'])['Prediction'].mean()



plot_pandemic_state= feature_val_pand_state_after-feature_val_pand_state_before
plot_pandemic_state=plot_pandemic_state.reset_index()

#%%

# Just start with the basic plot

all_features = feature_val_overall.reset_index()


tfidf = pd.read_csv(wd+'\\Data\\Tfidf\\All_reviews.csv', index_col=0)

tfidf = tfidf.iloc[:10, :]
tfidf = tfidf.rename({'Features':'Feature'}, axis=1)

filtered_caring = pd.merge(tfidf,all_features, on=['Feature'], how='left')

hue_order = ['NEGATIVE', 'POSITIVE', ]
hue_colors = ['#6065CC','#65CC60']

fig = plt.figure(figsize=(14, 6))
sns.catplot(data=filtered_caring, y='Prediction', x='Feature', 
            hue='Sentiment', hue_order=hue_order, palette=hue_colors,
            kind='bar',
            height = 5, aspect=3,)



plt.ylabel("Predicted Rating")
plt.ylim(1,5)
plt.show()

#%%

##
# 
# This plot does the top 10 tfidf features as paired barplots
#
##


feature_val_pand_before_plot = feature_val_pand_before.reset_index().copy()
feature_val_pand_after_plot = feature_val_pand_after.reset_index().copy()


def pre_post_replace(df_in, pre_post):
    
    df_in.Sentiment = df_in.Sentiment.str.replace('POSITIVE',pre_post.upper()+'_POSITIVE')
    df_in.Sentiment = df_in.Sentiment.str.replace('NEGATIVE',pre_post.upper()+'_NEGATIVE')

    return df_in


feature_val_pand_before_plot = pre_post_replace(feature_val_pand_before_plot, 'pre')
feature_val_pand_after_plot = pre_post_replace(feature_val_pand_after_plot, 'post')

plot_pandemic=pd.concat([feature_val_pand_before_plot,feature_val_pand_after_plot], axis=0 )



tfidf = pd.read_csv(wd+'\\Data\\Tfidf\\All_reviews.csv', index_col=0)

tfidf = tfidf.iloc[:10, :]
tfidf = tfidf.rename({'Features':'Feature'}, axis=1)

filtered_caring = pd.merge(tfidf,plot_pandemic, on=['Feature'], how='left')


hue_order = ['PRE_NEGATIVE', 'POST_NEGATIVE', 'PRE_POSITIVE', 'POST_POSITIVE']
hue_colors = ['#6065CC',       '#A1A3E0',       '#65CC60',       '#A3E0A1',]

fig = plt.figure(figsize=(14, 6))
sns.catplot(data=filtered_caring, y='Prediction', x='Feature', 
            hue='Sentiment', hue_order=hue_order, palette=hue_colors,
            kind='bar',
            height = 5, aspect=3,)

plt.ylabel("Predicted Rating")
plt.ylim(1,5)
plt.show()





#%%

##
# 
# This plot does the top 25 as a change - dots
#
##

feature_val_pand_before_plot = feature_val_pand_before.copy()
feature_val_pand_after_plot = feature_val_pand_after.copy()


plot_pandemic= feature_val_pand_after_plot-feature_val_pand_before_plot
plot_pandemic=plot_pandemic.reset_index()


tfidf = pd.read_csv(wd+'\\Data\\Tfidf\\All_reviews.csv', index_col=0)

tfidf = tfidf.iloc[:25, :]
tfidf = tfidf.rename({'Features':'Feature'}, axis=1)

filtered_caring = pd.merge(tfidf,plot_pandemic, on=['Feature'], how='left')

#filtered_caring =filtered_caring.sort_values(by='Prediction', axis=0, ascending=False)

hue_order = ['NEGATIVE', 'POSITIVE', ]
hue_colors = ['#6065CC','#65CC60']

fig = plt.figure(figsize=(14, 6))
sns.catplot(data=filtered_caring, y='Prediction', x='Feature', 
            hue='Sentiment', hue_order=hue_order, palette=hue_colors,
            kind='swarm', s=8,
            height = 5, aspect=3,)

plt.hlines(0,-100, 100, linestyles='dotted', colors='black', linewidths=1)


plt.ylabel("Change in Predicted Rating")
plt.ylim(-1,1)
plt.show()



#%%
##
# 
# Show the top 10 most changed features - positive and negative
#
##

feature_val_pand_before_plot = feature_val_pand_before.copy()
feature_val_pand_after_plot = feature_val_pand_after.copy()


plot_pandemic= feature_val_pand_after_plot-feature_val_pand_before_plot
plot_pandemic=plot_pandemic.reset_index()


filtered_caring =plot_pandemic.sort_values(by='Prediction', axis=0, ascending=False).reset_index(drop=True)
filtered_caring =filtered_caring.iloc[:10,:]

hue_order = ['NEGATIVE', 'POSITIVE', ]
hue_colors = ['#6065CC','#65CC60']

fig = plt.figure(figsize=(14, 6))
sns.catplot(data=filtered_caring, y='Prediction', x='Feature', 
            hue='Sentiment', hue_order=hue_order, palette=hue_colors,
            kind='swarm', s=8,
            height = 5, aspect=1.5,)

plt.hlines(0,-100, 100, linestyles='dotted', colors='black', linewidths=1)


plt.ylabel("Change in Predicted Rating")
plt.ylim(-1,1)
plt.show()



filtered_caring =plot_pandemic.sort_values(by='Prediction', axis=0, ascending=True).reset_index(drop=True)
filtered_caring =filtered_caring.iloc[:10,:]

hue_order = ['NEGATIVE', 'POSITIVE', ]
hue_colors = ['#6065CC','#65CC60']

fig = plt.figure(figsize=(14, 6))
sns.catplot(data=filtered_caring, y='Prediction', x='Feature', 
            hue='Sentiment', hue_order=hue_order, palette=hue_colors,
            kind='swarm', s=8,
            height = 5, aspect=1.5,)

plt.hlines(0,-100, 100, linestyles='dotted', colors='black', linewidths=1)


plt.ylabel("Change in Predicted Rating")
plt.ylim(-1,1)
plt.show()


#%%

##
# 
# Look at the most difference by state
#
##

feature_val_state_plot =  feature_val_state.reset_index()
feature_val_state_plot.head(20)


pos = {}
neg = {}

for feature in feature_val_state_plot.Feature.unique():
    filt_df = feature_val_state_plot[feature_val_state_plot.Feature == feature]
    
    #Filters for positive
    diff = filt_df[filt_df.Sentiment == 'POSITIVE'].Prediction.max() - filt_df[filt_df.Sentiment == 'POSITIVE'].Prediction.min()
    pos[feature] = diff
    
    #Filters for negative
    diff = filt_df[filt_df.Sentiment == 'NEGATIVE'].Prediction.max() - filt_df[filt_df.Sentiment == 'NEGATIVE'].Prediction.min()
    neg[feature] = diff
    
pos = pd.DataFrame.from_dict(pos, orient='index').reset_index()
pos = pos.rename({'index':'Feature', 0:'Difference'}, axis=1)
pos = pos.sort_values(by='Difference', axis=0, ascending=False).reset_index(drop=True)

neg = pd.DataFrame.from_dict(neg, orient='index').reset_index()
neg = neg.rename({'index':'Feature', 0:'Difference'}, axis=1)
neg = neg.sort_values(by='Difference', axis=0, ascending=False).reset_index(drop=True)
#%%



focus_word = 'new'#neg.Feature.iloc[0]
sentiment = 'NEGATIVE'
#sentiment = 'POSITIVE'

review_mean = feature_val_state_plot[(feature_val_state_plot.Sentiment == sentiment) &(feature_val_state_plot.Feature == focus_word)]

review_mean=review_mean.groupby('State').mean()


#Create the normalized gradient centered on the zero between the max and negative max
#Find the max distance from 0 

norm = Normalize(vmin=1, vmax=5)
color_vals=[cm.jet_r(norm(val),) for val in review_mean.Prediction ]


#Add a column to the dataframe for the RGBA values we calculated
review_mean['color'] = color_vals





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
title_str='Average Review Score Scraped per State\nMost Populous City in each State Scraped'
ax_us.set_title(title_str, fontsize=15)


#Loop through each state and paint the borders and facecolor according to the RGBA values we calculated
for astate in shpreader.Reader(states_shp).records():
    try:
        #Get hotel info and colour for the state
        state_abbrev = astate.attributes['postal']
        
        
        if state_abbrev == "AK":
            hotel_color = review_mean.loc['AK','color']
            ax_ak.add_geometries([astate.geometry], ccrs.PlateCarree(),
                      facecolor=hotel_color, edgecolor='grey')
            ax_ak.background_patch.set_visible(False)
            x = astate.geometry.centroid.x        
            y = astate.geometry.centroid.y
            ax_ak.text(x+2.4, y, round(review_mean.loc['AK','Prediction'],1), color='White', size=11, ha='center', va='center', transform=ccrs.PlateCarree(), 
                    path_effects=[PathEffects.withStroke(linewidth=3, foreground="k", alpha=.8)])
            
        elif state_abbrev == "HI":
            hotel_color = review_mean.loc['HI','color']
            ax_hi.background_patch.set_visible(False)
            ax_hi.add_geometries([astate.geometry], ccrs.PlateCarree(),
                      facecolor=hotel_color, edgecolor='grey')
            x = astate.geometry.centroid.x        
            y = astate.geometry.centroid.y
            ax_hi.text(x+2.4, y, round(review_mean.loc['HI','Prediction'],1), color='White', size=11, ha='center', va='center', transform=ccrs.PlateCarree(), 
                    path_effects=[PathEffects.withStroke(linewidth=3, foreground="k", alpha=.8)])
        else:
            hotel_color = review_mean.loc[state_abbrev,'color']
            ax_us.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor=hotel_color, edgecolor='grey')
            x = astate.geometry.centroid.x        
            y = astate.geometry.centroid.y
            ax_us.text(x, y, round(review_mean.loc[state_abbrev,'Prediction'],1), color='White', size=11, ha='center', va='center', transform=ccrs.PlateCarree(), 
                    path_effects=[PathEffects.withStroke(linewidth=3, foreground="k", alpha=.8)])
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
cbar = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=cm.jet_r, orientation = 'vertical', ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['3 Stars', '4 Stars', '5 Stars'])

plt.show()



#%%
##
# 
# Look at the most difference by state, pandemic
#
##

feature_val_state_plot =  plot_pandemic_state.reset_index()
feature_val_state_plot.head(20)


pos = {}
neg = {}

for feature in feature_val_state_plot.Feature.unique():
    filt_df = feature_val_state_plot[feature_val_state_plot.Feature == feature]
    
    #Filters for positive
    diff = filt_df[filt_df.Sentiment == 'POSITIVE'].Prediction.max() - filt_df[filt_df.Sentiment == 'POSITIVE'].Prediction.min()
    pos[feature] = diff
    
    #Filters for negative
    diff = filt_df[filt_df.Sentiment == 'NEGATIVE'].Prediction.max() - filt_df[filt_df.Sentiment == 'NEGATIVE'].Prediction.min()
    neg[feature] = diff
    
pos = pd.DataFrame.from_dict(pos, orient='index').reset_index()
pos = pos.rename({'index':'Feature', 0:'Difference'}, axis=1)
pos = pos.sort_values(by='Difference', axis=0, ascending=False).reset_index(drop=True)

neg = pd.DataFrame.from_dict(neg, orient='index').reset_index()
neg = neg.rename({'index':'Feature', 0:'Difference'}, axis=1)
neg = neg.sort_values(by='Difference', axis=0, ascending=False).reset_index(drop=True)



#%%

focus_word = 'staff'#neg.Feature.iloc[0]
sentiment = 'NEGATIVE'
sentiment = 'POSITIVE'

review_mean = feature_val_state_plot[(feature_val_state_plot.Sentiment == sentiment) &(feature_val_state_plot.Feature == focus_word)]

review_mean=review_mean.groupby('State').mean()


#Create the normalized gradient centered on the zero between the max and negative max
#Find the max distance from 0 

norm = Normalize(vmin=-1, vmax=1)
color_vals=[cm.jet_r(norm(val),) for val in review_mean.Prediction ]


#Add a column to the dataframe for the RGBA values we calculated
review_mean['color'] = color_vals





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
title_str='Average Review Score Scraped per State\nMost Populous City in each State Scraped'
ax_us.set_title(title_str, fontsize=15)


#Loop through each state and paint the borders and facecolor according to the RGBA values we calculated
for astate in shpreader.Reader(states_shp).records():
    try:
        #Get hotel info and colour for the state
        state_abbrev = astate.attributes['postal']
        
        
        if state_abbrev == "AK":
            hotel_color = review_mean.loc['AK','color']
            ax_ak.add_geometries([astate.geometry], ccrs.PlateCarree(),
                      facecolor=hotel_color, edgecolor='grey')
            ax_ak.background_patch.set_visible(False)
            x = astate.geometry.centroid.x        
            y = astate.geometry.centroid.y
            ax_ak.text(x+2.4, y, round(review_mean.loc['AK','Prediction'],1), color='White', size=11, ha='center', va='center', transform=ccrs.PlateCarree(), 
                    path_effects=[PathEffects.withStroke(linewidth=3, foreground="k", alpha=.8)])
            
        elif state_abbrev == "HI":
            hotel_color = review_mean.loc['HI','color']
            ax_hi.background_patch.set_visible(False)
            ax_hi.add_geometries([astate.geometry], ccrs.PlateCarree(),
                      facecolor=hotel_color, edgecolor='grey')
            x = astate.geometry.centroid.x        
            y = astate.geometry.centroid.y
            ax_hi.text(x+2.4, y, round(review_mean.loc['HI','Prediction'],1), color='White', size=11, ha='center', va='center', transform=ccrs.PlateCarree(), 
                    path_effects=[PathEffects.withStroke(linewidth=3, foreground="k", alpha=.8)])
        else:
            hotel_color = review_mean.loc[state_abbrev,'color']
            ax_us.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor=hotel_color, edgecolor='grey')
            x = astate.geometry.centroid.x        
            y = astate.geometry.centroid.y
            ax_us.text(x, y, round(review_mean.loc[state_abbrev,'Prediction'],1), color='White', size=11, ha='center', va='center', transform=ccrs.PlateCarree(), 
                    path_effects=[PathEffects.withStroke(linewidth=3, foreground="k", alpha=.8)])
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
cbar = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=cm.jet_r, orientation = 'vertical', ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['-1 Stars', ' 0', '+1 Stars'])

plt.show()


