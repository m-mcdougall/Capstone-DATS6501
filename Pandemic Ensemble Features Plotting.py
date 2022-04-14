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


#features = pd.read_csv(wd+'\\Data\\Features\\Sentences\\feature_sentence_matrix.csv', index_col=0)
features = pd.read_csv(wd+'\\Data\\Features\\Sentences\\reorder_feature_sentence_matrix.csv', index_col=0)
#predictions = pd.read_csv(wd+'\\Data\\Features\\Results\\ensemble_predictions.csv',index_col=0)
predictions = pd.read_csv(wd+'\\Data\\Features\\Results\\pandemic_reordered_ensemble_predictions.csv',index_col=0,)


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



feature_val_pand_before = features[features.Pandemic_Timing == 'Before']
feature_val_pand_after = features[features.Pandemic_Timing == 'After']
feature_val_pand_after = feature_val_pand_after.groupby([ 'Feature','Sentiment'])['Prediction'].mean()
feature_val_pand_before = feature_val_pand_before.groupby([ 'Feature','Sentiment'])['Prediction'].mean()

feature_val_pand = feature_val_pand_before - feature_val_pand_after


#%%

#Changes during the pandemic, by state

feature_val_pand_state_after = features[features.Pandemic_Timing == 'After'].groupby([ 'Feature','State','Sentiment'])['Prediction'].mean()
feature_val_pand_state_before = features[features.Pandemic_Timing == 'Before'].groupby([ 'Feature','State','Sentiment'])['Prediction'].mean()



plot_pandemic_state= feature_val_pand_state_after-feature_val_pand_state_before
plot_pandemic_state=plot_pandemic_state.reset_index()

#%%

# Just start with the basic plot

all_features = feature_val_overall.reset_index()


tfidf = pd.read_csv(wd+'\\Data\\Tfidf\\All_reviews.csv', index_col=0)

tfidf = tfidf.iloc[:15, :]
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
plt.title('Mean Predicted Top 15 Tfidf Features Ratings')
plt.show()

#%%
 
#As above but with ci

all_features = features

tfidf = pd.read_csv(wd+'\\Data\\Tfidf\\All_reviews.csv', index_col=0)

tfidf = tfidf.iloc[:15, :]
tfidf = tfidf.rename({'Features':'Feature'}, axis=1)

filtered_caring = pd.merge(tfidf,all_features, on=['Feature'], how='left')

hue_order = ['NEGATIVE', 'POSITIVE', ]
hue_colors = ['#6065CC','#65CC60']

fig = plt.figure(figsize=(14, 6))
sns.catplot(data=filtered_caring, y='Prediction', x='Feature', 
            hue='Sentiment', hue_order=hue_order, palette=hue_colors,
            kind='bar', 
            ci=95, capsize=0.05,errwidth=1.5,
            height = 5, aspect=3,)


#plt.yticks(np.arange(0, 5)+1, 1.0)
plt.title('Top 15 Tfidf Features\nMean Predicted Ratings', fontsize=18)

plt.ylabel("Predicted Rating")
plt.ylim(1,5)
plt.yticks(np.arange(0, 5)+1, 1)

plt.show()

#%%

# Try to make a dot text plot for just the basic scores 

all_features = feature_val_overall.reset_index()


tfidf = pd.read_csv(wd+'\\Data\\Tfidf\\All_reviews.csv', index_col=0)

tfidf = tfidf.iloc[:100, :]
tfidf = tfidf.rename({'Features':'Feature'}, axis=1)
tfidf = tfidf.drop('TFIDF', axis=1)

filtered_caring = pd.merge(tfidf,all_features, on=['Feature'], how='left')
filtered_caring = filtered_caring.set_index(['Feature','Sentiment' ]).unstack().reset_index()
filtered_caring.columns = filtered_caring.columns.droplevel()
filtered_caring = filtered_caring.rename({'':'Feature'}, axis=1)


fig = plt.figure(figsize=(12, 6))

for word in filtered_caring.Feature.unique():
    filt_df=filtered_caring[filtered_caring.Feature == word]
    plt.scatter(filt_df['NEGATIVE'], filt_df['POSITIVE'], marker='.', color='red')
    plt.text(filt_df['NEGATIVE']+.005, filt_df['POSITIVE']-.005, word, fontsize=9)



plt.xlim(1, 5)
plt.ylim(1, 5)

plt.xlabel('Mean Predicted Rating in Negative Context')
plt.ylabel('Mean Predicted Rating in Positive Context')

plt.hlines(3, -1.5, 5.5)
plt.vlines(3, -1.5, 5.5)

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
plt.title('Changes in Top 10 Tfidf Features Ratings\nPre vs Pandemic Predicted Ratings')
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

tfidf = tfidf.iloc[:15, :]
tfidf = tfidf.rename({'Features':'Feature'}, axis=1)

filtered_caring = pd.merge(tfidf,plot_pandemic, on=['Feature'], how='left')

#filtered_caring =filtered_caring.sort_values(by='Prediction', axis=0, ascending=False)

hue_order = ['NEGATIVE', 'POSITIVE', ]
hue_colors = ['#6065CC','#65CC60']

fig = plt.figure(figsize=(14, 6))
sns.catplot(data=filtered_caring, y='Prediction', x='Feature', 
            hue='Sentiment', hue_order=hue_order, palette=hue_colors,
            kind='swarm', s=8,
            height = 5, aspect=2,)

plt.hlines(0,-100, 100, linestyles='dotted', colors='black', linewidths=1)

plt.title('Changes in Top 15 Tfidf Features Ratings\nPre vs Pandemic Predicted Ratings')
plt.ylabel("Change in Predicted Rating")
plt.ylim(-1,1)
plt.show()
#%%

#
#
#Trying to the the change in pandemic dot plot as overlapping bar plot
#
#
#


feature_val_pand_before2 = features[features.Pandemic_Timing == 'Before']
feature_val_pand_after2 = features[features.Pandemic_Timing == 'After']
#feature_val_pand_after = feature_val_pand_after.groupby([ 'Feature','Sentiment'])['Prediction'].mean()
#feature_val_pand_before = feature_val_pand_before.groupby([ 'Feature','Sentiment'])['Prediction'].mean()

feature_val_pand = feature_val_pand_before2.filter(['Feature','Sentiment'], axis=1)
feature_val_pand['Prediction'] = feature_val_pand_after2['Prediction'].array.astype(float) - feature_val_pand_before2['Prediction'].array.astype(float)

tfidf = pd.read_csv(wd+'\\Data\\Tfidf\\All_reviews.csv', index_col=0)

tfidf = tfidf.iloc[:15, :]
tfidf = tfidf.rename({'Features':'Feature'}, axis=1)

filtered_caring = pd.merge(tfidf,feature_val_pand, on=['Feature'], how='left')



hue_order = ['NEGATIVE', 'POSITIVE', ]
hue_colors = ['#6065CC','#65CC60']

fig = plt.figure(figsize=(14, 6))

pos_bars = filtered_caring[filtered_caring['Sentiment'] == 'POSITIVE']
neg_bars = filtered_caring[filtered_caring['Sentiment'] == 'NEGATIVE']

sns.barplot(data=pos_bars, y='Prediction', x='Feature', color ='#65CC60',
            ci=95,capsize=0.1,errwidth=1.8, errcolor = '#0F690B', label="POSITIVE" )
sns.barplot(data=neg_bars, y='Prediction', x='Feature',color ='#6065CC',
            ci=95,capsize=0.07,errwidth=1.8,errcolor = '#2F3264' ,label="NEGATIVE")

plt.hlines(0,-5, 20, linestyles='dotted', colors='black', linewidths=1)
plt.xlim(-1,15)
plt.title('Changes in Top 15 Tfidf Features Ratings\nPre vs Pandemic Predicted Ratings',fontsize=18)
plt.ylabel("Change in Predicted Rating")
plt.ylim(-1.5,1.5)

plt.legend(loc="upper left", title='Sentiment')

plt.show()




#%%

#
#
#  Change in pandemic dot plot as text swarm
#
#
#
feature_val_pand_before_plot = feature_val_pand_before.copy()
feature_val_pand_after_plot = feature_val_pand_after.copy()


plot_pandemic= feature_val_pand_after_plot-feature_val_pand_before_plot
plot_pandemic=plot_pandemic.reset_index()


tfidf = pd.read_csv(wd+'\\Data\\Tfidf\\All_reviews.csv', index_col=0)

tfidf = tfidf.iloc[:50, :]
tfidf = tfidf.rename({'Features':'Feature'}, axis=1)
tfidf = tfidf.drop('TFIDF', axis=1)

filtered_caring = pd.merge(tfidf,plot_pandemic, on=['Feature'], how='left')
filtered_caring = filtered_caring.set_index(['Feature','Sentiment' ]).unstack().reset_index()
filtered_caring.columns = filtered_caring.columns.droplevel()
filtered_caring = filtered_caring.rename({'':'Feature'}, axis=1)


fig = plt.figure(figsize=(15, 7))

for word in filtered_caring.Feature.unique():
    filt_df=filtered_caring[filtered_caring.Feature == word]
    plt.scatter(filt_df['NEGATIVE'], filt_df['POSITIVE'], marker='*', color='red')
    plt.text(filt_df['NEGATIVE']+.005, filt_df['POSITIVE']-.005, word, fontsize=9)


plt.xlabel("Negative Sentiment")
plt.ylabel("Positive Sentiment")

plt.xlim(-0.60, 0.55)
plt.ylim(-0.3,0.8)

plt.hlines(0, -1.5, 1.5, linestyles='dotted', colors='#6065CC', linewidths=1)
plt.vlines(0, -1.5, 1.5, linestyles='dotted', colors='#65CC60', linewidths=1)

plt.title('Changes in Top 50 Tfidf Features Ratings\nPre vs Pandemic Predicted Ratings',fontsize=18)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(-0.52, 0.75, 'More Polarized', fontsize=10, verticalalignment='top', bbox=props)
plt.text(-0.52, -0.23, 'More Negative', fontsize=10, verticalalignment='top', bbox=props)

plt.text(0.43, 0.75, 'More Positive', fontsize=10, verticalalignment='top', bbox=props)
plt.text(0.42, -0.23, 'Less Polarized', fontsize=10, verticalalignment='top', bbox=props)

plt.show()
#%%


#
#
#  Change in pandemic dot plot as text swarm - Ben Alternate
#
#   Looks terrible
#

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

tfidf = tfidf.iloc[:20, :]
tfidf = tfidf.rename({'Features':'Feature'}, axis=1)

filtered_caring = pd.merge(tfidf,plot_pandemic, on=['Feature'], how='left')
filtered_caring = filtered_caring.drop('TFIDF', axis=1)




sentiment = 'POSITIVE'
#sentiment = 'NEGATIVE'

filtered_caring_plot = filtered_caring[filtered_caring.Sentiment.str.contains(sentiment)]
filtered_caring_plot.Sentiment=filtered_caring_plot.Sentiment.str.replace('PRE_'+sentiment,'0')
filtered_caring_plot.Sentiment=filtered_caring_plot.Sentiment.str.replace('POST_'+sentiment,'1')
filtered_caring_plot.Sentiment=filtered_caring_plot.Sentiment.astype(int)




sns.lineplot(data=filtered_caring_plot, x='Sentiment', y='Prediction', hue='Feature')






#%%


fig = plt.figure(figsize=(14, 6))

for word in filtered_caring.Feature.unique():
    filt_df=filtered_caring[filtered_caring.Feature == word]
    plt.scatter(filt_df['NEGATIVE'], filt_df['POSITIVE'], marker='.', color='red')
    plt.text(filt_df['NEGATIVE']+.005, filt_df['POSITIVE']-.005, word, fontsize=9)


plt.xlabel("Change in Ratings: Negative Sentiment")
plt.ylabel("Change in Ratings: Positive Sentiment")

plt.xlim(-0.55, 0.55)
plt.ylim(-0.5,1)

plt.hlines(0, -1.5, 1.5, linestyles='dotted', colors='black', linewidths=1)
plt.vlines(0, -1.5, 1.5, linestyles='dotted', colors='black', linewidths=1)

plt.title('Changes in Top 50 Tfidf Features Ratings\nPre vs Pandemic Predicted Ratings',fontsize=18)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(-0.52, 0.9, 'More Polarized', fontsize=10, verticalalignment='top', bbox=props)
plt.text(-0.52, -0.4, 'More Negative', fontsize=10, verticalalignment='top', bbox=props)

plt.text(0.43, 0.9, 'More Positive', fontsize=10, verticalalignment='top', bbox=props)
plt.text(0.42, -0.4, 'Less Polarized', fontsize=10, verticalalignment='top', bbox=props)

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


focus_word = 'pool'#neg.Feature.iloc[0]


for sentiment in ['POSITIVE', 'NEGATIVE', ]:
    
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
    title_str='Mean Predicted Review Score per State\nTfidf Feature: '+focus_word.capitalize()+', '+sentiment.capitalize()+' Context'
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
    cbar.ax.set_yticklabels(['1 Stars', '3 Stars', '5 Stars'])
    
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

#focus_word = 'city'#neg.Feature.iloc[0]

for focus_word in neg.Feature.iloc[0:13]:
    
    
    for sentiment in ['POSITIVE', 'NEGATIVE', ]:
        
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
        title_str='Difference in Predicted Review Stars per State\nFeature: '+focus_word.upper()+" with a "+sentiment.upper()+' Context'
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

#%%

##
# 
# Look at walkablility
#
##

# Extract the walkability

# Now time to groupby for analysis
feature_val_walk=features.copy()
feature_val_walk['temp'] = feature_val_walk.Sentence.str.find('Walkability ')+12

feature_val_walk['Walk'] = feature_val_walk.Sentence.str.slice(37,39)
feature_val_walk['Walk'] = feature_val_walk['Walk'].str.strip()
feature_val_walk['Walk'] = feature_val_walk['Walk'].str.replace('\.','', regex=True)
feature_val_walk.head()
feature_val_walk.tail()

#%%

# Change over the pandemic
feature_val_walk_state = feature_val_walk.groupby(['State','Walk'])['Prediction'].mean()
feature_val_walk_state_sent = feature_val_walk.groupby(['State','Sentiment','Walk'])['Prediction'].mean()
feature_val_walk_state_sent=feature_val_walk_state_sent.reset_index()


feature_val_walk_pand_before = feature_val_walk[feature_val_walk.Pandemic_Timing == 'Before']
feature_val_walk_pand_after = feature_val_walk[feature_val_walk.Pandemic_Timing == 'After']
feature_val_walk_pand_after = feature_val_walk_pand_after.groupby([ 'Walk'])['Prediction'].mean()
feature_val_walk_pand_before = feature_val_walk_pand_before.groupby([ 'Walk'])['Prediction'].mean()
#feature_val_walk_pand_before = feature_val_walk_pand_before.groupby([ 'Walk','Sentiment'])['Prediction'].mean()


feature_walk_pand = feature_val_walk_pand_after - feature_val_walk_pand_before

#%%

# Change over the pandemic, by state

feature_val_walk_pand_before = feature_val_walk[feature_val_walk.Pandemic_Timing == 'Before']
feature_val_walk_pand_after = feature_val_walk[feature_val_walk.Pandemic_Timing == 'After']
feature_val_walk_pand_after = feature_val_walk_pand_after.groupby([ 'Walk','State'])['Prediction'].mean()
feature_val_walk_pand_before = feature_val_walk_pand_before.groupby([ 'Walk','State'])['Prediction'].mean()
#feature_val_walk_pand_before = feature_val_walk_pand_before.groupby([ 'Walk','Sentiment'])['Prediction'].mean()

feature_walk_state_pand = feature_val_walk_pand_after - feature_val_walk_pand_before

#%%



feature_val_walk_low = feature_val_walk[feature_val_walk.Walk == '0']
feature_val_walk_high = feature_val_walk[feature_val_walk.Walk == '3']


feature_val_walk_low = feature_val_walk_low.groupby([ 'State',])['Prediction'].mean()
feature_val_walk_high = feature_val_walk_high.groupby(['State',])['Prediction'].mean()

feature_val_walk_diff = feature_val_walk_high - feature_val_walk_low


#%%

##
# 
# There is no connection with walkability and regionality.
#
##

for walk_rating in ['0', '1','2','3']:
    
    #walk_rating = '3'#neg.Feature.iloc[0]
    #sentiment = 'NEGATIVE'
    sentiment = 'POSITIVE'
    
    #Show for only that walkability
    review_mean_walk = feature_val_walk_state_sent[(feature_val_walk_state_sent.Walk == walk_rating)]
    #review_mean_walk = feature_val_walk_state_sent[(feature_val_walk_state_sent.Sentiment == sentiment) &(feature_val_walk_state_sent.Walk == walk_rating)]

    #Show for all walkability, only that sentiment
    review_mean_all = feature_val_walk_state_sent
    #review_mean_all = feature_val_walk_state_sent[(feature_val_walk_state_sent.Sentiment == sentiment)]
    
    review_mean_walk=review_mean_walk.groupby('State').mean()
    review_mean_all=review_mean_all.groupby('State').mean()
    
    review_mean=review_mean_walk - review_mean_all
    
    
    #Create the normalized gradient centered on the zero between the max and negative max
    #Find the max distance from 0 
    
    norm = Normalize(vmin=-0.5, vmax=0.5)
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
    title_str='Difference in Predicted Review Stars from Mean per State\nWalkability: '+walk_rating+" with a "+sentiment.upper()+' Context'
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

#%%


train_set = pd.read_csv(wd+'\\Data\\Cleaned\\Split\\New_Samples_pandemic_reordered_3000_train.csv', index_col = 0)
x=train_set.sample(30)

# Extract the walkability

# Now time to groupby for analysis



train_set['Walk'] = train_set.Review_Body.str.slice(38,40)
train_set['Walk'] = train_set['Walk'].str.strip()
train_set['Walk'] = train_set['Walk'].str.replace('\.','', regex=True)

train_set['State'] = train_set.Review_Body.str.slice(5,8).str.strip()

train_set.head()
train_set.tail()


#%%

total_examples = train_set.groupby(['State'])['Walk'].value_counts()
total_examples.name = 'Walk_count'
total_examples = total_examples.reset_index()

collect = {}

for state in total_examples.State.unique():
    filt_df = total_examples[total_examples.State == state]
    #most_freq = total_examples.iloc[filt_df.Walk_count.idxmax(),:]['Walk']
    
    #most_freq = filt_df.Walk_count.max()/filt_df.Walk_count.sum()
    try:
        most_freq = filt_df[filt_df.Walk=='3']['Walk_count'].iloc[0]/filt_df.Walk_count.sum()
    except:
        most_freq=0.0
    collect[state] = float(most_freq)
    
total_freq_walk = pd.DataFrame.from_dict(collect, orient='index').reset_index() 
total_freq_walk = total_freq_walk.rename({'index':'State', 0:'Prediction'}, axis=1)

total_ratngs = train_set.groupby(['State'])['Review_rating'].mean()
total_ratngs = total_ratngs.reset_index() 
total_ratngs = total_ratngs.rename({'Review_rating':'Prediction'}, axis=1)


#Correlation between freq walkability and mean review
np.corrcoef(total_freq_walk.Prediction,total_ratngs.Prediction)

#Moderate degree of correlation between the frequency of high walkability and the mean review score 0.55788184

#%%

np.corrcoef(train_set.Walk.astype(float),train_set.Review_rating)

#Very low correlation betwwne the raw walk score and the rating  0.12247636



#%%

review_mean=total_freq_walk
review_mean=review_mean.set_index('State')

#Create the normalized gradient centered on the zero between the max and negative max
#Find the max distance from 0 

norm = Normalize(vmin=0, vmax=5)
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
#title_str='Difference in Predicted Review Stars per State\nWalkability: '+walk_rating+" with a "+sentiment.upper()+' Context'
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
            ax_us.text(x, y, round(review_mean.loc[state_abbrev,'Prediction'],2), color='White', size=11, ha='center', va='center', transform=ccrs.PlateCarree(), 
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

#%%


feature_val_walk_plain = feature_val_walk.groupby(['Sentiment','Walk'])['Prediction'].mean().reset_index()

sns.lineplot(data=feature_val_walk_plain, x='Walk', y='Prediction', hue='Sentiment')





#%%


#
#   Tryiing a wordcloud
#
#

from wordcloud import WordCloud

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

tfidf = tfidf.iloc[:20, :]
tfidf = tfidf.rename({'Features':'Feature'}, axis=1)

filtered_caring = pd.merge(tfidf,plot_pandemic, on=['Feature'], how='left')
filtered_caring = filtered_caring.drop('TFIDF', axis=1)



#%%

#
#
#  Change in pandemic dot plot as text swarm
#
#
#
feature_val_pand_before_plot = feature_val_pand_before.copy()
feature_val_pand_after_plot = feature_val_pand_after.copy()


plot_pandemic= feature_val_pand_after_plot-feature_val_pand_before_plot
plot_pandemic=plot_pandemic.reset_index()


all_tfidf=[]

for file in ['All_reviews.csv','PostPandemic_reviews.csv','PrePandemic_reviews.csv']:
    #Extract only the top 100 features for each list
    all_tfidf.append(pd.read_csv(wd+'\\Data\\Tfidf\\'+file, index_col=0).iloc[0:26, :])

all_tfidf = pd.concat(all_tfidf, ignore_index=True)

#Get the unique values, so we don't repeat
all_tfidf = all_tfidf.drop_duplicates(subset='Features')



tfidf = all_tfidf.rename({'Features':'Feature'}, axis=1)
tfidf = tfidf.drop('TFIDF', axis=1)

filtered_caring = pd.merge(tfidf,plot_pandemic, on=['Feature'], how='left')
filtered_caring = filtered_caring.set_index(['Feature','Sentiment' ]).unstack().reset_index()
filtered_caring.columns = filtered_caring.columns.droplevel()
filtered_caring = filtered_caring.rename({'':'Feature'}, axis=1)


top_15 = all_tfidf.iloc[:15,:].Features.to_list()
add_words = [ 'bed', 'comfortable', 'restaurant', 'helpful', 'area', 'desk', 'front desk',\
 'close', 'pool', 'food', 'view', 'parking', 'experience', 'bar', 'floor', 'everything', 'great location',\
 'bathroom', 'recommend', 'free']
#[top_15.append(word) for word in add_words]

fig = plt.figure(figsize=(15, 7))

for word in filtered_caring.Feature.unique():
    filt_df=filtered_caring[filtered_caring.Feature == word]
    plt.scatter(filt_df['NEGATIVE'], filt_df['POSITIVE'], marker='*', color='red')
    #if word in top_15:
    plt.text(filt_df['NEGATIVE']+.005, filt_df['POSITIVE']-.005, word, fontsize=9)


plt.xlabel("Negative Sentiment")
plt.ylabel("Positive Sentiment")

plt.xlim(-0.80, 0.65)
plt.ylim(-0.3,0.8)

plt.hlines(0, -1.5, 1.5, linestyles='dotted', colors='#6065CC', linewidths=1)
plt.vlines(0, -1.5, 1.5, linestyles='dotted', colors='#65CC60', linewidths=1)

plt.title('Changes in Top 50 Tfidf Features Ratings\nPre vs Pandemic Predicted Ratings',fontsize=18)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(-0.52, 0.75, 'More Polarized', fontsize=10, verticalalignment='top', bbox=props)
plt.text(-0.52, -0.23, 'More Negative', fontsize=10, verticalalignment='top', bbox=props)

plt.text(0.43, 0.75, 'More Positive', fontsize=10, verticalalignment='top', bbox=props)
plt.text(0.42, -0.23, 'Less Polarized', fontsize=10, verticalalignment='top', bbox=props)

plt.show()