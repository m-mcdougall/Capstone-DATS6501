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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


#Flatten list utility function
def flatten_list(list_in):
    return [item for sublist in list_in for item in sublist]


#Set the working directory
wd=os.path.abspath('C://Users//Mariko//Documents//GitHub//Capstone-DATS6501')
os.chdir(wd)

#%%

# Import the city's Hotels and Reviews


###
# Load in all Hotels
###

#Loop through all cities - Collect all Hotels
all_files = os.listdir(wd+'\\Data\\Cleaned\\')
all_cities = [file for file in all_files if '.pkl' in file and 'Hotels' in file]

#Start with just one
#all_cities = [all_cities[1]]

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


###
# Load in all Reviews
###

#Loop through all cities - Collect all Hotels
all_files = os.listdir(wd+'\\Data\\Cleaned\\')
all_cities = [file for file in all_files if '.pkl' in file and 'Reviews' in file]

#Start with just one
#all_cities = [all_cities[1]]

hotels = []
for city in tqdm(all_cities):
    hotels_file =  pd.read_pickle(wd+'\\Data\\Cleaned\\'+city)
    
    #Drop some columns to make the dataframe lighter
    hotels_file.drop(['Review_date', 'Reviewer_loc','Review_stay_date', 'tokens','tokens_joined',
                       'Review_Year', 'Review_Month','Stay_Year', 'Stay_Month',
                       #'Review_title', 'Review_text'
                       ], 
                     axis=1, inplace=True)
    
    #Merge the title and body, then drop the individuals
    hotels_file['Review_Body'] = hotels_file['Review_title']+'. '+hotels_file['Review_text'] 
    hotels_file.drop(['Review_title', 'Review_text'], 
                     axis=1, inplace=True)    
    
    #Add to the list
    hotels.append(hotels_file)


#Merge into one dataframe
reviews_df = pd.concat(hotels)
reviews_df.rename({'Hotel_ID':'hotel_ID'}, axis=1, inplace=True)

#Delete intermediaries to free up memory
del all_files, all_cities, city, hotels, hotels_file


## Merge the states to the Reviews


#Get the states 
state_df = hotels_df.filter(['hotel_ID', 'State', 'hotel_location_walk']).copy()

#Merge the states into each review, for grouping.
reviews_df = reviews_df.merge(state_df, on = 'hotel_ID')

#Drop the hotel ID - not needed
reviews_df.drop(['hotel_ID'],axis=1, inplace=True)  

demo = reviews_df.sample(n=2000, random_state=42)
print(demo.head())

#%%



################################
#
#  Subsample the data 
#  Equal samples from each state
#  Sample size: 1000
#
################################

collect_sampler = []
all_methods_mse_collector = {}
all_methods_acc_collector = {}

for state in tqdm(reviews_df.State.unique()):
    
    
    #subset_state = (reviews_df[reviews_df.State == state]).sample(n=1000, replace=False, random_state=42)
    subset_state = (reviews_df[reviews_df.State == state])
    subset_state.groupby('Grade', group_keys=False).apply(lambda x: x.sample(1500, replace=False, random_state=42))
    collect_sampler.append(subset_state)



collect_sampler = pd.concat(collect_sampler)
collect_sampler.reset_index(drop=True, inplace=True)

#Adjust so the ratings start at 0
collect_sampler['Review_rating'] = collect_sampler['Review_rating'] -1

#Split into train, test and validation
train, test = train_test_split(collect_sampler, test_size=0.2, random_state=42)
#%%



## Check how it would look if only predict 4

train.Review_rating.plot.hist()




# Calculate the F1 score.
f1 = accuracy_score(y_true=test["Review_rating"], y_pred=[4]*test.shape[0])
mse_score = mean_squared_error(y_true=test["Review_rating"], y_pred=[4]*test.shape[0])

print('Accurancy if you always guess the most frequent rating (4)')
print('Accuracy: %.4f' % f1)
print('MSE: %.4f' % mse_score)

all_methods_mse_collector['Guess_Most_Freq'] = mse_score
all_methods_acc_collector['Guess_Most_Freq'] = f1

#Accuracy: 0.4879
#MSE: 2.4058
#%%


def add_walkbad(df_in):
    def walkability_str_gen(num):
        if num <= 50:
            
            if num > 25:
                walk_str= 'Not walkable. '
            else:
                walk_str= 'Not walkable at all. '
                
        else: #Num score 50+
            if num > 75: #most walkable
                walk_str= 'Very walkable. '
            else:
                walk_str= 'Fairly walkable. '
        return walk_str
    
    df_in = df_in.copy()
    #Filter for review_prepandemic col
    
    #Create additional text to add to the review
    new_text = df_in.hotel_location_walk.map(walkability_str_gen)

    df_in.Review_Body = new_text + df_in.Review_Body 
    
    return df_in

def add_statebad(df_in):
    
    df_in = df_in.copy()
    
    #Create additional text to add to the review
    new_text = 'This hotel is in '+df_in.State+'. '

    df_in.Review_Body = new_text + df_in.Review_Body 
    
    return df_in


def add_pandemicbad(df_in):
    
    df_in = df_in.copy()
    #Filter for review_prepandemic col
    boolean_filter = {True:'before', False:'after'}
    
    #Create additional text to add to the review
    new_text = 'I stayed '+ df_in.Review_PrePandemic.map(boolean_filter) + ' the pandemic'

    df_in.Review_Body = new_text + df_in.Review_Body 
    
    return df_in

#%%

## Version 2

def add_walk(df_in):
    def walkability_str_gen(num):
        if num <= 50:
            
            if num > 25:
                walk_str= 'Walkability 0. '
            else:
                walk_str= 'Walkability 1. '
                
        else: #Num score 50+
            if num > 75: #most walkable
                walk_str= 'Walkability 3. '
            else:
                walk_str= 'Walkability 2. '
        return walk_str
    
    df_in = df_in.copy()
    #Filter for review_prepandemic col
    
    #Create additional text to add to the review
    new_text = df_in.hotel_location_walk.map(walkability_str_gen)

    df_in.Review_Body = new_text + df_in.Review_Body 
    
    return df_in

def add_state(df_in):
    
    df_in = df_in.copy()
    
    #Create additional text to add to the review
    new_text = 'State '+df_in.State+'. '

    df_in.Review_Body = new_text + df_in.Review_Body 
    
    return df_in


def add_pandemic(df_in):
    
    df_in = df_in.copy()
    #Filter for review_prepandemic col
    boolean_filter = {True:'Before', False:'After'}
    
    #Create additional text to add to the review
    new_text = df_in.Review_PrePandemic.map(boolean_filter) + ' pandemic. '

    df_in.Review_Body = new_text + df_in.Review_Body 
    
    return df_in


#%%

def save_samples(baseline_save, train_in, test_in):
    train_in=train_in.filter(['Review_Body', 'Review_rating'], axis=1)
    train_in.to_csv(wd+'\\Data\\Cleaned\\Split\\New_baselines\\new_min'+baseline_save+'_train.csv')
                    
    test_in=test_in.filter(['Review_Body', 'Review_rating'], axis=1)
    test_in.to_csv(wd+'\\Data\\Cleaned\\Split\\New_baselines\\new_min'+baseline_save+'_test.csv')
    

baseline = 'review_only'
save_samples(baseline, train, test)


#Walk only
baseline = 'review_walk'
train_mod = train.copy()
test_mod = test.copy()

train_mod = add_walk(train_mod)
test_mod = add_walk(test_mod)

save_samples(baseline, train_mod, test_mod)


#Pandemic only
baseline = 'review_pand'
train_mod = train.copy()
test_mod = test.copy()

train_mod = add_pandemic(train_mod)
test_mod = add_pandemic(test_mod)

save_samples(baseline, train_mod, test_mod)


#State only
baseline = 'review_state'
train_mod = train.copy()
test_mod = test.copy()

train_mod = add_state(train_mod)
test_mod = add_state(test_mod)

save_samples(baseline, train_mod, test_mod)


#Walk+State only
baseline = 'review_walk_state'
train_mod = train.copy()
test_mod = test.copy()

train_mod = add_walk(train_mod)
test_mod = add_walk(test_mod)

train_mod = add_state(train_mod)
test_mod = add_state(test_mod)

save_samples(baseline, train_mod, test_mod)


#Walk+Pandemic only
baseline = 'review_walk_pand'
train_mod = train.copy()
test_mod = test.copy()

train_mod = add_walk(train_mod)
test_mod = add_walk(test_mod)

train_mod = add_pandemic(train_mod)
test_mod = add_pandemic(test_mod)

save_samples(baseline, train_mod, test_mod)




#Pandemic+State
baseline = 'review_pand_state'
train_mod = train.copy()
test_mod = test.copy()

train_mod = add_pandemic(train_mod)
test_mod = add_pandemic(test_mod)

train_mod = add_state(train_mod)
test_mod = add_state(test_mod)

save_samples(baseline, train_mod, test_mod)



#Walk only
baseline = 'review_walk_pand_state'
train_mod = train.copy()
test_mod = test.copy()

train_mod = add_walk(train_mod)
test_mod = add_walk(test_mod)

train_mod = add_pandemic(train_mod)
test_mod = add_pandemic(test_mod)

train_mod = add_state(train_mod)
test_mod = add_state(test_mod)

save_samples(baseline, train_mod, test_mod)


#%%



################################
#
#  RUN THIS TO GENERATE THE ACTUAL TRAINING/TEST/VAL SET
#  Subsample the data 
#  Equal samples from each state
#  Sample size: 1000
#
################################




collect_sampler = []
all_methods_mse_collector = {}
all_methods_acc_collector = {}

for state in tqdm(reviews_df.State.unique()):
    
    
    #subset_state = (reviews_df[reviews_df.State == state]).sample(n=1000, replace=False, random_state=42)
    subset_state = (reviews_df[reviews_df.State == state])
    try:
        pre = subset_state[subset_state.Review_PrePandemic == True].sample(1500, replace=False, random_state=42)
        post = subset_state[subset_state.Review_PrePandemic == False].sample(1500, replace=False, random_state=42)
        #print(f'{state}: Post is only {post.shape[0]}')
        collect_sampler.append(pre)
        collect_sampler.append(post)
    except:
        pre = subset_state[subset_state.Review_PrePandemic == True].sample(1500, replace=False, random_state=42)
        post = subset_state[subset_state.Review_PrePandemic == False]
        print(f'{state}: Post is only {post.shape[0]}')
        collect_sampler.append(pre)
        collect_sampler.append(post)




collect_sampler = pd.concat(collect_sampler)
collect_sampler.reset_index(drop=True, inplace=True)

#Adjust so the ratings start at 0
collect_sampler['Review_rating'] = collect_sampler['Review_rating'] -1

#Split into train, test and validation
train, test = train_test_split(collect_sampler, test_size=0.4, stratify=collect_sampler['Review_rating'], random_state=42)
validation, test = train_test_split(test, test_size=0.5, stratify=test['Review_rating'], random_state=42)

#%%


def add_all_features(df_in, save_name=''):

    df_in = df_in.copy()
    
    df_in = add_walk(df_in)
    df_in = add_pandemic(df_in)
    df_in = add_state(df_in)

    df_in=df_in.filter(['Review_Body', 'Review_rating'], axis=1)
    df_in.to_csv(wd+'\\Data\\Cleaned\\Split\\New_Samples_pandemic_reordered_3000_'+save_name+'.csv')
    
    return df_in


train_done = add_all_features(train, save_name='train')
test_done = add_all_features(test, save_name='test')
validation_done = add_all_features(validation, save_name='validation')


