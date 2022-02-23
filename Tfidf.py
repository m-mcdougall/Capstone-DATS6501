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
sample = reviews_df.sample(n=2000)


#%%


from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(ngram_range=(1,2),token_pattern = r'[\w\']+', 
                             max_features=10000, strip_accents='ascii')

vectorized_features = vectorizer.fit_transform(sample.tokens_joined)


feature_names = np.array(vectorizer.get_feature_names())
feature_tfidf = np.asarray(vectorized_features.sum(axis=0)).ravel()
features_df = pd.DataFrame([feature_names, feature_tfidf]).T
features_df = features_df.rename(columns={0:"Features", 1:'TFIDF'})
features_df = features_df.sort_values(by='TFIDF', ascending=False).reset_index(drop=True)


#%%


def get_features(df_in):
    
    vectorizer = TfidfVectorizer(ngram_range=(1,2),token_pattern = r'[\w\']+', 
                             max_features=10000, strip_accents='ascii', 
                             stop_words=['u', 'h', "i've", "i'm", 'm', 'd', "i'd"])

    vectorized_features = vectorizer.fit_transform(df_in.tokens_joined)
    
    
    feature_names = np.array(vectorizer.get_feature_names())
    feature_tfidf = np.asarray(vectorized_features.sum(axis=0)).ravel()
    features_df = pd.DataFrame([feature_names, feature_tfidf]).T
    features_df = features_df.rename(columns={0:"Features", 1:'TFIDF'})
    features_df = features_df.sort_values(by='TFIDF', ascending=False).reset_index(drop=True)
    
    return features_df

#%%

# Get features for the sample

sample_features = get_features(sample)
sample_features.to_csv(wd+'\\Data\\Tfidf\\Sample_reviews.csv')

#%%

# All reviews - Warning, takes a long time to run

all_features = get_features(reviews_df)
all_features.to_csv(wd+'\\Data\\Tfidf\\All_reviews.csv')

print("Full Set done")
#%%

# Get features for the sample

sample_features = get_features(sample)
sample_features.to_csv(wd+'\\Data\\Tfidf\\Sample_reviews.csv')

#%%


# Compare features for pre/post Covid

data_in = reviews_df[reviews_df.Stay_PrePandemic==True]
pre_pand_features = get_features(data_in)
pre_pand_features.to_csv(wd+'\\Data\\Tfidf\\PrePandemic_reviews.csv')

print("Pre done")


data_in = reviews_df[reviews_df.Stay_PrePandemic==False]
pre_pand_features = get_features(data_in)
pre_pand_features.to_csv(wd+'\\Data\\Tfidf\\PostPandemic_reviews.csv')

print("Post done")

#%%

# Compare features for good reviews vs bad reviews

data_in = reviews_df[reviews_df.Review_rating<=2]
quality_features = get_features(data_in)
quality_features.to_csv(wd+'\\Data\\Tfidf\\Bad_reviews.csv')

print("Bad Stay done")


data_in = reviews_df[reviews_df.Review_rating>=4]
quality_features = get_features(data_in)
quality_features.to_csv(wd+'\\Data\\Tfidf\\Good_reviews.csv')

print("Good Stay done")

#%%

# Load the tfidfs from file and compare

rankings = []

for file in os.listdir(wd+'\\Data\\Tfidf\\'):
    file_tfidf = pd.read_csv(wd+'\\Data\\Tfidf\\'+file, index_col=0)
    file_tfidf=file_tfidf.rename(columns={'Features':file[0:file.find('_')]})
    file_tfidf=file_tfidf.drop('TFIDF', axis=1)
    rankings.append(file_tfidf)
    

feature_comparison=pd.concat(rankings, axis=1)
feature_comparison=feature_comparison.reset_index()
feature_comparison=feature_comparison.rename(columns={'index':'Ranking'})



#%%















