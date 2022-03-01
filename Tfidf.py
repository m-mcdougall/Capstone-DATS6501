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
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from  matplotlib.ticker import FuncFormatter


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
for year in tqdm(range(2005,2023)):
    print(f'\n    Year: {year}\n########################\n')
    annual_reviews = reviews_df[reviews_df.Stay_Year == year]
    annual_features = get_features(annual_reviews)
    annual_features.to_csv(wd+'\\Data\\Tfidf\\Annual\\'+str(year)+'_reviews.csv')


#%%



rankings = []

for file in os.listdir(wd+'\\Data\\Tfidf\\Annual\\'):
    file_tfidf = pd.read_csv(wd+'\\Data\\Tfidf\\Annual\\'+file, index_col=0)
    file_tfidf=file_tfidf.rename(columns={'Features':file[0:file.find('_')]})
    file_tfidf=file_tfidf.drop('TFIDF', axis=1)
    rankings.append(file_tfidf)
    

feature_comparison=pd.concat(rankings, axis=1)
feature_comparison=feature_comparison.reset_index()
feature_comparison=feature_comparison.rename(columns={'index':'Ranking'})



#%%

##
#
#  Plot trends in features annually
#
##

def extract_annual_features(df_in, feature_select):
    """
    Takes the annual features dataframe, and finds the ranking for the selected feature
    """
        
    collect = {}
    
    
    for year in range(2005,2023):
        
        rank = df_in[df_in[str(year)]==feature_select].Ranking.iloc[0]
        
        collect[year] = rank
        
    return pd.Series(collect)



# Extract words of interest

collect = []

for word in ['clean', 'breakfast', 'staff', 'location']:
    annual = extract_annual_features(feature_comparison, word)
    annual = annual.rename(word)
    collect.append(annual)

selected_annual = pd.concat(collect, axis=1)



# Plot the extracted trends 

selected_annual.plot(color = ['#5ACBD0', '#D05F5A', '#9A5AD0', '#90D05A'], figsize=(8,5))
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
plt.vlines(2020, 20, 1, linestyles='dotted', colors='#3F4040')
plt.ylim(2,18)
plt.gca().invert_yaxis()
plt.ylabel('Feature Ranking')
plt.xlabel('Year')
plt.title('Feature Importance over Time', fontsize=14)
plt.legend(bbox_to_anchor=(0.25, 0.7, 1, 0),  fontsize=11, frameon=False)



#%%%

##
#
#  By State
#
##

#Loop through by State and extract featues
for state in tqdm(reviews_df.State.unique()):
    
    state_reviews = reviews_df[reviews_df.State == state]
    state_features = get_features(state_reviews)
    state_features.to_csv(wd+'\\Data\\Tfidf\\State\\'+str(state)+'_reviews.csv')
    
    print(f'\n {state} Top Ten\n-------------')
    print(state_features.head(10))


#%%

# Load the tfidfs from file and compare - By State

rankings = []

for file in os.listdir(wd+'\\Data\\Tfidf\\State\\'):
    file_tfidf = pd.read_csv(wd+'\\Data\\Tfidf\\State\\'+file, index_col=0)
    file_tfidf=file_tfidf.rename(columns={'Features':file[0:file.find('_')]})
    file_tfidf=file_tfidf.drop('TFIDF', axis=1)
    rankings.append(file_tfidf)
    

feature_comparison=pd.concat(rankings, axis=1)
feature_comparison=feature_comparison.reset_index()
feature_comparison=feature_comparison.rename(columns={'index':'Ranking'})



#%%

##
#
#  By Region
#
##

#Define the regions and the component states
regions={"Pacific":['WA', 'OR','CA','NV','AK','HI',],
         'Rocky_Mountains':['MT','ID','WY','UT','CO', 'AZ','NM',],
         'Midwest':['ND','SD','NE','KS','MN','IA','MO','WI','IL','MI','IN','OH'],
         'Southwest':['TX','OK', 'AR','LA',],
         'Southeast':['KY','TN','MS','AL','WV','VA','NC','SC','GA','FL',],
         'Northeast':['ME','NH','VT','NY','MA','RI','CT','PA','NJ','DE','MD','DC']}    

#Flip the key:value pairs, to assign one region for each state
region_flip={}
for key in regions:
    for state in regions[key]:
        region_flip[state]=key


#Assign a region column based on the state
reviews_df["Region"] = reviews_df["State"].copy()
reviews_df["Region"] = reviews_df["Region"].replace(region_flip)

#Retake a sample, now including the region
sample = reviews_df.sample(n=1000)

#Loop through by region and extract featues
for region in tqdm(reviews_df.Region.unique()):
    
    region_reviews = reviews_df[reviews_df.Region == region]
    region_features = get_features(region_reviews)
    region_features.to_csv(wd+'\\Data\\Tfidf\\Region\\'+str(region)+'_reviews.csv')
    
    print(f'\n {region} Top Ten\n-------------')
    print(region_features.head(10))
    
    
#%%


# Load the tfidfs from file and compare - By Region

rankings = []

for file in os.listdir(wd+'\\Data\\Tfidf\\Region\\'):
    file_tfidf = pd.read_csv(wd+'\\Data\\Tfidf\\Region\\'+file, index_col=0)
    file_tfidf=file_tfidf.rename(columns={'Features':file[0:file.find('_')]})
    file_tfidf=file_tfidf.drop('TFIDF', axis=1)
    rankings.append(file_tfidf)
    

feature_comparison=pd.concat(rankings, axis=1)
feature_comparison=feature_comparison.reset_index()
feature_comparison=feature_comparison.rename(columns={'index':'Ranking'})

#%%

def extract_category_features(df_in, feature_select):
    """
    Takes the annual features dataframe, and finds the ranking for the selected feature
    """
        
    collect = {}
    
    
    for category in df_in.columns:
        
        try:
            rank = df_in[df_in[category]==feature_select].Ranking.iloc[0]
        except:
            rank = 1001
        
        collect[category] = rank
        
    return pd.Series(collect).drop('Ranking', axis=0)



# Extract words of interest

collect = []

for word in ['clean', 'breakfast', 'pool', 'location']:
    annual = extract_category_features(feature_comparison, word)
    annual = annual.rename(word)
    collect.append(annual)

selected_annual = pd.concat(collect, axis=1)
