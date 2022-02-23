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

##
#  NLP Token Parsing Functions
##  

def custom_tokenizer(review_in):
    """
    Tokenizes the sentences usung a casual tokenizer, lowercases
    """

    from nltk.tokenize.casual import casual_tokenize
    
    review_in = re.sub('’', '\'', review_in)
    reviews_token = casual_tokenize(review_in, reduce_len=True, preserve_case=False)
    
    return reviews_token



def remove_stops(review_in):
    """
    Removes stopwords from tokenized sentences
    """
    from nltk.corpus import stopwords
    
    stopwords_eng = stopwords.words('english')
    reviews_token_stop = [word for word in review_in if word not in stopwords_eng] 
    
    return reviews_token_stop



def custom_lemmatizer(review_in):
    """
    Lemmatizes all tokens
    """
    from nltk.stem import WordNetLemmatizer
    
    lemmatizer = WordNetLemmatizer()
    reviews_token_lem = [lemmatizer.lemmatize(word) for word in review_in]
    
    return reviews_token_lem



def remove_punct(review_in):
    """
    Replaces special characters and removes punctuation from all tokens
    """
    #Remove punctuation
    import string
    punc = string.punctuation + '’' + "”" + "“" + '…' + 's'
    reviews_token_pun = [word for word in review_in if word not in punc]
    
    
    #Remove elipses
    import re
    reviews_token_dots = [re.sub(r'\.[\.]+', '', word) for word in reviews_token_pun] 
    reviews_token_dots = [word for word in reviews_token_dots if word != ''] 
    
    return reviews_token_dots



def number_remover(review_in):
    """
    Removes isolated numbers that are not dates, and number-word fragments (eg 10am, 2f) 
    """
    #Remove all numbers, and words beginning with numbers (eg, 9am)
    import re
    reviews_numbers = [re.sub(r'[\d]+[\w]?', '', word) for word in review_in] 
    reviews_numbers = [re.sub(r'[\_]', '', word) for word in reviews_numbers] 
    
    return reviews_numbers



def joiner(review_in, join_with = ' '):
    """
    Joins tokens into one string for parsing
    """
    return join_with.join(review_in) 




##
#  Data Management Functions
## 


def date_splitter(df_in, column_in):
    """
    Splits the captured date columns in the reviews. 
    Needs to run twice, once for the review date column and once for the date of stay column, as they
    are parsed differently.
    """
    
    
    import datetime
    
    if column_in == 'Review_date':
        
        prefix = 'Review'
        month_abbrev = '%b'
        
    elif column_in == 'Review_stay_date':
        
        prefix = 'Stay'
        month_abbrev = '%B'
        
    else:
        raise ValueError('Incorrect column')
        
    
    df_in[column_in].replace('Today', 'Feb 2022', inplace=True)
    df_in[column_in].replace('Yesterday', 'Feb 2022', inplace=True)
    df_in[column_in]=df_in[column_in].str.replace(' wrote a review ', '',)
    df_in[column_in]=df_in[column_in].str.strip()
    
    #Split review date into month and year
    new=df_in[column_in].str.split(' ', expand = True)
    months = new[0]
    year = new[1]
    
    #Reviews from the current month have the day, rather than the year. Convert to correct year (2022)
    year.replace(to_replace = r'\b[\d][\d]?\b', regex=True, value = 2022, inplace=True)
    year = year.astype(int)
    
    #Convert Months from abbreviation to integer
    months = months.apply(lambda x: datetime.datetime.strptime(x, month_abbrev).month)
    
    #Add Years and months to dataframe
    df_in[prefix+'_Year'] = year
    df_in[prefix+'_Month'] = months
                                          
    #Index if the Review was written pre-pandemic
    df_in[prefix+'_PrePandemic'] = ((year<2020)|((year==2020) & (months<2)))


    return df_in



def location_calculator(df_in, column_in):
    """
    Calculates the number of locations per distance - not an exact value, but extrapolation
    Works for 'hotel_location_food' and 'hotel_location_attract'
    """
    
    if column_in not in ['hotel_location_food', 'hotel_location_attract']:
        raise ValueError('Incorrect column')
    
    
    #Replace the 0 value so that they parse correctly (still calculates to 0)    
    df_in[column_in] = df_in[column_in].astype(str).str.replace(r'^0$', '0 within 1 miles', regex=True)
    
    #Split the text and parse as values
    new = df_in[column_in].str.split(' within ', expand = True)
    locations = new[0]
    distance = new[1]
    distance = distance.str.replace(' miles', '')
    
    #Calculate the locations per distance
    relative = locations.astype(float)/distance.astype(float)
    
    #Return results
    df_in[column_in+'_calc'] = relative
    return df_in
    




#%%


#########
#
#  Parse a City's Hotel Reviews
#  Calculate stats, clean text and save to file.
#
########


# Import the city's Hotels and Reviews

#city_id = 'g35394'
city_id = 'g60763'


hotels_file = 'Hotels_' + city_id + '.csv'
reviews_file = 'Reviews_' + city_id + '.csv'

hotels_df = pd.read_csv(wd+'\\Data\\'+hotels_file)
reviews_df = pd.read_csv(wd+'\\Data\\'+reviews_file)


#%%

# Basic statistics on the review ratings

reviews_df.Review_rating.unique()


reviews_df.groupby('Hotel_ID').Review_rating.mean().head()

review_stats_count = reviews_df.groupby('Hotel_ID').Review_rating.count().rename('Number_reviews')
review_stats_mean = reviews_df.groupby('Hotel_ID').Review_rating.mean().rename('Review_rating_mean')
review_stats_std = reviews_df.groupby('Hotel_ID').Review_rating.std().rename('Review_rating_std')
review_stats_sem = reviews_df.groupby('Hotel_ID').Review_rating.sem().rename('Review_rating_sem')

#%%

# Merge each hotel's calculated metrics into the main dataframe

hotels_df=hotels_df.join(review_stats_count, on='hotel_ID')
hotels_df=hotels_df.join(review_stats_mean, on='hotel_ID')
hotels_df=hotels_df.join(review_stats_std, on='hotel_ID')
hotels_df=hotels_df.join(review_stats_sem, on='hotel_ID')


#%%


## Fill Na Review Titles
reviews_df['Review_title'].fillna('', inplace=True)

## Combine the Review title and the Text
reviews_df['tokens'] = reviews_df['Review_title']+' . '+reviews_df['Review_text'] 


## Tokenize and parse the Reviews
reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: custom_tokenizer(x))
reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: remove_stops(x))
reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: custom_lemmatizer(x))
reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: number_remover(x))
reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: remove_punct(x))

reviews_df['tokens_joined'] = reviews_df.tokens.apply(lambda x: joiner(x, join_with=','))


#%%

#Split the dates
reviews_df=date_splitter(reviews_df, 'Review_date')
reviews_df=date_splitter(reviews_df, 'Review_stay_date')



#%%

#Calculate the relaive location density
hotels_df=location_calculator(hotels_df, 'hotel_location_food')
hotels_df=location_calculator(hotels_df, 'hotel_location_attract')



#%%

#Fill in Nans for Reviewer_loc
reviews_df['Reviewer_loc'].fillna('Unknown', inplace=True)
reviews_df['Reviewer_loc'] = reviews_df['Reviewer_loc'].str.replace('1 contribution', 'Unknown')



#%%

#Save Cleaned files as pickles
hotels_df.to_pickle(wd+'\\Data\\Cleaned\\'+hotels_file[0:-4]+'.pkl')
reviews_df.to_pickle(wd+'\\Data\\Cleaned\\'+reviews_file[0:-4]+'.pkl')



#sample = reviews_df.iloc[0:250]


#%%

issues = []

#Loop through all cities
all_files = os.listdir(wd+'\\Data\\')
#all_cities = [file[len('Hotels_'):-4] for file in all_files if '.csv' in file and 'Hotels' in file]
all_cities = ['g45963', 'g60805', 'g60982', 'g49022']

for city_id in tqdm(all_cities):
    
    try:
        # Load the csv files
        hotels_file = 'Hotels_' + city_id + '.csv'
        reviews_file = 'Reviews_' + city_id + '.csv'
        
        hotels_df = pd.read_csv(wd+'\\Data\\'+hotels_file)
        reviews_df = pd.read_csv(wd+'\\Data\\'+reviews_file)
        
        
        # Print out for user 
        #print('#############################\n')
        print(f'\n\n      City: {hotels_df.hotel_city.value_counts().index[0]} - {city_id}')
        print('#########################################\n')
        
        
        
        # Basic statistics on the review ratings
        
        reviews_df.Review_rating.unique()
        
        
        reviews_df.groupby('Hotel_ID').Review_rating.mean().head()
        
        review_stats_count = reviews_df.groupby('Hotel_ID').Review_rating.count().rename('Number_reviews')
        review_stats_mean = reviews_df.groupby('Hotel_ID').Review_rating.mean().rename('Review_rating_mean')
        review_stats_std = reviews_df.groupby('Hotel_ID').Review_rating.std().rename('Review_rating_std')
        review_stats_sem = reviews_df.groupby('Hotel_ID').Review_rating.sem().rename('Review_rating_sem')
        
    
        
        # Merge each hotel's calculated metrics into the main dataframe
        
        hotels_df=hotels_df.join(review_stats_count, on='hotel_ID')
        hotels_df=hotels_df.join(review_stats_mean, on='hotel_ID')
        hotels_df=hotels_df.join(review_stats_std, on='hotel_ID')
        hotels_df=hotels_df.join(review_stats_sem, on='hotel_ID')
        
        print('Completed: Stats')
        
        
        ## Fill Na Review Titles
        reviews_df['Review_title'].fillna('', inplace=True)
        
        ## Combine the Review title and the Text
        reviews_df['tokens'] = reviews_df['Review_title']+' . '+reviews_df['Review_text'] 
        
        
        ## Tokenize and parse the Reviews
        reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: custom_tokenizer(x))
        reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: remove_stops(x))
        reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: custom_lemmatizer(x))
        reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: number_remover(x))
        reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: remove_punct(x))
        reviews_df['tokens_joined'] = reviews_df.tokens.apply(lambda x: joiner(x, join_with=','))
        print('Completed: Tokens')
        
        
        
        #Split the dates
        reviews_df=date_splitter(reviews_df, 'Review_date')
        reviews_df=date_splitter(reviews_df, 'Review_stay_date')
        print('Completed: Dates')
        
        
        
        #Calculate the relaive location density
        hotels_df=location_calculator(hotels_df, 'hotel_location_food')
        hotels_df=location_calculator(hotels_df, 'hotel_location_attract')
        print('Completed: Locations')
    
        
        #Fill in Nans for Reviewer_loc
        reviews_df['Reviewer_loc'].fillna('Unknown', inplace=True)
        reviews_df['Reviewer_loc'] = reviews_df['Reviewer_loc'].str.replace('1 contribution', 'Unknown')
        
        
    
        #Save Cleaned files as pickles
        hotels_df.to_pickle(wd+'\\Data\\Cleaned\\'+hotels_file[0:-4]+'.pkl')
        reviews_df.to_pickle(wd+'\\Data\\Cleaned\\'+reviews_file[0:-4]+'.pkl')
        

    except:
        issues.append(city_id)
        print(f'There are now {len(issues)} Problem Files')
        print(city_id)


#%%

issues_copy = ['g32655', 'g45963', 'g60763', 'g60805', 'g60982']
#No reviews, Review headers missing,  Review headers missing, No reviews,  Review headers missing,

city_id = 'g60982'
issues_copy = ['g45963', 'g60805']








