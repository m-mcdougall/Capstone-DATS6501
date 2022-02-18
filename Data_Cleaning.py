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

city_id = 'g35394'
city_id = 'g35805'


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

#%%

## Tokenize and parse the Reviews

reviews_df['tokens'] = reviews_df.Review_text.apply(lambda x: custom_tokenizer(x))
reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: remove_stops(x))
reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: custom_lemmatizer(x))
reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: number_remover(x))
reviews_df['tokens'] = reviews_df.tokens.apply(lambda x: remove_punct(x))

reviews_df['tokens_joined'] = reviews_df.tokens.apply(lambda x: joiner(x, join_with=','))


#%%


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



#%%

#Split the dates
reviews_df=date_splitter(reviews_df, 'Review_date')
reviews_df=date_splitter(reviews_df, 'Review_stay_date')


#%%


def location_calculator(df_in, column_in):
    """
    Calculates the number of locations per distance - not an exact value, but extrapolation
    Works for 'hotel_location_food' and 'hotel_location_attract'
    """
    
    if column_in not in ['hotel_location_food', 'hotel_location_attract']:
        raise ValueError('Incorrect column')
    
    
    #Replace the 0 value so that they parse correctly (still calculates to 0)    
    df_in[column_in] = df_in[column_in].str.replace(r'^0$', '0 within 1 miles', regex=True)
    
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

#Calculate the relaive location density
hotels_df=location_calculator(hotels_df, 'hotel_location_food')
hotels_df=location_calculator(hotels_df, 'hotel_location_attract')




#%%
sample = demo_eel.tokens_joined.iloc[0:550]

from sklearn.feature_extraction.text import CountVectorizer

cv= CountVectorizer(ngram_range=(1,2), token_pattern = r'[\w\']+', max_features=10000, strip_accents='ascii')

word_count_vector=cv.fit_transform(sample)

list(cv.vocabulary_.keys())[:10]

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)




#%%


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

# you only needs to do this once, this is a mapping of index to 
feature_names=cv.get_feature_names()
# get the document that we want to extract keywords from
doc= list(demo_eel.tokens_joined.iloc[150:600].array)
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform(doc))
#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,20)
# now print the results
print("\n=====Doc=====")
print(doc)
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])
    
    
#%%


doc= list(demo_eel.tokens_joined.iloc[560:600].array)
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform(doc))


#%%










