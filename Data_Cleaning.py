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
city_id = 'g28970'


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

    from nltk.tokenize.casual import casual_tokenize
    
    review_in = re.sub('’', '\'', review_in)
    reviews_token = casual_tokenize(review_in, reduce_len=True, preserve_case=False)
    
    return reviews_token



def remove_stops(review_in):
    
    from nltk.corpus import stopwords
    
    stopwords_eng = stopwords.words('english')
    reviews_token_stop = [word for word in review_in if word not in stopwords_eng] 
    
    return reviews_token_stop



def custom_lemmatizer(review_in):
    
    from nltk.stem import WordNetLemmatizer
    
    lemmatizer = WordNetLemmatizer()
    reviews_token_lem = [lemmatizer.lemmatize(word) for word in review_in]
    
    return reviews_token_lem


def remove_punct(review_in):
    
    #Remove punctuation
    import string
    punc = string.punctuation + '’' + "”" + "“" + '…' 
    reviews_token_pun = [word for word in review_in if word not in punc]
    
    
    #Remove elipses
    import re
    reviews_token_dots = [re.sub(r'\.[\.]+', '', word) for word in reviews_token_pun] 
    reviews_token_dots = [word for word in reviews_token_dots if word != ''] 
    
    return reviews_token_dots


def number_remover(review_in):
    
    #Remove all numbers, and words beginning with numbers (eg, 9am)
    import re
    reviews_numbers = [re.sub(r'[\d]+[\w]+', '', word) for word in review_in] 
    reviews_numbers = [re.sub(r'[\_]', '', word) for word in reviews_numbers] 
    
    return reviews_numbers


def joiner(review_in):
    return ' '.join(review_in) 

#%%

demo_eel = reviews_df.iloc[: , :]
demo_eel['tokens'] = demo_eel.Review_text.apply(lambda x: custom_tokenizer(x))
demo_eel['tokens'] = demo_eel.tokens.apply(lambda x: remove_stops(x))
demo_eel['tokens'] = demo_eel.tokens.apply(lambda x: custom_lemmatizer(x))
demo_eel['tokens'] = demo_eel.tokens.apply(lambda x: remove_punct(x))
demo_eel['tokens'] = demo_eel.tokens.apply(lambda x: number_remover(x))


demo_eel['tokens_joined'] = demo_eel.tokens.apply(lambda x: joiner(x))

#%%

sample = demo_eel.tokens_joined#.iloc[0:250]

from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize training and testing data
# Train vectorizor using the parameters from our gridsearch
tfidf= TfidfVectorizer(binary=True, norm='l1')
test = tfidf.fit_transform(sample)

#x=pd.DataFrame(test.toarray(), columns=tfidf.get_feature_names())


names = tfidf.get_feature_names()
#%%


y=x.sum()



#%%

from nltk.tag import pos_tag

sentence = ' '.join(names)
tagged_sent = pos_tag(sentence.split())

propernouns = [word for word,pos in tagged_sent if pos == 'NNP']


exceptions = ['barber', 'keyboard', 'keycard', 'slumber', 'uber', 'yoga', 'kichenette', 'rubber', 'somber',
              'yogurt', 'zombie', 'invoice', ]



#%%



small_sample = demo_eel.iloc[0:250]










