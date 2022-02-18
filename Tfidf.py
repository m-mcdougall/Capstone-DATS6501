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


hotels_df_load = pd.read_pickle(wd+'\\Data\\Cleaned\\'+hotels_file[0:-4]+'.pkl')
reviews_df_load = pd.read_pickle(wd+'\\Data\\Cleaned\\'+reviews_file[0:-4]+'.pkl')




#%%
sample = reviews_df.tokens_joined.iloc[0:250]

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
