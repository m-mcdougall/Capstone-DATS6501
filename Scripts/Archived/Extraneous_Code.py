# -*- coding: utf-8 -*-
#%%

sample = demo_eel.tokens_joined#.iloc[0:250]

## First draft of Tfidf

from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize training and testing data
# Train vectorizor using the parameters from our gridsearch
tfidf= TfidfVectorizer(binary=True, norm='l1')
test = tfidf.fit_transform(sample)

#x=pd.DataFrame(test.toarray(), columns=tfidf.get_feature_names())

#All features
names = tfidf.get_feature_names()



## Remove proper nouns 

from nltk.tag import pos_tag

sentence = ' '.join(names)
tagged_sent = pos_tag(sentence.split())

propernouns = [word for word,pos in tagged_sent if pos == 'NNP']

#Some words get incorrectly categorized, so remove them
exceptions = ['barber', 'keyboard', 'keycard', 'slumber', 'uber', 'yoga', 'kichenette', 'rubber', 'somber',
              'yogurt', 'zombie', 'invoice', ]

#%%