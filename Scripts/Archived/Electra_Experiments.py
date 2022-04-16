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



#Flatten list utility function
def flatten_list(list_in):
    return [item for sublist in list_in for item in sublist]


#Set the working directory
wd=os.path.abspath('C://Users//Mariko//Documents//GitHub//Capstone-DATS6501')
os.chdir(wd)

#%%


def add_features_no_walk(df_in):
    
    df_in = df_in.copy()
    #Filter for review_prepandemic col
    boolean_filter = {True:'before', False:'after'}
    
    #Create additional text to add to the review
    new_text = '.'+' This hotel is in '+df_in.State+'.'\
    + ' I stayed '+ df_in.Review_PrePandemic.map(boolean_filter) + ' the pandemic.'

    df_in.Review_Body = df_in.Review_Body + new_text
    
    return df_in

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
                       'Review_Year', 'Review_Month','Stay_Year', 'Stay_Month'], 
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



#Get a sample, because you canot open the full dataset
sample = reviews_df.sample(n=5000, random_state=42)



#%%

##Save the sample as a test for the Electra Model
# 60/20/20 Split

sample['Review_rating'] = sample['Review_rating'] -1


train, test = train_test_split(sample, test_size=0.4, random_state=42)
test, validation = train_test_split(test, test_size=0.5, random_state=42)

train.to_csv(wd+'\\Data\\Cleaned\\Split\\train_sample.csv', index=False)
test.to_csv(wd+'\\Data\\Cleaned\\Split\\test_sample.csv', index=False)
validation.to_csv(wd+'\\Data\\Cleaned\\Split\\validation_sample.csv', index=False)


#%%

## Check how it would look if only predict 4

train.Review_rating.plot.hist()

from sklearn.metrics import accuracy_score


# Calculate the F1 score.
f1 = accuracy_score(y_true=test["Review_rating"], y_pred=[4]*test.shape[0])

print('Accurancy if you always guess the most frequent rating (4)')
print('Accuracy: %.4f' % f1)

#Accuracy f1: 0.4930

#%%%

# Check how often people's reviews and stays were on opposite sides of the pandemic


print(f'Total differences between time of review and time of stay: {(reviews_df.Review_PrePandemic != reviews_df.Stay_PrePandemic).sum()}')
print(f'Total reviews: {reviews_df.shape[0]}')
print(f'Percent differences between time of review and time of stay: {100*(reviews_df.Review_PrePandemic != reviews_df.Stay_PrePandemic).sum()/reviews_df.shape[0]}%')

# Less than 1% of reviews were written at a different Covid period than the stay

# Excluding the time of review variable.

#%%

def walkability_str_gen(num):
    if num <= 50:
        
        if num > 10:
            walk_str= 'not very walkable.'
        else:
            walk_str= 'not walkable at all.'
            
    else: #Num score 50+
        if num > 80: #most walkable
            walk_str= 'very walkable.'
        else:
            walk_str= 'fairly walkable.'
    return walk_str

def add_features(df_in):
    
    df_in = df_in.copy()
    #Filter for review_prepandemic col
    boolean_filter = {True:'before', False:'after'}
    
    #Create additional text to add to the review
    new_text = '.'+' This hotel is in '+df_in.State+'.'\
    + ' I stayed '+ df_in.Review_PrePandemic.map(boolean_filter) + ' the pandemic.'\
    + ' This hotel was ' + df_in.hotel_location_walk.map(walkability_str_gen)

    df_in.Review_Body = df_in.Review_Body + new_text
    
    return df_in


def add_features_no_walk(df_in):
    
    df_in = df_in.copy()
    #Filter for review_prepandemic col
    boolean_filter = {True:'before', False:'after'}
    
    #Create additional text to add to the review
    new_text = '.'+' This hotel is in '+df_in.State+'.'\
    + ' I stayed '+ df_in.Review_PrePandemic.map(boolean_filter) + ' the pandemic.'

    df_in.Review_Body = df_in.Review_Body + new_text
    
    return df_in


train_feature_add = add_features(train)
test_feature_add = add_features(test)
validation_feature_add = add_features(validation)


train_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\train_features_sample.csv', index=False)
test_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\test_features_sample.csv', index=False)
validation_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\validation_features_sample.csv', index=False)

#%%

    

#%%

##Save the full dataset for the Electra Model

reviews_df['Review_rating'] = reviews_df['Review_rating'] -1


train, test = train_test_split(reviews_df, test_size=0.4, random_state=42)
test, validation = train_test_split(test, test_size=0.5, random_state=42)

train.to_csv(wd+'\\Data\\Cleaned\\Split\\train_all.csv', index=False)
test.to_csv(wd+'\\Data\\Cleaned\\Split\\test_all.csv', index=False)
validation.to_csv(wd+'\\Data\\Cleaned\\Split\\validation_all.csv', index=False)



#%%


################################
#
#  Subsample the data- Small
#  Equal samples from each state
#  Sample size: 2000
#
################################

collect_sampler = []


for state in tqdm(reviews_df.State.unique()):
    subset_state = (reviews_df[reviews_df.State == state]).sample(n=2000, replace=False, random_state=42)
    collect_sampler.append(subset_state)



collect_sampler = pd.concat(collect_sampler)

#Adjust so the ratings start at 0
collect_sampler['Review_rating'] = collect_sampler['Review_rating'] -1

#Split into train, test and validation
train, test = train_test_split(collect_sampler, test_size=0.4, random_state=42)
test, validation = train_test_split(test, test_size=0.5, random_state=42)


#Add the features
train_feature_add = add_features_no_walk(train)
test_feature_add = add_features_no_walk(test)
validation_feature_add = add_features_no_walk(validation)

#Save the sampled data
train_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\sm_sampled_data_train_features.csv', index=False)
test_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\sm_sampled_data_test_features.csv', index=False)
validation_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\sm_sampled_data_validation_features.csv', index=False)

#%%


################################
#
#  Subsample the data- Medium
#  Equal samples from each state
#  Sample size: 5000
#
################################

collect_sampler = []


for state in tqdm(reviews_df.State.unique()):
    subset_state = (reviews_df[reviews_df.State == state]).sample(n=5000, replace=False, random_state=42)
    collect_sampler.append(subset_state)



collect_sampler = pd.concat(collect_sampler)

#Adjust so the ratings start at 0
collect_sampler['Review_rating'] = collect_sampler['Review_rating'] -1

#Split into train, test and validation
train, test = train_test_split(collect_sampler, test_size=0.4, random_state=42)
test, validation = train_test_split(test, test_size=0.5, random_state=42)


#Add the features
train_feature_add = add_features_no_walk(train)
test_feature_add = add_features_no_walk(test)
validation_feature_add = add_features_no_walk(validation)

#Save the sampled data
train_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\m_sampled_data_train_features.csv', index=False)
test_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\m_sampled_data_test_features.csv', index=False)
validation_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\m_sampled_data_validation_features.csv', index=False)

#%%


################################
#
#  Subsample the data- Medium - Stratifiied
#  Equal samples from each state
#  Sample size: 5000
#
################################

collect_sampler = []



for state in tqdm(reviews_df.State.unique()):
    subset_state = (reviews_df[reviews_df.State == state])#.sample(n=5000, replace=False, random_state=42)
    
    #Stratify to ensure equal representation of all ratings
    subset_state=subset_state.groupby('Review_rating').apply(lambda x: x.sample(n=500, replace=False, random_state=42))
    collect_sampler.append(subset_state)

#Total size = 127500

collect_sampler = pd.concat(collect_sampler)

#Adjust so the ratings start at 0
collect_sampler['Review_rating'] = collect_sampler['Review_rating'] -1


#Split into train, test and validation
train, test = train_test_split(collect_sampler, test_size=0.4, random_state=42, stratify=collect_sampler.Review_rating)
test, validation = train_test_split(test, test_size=0.5, random_state=42, stratify=test.Review_rating)


#Add the features
train_feature_add = add_features_no_walk(train)
test_feature_add = add_features_no_walk(test)
validation_feature_add = add_features_no_walk(validation)

#Save the sampled data
train_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\ms_sampled_data_train_features.csv', index=False)
test_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\ms_sampled_data_test_features.csv', index=False)
validation_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\ms_sampled_data_validation_features.csv', index=False)

#All calss weightings the same now
#Accurancy if you always guess 4
#Accuracy: 0.2000

#%%


################################
#
#  Subsample the data- large
#  Equal samples from each state
#  Sample size: 8000
#
################################

collect_sampler = []


for state in tqdm(reviews_df.State.unique()):
    subset_state = (reviews_df[reviews_df.State == state]).sample(n=8000, replace=False, random_state=42)
    collect_sampler.append(subset_state)



collect_sampler = pd.concat(collect_sampler)

#Adjust so the ratings start at 0
collect_sampler['Review_rating'] = collect_sampler['Review_rating'] -1

#Split into train, test and validation
train, test = train_test_split(collect_sampler, test_size=0.4, random_state=42)
test, validation = train_test_split(test, test_size=0.5, random_state=42)


#Add the features
train_feature_add = add_features_no_walk(train)
test_feature_add = add_features_no_walk(test)
validation_feature_add = add_features_no_walk(validation)

#Save the sampled data
train_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\lg_sampled_data_train_features.csv', index=False)
test_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\lg_sampled_data_test_features.csv', index=False)
validation_feature_add.to_csv(wd+'\\Data\\Cleaned\\Split\\lg_sampled_data_validation_features.csv', index=False)







#%%

################################
#
#  Modeling with ELECTRA
#
################################


#Load Electra - Model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
model = AutoModelForSequenceClassification.from_pretrained("google/electra-base-discriminator")


#Load Data
dataset = load_dataset('csv', data_files={'train': wd+'\\Data\\Cleaned\\Split\\train_features_sample.csv',
                                          'test': wd+'\\Data\\Cleaned\\Split\\test_features_sample.csv',
                                          'validation':wd+'\\Data\\Cleaned\\Split\\validation_features_sample.csv'})


###  Tokenize  ###
def tokenize_function(example):
    return tokenizer(example["Review_Body"],truncation=True)


#Prepare the data
tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Remove raw text - already tokenized and masked
tokenized_datasets = tokenized_datasets.remove_columns(["Review_Body"])

#Note: Removing all non-text columns?
tokenized_datasets = tokenized_datasets.remove_columns(['State',"Review_PrePandemic", 'Stay_PrePandemic', 'hotel_location_walk'])

#Rename and reformat columns
tokenized_datasets = tokenized_datasets.rename_column("Review_rating", "labels")
tokenized_datasets.set_format("torch", columns=tokenized_datasets["train"].column_names)
tokenized_datasets["train"].column_names



###  Dataloader  ###
train_dataloader = DataLoader(tokenized_datasets["train"],
                              shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["validation"],
                             batch_size=8, collate_fn=data_collator)


for batch in train_dataloader:
    break

#%%



###  Set-up  ###
num_epochs = 5
checkpoint = "google/electra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)



###  Model Set-up  ###
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5)
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler( "linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps)
print(num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device

#Tqdm
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    
    #Save at the end of each epoch
    model.save_pretrained(wd+'//models//epoch_'+str(epoch)+'_features_model_save')

###  Evaluations  ###
metric = load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

#%%

#Test model Predictions

sample_sentence = "This hotel could be cleaner. This hotel is in HI. I stayed before the pandemic."

input_ids = torch.tensor(tokenizer.encode(sample_sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
scores = model(input_ids)[0]
print(scores)

#Print the predicted score
prediction = torch.argmax(model(input_ids).logits, dim=-1)
print(prediction)











#cp ./models/epoch_0_features_model_save_cpu epoch_0_features_model_save_cpu
#cp /notebooks/models/epoch_0_features_model_save_cpu /notebooks/epoch_0_features_model_save_cpu


#model = AutoModelForSequenceClassification.from_pretrained("captest_epoch_4_features_model_save_cpu")
#model = AutoModelForSequenceClassification.from_pretrained("/home/ec2-user/notebooks/models/captest_epoch_4_features_model_save_cpu/")











