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

#Web-Scraping Imports
import requests
from bs4 import BeautifulSoup


#Set the working directory
wd=os.path.abspath('C://Users//Mariko//Documents//GitHub//Capstone-DATS6501')
os.chdir(wd)


# https://github.com/m-mcdougall/BicycleTheft-DATS6103/blob/main/Web%20Scraper%20and%20Updater.ipynb
# https://github.com/m-mcdougall/Remote_Careers-DATS6401/blob/main/Websraper.py

#%%

"""

  To Do
---------

1. Search fo the city, since the htmls are custom
2. Navigate to the hotels page
3. Collect hotels for various star ranges
4. Navigate to each otel page
5. Save the location, and a selection of reviews

"""

#%%

def gen_property_pages(soup_in):
    """
    Generates the URL adders for the number of pages of hotels
    eg, show page 3 of the results
    
    Parameters
    ----------
    soup_in : Beautiful Soup Page
        The full first page's beautiful soup'

    Returns
    -------
    output : List
        A list of all the url page number additives for all pages beyond the first.

    """
    
    #Find number of hotel properties
    prop_raw = soup_in.find('span',{'class':'eMoHQ'}).text
    properties = int(prop_raw[0:prop_raw.find(' prop')])
    
    
    #Generate addresses in increments of 30
    output = []
    count=30
    while count < properties:
        output.append('-oa'+str(count)+'-')
        count+=30
        
    return output

#%%


cat = [1]

for i in cat:
    if i ==1:
        cat.extend([2,3])
    print(i)


#%%

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36',
}

#First page
page_url = ['-']


links = []
skip_sequential = 100

for pagecount in page_url:
    
    # Once we get to the non-reviewed hotels, stop scraping.
    if skip_sequential >5:
        continue
    
    url= 'https://www.tripadvisor.com/Hotels-g28970'+pagecount+'Washington_DC_District_of_Columbia-Hotels.html'
    
    print('\n\n'+url)
    

    
    #Download the page info
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # Parse the number of properties, and add the 
    # coorresponding number of pages to the search
    # NOTE: Only do this on the first page, or it will continue infinitely
    if pagecount == '-':
        page_url.extend(gen_property_pages(soup))
    
    
    #Extract only the hotel listings div
    results = soup.find("div", {"id":"taplc_hsx_hotel_list_lite_dusty_hotels_combined_sponsored_0"})
    
    
    #Collect all hotel listing divs
    listings_all = results.find_all("div", {"class":"meta_listing ui_columns large_thumbnail_mobile"})
    
    
    for listing in listings_all:
    
        # Only collect links for hotels with sufficient reviews
    
    
        # First, check review count
        try:
            review_raw = listing.find("a", {"class":"review_count"}).text
            review_raw = review_raw[0:review_raw.find(' review')]
            review_raw = int(review_raw.replace(',',''))
        except:
            print('ERROR: Skipping')
            print(listing)
            review_raw = 0
    
        # Filter for review count
        if review_raw < 100:
            print(f'Insufficient review count: {review_raw}. Skipping...')
            skip_sequential +=1 
        else:
            print(f'Sufficient reviews: {review_raw}.')
        
            #Collect the hotel's url
            listing_link = listing.find("div", {"class":"listing_title"})
            links.append(listing_link.find("a")['href'])
            skip_sequential = 0
        
    print('Got the page. Sleeping...')
    time.sleep(10)
#%%



#%%



