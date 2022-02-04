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

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36',
}

#First page
page_url = ['-']


links = []
skip_sequential = 0

for pagecount in page_url:
    
    # Once we get to the non-reviewed hotels, stop scraping.
    if skip_sequential <5:
        
        
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
        results = soup.find("div", {"class":"bodycon_main"})
        
        
        #Collect all hotel listing divs
        listings_all = results.find_all("div", {"class":"meta_listing ui_columns large_thumbnail_mobile"})
        skip_sequential=0
        
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
                skip_sequential += -1
            
        print('Got the page. Sleeping...')
        time.sleep(10)

#Convert to set to remove duplicates (Sponsored hotels)
links_set = list(set(links))

#%%

link_hotel = 'https://www.tripadvisor.com/'+links_set[0]

link_hotel = 'https://www.tripadvisor.com/Hotel_Review-g28970-d84083-Reviews-Washington_Marriott_Georgetown-Washington_DC_District_of_Columbia.html'


"""
#Hotel info
------------

--Header--
Hotel name
Hotel city
Address

--About--
Hotel user rating
Hotel blurb
Hotel ameneties
Hotel room features
Hotel star rating

--Location--
Location walker
Location resturaunt
Location Attraction


#Need to get number of reviews
 - Then can use the counter function from above
 

#Review info
------------

Date of review
Date of stay
Star rating
Review Title
Review full text
User location?

"""


#%%

link_hotel = 'https://www.tripadvisor.com/Hotel_Review-g28970-d84083-Reviews-Washington_Marriott_Georgetown-Washington_DC_District_of_Columbia.html'

#link_hotel = 'https://www.tripadvisor.com/Hotel_Review-g28970-d23149085-Reviews-Lyle_Washington_DC-Washington_DC_District_of_Columbia.html'

#Download the page info
page = requests.get(link_hotel, headers=headers)
soup = BeautifulSoup(page.content, 'html.parser')
#%%

#First, basic hotel info

results_header = soup.find('div', {'id':'component_3'})

hotel_name = results_header.find('h1', {'id':'HEADING'}).text
hotel_city = results_header.find('div', {'class':'KeVaw'}).find("a").text[10::]
hotel_address = results_header.find('span', {'class':'ceIOZ yYjkv'}).text



#%%

# Next, the about section

def amenity_collector(amenity_div):
    """
    Collects all amenities offered by the hotel
    
    Parameters
    ----------
    amenity_div : Soup div for the amenity table

    Returns
    -------
    collect : list
        List of all amenities, given each tag is <200 char.
        Limit reduces errors in the amenities.

    """
    
    collect = []
    for tags in amenity_div.find_all('div'):
        amenity = tags.text
        if len(amenity) < 140:
            collect.append(amenity)
        else:
            print('ERROR: AMENITY TOO LARGE:'+amenity+'\n')
    return collect


'''
--About--
Hotel user rating
Hotel blurb
Hotel ameneties
Hotel room features
Hotel star rating
'''

results_about = soup.find('div', {'id':'ABOUT_TAB'})

hotel_user_rating = float(results_about.find('span', {'class':'bvcwU P'}).text)
hotel_blurb = results_about.find('div', {'class':'pIRBV _T'}).text
hotel_prop_amenity = amenity_collector(results_about.find('div', {'class':'exmBD K'}))
hotel_room_amenity = amenity_collector(results_about.find_all('div', {'class':'exmBD K'})[1])
hotel_stars = results_about.find('div', {'class':'drcGn _R MC S4 _a H'}).find('svg')['title']
hotel_stars = hotel_stars[0:hotel_stars.find(' bubbles')]

#%%

"""
--Location--
Location walker
Location resturaunt
Location Attraction

"""
results_location = soup.find('div', {'id':'LOCATION'}).find('div', {'class':'ui_columns'})

hotel_location_walk = results_location.find_all('div', {'class':'eaCqs u v ui_column is-4'})[0]
hotel_location_walk = hotel_location_walk.find('span', {'class':'bpwqy dfNPK'}).text


hotel_location_food = results_location.find_all('div', {'class':'eaCqs u v ui_column is-4'})[1]
hotel_location_foodA = hotel_location_food.find('span', {'class':'bpwqy VyMdE'}).text
hotel_location_foodB = hotel_location_food.find('span', {'class':'ehKIl'}).text
hotel_location_food = hotel_location_foodA + ' ' + hotel_location_foodB


hotel_location_attract = results_location.find_all('div', {'class':'eaCqs u v ui_column is-4'})[2]
hotel_location_attractA = hotel_location_attract.find('span', {'class':'bpwqy eKwbS'}).text
hotel_location_attractB = hotel_location_attract.find('span', {'class':'ehKIl'}).text
hotel_location_attract = hotel_location_attractA + ' ' + hotel_location_attractB


#%%




