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

#Flatten list utility function
def flatten_list(list_in):
    return [item for sublist in list_in for item in sublist]


#Set the working directory
wd=os.path.abspath('C://Users//Mariko//Documents//GitHub//Capstone-DATS6501')
os.chdir(wd)


#Set the headers for the scraper
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36',
}


# https://github.com/m-mcdougall/BicycleTheft-DATS6103/blob/main/Web%20Scraper%20and%20Updater.ipynb
# https://github.com/m-mcdougall/Remote_Careers-DATS6401/blob/main/Websraper.py

#%%

"""

  To Do
---------

1. Search fo the city, since the htmls are custom
2. Navigate to the hotels page
3. Collect hotels for various star ranges
4. Navigate to each hotel page
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

url_prefix = 'https://www.tripadvisor.com/Hotels-g60811'
url_suffix = 'Baltimore_Maryland-Hotels.html'

def city_hotel_links_scraper(url_prefix_in, url_suffix_in):
    #Initialize variables
    
    page_url = ['-'] #First page
    links = []
    skip_sequential = 0
    
    
    
    for pagecount in page_url:
        
        # Once we get to the non-reviewed hotels, stop scraping.
        if skip_sequential <5:
            
            
            url= url_prefix_in+pagecount+url_suffix_in
            
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
    
    return links_set

links_set = city_hotel_links_scraper(url_prefix, url_suffix)

#%%




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

--Header--
Date of review
User location?

--Body--
Date of stay
Star rating
Review Title
Review full text


"""


#%%


def hotel_and_review_scraper(link_hotel):
    #Download the page info
    page = requests.get(link_hotel, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    
    
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
    
    
    ## Basic Info Section
    
    hotel_ID  = link_hotel[link_hotel.find('Hotel_Review-')+len('Hotel_Review-'):link_hotel.find('-Reviews')]
    
    results_header = soup.find('div', {'id':'component_3'})
    
    hotel_name = results_header.find('h1', {'id':'HEADING'}).text
    hotel_city = results_header.find('div', {'class':'KeVaw'}).find("a").text[10::]
    hotel_address = results_header.find('span', {'class':'ceIOZ yYjkv'}).text
    
    
    
    ## About Section
    
    results_about = soup.find('div', {'id':'ABOUT_TAB'})
    
    hotel_user_rating = float(results_about.find('span', {'class':'bvcwU P'}).text)
    hotel_blurb = results_about.find('div', {'class':'pIRBV _T'}).text
    hotel_prop_amenity = amenity_collector(results_about.find('div', {'class':'exmBD K'}))
    hotel_room_amenity = amenity_collector(results_about.find_all('div', {'class':'exmBD K'})[1])
    hotel_stars = results_about.find('div', {'class':'drcGn _R MC S4 _a H'}).find('svg')['title']
    hotel_stars = hotel_stars[0:hotel_stars.find(' bubbles')]
    
    
    
    ## Location Section
    
    results_location = soup.find('div', {'id':'LOCATION'}).find('div', {'class':'ui_columns'})
    
    try:
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
    
    except:
        print('This place is far:\n'+link_hotel)
        hotel_location_walk=hotel_location_food=hotel_location_attract ='0'
    
    
    hotel_series = pd.Series({"hotel_ID":hotel_ID, "hotel_name":hotel_name, "hotel_city":hotel_city,
                              "hotel_address":hotel_address, "hotel_user_rating":hotel_user_rating,
                              "hotel_blurb":hotel_blurb, "hotel_prop_amenity":hotel_prop_amenity, 
                              "hotel_room_amenity":hotel_room_amenity, "hotel_stars":hotel_stars, 
                              "hotel_location_walk":hotel_location_walk, "hotel_location_food":hotel_location_food,
                              "hotel_location_attract":hotel_location_attract})
    
    hotel_series = pd.DataFrame(hotel_series).T
    
    
    ## Save the Hotel Data to City file
    
    city_ID = hotel_ID[0:hotel_ID.find('-')]
    
    # Check if city already exists
    # Append if file exists, new file if not
    
    if 'Hotels_'+city_ID+'.csv' not in os.listdir(wd+'//Data'):
        hotel_series.to_csv(wd+'//Data//Hotels_'+city_ID+'.csv', mode = 'w', index=False, header=True)
    else:
        hotel_series.to_csv(wd+'//Data//Hotels_'+city_ID+'.csv', mode = 'a', index=False, header=False)
    
    
    
    
    #### Reviews  #####
    
   
    ## Reviews Section
    hotel_reviews = int(soup.find('div', {'id':'REVIEWS'}).find('span', {'class':'cdKMr Mc _R b'}).text.replace(',',''))
    
    
    
    
    def gen_review_pages(review_url, review_count):
        """
        Generates the URL adders for the number of pages of reviews
        eg, show page 3 of the results
        
        Parameters
        ----------
        review_url : URL of the hotel's base review page
        review_count : The number of reviews for the hotel
    
        Returns
        -------
        output : List
            A list of all the urls for pages beyond the first.
    
        """
        
        #Split the url into segments
        split = review_url.find('-Reviews-')
        
        review_url_pre = review_url[0:split]
        review_url_post = review_url[split::]
        
        
        #Generate addresses in increments of 30
        output = []
        count=5
        while count < review_count:
            output.append(review_url_pre+'-or'+str(count)+review_url_post)
            count+=5
            
        return output
    
    
    def review_page_collector(reviews_div_in, hotel_ID_in):
        """
        Collects the contents of the entire reviews_div,
        collecting all the informaion from all reviews on the page
    
        Parameters
        ----------
        reviews_div_in : soup div for the reviews
    
        Returns
        -------
        collect : list of pandas Series
            Contains all the reviews as a series each.
    
        """
    
        def review_span_collector(review_text_in):
            """
            Collects all the review spans and returns as a list
        
            Parameters
            ----------
            review_text_in : Soup q containing the individial review's spans'
        
            Returns
            -------
            collect : String
                String containing the contents of all the spans in the review.    
            """
        
            #Loop through all spans and extract the text
            #Linebreaks are represented by seperate spans
            collect = []
            for text_block in review_text_in.find_all('span'):
                collect.append(text_block.text)
        
            #Return only the string - join if needed
            if len(collect) == 1:
                return collect[0]
            else:
               return ' '.join(collect)
    
    
        collect = []
        
        # Individual review
        for review_in in reviews_div_in:
           
            try:
                ### Header Information
                
                review_header = review_in.find('div', {'class':'xMxrO'})
                
                review_date = review_header.find('div', {'class':'bcaHz'}).find('span').text
                review_date = review_date[review_date.find('a review')+len('a review ')::]
                
                review_home_loc = review_header.find('div', {'class':'BZmsN'}).find('span').text
                # If reviewer home is not listed, the first div contains their contributions, if so, skip.
                if 'contributions' in review_home_loc:
                    review_home_loc = 'N/A'
                
                
                ### Body Information
                
                review_body = review_in.find('div', {'class':'cqoFv _T'})
                
                review_rating = review_body.find('div', {'data-test-target':'review-rating'}).find('span')['class'][1]
                review_rating = int(review_rating[len('bubble_'):-1])
                
                review_title = review_body.find('div', {'data-test-target':'review-title'}).find('span').text
                
                review_text = review_span_collector(review_body.find('q', {'class':'XllAv H4 _a'}))
                
                
                review_stay_date = review_body.find('span', {'class':'euPKI _R Me S4 H3'}).text
                review_stay_date = review_stay_date[review_stay_date.find(": ")+2::]
                
                out = pd.Series({'Hotel_ID':hotel_ID_in,'Review_date':review_date, 'Reviewer_loc':review_home_loc, 
                                 'Review_rating':review_rating,'Review_stay_date':review_stay_date,
                                 'Review_title':review_title, 'Review_text':review_text})
                
                collect.append(out)
            except:
                print('ERROR READING. SKIPPING REVIEW.')
        
        return collect
    
    
    reviews_pages_2plus = [link_hotel] + gen_review_pages(link_hotel, hotel_reviews)
    
    all_reviews = []
    
    for review_i in tqdm(reviews_pages_2plus):
        
        if review_i != link_hotel:
            #Download the page info
            page = requests.get(review_i, headers=headers)
            soup = BeautifulSoup(page.content, 'html.parser')
            time.sleep(1.24)
    
        reviews_div = soup.find('div', {'id':'component_16'}).find_all('div', {'class':'cWwQK MC R2 Gi z Z BB dXjiy'})
        
        
        all_reviews.append(review_page_collector(reviews_div, hotel_ID))
    
    
    all_reviews_flat = flatten_list(all_reviews)
    all_reviews_flat = pd.DataFrame(all_reviews_flat)
    
    

    
    ## Save the Review Data to City file
    
    city_ID = hotel_ID[0:hotel_ID.find('-')]
    
    # Check if city already exists
    # Append if file exists, new file if not
    
    if 'Reviews_'+city_ID+'.csv' not in os.listdir(wd+'//Data'):
        all_reviews_flat.to_csv(wd+'//Data//Reviews_'+city_ID+'.csv', mode = 'w', index=False, header=True)
    else:
        all_reviews_flat.to_csv(wd+'//Data//Reviews_'+city_ID+'.csv', mode = 'a', index=False, header=False)


#%%

for h in range(4,60):
    
    print(f'\n\n\n   Now working on {h}/60\n###########################')
    
    link_hotel_test = 'https://www.tripadvisor.com/'+links_set[h]
    
    
    #link_hotel_test = 'https://www.tripadvisor.com/Hotel_Review-g28970-d84083-Reviews-Washington_Marriott_Georgetown-Washington_DC_District_of_Columbia.html'
    #link_hotel_test = 'https://www.tripadvisor.com/Hotel_Review-g28970-d939976-Reviews-Hotel_Zena_A_Viceroy_Urban_Retreat-Washington_DC_District_of_Columbia.html'
    
    hotel_and_review_scraper(link_hotel_test)
    








