# -*- coding: utf-8 -*-

all_cities = ['',
              '',
              '',
              '',
              ]

problem_children = []

for city_url in all_cities:
    
    # Split the URL
    url_prefix, url_suffix = hotel_page_url_splitter(city_url)
    
    # Get all the hotel links for the city
    links_set = city_hotel_links_scraper(url_prefix, url_suffix)
    
    
    # Loop through each hotel and download all info

    
    for h in range(0,len(links_set)):
    #for h in range(0,1):
        
        print(f'\n\n\n   {url_suffix.split("-Hotels")[0]}: Now working on {h}/{len(links_set)-1}\n############################################')
        
        link_hotel_test = 'https://www.tripadvisor.com/'+links_set[h]
        
        
        #link_hotel_test = 'https://www.tripadvisor.com/Hotel_Review-g28970-d84083-Reviews-Washington_Marriott_Georgetown-Washington_DC_District_of_Columbia.html'
        #link_hotel_test = 'https://www.tripadvisor.com/Hotel_Review-g28970-d939976-Reviews-Hotel_Zena_A_Viceroy_Urban_Retreat-Washington_DC_District_of_Columbia.html'
        
        try:
            hotel_and_review_scraper(link_hotel_test)
        except:
            problem_children.append(link_hotel_test)
            print(f'There are now {len(problem_children)} Problem Children')
            print(link_hotel_test)
            
            
#%%

