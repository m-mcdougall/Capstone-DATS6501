# -*- coding: utf-8 -*-

runfile('C:/Users/Mariko/Documents/GitHub/Capstone-DATS6501/Webscraper.py', wdir='C:/Users/Mariko/Documents/GitHub/Capstone-DATS6501')


all_cities = [#'https://www.tripadvisor.com/Hotels-g60880-Anchorage_Alaska-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g60766-Little_Rock_Arkansas-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g33726-Bridgeport_Connecticut-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g34059-Wilmington_Delaware-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g32655-Los_Angeles_California-Hotels.html', 
              #'https://www.tripadvisor.com/Hotels-g60898-Atlanta_Georgia-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g60864-New_Orleans_Louisiana-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g35394-Boise_Idaho-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g42139-Detroit_Michigan-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g45086-Billings_Montana-Hotels.html'
              #'https://www.tripadvisor.com/Hotels-g60878-Seattle_Washington-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g60946-Providence_Rhode_Island-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g60097-Milwaukee_Wisconsin-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g57201-Burlington_Vermont-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g28970-Washington_DC_District_of_Columbia-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g31310-Phoenix_Arizona-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g37209-Indianapolis_Indiana-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g37835-Des_Moines_Iowa-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g60982-Honolulu_Oahu_Hawaii-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g60933-Albuquerque_New_Mexico-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g60745-Boston_Massachusetts-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g54171-Charleston_South_Carolina-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g58947-Charleston_West_Virginia-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g60439-Cheyenne_Wyoming-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g50226-Columbus_Ohio-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g49785-Fargo_North_Dakota-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g56003-Houston_Texas-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g43833-Jackson_Mississippi-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g44535-Kansas_City_Missouri-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g60766-Little_Rock_Arkansas-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g60880-Anchorage_Alaska-Hotels.html',
              #'https://www.tripadvisor.com/Hotels-g33726-Bridgeport_Connecticut-Hotels.html',
              'https://www.tripadvisor.com/Hotels-g45963-Las_Vegas_Nevada-Hotels.html',
              'https://www.tripadvisor.com/Hotels-g32655-Los_Angeles_California-Hotels.html',
              'https://www.tripadvisor.com/Hotels-g39604-Louisville_Kentucky-Hotels.html',
              'https://www.tripadvisor.com/Hotels-g46152-Manchester_New_Hampshire-Hotels.html',
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


"""
problem_errors = []

for h in range(0,len(problem_children)-1):
#for h in range(0,1):
    
    print(f'\n\n\n   Now working on {h}/{len(problem_children)-1}\n###########################')
    
    link_hotel_test = 'https://www.tripadvisor.com/'+problem_children[h]
    
    
    #link_hotel_test = 'https://www.tripadvisor.com/Hotel_Review-g28970-d84083-Reviews-Washington_Marriott_Georgetown-Washington_DC_District_of_Columbia.html'
    #link_hotel_test = 'https://www.tripadvisor.com/Hotel_Review-g28970-d939976-Reviews-Hotel_Zena_A_Viceroy_Urban_Retreat-Washington_DC_District_of_Columbia.html'
    
    try:
        hotel_and_review_scraper(link_hotel_test)
    except:
        problem_errors.append(link_hotel_test)
        print(f'There are now {len(problem_errors)} Double Problems')
        print(link_hotel_test)
    
"""