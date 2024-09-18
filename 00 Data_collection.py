import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine

def read_csv_hotels():

    # Positive reviews dataframe
    df_csv = pd.read_csv('Hotel_Reviews.csv')
    df_pos = pd.concat([df_csv.pop(x) for x in ['Reviewer_Score', 'Positive_Review',"Hotel_Name"]], axis=1)
    df_pos = df_pos.rename({"Positive_Review": "review"}, axis="columns")
    df_pos=df_pos[(df_pos["review"]!="No Positive")&(df_pos["review"]!="Nothing")]

    # Negative reviews dataframe
    df_csv = pd.read_csv('Hotel_Reviews.csv')
    df_neg = pd.concat([df_csv.pop(x) for x in ['Reviewer_Score', 'Negative_Review',"Hotel_Name"]], axis=1)
    df_neg = df_neg.rename({"Negative_Review": "review"}, axis="columns")
    df_neg = df_neg[(df_neg["review"]!="No Negative")&(df_neg["review"]!="Nothing")]

    df_pos = df_pos.assign(label='POSITIVE')
    df_neg = df_neg.assign(label='NEGATIVE')

    df_kaggle = pd.concat([df_pos,df_neg])

    df_kaggle=df_kaggle.rename({"Reviewer_Score": "score"}, axis="columns")
    df_kaggle = df_kaggle.assign(site='Booking')

    return df_kaggle

def handwritten_reviews():
    # Not actually handwritten by me, but from Google Reviews (not used in Webscrapping)
    handwritten_reviews = {
        'score':[9.0, 8.0, 7.5, 9.0, 6.0,
        1.0, 2.5, 3.0, 4.5, 1.0],
        'label': ["POSITIVE", "POSITIVE", "POSITIVE", "POSITIVE","POSITIVE",
        "NEGATIVE","NEGATIVE","NEGATIVE","NEGATIVE","NEGATIVE"],
        'review': ['The hotel room was luxurious and clean. This trip was part business & part hang out with friends. I felt like the hotel catered to all of my needs. Decent service. A high-class accommodation for those wanting the best of the best while in Las Vegas.', 
        'The Atmosphere is amazing and so love the hotel. The staff was very helpful and the rooms are nice. Our first time staying here and loved it. I highly recommend this place', 
        'A luxurious stay in one of the most elegant casino hotels in Vegas. Check-in was simple, the rooms were spacious, and the pool was tropical. The casino itself has a lot to offer in terms of tables, and there are plenty of food options available.', 
        "This place is huge, so prepare to walk. Staff was friendly and helpful. Room was outdated a bit, but tidy and overall clean if you don't look too hard in the corners. Great centralized location on the strip and easy enough to get to from the airport", 
        'Great classic place right in the middle of the strip. We stayed in the palace tower in a premium room. Everything was nice and new and clean. They were doing construction at the front entrance while we were there but the detour was easily accommodating.', 
        "They neglected to tell me the place was under construction. Their customer service is completely down the drain. We had stains all over our bathroom floor that they never cleaned. Basically this is a luxury resort that is providing service of a one star hotel.",
        'The hotel fails to mention the enormous construction project it has which forces its patrons to walk extremely far. Had I known about the construction we wouldn’t have booked our stay here. The location of the Uber pickup / drop off is so far from the tower I stayed at. The tower I chose was done so to prevent walking through the casino to get the room but you are forced to walk through it in a narrow mess. No complementary water. Basic amenities, do yourself a favor and book at the Wynn', 
        'I attended the Keith Urban concert on Saturday night, the concert was awesome. The problem I had was with the prices of the drinks. I ordered two drinks at the bar and was charged over 70 dollars. Two drinks 70 dollars, really ?. I understand drinks are usually expensive at concerts and sporting events but this is crazy. I will never attend an event there again.', 
        "The bed was comfortable and the bathroom was very nice. However, the couch was stained and there were drips of something on the wall they looked like dried soda. We had a broken hairdryer and found the lobby confusing to navigate. After paying similar prices at other hotels on/near the strip with cleaner rooms, we aren't impressed with Caesar's.", 
        "The lady at the front desk was very rude to us when the Caesar's made a mistake in our reservation. They charge for everything, even for water and coffee. The rooms were ok, kind of old but we expected more from the “best hotel” in Vegas (it clearly isn’t) and it was also very expensive and we didn’t even get a good view. The housekeeping workers were very nice tho. I wouldn’t recommend staying here, just take a look since the architecture is beautiful and the pools were very nice too."]
        }
    df_handwritten = pd.DataFrame(handwritten_reviews)
    df_handwritten =df_handwritten.assign(site='Google')
    df_handwritten =df_handwritten.assign(Hotel_Name='Caesars Palace')

    return df_handwritten



# Unvalid sites for webscrapping: Expedia, Trip.com, Google Hotels, Trivago, Travelocity, Orbitz, Wotif
# Valid sites for webscrapping: Nl.hotels, Yelp, Kayak, Trivago (unused)

# This function can access hotels.nl and get a revAmount (up to 10)
# number of reviews from the hotel specified with the url
def nl_hotels_webscrapper(reviewAmount,url="https://nl.hotels.com/ho124363?modal=property-reviews"):

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    entries = soup.find_all("li",class_="_1BIjNY")

    idx=0
    score_list=[]
    label_list=[]
    review_list=[]
    site_list=[]
    hotel_list=[]

    for e in entries:

        # Score and review fetch
        score = e.find("span",class_="_1biq31")
        review = e.find("p",class_="oZl9tt")
        
        # Format of scores are "9,0." Let's reshape it 
        if score!=None:
            s=float(score.text.strip(".").replace(",","."))
            if (s>5):
                label="POSITIVE"
            elif(s<5):
                label="NEGATIVE"
            else:
                label=0

            score_list.append(s)
            label_list.append(label)
            site_list.append("hotels.nl")
            hotel_list.append("Caesars Palace")
            review_list.append(review.text)

            # Amount of iterations (fisrt 2 return None)
            idx+=1
            if idx==reviewAmount+2:
                break;

    nl_hotels = pd.DataFrame({
    "label": label_list,
    "Hotel_Name":hotel_list,
    "score": score_list,
    "review": review_list,
    "site": site_list})
    return nl_hotels

def yelp_webscrapper():

    yelp_urls=[
    "https://www.yelp.com/biz/caesars-palace-las-vegas-10?sort_by=date_desc",
    "https://www.yelp.com/biz/caesars-palace-las-vegas-10?start=10&sort_by=date_desc",
    "https://www.yelp.com/biz/caesars-palace-las-vegas-10?start=20&sort_by=date_desc",
    "https://www.yelp.com/biz/caesars-palace-las-vegas-10?start=30&sort_by=date_desc",
    "https://www.yelp.com/biz/caesars-palace-las-vegas-10?start=40&sort_by=date_desc",
    "https://www.yelp.com/biz/caesars-palace-las-vegas-10?start=50&sort_by=date_desc",
    "https://www.yelp.com/biz/caesars-palace-las-vegas-10?start=60&sort_by=date_desc",
    "https://www.yelp.com/biz/caesars-palace-las-vegas-10?start=70&sort_by=date_desc",
    "https://www.yelp.com/biz/caesars-palace-las-vegas-10?start=80&sort_by=date_desc",
    "https://www.yelp.com/biz/caesars-palace-las-vegas-10?start=90&sort_by=date_desc",
    "https://www.yelp.com/biz/caesars-palace-las-vegas-10?start=100&sort_by=date_desc"
    ]

    score_list=[]
    label_list=[]
    review_list=[]
    site_list=[]
    hotel_list=[]

    for url in yelp_urls:

        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        entries = soup.find_all("li",class_="margin-b5__373c0__3ho0z border-color--default__373c0__1WKlL")

        for e in entries:
            # Score fetching
            scr = e.find(class_="margin-t1__373c0__1zX1r margin-b1-5__373c0__jjw8Y border-color--default__373c0__1WKlL")
            if scr.find(attrs={"aria-label": "5 star rating"}):
                s=10.0
                label="POSITIVE"
            elif scr.find(attrs={"aria-label": "4 star rating"}):
                s=8.0
                label="POSITIVE"
            elif scr.find(attrs={"aria-label": "2 star rating"}):
                s=4.0
                label="NEGATIVE"
            elif scr.find(attrs={"aria-label": "1 star rating"}):
                s=2.0
                label="NEGATIVE"
            else:
                s=0
                label=0
            score_list.append(s)
            label_list.append(label)
            hotel_list.append("Caesars Palace")
            site_list.append("yelp")

            # Review fetching
            rev = e.find("span",class_="raw__373c0__tQAx6")
            review_list.append(rev.text)

    yelp_reviews = pd.DataFrame({
    "label": label_list,
    "Hotel_Name":hotel_list,
    "score": score_list,
    "review": review_list,
    "site": site_list})
    return yelp_reviews

df = pd.concat([read_csv_hotels(),handwritten_reviews(),yelp_webscrapper(),nl_hotels_webscrapper(4)])
# Drop rows without label (Score=5/10)
df = df.replace(0, np.nan).dropna()
df = df.reset_index()
# -55s


# Insertion into MySQL database


# create db first in MySQL
engine = create_engine('mysql+mysqlconnector://root:root@localhost:3308/hotel_reviews')

df.to_sql(name='hotel_reviews',con=engine,if_exists='replace',index=False, method="multi" ,chunksize=500) 

# engine.execute("SELECT * FROM hotel_reviews").fetchall()
