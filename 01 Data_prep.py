from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

engine = create_engine('mysql+mysqlconnector://root:root@localhost:3308/hotel_reviews')
connection = engine.raw_connection()

df = pd.read_sql("SELECT * FROM hotel_reviews", con=engine)

conn = engine.raw_connection()
cur = conn.cursor()

# Some data insights at first glance. The fetching must be parametrized. 
cur.callproc('get_all_reviews')
for row in cur.stored_results():
    results=row.fetchall()
    colNamesList=row.description
colNamesList=[i[0] for i in colNamesList]

df_raw=pd.DataFrame(results, columns=colNamesList)
df_raw.head()

# Describe
df_raw.describe()
df_raw.shape


# See what reviews are too empty 
df_sorted=df.sort_values(by="review", key=lambda x: x.str.len())

# Wordcloud with df_raw

text_raw = df_raw['review'].values 
# Create and generate a word cloud image:
wordcloud_raw = WordCloud(background_color='white').generate(str(text_raw))

# Display the generated image:
plt.imshow(wordcloud_raw, interpolation='bilinear')
plt.axis("off")
plt.show()

# Get review samples. 800000/8000 = 100 reviews
cur.callproc('get_reviews_by_module',['8000'])
for row in cur.stored_results():
    results=row.fetchall()
    colNamesList=row.description
colNamesList=[i[0] for i in colNamesList]

df_sample=pd.DataFrame(results, columns=colNamesList)
df_sample.head()

# Data cleaning/wrangling in SQL: Remove empty reviews(NO negative, etc) punctuation, lowercase and stop words
# results = engine.execute('get_all_reviews')


# Cleaning: Remove reviews with <12 characters, punctuation, and to lowercase
cur.callproc('remove_empty_reviews')  
cur.callproc('remove_punctuation')  
cur.callproc('to_lowercase') 

df = pd.read_sql("SELECT * FROM hotel_reviews", con=engine)
df.head()

# Default stopwords set
stopwords=set(STOPWORDS)
# Adding stopwords from the wordcloud
context_stopwords=["hotel","room","rooms","staff","upon","bit","payment","will",
"location","checkout","outside","check","food"]
stopwords.update(context_stopwords)
df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))


# Describe
df.describe()

# Wordcloud with cleaned dataset

df.head()
text = df['review'].values 
# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(str(text))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


df.to_csv("Clean_Reviews.csv")


# -----------------------------------
# PLOTS
# -----------------------------------


#  Positive/Negative cheeseplot by Website.
cur.callproc('label_ratio_by_website')
for row in cur.stored_results():
    results=row.fetchall()
    colNamesList=row.description
colNamesList=[i[0] for i in colNamesList]
df_pie=pd.DataFrame(results, columns=colNamesList)
# Transpose, set sites as columns and remove the duplicate sites row
df_pie=df_pie.transpose().set_axis(df_pie.site, axis='columns').iloc[1:,:]
plot = df_pie.plot(kind="pie",legend=False , autopct='%1.0f%%',subplots=True,rot=1,figsize=(11, 3), title="Positive/Negative review ratio per review site", colors = ['#DC143C', '#2E8B57'])


# Barplot with labels grouped by score ranges.
cur.callproc('score_ranges')
for row in cur.stored_results():
    results=row.fetchall()
    colNamesList=row.description
colNamesList=[i[0] for i in colNamesList]
score_range=pd.DataFrame(results, columns=colNamesList)
score_range

# Rearrange the df so it has the counts as values, negative/positive as columns and scorerange as index
scores=pd.unique(score_range["scorerange"].values)
neg_scores=score_range[score_range["label"] == "NEGATIVE"]["count"]
pos_scores=score_range[score_range["label"] == "POSITIVE"]["count"]
rev_score_range=pd.DataFrame({'NEGATIVE':neg_scores.values, 'POSITIVE':pos_scores.values} , index=scores)
rev_score_range

rev_score_range.plot(kind="bar",title="Reviews amount by score range",rot=0,color={"NEGATIVE": "#DC143C", "POSITIVE": "#2E8B57"})


# -1m 4s