import dash 
from dash import html
from pymongo import MongoClient
import pandas as pd
from matplotlib import pyplot as plt

# Load data (8m 16s)
client = MongoClient("localhost:27017")
db=client.hotel_reviews
result=db.hotel_reviews.find({})
source=list(result)
df=pd.DataFrame(source)
df.head()

df=df.sample(5000, random_state=1)
df

df["Hotel_Name"].hist()
plt.show()

df["label"].hist()
plt.show()

app = dash.Dash(__name__)
application=app.server

app.layout=html.H1("Hello Dash")

import plotly.express as px
df = px.data.gapminder().query("year == 2007")
fig = px.scatter_geo(df, locations="iso_alpha",
                     size="pop", # size of markers, "pop" is one of the columns of gapminder
                     )
fig.show()

if __name__=="__main__":
    app.run_server(debug=False)