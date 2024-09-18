import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from pymongo import MongoClient
from wordcloud import WordCloud as wc
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt


# # Data fetch
# client = MongoClient("localhost:27017")
# db=client.hotel_reviews
# result=db.hotel_reviews.find({})
# source=list(result)
# df=pd.DataFrame(source)

# Dash application defining
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv("Hotel_Reviews.csv")

# All reviews from Amsterdam
amsdf = df[df["Hotel_Address"].str.contains("Amsterdam Netherlands") == True]
# Get one review per hotel, hence getting unique hotels
hotelsdf = amsdf.drop_duplicates(subset=["Hotel_Address"])


# Function for data points coloring
def SetColor(x):
    if x < 7.5:
        return "red"
    elif 7.5 <= x <= 8:
        return "orange"
    elif 8 < x <= 8.5:
        return "yellow"
    elif 8.5 < x <= 9:
        return "GreenYellow"
    elif x > 9:
        return "green"


# ------------------------------------------------------------------------------
# App layout

app.layout = html.Div(
    [
        # Title
        html.H1(
            "Best Hotels in Amsterdam",
            style={
                "padding-left": "25%",
                "padding-right": "25%",
                "text-align": "center",
            },
        ),
        dcc.Tabs(
            id="tabs-example",
            value="tab-1-example",
            children=[
                dcc.Tab(
                    label="Map",
                    value="tab-1-example",
                    children=[
                        # Slider
                        html.Div(
                            "Score selector",
                            style={
                                "padding-left": "25%",
                                "padding-right": "25%",
                                "text-align": "center",
                                "fontSize": 18,
                                "color": "gray",
                            },
                        ),
                        html.Div(
                            [
                                dcc.RangeSlider(
                                    id="score-slider",
                                    min=0,
                                    max=10,
                                    value=[0, 10],
                                    marks={
                                        "0": 0,
                                        "1": 1,
                                        "2": 2,
                                        "3": 3,
                                        "4": 4,
                                        "5": 5,
                                        "6": 6,
                                        "7": 7,
                                        "8": 8,
                                        "9": 9,
                                        "10": 10,
                                    },
                                    step=0.1,
                                    tooltip={"always_visible": True},
                                ),
                            ],
                            style={
                                "width": "50%",
                                "padding-left": "25%",
                                "padding-right": "25%",
                            },
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Review Worclouds",
                    value="tab-2-example",
                    children=[
                        html.Div(
                            "Hotel selector",
                            style={
                                "padding-left": "25%",
                                "padding-right": "25%",
                                "text-align": "center",
                                "fontSize": 18,
                                "color": "gray",
                            },
                        ),
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="dropdown",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in hotelsdf.Hotel_Name.unique()
                                    ],
                                    multi=False,
                                    placeholder="(All hotels)",
                                ),
                            ],
                            style={
                                "width": "50%",
                                "padding-left": "25%",
                                "padding-right": "25%",
                            },
                        ),
                    ],
                ),
            ],
        ),
        html.Div(id="tabs-content-example"),
    ]
)

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    Output("tabs-content-example", "children"),
    Input("tabs-example", "value"),
    Input("score-slider", "value"),
    Input("dropdown", "value"),
)
def render_content(tab, selected_score=[0, 10], hotel_name="All"):

    # Tab 1: Map
    if tab == "tab-1-example":

        filtered_df = hotelsdf[
            (hotelsdf["Average_Score"] > selected_score[0])
            & (hotelsdf["Average_Score"] < selected_score[1])
        ]
        fig = px.scatter_mapbox(
            data_frame=filtered_df,
            lat=filtered_df["lat"],
            lon=filtered_df["lng"],
            hover_name=filtered_df["Hotel_Name"],
            zoom=12.2,
            width=1500,
            height=700,
            hover_data=dict(Hotel_Name=False, Average_Score=True, lat=False, lng=False),
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_traces(
            marker=dict(
                size=15,
                opacity=0.7,
                color=list(map(SetColor, filtered_df["Average_Score"])),
            )
        )

        # Legend
        fig.add_trace(
            go.Bar(
                name="Scored 7.5 or lower",
                marker=dict(color=list(map(SetColor, hotelsdf["Average_Score"]))[26]),
                x=[""],
                y=[""],
            )
        )
        fig.add_trace(
            go.Bar(
                name="Score between 7.5 and 8",
                marker=dict(color=list(map(SetColor, hotelsdf["Average_Score"]))[7]),
                x=[""],
                y=[""],
            )
        )
        fig.add_trace(
            go.Bar(
                name="Score between  and 8.5",
                marker=dict(color=list(map(SetColor, hotelsdf["Average_Score"]))[0]),
                x=[""],
                y=[""],
            )
        )
        fig.add_trace(
            go.Bar(
                name="Score between 8.5 and 9",
                marker=dict(color=list(map(SetColor, hotelsdf["Average_Score"]))[2]),
                x=[""],
                y=[""],
            )
        )
        fig.add_trace(
            go.Bar(
                name="Scored 9 or higher",
                marker=dict(color=list(map(SetColor, hotelsdf["Average_Score"]))[28]),
                x=[""],
                y=[""],
            )
        )

        tab_content = html.Div([dcc.Graph(id="graph-1-tabs", figure=fig)])

# Tab 2: Worcloud
    elif tab == "tab-2-example":
        selected_reviewsdf = amsdf[amsdf["Hotel_Name"] == hotel_name]

        # Wordcloud with all hotels, if none chosen
        if len(selected_reviewsdf) < 1:
            selected_reviewsdf = amsdf

        neg_text = selected_reviewsdf["Negative_Review"].values
        pos_text = selected_reviewsdf["Positive_Review"].values

        stopwords = set(STOPWORDS)
        stopwords.update(["hotel", "room", "positive", "Negative"])

        neg_wordcloud = wc(stopwords=stopwords,background_color='white').generate(str(neg_text))
        pos_wordcloud = wc(stopwords=stopwords,background_color='white').generate(str(pos_text))
        
        neg_wordcloud.recolor(colormap="Reds", random_state=1)
        pos_wordcloud.recolor(colormap="Greens", random_state=1)

        neg_fig = px.imshow(neg_wordcloud)
        neg_fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        neg_fig.update_layout(title_text="Negative Reviews Wordcloud", title_x=0.5)

        pos_fig = px.imshow(pos_wordcloud)
        pos_fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        pos_fig.update_layout(title_text="Positive Reviews Wordcloud", title_x=0.5)

        tab_content = html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(id="graph-2-tabs", figure=pos_fig),
                            ],
                            className="six columns",
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="graph-3-tabs",
                                    figure=neg_fig,
                                ),
                            ],
                            className="six columns",
                        ),
                    ],
                    className="row",
                )
            ]
        )

    return tab_content


# ------------------------------------------------------------------------------
# Run
if __name__ == "__main__":
    app.run_server(debug=False)
