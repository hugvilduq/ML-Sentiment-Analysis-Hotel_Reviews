# Sentiment Analysis on Hotel Reviews
## Project Overview
This project focuses on performing sentiment analysis on hotel reviews. The goal is to collect, clean, and analyze review data to develop a machine learning model that can predict the sentiment (positive, negative, or neutral) of the reviews. The project involves three key stages: data collection, data cleaning, and building a sentiment analysis machine learning model.

## Table of Contents
1. Project Overview
2. Requirements
3. Data Collection
4. Data Cleaning
5. Exploratory Data Analysis (EDA)
6. Model Training
7. Evaluation
8. How to Run
9. File Structure
10. Contributors

## Requirements
To run this project, you will need the following libraries installed:

- Python 3.x
- Pandas
- NumPy
- Pickle (Model loading)
- Scikit-learn
- NLTK (Natural Language Toolkit)
- BeautifulSoup (Web scraping)
- Wordcloud
- SQLalchemy
- Pymongo
- Matplotlib (for data visualization)


You can install all dependencies by running:
```
bash
Copy code
pip install -r requirements.txt
```

## Data Collection

### Source:
Hotel reviews can be collected from various sources, such as:

1. Web scraping hotel booking platforms (e.g., TripAdvisor, Yelp.com). See 00 Data Collection for details
2. Public datasets from sources like Kaggle (Booking hotel reviews dataset).

## Data Cleaning
Raw data may contain noise, missing values, or irrelevant information. The data_cleaning.py script handles data preprocessing, which includes:

- Removing duplicates
- Filling or removing missing values
- Normalizing text (removing punctuation, converting to lowercase)
- Tokenization, stopword removal, and lemmatization using NLTK or SpaCy
- Encoding sentiment labels (e.g., Positive = 1, Negative = 0)

## Model Training
For sentiment analysis, several machine learning models can be used, including:

- Logistic Regression
- Support Vector Machine (SVM)
- K-nearest neighbor

The model_training.py script contains the process of training the model. It includes:

1. Vectorization: Use TF-IDF or Word2Vec for text vectorization.
2. Model Training: Train the machine learning model using labeled data.
3. Hyperparameter Tuning: Apply GridSearchCV or RandomSearch for optimizing hyperparameters.
Example:

## Evaluation
After training, the model is evaluated using various metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Contributors
Hugo Villanueva (hvilladuque@gmail.com)
Feel free to open an issue or contribute to the project!
