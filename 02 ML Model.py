from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
import pickle


# -------------------------------------------
# Fetching dataset (57s)
# -------------------------------------------

engine = create_engine('mysql+mysqlconnector://root:root@localhost:3308/hotel_reviews')
connection = engine.raw_connection()

df = pd.read_sql("SELECT * FROM hotel_reviews", con=engine)

conn = engine.raw_connection()
cur = conn.cursor()
# Get review samples. ~800,000/80 = ~100,000 reviews
cur.callproc('get_reviews_by_module',['8'])
for row in cur.stored_results():
    results=row.fetchall()
    colNamesList=row.description
colNamesList=[i[0] for i in colNamesList]

df=pd.DataFrame(results, columns=colNamesList)

df["label"]=df.apply(lambda df: 1 if df["label"]=="POSITIVE" else 0, axis=1)

# -------------------------------------------
# Preparing the data
# -------------------------------------------

# Remove stopwords
stopwords=set(STOPWORDS)
context_stopwords=["hotel","park","room","rooms","staff","upon","bit","payment","will","location","checkout","outside","check","food","caesar"]
stopwords.update(context_stopwords)
df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

# Tokenization
df['review'] = df.apply(lambda x: word_tokenize(x['review']), axis=1)

# Stemming
porter = PorterStemmer()
df['review'] = df['review'].apply(lambda x: [porter.stem(y) for y in x]) # Stem every word.

# Convert tokens back into one string, to perform Tfidf
df['review'] = df['review'].apply(lambda x: ' '.join([str(i) for i in x]))

# -------------------------------------------
# Vectorizing and defining the sets
# -------------------------------------------


# Get x,y;vectorize and transform; split data; model fit; pos_test;score; crossfold?
# Max features(up to 25000) in Tfidf vectorizer doesnt enhance the results. Tfidf and CountVectorizer are almost equal good
# ngram_range=(1,2), these features dont affect at all:max_df=0.2, min_df=0.0001
vect=TfidfVectorizer(ngram_range=(1,2))
X = vect.fit_transform(df.review)
y = df['label']

# Dataset split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1,stratify=y)

print("Vocabulary size: {}".format(len(vect.vocabulary_)))

##############################################################################
# Model building
##############################################################################


# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# MODEL 1:Logistic Regression
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Model loading
filename = 'log_reg_model.sav'
log_reg = pickle.load(open(filename, 'rb'))

# OR model creating
log_reg = LogisticRegression(C=5,penalty="l2",solver="liblinear")
log_reg.fit(X_train, y_train)

# _______________________________________________
# Testing log_reg (10s)

y_pred = log_reg.predict(X_test)

# Crossfold validation with 5 folds (accuracy)
cv_results = cross_val_score(log_reg, X_train, y_train, cv=5)
print("Mean accuracy with cv=5:",np.mean(cv_results))

# Confusion matrix 
print(metrics.confusion_matrix(y_test, y_pred)/len(y_test))

# Accuracy
metrics.accuracy_score(y_test, y_pred)

# F1 score
metrics.f1_score(y_test, y_pred)

# Area under ROC curve
print("AUC:",metrics.roc_auc_score(y_test, y_pred))


# --------------------------------------
# Log_reg Hyperparameter tuning
# --------------------------------------
# To know what scoring methods to use in GridSearch
# metrics.SCORERS.keys()

# make a dictionary of hyperparameter values to search. Parameters of model
search_space = {
    "C" : [10,5,1,0.1],
    "penalty" : ["l1","l2","elasticnet"],
    "solver" : ["newton-cg", "lbfgs", "liblinear"]
}

from sklearn.model_selection import GridSearchCV
# make a GridSearchCV object
GS = GridSearchCV(estimator = log_reg,
                  param_grid = search_space,
                  scoring = ["accuracy"], 
                  refit = "accuracy",
                  cv = 2,
                  verbose = 2)

GS.fit(X_train, y_train)

print(GS.best_estimator_) # to get the complete details of the best model
print(GS.best_params_) # to get only the best hyperparameter values that we searched for
print(GS.best_score_) # score according to the metric we passed in refit

df_gridsearch = pd.DataFrame(GS.cv_results_)
df_gridsearch[["mean_score_time","params","mean_test_accuracy"]].sort_values("mean_test_accuracy",ascending=False).head()


# Save model
pickle.dump(log_reg, open(filename, 'wb'))


# --------------------------------------
# Log_reg Prediction
# --------------------------------------

one_pos = ["Excellent service from all. Clean hotel. Room was phenomenal. Shower and beds. They rock. Stay here cannot beat it."
 "Highly recommend. Lots of food choices. Carry out delivery available. Staff very helpful. Even have tallvehicle parking."]

print("Pos prediction: {}". format(log_reg.predict(vect.transform(one_pos))))


# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# MODEL 2: KNN, majority vote (1min to run at 100k)
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Model loading
filename = 'knn_model.sav'
knn = pickle.load(open(filename, 'rb'))

# OR model building
knn = KNeighborsClassifier(n_neighbors=3,p=2,leaf_size=10)
knn.fit(X_train, y_train)

# _______________________________________________
# Testing KNN (4min 13s)

y_pred = knn.predict(X_test)

# First evaluation
knn_score = knn.score(X_test, y_test)


# 0.61 with 100k reviews; 3min67s
print("Results for KNN Classifier with tfidf")
print(knn_score)

cv_results = cross_val_score(knn, X_train, y_train, cv=5)
np.mean(cv_results)

# Accuracy
metrics.accuracy_score(y_test, y_pred)

# F1 score
metrics.f1_score(y_test, y_pred)

# Area under ROC curve
metrics.roc_auc_score(y_test, y_pred)

# Confusion matrix 
metrics.confusion_matrix(y_test, y_pred)/len(y_test)


# --------------------------------------
# KNN Hyperparameter tuning
# --------------------------------------
# To know what scoring methods to use in GridSearch
# metrics.SCORERS.keys()

# make a dictionary of hyperparameter values to search. Parameters of model
search_space = {
    "n_neighbors" : [3,5,7,9],
    "leaf_size" : [10,30,50],
    "p" : [1,2]
}


# make a GridSearchCV object
GS = GridSearchCV(estimator = knn,
                  param_grid = search_space,
                  scoring = ["accuracy"], 
                  refit = "accuracy",
                  cv = 2,
                  verbose = 2)

GS.fit(X_train, y_train)

print(GS.best_estimator_) # to get the complete details of the best model
print(GS.best_params_) # to get only the best hyperparameter values that we searched for
print(GS.best_score_) # score according to the metric we passed in refit

df_gridsearch = pd.DataFrame(GS.cv_results_)
df_gridsearch[["mean_score_time","params","mean_test_accuracy"]].sort_values("mean_test_accuracy",ascending=False).head()

# Prediction
one_pos = ["mean_score_time","params","mean_test_accuracy"]
print("Pos prediction: {}". format(knn.predict(vect.transform(one_pos))))

# Save model, wb means write mode
pickle.dump(knn, open(filename, 'wb'))


# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# MODEL 3: SVM: Hyperplane separator
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Load the model
filename = 'svm_model.sav'
svmach = pickle.load(open(filename, 'rb'))

#OR Create a svm classifier
svmach = svm.SVC(C=1.0, kernel='linear')
svmach.fit(X_train, y_train)

# ____________________________________________


y_pred = svmach.predict(X_test)

# Confusion matrix
metrics.confusion_matrix(y_test, y_pred)/len(y_test)

# About 10 minutes
cv_results = cross_val_score(svmach, X_train, y_train, cv=2)
print(np.mean(cv_results))

# Accuracy
metrics.accuracy_score(y_test, y_pred)

# F1 score
metrics.f1_score(y_test, y_pred)

# Area under ROC curve
metrics.roc_auc_score(y_test, y_pred)


# --------------------------------------
# SVM tuning
# --------------------------------------
# To know what scoring methods to use in GridSearch
# metrics.SCORERS.keys()

# make a dictionary of hyperparameter values to search. Parameters of model
search_space = {
    "C" : [100,10,1,0.1,0.01],
    "kernel" : ["linear","poly","sigmoid"]
}

from sklearn.model_selection import GridSearchCV
# make a GridSearchCV object
GS = GridSearchCV(estimator = svmach,
                  param_grid = search_space,
                  scoring = ["accuracy"], 
                  refit = "accuracy",
                  cv=2,
                  verbose = 2)

GS.fit(X_train, y_train)

print(GS.best_estimator_) # to get the complete details of the best model
print(GS.best_params_) # to get only the best hyperparameter values that we searched for
print(GS.best_score_) # score according to the metric we passed in refit

df_gridsearch = pd.DataFrame(GS.cv_results_)
df_gridsearch[["mean_fit_time","mean_score_time","params","mean_test_accuracy"]].sort_values("mean_test_accuracy",ascending=False).head()


# --------------------------------------
# Model storage (15min to run at 100k)
# --------------------------------------

# Save model, wb means write mode

pickle.dump(svmach, open(filename, 'wb'))
 
 
# load the model from disk

# 1.45 min to get result from loaded svm
result = svmach.score(X_test, y_test)
print(result)

# Area under ROC curve
metrics.roc_auc_score(y_test, y_pred)


one_neg = ["We have previously stayed in hotels in Spain, Portugal, France, United Kingdom and Ireland, and this is the first time that a hotel has asked us to pay a 75€ deposit (plus the full price of the room) before entering the room."
"Also, before we left, the staff made us wait 10 minutes in the lobby while they confirmed that the room was in order before releasing our deposit."
"We did not feel welcome at any time during our stay."
"Some of the staff seemed more concerned about their own makeup than the hostel guests."
"Buffet style breakfast not according to a 4 star hotel."
"Avoid it if possible."]


print("Neg prediction: {}". format(svmach.predict(vect.transform(one_neg))))


# """"""""""""""""""""""""""""""""""""""""""""""""
# More predictions, all models
# """"""""""""""""""""""""""""""""""""""""""""""""

pos_test=["It was a pleasant stay at the hotel. Covid rules were enforced, which reassured us of our safety. The (bath)room is overall clean and the staff were friendly. Kudos to the gentleman at the restaurant for bringing the croissants to our room after a mistake when placing an order"]

print("Pos prediction by log_reg: {}". format(log_reg.predict(vect.transform(pos_test))))
print("Pos prediction by knn: {}". format(knn.predict(vect.transform(pos_test))))
print("Pos prediction by svm: {}". format(svmach.predict(vect.transform(pos_test))))


neg_test = ["This is the dirtiest hotel we have ever been to. Everything was dirty. Bedsheets and Towels even after the fourth change, every new one had dirty spots. Dust everywhere, Stains on the sink, cups, glasses ... We left one day earlier because my boyfriend did not want to spend his birthday in this messy hotel. We felt very uncomfortable even though it is a hotel with four stars."]

print("Neg prediction by log_reg: {}". format(log_reg.predict(vect.transform(neg_test))))
print("Neg prediction by knn: {}". format(knn.predict(vect.transform(neg_test))))
print("Neg prediction by svm: {}". format(svmach.predict(vect.transform(one_neg))))


