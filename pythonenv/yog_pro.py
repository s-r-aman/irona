import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import pickle

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score

pd.set_option('display.max_colwidth', 300)

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

# function for text cleaning 
def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text

meta = pd.read_csv("movie.metadata.tsv", sep = '\t', header = None)

#print(meta.head())

# rename columns
meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]

plots = []

with open("plot_summaries.txt", 'r') as f:
       reader = csv.reader(f, dialect='excel-tab') 
       for row in tqdm(reader):
            plots.append(row)

movie_id = []
plot = []

# extract movie Ids and plot summaries
for i in tqdm(plots):
    movie_id.append(i[0])
    plot.append(i[1])

# create dataframe
movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})

# change datatype of 'movie_id'
meta['movie_id'] = meta['movie_id'].astype(str)

# merge meta with movies
movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id')

# an empty list
genres = [] 

# extract genres
for i in movies['genre']: 
    genres.append(list(json.loads(i).values())) 

# add to 'movies' dataframe  
movies['genre_new'] = genres

# remove samples with 0 genre tags
movies_new = movies[~(movies['genre_new'].str.len() == 0)]

# get all genre tags in a list
all_genres = sum(genres,[])
#print(len(set(all_genres)))

all_genres = nltk.FreqDist(all_genres) 

# create dataframe
all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 
                              'Count': list(all_genres.values())})

movies_new['clean_plot'] = movies_new['plot'].apply(lambda x: clean_text(x))

stop_words = set(stopwords.words('english'))


movies_new['clean_plot'] = movies_new['clean_plot'].apply(lambda x: remove_stopwords(x))

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies_new['genre_new'])

# transform target variable
y = multilabel_binarizer.transform(movies_new['genre_new'])

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

# split dataset into training and validation set
xtrain, xval, ytrain, yval = train_test_split(movies_new['clean_plot'], y, test_size=0.2, random_state=9)

# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

lr = LogisticRegression()
clf = OneVsRestClassifier(lr)

# fit model on train data
clf.fit(xtrain_tfidf, ytrain)

# make predictions for validation set
y_pred = clf.predict(xval_tfidf)

lab = multilabel_binarizer.inverse_transform(y_pred)[3]
print(lab)

# evaluate performance
print(f1_score(yval, y_pred, average="micro"))

clf_file = open('clf.pkl', 'wb')
pickle.dump(clf, clf_file)
clf_file.close()

tfidf_vectorizer_file = open('tfidf_vectorizer.pkl', 'wb')
pickle.dump(tfidf_vectorizer, tfidf_vectorizer_file)
tfidf_vectorizer_file.close()

multilabel_binarizer_file = open('multilabel_binarizer.pkl', 'wb')
pickle.dump(multilabel_binarizer, multilabel_binarizer_file)
multilabel_binarizer_file.close()
