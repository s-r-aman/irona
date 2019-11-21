import pickle
import re
import argparse
from nltk.corpus import stopwords
ap = argparse.ArgumentParser()
ap.add_argument('input')
args = vars(ap.parse_args())

input = args['input']

tfidf_vectorizer_file = open('tfidf_vectorizer.pkl', 'rb') 
clf_file = open('clf.pkl', 'rb') 
multilabel_binarizer_file = open('multilabel_binarizer.pkl', 'rb') 

tfidf_vectorizer = pickle.load(tfidf_vectorizer_file)
clf = pickle.load(clf_file)
multilabel_binarizer = pickle.load(multilabel_binarizer_file)

stop_words = set(stopwords.words('english'))

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

def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)

sen = "Katherine Sullivan, a severe agoraphobic, witnesses the murder of her husband and speaks with the investigating detective. Then both the body and the detective disappear. Katherine hires private investigator Jack Mize to figure out, only Mize isn't so sure Katherine's version of reality is the truth."
res = infer_tags(input)
print(res)