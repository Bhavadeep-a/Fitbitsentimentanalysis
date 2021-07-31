from flask import Flask,render_template,request,send_file
import pickle
import pandas as pd
import numpy as np
import nltk
import re
import string
import spacy
from nltk.corpus import stopwords
nltk.tokenize.punkt
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
#import tensorflow as tf
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')



model = pickle.load(open('model.pkl','rb'))

data_raw = pd.read_csv('Fitbit_versa2_reviews_English.csv')

## Definition for eda sentiment
def EDA_sentiment (text_data = None):
    
    review_list = [i.strip() for i in text_data]
    
    afinn = pd.read_csv('Afinn.csv',sep=',', encoding='latin-1')
    affinity_score = afinn.set_index('word')['value'].to_dict()
    sentences = nltk.tokenize.sent_tokenize(' '.join(review_list))

    lemmatizer = WordNetLemmatizer()
    sentences_lemma = []
    
    for i in sentences:
        sentences_lemma.append(lemmatizer.lemmatize(i))
    sentences_df = pd.DataFrame(sentences_lemma, columns = ['Sentences'])
    
    nlp = spacy.load('en_core_web_sm')    
    sentiment_lexicon = affinity_score
    def calculate_sentiment(text: str = None):
        sentiment_score = 0
        sentence = nlp(text)
        for word in sentence:
            sentiment_score += sentiment_lexicon.get(word.lemma_, 0)
        
        return sentiment_score
    
    sentences_df['Sentiment Value'] = sentences_df['Sentences'].apply(calculate_sentiment)
    sentences_df['Sentence Length'] = sentences_df['Sentences'].apply(len)
    sentences_df['Polarity'] = sentences_df['Sentiment Value'].apply(lambda x: 'Positive' if x > 0 else 'Neutral' if x == 0 else 'Negative')
    
    return sentences_df # Use this as an input for data_prep and visualization
    
sentences_df_model_train = EDA_sentiment(data_raw['Reviews'])

def data_prep (text_data, sentences_df = None):
    def text_processed(text):
        Stopwords = stopwords.words('english') # loading stop words
        Stopwords.extend(['fitbit', 'nt', 'watch'])
        no_punc = [i for i in text if i not in string.punctuation]
        no_punc = ''.join(no_punc)
        return ' '.join([i for i in no_punc.split() if i.lower() not in Stopwords])
    X = sentences_df['Sentences'].apply(text_processed) # testing/ data pssed in the function
    y = sentences_df['Polarity'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2}) # testing/data passed in the function
    
    X_train = sentences_df_model_train['Sentences'].apply(text_processed) # training data
    y_train = sentences_df_model_train['Polarity'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2}) #training data

    
    cv = CountVectorizer()
    X_cv_train = cv.fit_transform(X_train) # Fitting CV on training data
    X_cv = cv.transform(X) # transforming testing data
    
    tfidf = TfidfTransformer()
    X_tfidf_train = tfidf.fit_transform(X_cv_train) # Fitting tfidf on training data
    X_tfidf = tfidf.transform(X_cv) # transforming testing data
    
    X_tfidf_train_array = X_tfidf_train.toarray() # Converting training data to array
    X_test_array = X_tfidf.toarray() # Converting testing data to array
    
    return X_test_array, y, X_tfidf_train_array, y_train # Only use X_test_array as input for model_random_forest

# Defining inputs and outputs for training data
X_tfidf_array_model_train = data_prep(data_raw['Reviews'], sentences_df_model_train)[2]
y_model_train = data_prep(data_raw['Reviews'], sentences_df_model_train)[3]

def model_random_forest(X_test_array= None):
    
    predictions = model.predict(X_test_array) # predicting on testing data
    pred_df = pd.DataFrame() # creating a df with actual values and predictions
    pred_df['Predicted'] = predictions
    
    return pred_df


app = Flask(__name__)
@app.route('/')
@app.route('/home')
def main():
    return render_template('home.html')

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/predict',methods = ['POST','GET'])
def predict():
    inp = request.form.get("review")
    a=[inp]
    dftest=EDA_sentiment(a)
    X_test_array = data_prep(a, dftest)[0]
    b=model_random_forest(X_test_array)
    c = b['Predicted'].max()
    if c == 2:
      result = "Great Work there! It's is a Positive Review 😃 with sentiment score 7"
    elif c == 1:
        result = "good! but there's room for improvement!It's a neutral Review😔 with sentiment score 0"
    else:
        result = "Try improving your product! It's a negative Review 😔 with sentiment score -2"
    # graphJSON  = visualization(dftest)
    return render_template('models.html',message=result)

if __name__ == '__main__':
    app.run()


