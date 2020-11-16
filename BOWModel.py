import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from bert_embedding import BertEmbedding


#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

import os
print(os.listdir("../IMDB Datasets/archive"))
import warnings
warnings.filterwarnings('ignore')

#Importing the training data
imdbTrainData = pd.read_csv('../IMDB Datasets/archive/Train.csv')
imdbTestData = pd.read_csv('../IMDB Datasets/archive/Test.csv')

#Printing some detailes from the training data
print(imdbTrainData.shape)
print(imdbTrainData.head(10))
print(imdbTrainData.describe())
print(imdbTrainData['label'].value_counts())

#Tokenization of text
tokenizer = ToktokTokenizer()

#Setting English stopwords
#nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english') # This command needed ===> python3 -m nltk.downloader stopwords

#Apply function on review column
imdbTrainData['text'] = imdbTrainData['text'].apply(denoise_text)
imdbTestData['text'] = imdbTestData['text'].apply(denoise_text)

#Apply function on review column
imdbTrainData['text'] = imdbTrainData['text'].apply(remove_special_characters)
imdbTestData['text'] = imdbTestData['text'].apply(remove_special_characters)

#Apply function on review column
imdbTrainData['text'] = imdbTrainData['text'].apply(simple_stemmer)
imdbTestData['text'] = imdbTestData['text'].apply(simple_stemmer)

#set stopwords to english
stop=set(stopwords.words('english'))
print(stop)

#Apply function on review column
imdbTrainData['text']=imdbTrainData['text'].apply(remove_stopwords)
imdbTestData['text']=imdbTestData['text'].apply(remove_stopwords)

#normalized train reviews
norm_train_reviews = imdbTrainData.text

#Normalized test reviewssirali

norm_test_reviews = imdbTestData.text

#Count vectorizer for bag of words
# cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
# #transformed train reviews
# cv_train_reviews=cv.fit_transform(norm_train_reviews)
# #transformed test reviews
# cv_test_reviews=cv.transform(norm_test_reviews)


bert_embedding = BertEmbedding()
cv_train_reviews = bert_embedding(norm_train_reviews, 'sum')

cv_test_reviews = bert_embedding(norm_test_reviews, 'sum')



# print('BOW_cv_train:',cv_train_reviews.shape)
# print('BOW_cv_test:',cv_test_reviews.shape)

#labeling the sentient data
lb=LabelBinarizer()
#transformed sentiment data
train_sentiments = lb.fit_transform(imdbTrainData['label'])
test_sentiments = lb.fit_transform(imdbTestData['label'])

#training the model
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)

#Fitting the model for Bag of words
lr_bow=lr.fit(cv_train_reviews,train_sentiments)
print(lr_bow)


#Predicting the model for bag of words
lr_bow_predict=lr.predict(cv_test_reviews)
print(lr_bow_predict)

#Accuracy score for bag of words
lr_bow_score=accuracy_score(test_sentiments,lr_bow_predict)
print("lr_bow_score :",lr_bow_score)


#Classification report for bag of words
lr_bow_report=classification_report(test_sentiments,lr_bow_predict,target_names=['Positive','Negative'])
print(lr_bow_report)

#confusion matrix for bag of words
cm_bow=confusion_matrix(test_sentiments,lr_bow_predict,labels=[1,0])
print(cm_bow)

#training the linear svm
# defining parameter range
param_grid = {'C': [1.0],
              'gamma': ['scale'],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=False, verbose=1)

#svm=SGDClassifier(loss='hinge',max_iter=500,random_state=42)

#fitting the svm for bag of words
svm_bow=grid.fit(cv_train_reviews,train_sentiments)
print(svm_bow)

#Predicting the model for bag of words
svm_bow_predict=grid.predict(cv_test_reviews)
print(svm_bow_predict)

#Accuracy score for bag of words
svm_bow_score=accuracy_score(test_sentiments,svm_bow_predict)
print("svm_bow_score :",svm_bow_score)

#Classification report for bag of words
svm_bow_report=classification_report(test_sentiments,svm_bow_predict,target_names=['Positive','Negative'])
print(svm_bow_report)

#confusion matrix for bag of words
cm_bow=confusion_matrix(test_sentiments,svm_bow_predict,labels=[1,0])
print(cm_bow)

#training the model
mnb=MultinomialNB()
#fitting the svm for bag of words
mnb_bow=mnb.fit(cv_train_reviews,train_sentiments)
print(mnb_bow)

#Predicting the model for bag of words
mnb_bow_predict=mnb.predict(cv_test_reviews)
print(mnb_bow_predict)

#Accuracy score for bag of words
mnb_bow_score=accuracy_score(test_sentiments,mnb_bow_predict)
print("mnb_bow_score :",mnb_bow_score)

#Classification report for bag of words
mnb_bow_report=classification_report(test_sentiments,mnb_bow_predict,target_names=['Positive','Negative'])
print(mnb_bow_report)





