import sys
import numpy as np
import random as rn
import pandas as pd
import torch
import nltk
from pytorch_pretrained_bert import BertModel
from torch import nn
# from torchnlp.datasets import imdb_dataset      # --> We are using our own uploaded dataset.
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from nltk.tokenize.toktok import ToktokTokenizer
from bert_embedding import BertEmbedding
bert_embedding = BertEmbedding()


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


rn.seed(321)
np.random.seed(321)
torch.manual_seed(321)
torch.cuda.manual_seed(321)

#Importing the training data
imdbTrainData = pd.read_csv('../IMDB Datasets/archive/Train.csv')
imdbTestData = pd.read_csv('../IMDB Datasets/archive/Test.csv')

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

imdbTrainData = imdbTrainData.to_dict(orient='records')
imdbTestData = imdbTestData.to_dict(orient='records')

train_texts, train_labels = list(zip(*map(lambda d: (d['text'], d['label']), imdbTrainData)))
test_texts, test_labels = list(zip(*map(lambda d: (d['text'], d['label']), imdbTestData)))

print(len(train_texts), len(train_labels), len(test_texts), len(test_labels))

# sentences = [len(sent) for sent in train_texts]
# print(sentences)
# plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
# plt.bar(range(1,5001), sentences, color = ['red'])
# plt.gca().set(title='No. of characters in each sentence', xlabel='Number of sentence', ylabel='Number of Characters in each sentence');

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

print(tokenizer.tokenize('Hi my name is Atul'))

train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], train_texts))
test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], test_texts))

train_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, train_tokens)), maxlen=512, truncating="post", padding="post", dtype="int")
test_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, test_tokens)), maxlen=512, truncating="post", padding="post", dtype="int")

train_y = train_labels
test_y = test_labels

train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

baseline_model = make_pipeline(CountVectorizer(ngram_range=(1,3)), LogisticRegression()).fit(train_texts, train_labels)
baseline_predicted = baseline_model.predict(test_texts)

print(classification_report(test_labels, baseline_predicted))