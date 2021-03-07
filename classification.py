import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers

path = 'dataset\mpst_full_data.csv'

def import_dataset(path):
    # return pd.read_csv(path, names=['Title', 'Conditions', 'ONCOLOGY'], skiprows=1)
    return pd.read_csv(path)

def max_len(x):
    a=x.split()
    return len(a)

def make_data_frames(full_df):

    train_df = pd.DataFrame(columns=['imdb_id', 'title','plot_synopsis', 'tags'])
    test_df = pd.DataFrame(columns=['imdb_id', 'title','plot_synopsis', 'tags'])
    val_df = pd.DataFrame(columns=['imdb_id', 'title','plot_synopsis', 'tags'])

    for index, row in full_df.iterrows():
        if row["split"] == "train":
            train_df = train_df.append({
                'imdb_id': full_df.loc[index, 'imdb_id'],
                 'title': full_df.loc[index, 'title'],
                 'plot_synopsis': full_df.loc[index, 'plot_synopsis'],
                 'tags': full_df.loc[index, 'tags']
                 }, ignore_index=True)
        elif row["split"] == "test":
            test_df = test_df.append({
                'imdb_id': full_df.loc[index, 'imdb_id'],
                 'title': full_df.loc[index, 'title'],
                 'plot_synopsis': full_df.loc[index, 'plot_synopsis'],
                 'tags': full_df.loc[index, 'tags']
                 }, ignore_index=True)
        elif row["split"] == "val":
            val_df = val_df.append({
                'imdb_id': full_df.loc[index, 'imdb_id'],
                'title': full_df.loc[index, 'title'],
                'plot_synopsis': full_df.loc[index, 'plot_synopsis'],
                'tags': full_df.loc[index, 'tags']
            }, ignore_index=True)

    train_df.to_csv('dataset\\train_dataset.csv')
    test_df.to_csv('dataset\\test_dataset.csv')
    val_df.to_csv('dataset\\validation_dataset.csv')
    return train_df, test_df, val_df

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

df = import_dataset(path)
pd.set_option('max_columns', None)

preprocessed_synopsis = []

for sentence in df['plot_synopsis'].values:
    sentence = decontracted(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
    preprocessed_synopsis.append(sentence.strip())

df['preprocessed_plots']=preprocessed_synopsis
df.to_csv('dataset\\dataset.csv')

def remove_spaces(x):
    x=x.split(",")
    nospace=[]
    for item in x:
        item=item.lstrip()
        nospace.append(item)
    return (",").join(nospace)

df['tags']=df['tags'].apply(remove_spaces)

train_df=df.loc[df.split=='train']
train_df=train_df.reset_index()
test_df=df.loc[df.split=='test']
test_df=test_df.reset_index()
val_df=df.loc[df.split=="val"]
val_df=val_df.reset_index()

vectorizer = CountVectorizer(tokenizer = lambda x: x.split(","), binary='true')
y_train = vectorizer.fit_transform(train_df['tags']).toarray()
y_test = vectorizer.transform(test_df['tags']).toarray()

print(max(df['plot_synopsis'].apply(max_len)))

vect=Tokenizer()
vect.fit_on_texts(train_df['plot_synopsis'])
vocab_size = len(vect.word_index) + 1
print(vocab_size)

encoded_docs_train = vect.texts_to_sequences(train_df['preprocessed_plots'])
max_length = vocab_size
padded_docs_train = pad_sequences(encoded_docs_train, maxlen=1200, padding='post')
print(padded_docs_train)

encoded_docs_test =  vect.texts_to_sequences(test_df['preprocessed_plots'])
padded_docs_test = pad_sequences(encoded_docs_test, maxlen=1200, padding='post')
encoded_docs_cv = vect.texts_to_sequences(val_df['preprocessed_plots'])
padded_docs_cv = pad_sequences(encoded_docs_cv, maxlen=1200, padding='post')

