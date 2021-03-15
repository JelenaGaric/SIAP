import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
import numpy as numpy
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score


path = 'dataset\mpst_full_data.csv'

def import_dataset(path):
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


results = set()
train_df['tags'].str.lower().str.split(",").apply(results.update)
categories_num = len(results)
print("Number of categories: ", categories_num)


# print(train_df['tags'].toarray())

vectorizer = CountVectorizer(tokenizer = lambda x: x.split(","), binary='true')
y_train = vectorizer.fit_transform(train_df['tags']).toarray()
y_test = vectorizer.transform(test_df['tags']).toarray()

# print(max(df['plot_synopsis'].apply(max_len)))

vect=Tokenizer()
vect.fit_on_texts(train_df['plot_synopsis'])
vocab_size = len(vect.word_index) + 1
# print(vocab_size)

encoded_docs_train = vect.texts_to_sequences(train_df['preprocessed_plots'])
max_length = vocab_size
padded_docs_train = pad_sequences(encoded_docs_train, maxlen=1200, padding='post')
#print(padded_docs_train)

encoded_docs_test =  vect.texts_to_sequences(test_df['preprocessed_plots'])
padded_docs_test = pad_sequences(encoded_docs_test, maxlen=1200, padding='post')
encoded_docs_cv = vect.texts_to_sequences(val_df['preprocessed_plots'])
padded_docs_cv = pad_sequences(encoded_docs_cv, maxlen=1200, padding='post')

def create_model():
    model = keras.Sequential()
    # Configuring the parameters
    model.add(layers.Embedding(vocab_size, output_dim=50, input_length=1200))
    model.add(layers.LSTM(128, return_sequences=True))
    # Adding a dropout layer
    model.add(layers.Dropout(0.5))
    model.add(layers.LSTM(64))
    model.add(layers.Dropout(0.5))
    # Adding a dense output layer with sigmoid activation
    model.add(layers.Dense(categories_num, activation='sigmoid'))

    print(model.summary())

    METRICS = [

        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
    return model

def train(model):
    history = model.fit(padded_docs_train, y_train,
                        epochs=10,
                        verbose=1,
                        validation_data=(padded_docs_test, y_test),
                        batch_size=16)

    model.save('model')

try:
    reconstructed_model = keras.models.load_model("model")
    model = reconstructed_model
    print("Loaded model.")
except:
    print("No model saved. Training a new one.")
    model = create_model();
    train(model)

predictions = model.predict([padded_docs_test])
thresholds = [0.1, 0.2, 0.3, 0.4]# ,0.5, 0.6, 0.7, 0.8, 0.9]

for val in thresholds:
    print("For threshold: ", val)
    pred = predictions.copy()

    pred[pred >= val] = 1
    pred[pred < val] = 0

    precision = precision_score(y_test, pred, average='micro')
    recall = recall_score(y_test, pred, average='micro')
    f1 = f1_score(y_test, pred, average='micro')

    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

def predict_sample():
    t = train_df.sample(1)
    encoded_docs = vect.texts_to_sequences(t['preprocessed_plots'])
    padded_docs = pad_sequences(encoded_docs, maxlen=1200, padding='post')
    pred = model.predict(padded_docs).tolist()

    for i in range(len(vectorizer.inverse_transform(pred[0])[0])):
        print(pred[0][i], "-->", vectorizer.inverse_transform(pred[0])[0][i])

    for i in range(len(pred[0])):
        if (pred[0][i] < 0.1):
            pred[0][i] = 0
        else:
            pred[0][i] = 1

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    print("Movie title -->", t['title'].values)
    print("Synopsis -->", t['plot_synopsis'].values)
    print("Original tags -->", t['tags'].values)
    print("Predicted tags -->", vectorizer.inverse_transform(pred[0])[0])


predict_sample()
