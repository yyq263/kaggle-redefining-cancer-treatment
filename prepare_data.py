import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer

def prepare_data():
    df_train = pd.read_csv('data/training_variants')
    df_train_text = pd.read_csv('data/training_text', sep='\|\|', header=None, skiprows=1, names=['ID', 'Text'])
    df_test = pd.read_csv('data/test_variants')
    df_test_text = pd.read_csv('data/test_text', sep='\|\|', header=None, skiprows=1, names=['ID', 'Text'])
    df_train = pd.merge(df_train, df_train_text, how="left", on="ID")
    df_test = pd.merge(df_test, df_test_text, how="left", on="ID")
    return df_train, df_test

df_train, df_test = prepare_data()
df_train = df_train.iloc[:50, :] 
df_test = df_test.iloc[:50, :]

def get_vocab(df_train, df_test):
    count_vectorizer = CountVectorizer(
        analyzer="word", tokenizer=nltk.word_tokenize,
        preprocessor=None, stop_words='english', max_features=None)
    docu = count_vectorizer.fit_transform(df_train.Text)
    return docu

docu = get_vocab(df_train, df_test)
print("vocabulary size: ", docu.shape) # how many unique words?





