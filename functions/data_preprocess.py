import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import re
import string

punct = set(string.punctuation)

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
# print(stop_words)

from nltk.stem.porter import *
stemmer = PorterStemmer()

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from bs4 import BeautifulSoup


def text_cleaning(df, review_column):
    df['tidy_review'] = df[f'{review_column}']

    for j in tqdm(range(df.shape[0])):
        # берем один отзыв
        text = df[f'{review_column}'][j]
        # все слова к одному регистру
        text = text.lower()
        # html
        text = BeautifulSoup(text, 'lxml').text
        # ненужные символы и знаки
        text = re.sub('[+\-*/]', '', text)
        text = re.sub('\d+(\.\d*)?', '', text)
        # убираем стоп-слова
        text = " ".join([word for word in text.split() if word not in stop_words])
        # пунктуация
        text = "".join([p for p in text if p not in punct])
        # достаем токены
        text = text.split()
        # стемминг
        text = [stemmer.stem(i) for i in text]
        # лемминг
        # text = [lemmatizer.lemmatize(i) for i in text]
        # соединяем обратно
        text = ' '.join(text)

        df['tidy_review'][j] = text

    return df


def light_text_cleaning(df, review_column):
    df['tidy_review'] = df[f'{review_column}']

    for j in tqdm(range(df.shape[0])):
        # берем один отзыв
        text = df[f'{review_column}'][j]
        # все слова к одному регистру
        # text = text.lower()
        # html
        text = BeautifulSoup(text, 'lxml').text
        # ненужные символы и знаки
        text = re.sub('[+\-*/]', '', text)
        text = re.sub('\d+(\.\d*)?', '', text)
        df['tidy_review'][j] = text

    return df
