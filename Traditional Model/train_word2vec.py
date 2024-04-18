import pandas as pd
import csv
import numpy as np
import nltk
import re
import os
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec


def Load_dataset():
    def parse_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return pd.DataFrame(data)
    
    train_data = parse_jsonl('..\\..\\dataset\\train.jsonl')
    test_data = parse_jsonl('..\\..\\dataset\\test.jsonl')
    
    x_train = train_data['string'].astype(str)
    y_train = train_data['label'].astype(str)
    x_test = test_data['string'].astype(str)
    y_test = test_data['label'].astype(str)
    
    label_map = {'background': 0, 'method': 1, 'result': 2}
    y_train = y_train.map(label_map)
    y_test = y_test.map(label_map)
    
    return x_train, y_train, x_test, y_test

def preprocess(text_series):
    text_series = text_series.str.lower()
    text_series = text_series.apply(lambda text: re.sub(r'\d+', '', text))  # Remove numbers
    text_series = text_series.apply(lambda text: re.sub(r'[^\w\s]', '', text))  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    text_series = text_series.apply(lambda text: ' '.join(word for word in word_tokenize(text) if word not in stop_words))
    return text_series

def train_and_save_word2vec_model(preprocessed_texts, model_filename):
    # Tokenize the sentences into words
    tokenized_data = [text.split() for text in preprocessed_texts]
    
    # Train the Word2Vec model
    model = Word2Vec(sentences=tokenized_data, vector_size=300, window=5, min_count=1, workers=4)

    # Save the model
    model.save(model_filename)
    print(f'Model saved to {model_filename}')

def main():
    print("Loading data...")
    x_train, y_train, x_test, y_test = Load_dataset()
    print("Done loading data.")

    print("Preparing the data...")
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    print("Data preparation complete.")

    # Train and save the Word2Vec model
    print("Training the Word2Vec model...")
    train_and_save_word2vec_model(x_train, "word2vec_model_300d.model")
    print("Word2Vec model training and saving complete.")

if __name__ == '__main__':
    main()
