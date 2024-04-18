import pandas as pd
import csv
import numpy as np
import nltk
import re
import os
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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


def Preprocess(text_series):
    # Convert text to lowercase
    text_series = text_series.str.lower()
    text_series = text_series.apply(lambda text: re.sub(r'\d+', '', text))  # Remove numbers
    text_series = text_series.apply(lambda text: re.sub(r'[^\w\s]', '', text))  # Remove punctuation
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text_series = text_series.apply(lambda text: ' '.join(word for word in word_tokenize(text) if word not in stop_words))
    
    return text_series

def train_and_save_doc2vec_model(preprocessed_texts, model_filename):
    tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(preprocessed_texts)]
    model = Doc2Vec(vector_size=200, alpha=0.025, min_alpha=0.00025, min_count=1, dm=1)
    model.build_vocab(tagged_data)
    for epoch in range(10):
        print(f'Iteration {epoch}')
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
    model.save(model_filename)
    print(f'Model saved to {model_filename}')

def load_and_use_model(model_filename, preprocessed_texts):
    if os.path.exists(model_filename):
        model = Doc2Vec.load(model_filename)
        print(f'Model loaded from {model_filename}')
        vectors = [model.infer_vector(text.split()) for text in preprocessed_texts]
        return vectors
    else:
        print("Model file not found!")
        return None


def main():
    
    print("loading data...")
    x_train, y_train, x_test, y_test = Load_dataset()
    print("done")

    print("Preparing the data...")
    x_train = Preprocess(x_train)
    x_test = Preprocess(x_test)
    print("done")

    print("Vectorising the text...")
    train_and_save_doc2vec_model(x_train, "doc2vec_model.model")

    # vectors = load_and_use_model("doc2vec_model.model", preprocessed_texts)
    
    print("saved")

if __name__ =='__main__':
    
    main()