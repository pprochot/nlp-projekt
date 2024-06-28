import os

import joblib
import nltk
import numpy
import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from dataset import get_train_data, get_test_data, get_polish_stopwords
from sklearn.naive_bayes import MultinomialNB
import eval

nlp = spacy.load('pl_core_news_sm')


def polish_tokenizer(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens


def setup_tfidf_vectorizer():
    return TfidfVectorizer(input='content', stop_words=get_polish_stopwords(), tokenizer=polish_tokenizer,
                           lowercase=True)

def train(svd_components):
    documents, tags = get_train_data()

    model = LinearSVC(class_weight='balanced')
    tfidf_vectorizer = setup_tfidf_vectorizer()
    svd = TruncatedSVD(n_components=svd_components)
    pipeline = make_pipeline(tfidf_vectorizer, svd)

    tfidf_matrix = pipeline.fit_transform(documents)
    # total_variance = svd.explained_variance_ratio_.sum()
    # print("Total variance explained by SVD: ", total_variance)

    model.fit(tfidf_matrix, tags)
    return model, pipeline


def test(model, pipeline):
    eval.evaluate(model, pipeline)


def predict(model, pipeline, text):
    tfidf_matrix = pipeline.transform([text])
    prediction = model.predict(tfidf_matrix)
    return prediction


if __name__ == '__main__':
    if os.path.isfile('pipeline.joblib') and os.path.isfile('model.joblib'):
        pipeline = joblib.load('pipeline.joblib')
        model = joblib.load('model.joblib')
    else:
        model, pipeline = train(svd_components=7100)  # 7100 => Variance > 0.95
        joblib.dump(pipeline, 'pipeline.joblib')
        joblib.dump(model, 'model.joblib')

    test(model, pipeline)
    # 0 (non-harmful), 1 (cyberbullying), 2 (hate-speech)
    print(predict(model, pipeline, "Obsrałeś znowu zbroje pajacu!"))  # 2
    print(predict(model, pipeline, "Żal ci biedaku??? Gdyby nie Kaczyński to by je twoi przyjaciele z PO rozkradl!"))  # 1
    print(predict(model, pipeline, "Żal ci biedaku??? Gdyby nie Tusk to by je twoi przyjaciele z PiS rozkradl!"))  # 1
    print(predict(model, pipeline, "Won bo psami poszczuje! Niech spier!"))  # 2 jeśli dodam pełne słowo
    print(predict(model, pipeline, "Idę zaraz do sklepu, kupić ci coś?"))  # 0
    print(predict(model, pipeline, "Chyba będziemy mieli ekpię, to jak, gramy zaraz?"))  # 0
    print(predict(model, pipeline, "W wakacje byliśmy w Barcelonie"))  # 0
    print(predict(model, pipeline, "Jesteś najgorszy, mam cię dosyć!"))  # 0
    print(predict(model, pipeline, "Jesteś najlepsza, cieszę się, że cię mam!"))  # 0

