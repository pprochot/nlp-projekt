import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset')
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
TEST_PATH = os.path.join(DATASET_PATH, 'test')

TRAIN_TAGS_PATH = os.path.join(TRAIN_PATH, 'training_set_clean_only_tags.txt')
TRAIN_TEXT_PATH = os.path.join(TRAIN_PATH, 'training_set_clean_only_text.txt')

def open_file(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


# ? https://towardsdatascience.com/word-embedding-techniques-word2vec-and-tf-idf-explained-c5d02e34d08
def get_train_data():
    tfidf_vectorizer = setup_tfidf_vectorizer()

    documents = open_file(TRAIN_TEXT_PATH)
    tags = open_file(TRAIN_TAGS_PATH)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    return tfidf_matrix, tags


def setup_tfidf_vectorizer():
    return TfidfVectorizer(input='content', tokenizer=word_tokenize)
    # ? Tokenizer needed here?

def get_test_data():
    open_file()


if __name__ == '__main__':
    nltk.download('punkt') # ! For some reason needed
    get_train_data()
