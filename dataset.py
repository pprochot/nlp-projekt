import os
import joblib
import nltk
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset')
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
TEST_PATH = os.path.join(DATASET_PATH, 'test/task 02')

TRAIN_TAGS_PATH = os.path.join(TRAIN_PATH, 'training_set_clean_only_tags.txt')
TRAIN_TEXT_PATH = os.path.join(TRAIN_PATH, 'training_set_clean_only_text.txt')

TEST_TEXT_PATH = os.path.join(TEST_PATH, 'test_set_only_text.txt')
TEST_TAGS_PATH = os.path.join(TEST_PATH, 'test_set_only_tags.txt')

POLISH_STOPWORDS_PATH = os.path.join(DATASET_PATH, 'polish.stopwords.txt')

def open_file(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


# ? https://towardsdatascience.com/word-embedding-techniques-word2vec-and-tf-idf-explained-c5d02e34d08
def get_train_data():
    documents = open_file(TRAIN_TEXT_PATH)
    tags = open_file(TRAIN_TAGS_PATH)
    return documents, tags


def get_test_data():
    documents = open_file(TEST_TEXT_PATH)
    tags = open_file(TEST_TAGS_PATH)
    return documents, tags


def get_polish_stopwords():
    return open_file(POLISH_STOPWORDS_PATH)

