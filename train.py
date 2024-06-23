from dataset import get_train_data


def train():
    tfidf_matrix, tags = get_train_data()
    print(tfidf_matrix, tags)

if __name__ == '__main__':
    train()