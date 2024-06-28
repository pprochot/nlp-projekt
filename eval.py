import subprocess

from dataset import get_test_data


def evaluate(model, pipeline):
    documents, tags = get_test_data()
    tfidf_matrix = pipeline.transform(documents)
    # Assume that 'model' is your trained model and 'test_data' is your test data
    predicted_labels = model.predict(tfidf_matrix)

    # Write the predicted labels to a file
    with open('predicted_labels.txt', 'w') as f:
        for label in predicted_labels:
            f.write(f"{label}\n")

    # Call the Perl script from Python
    result = subprocess.run(['perl', 'dataset/test/task 02/evaulate2.pl', 'predicted_labels.txt'], capture_output=True, text=True)
    print(result)
    # Print the output of the Perl script
    print(result.stdout)