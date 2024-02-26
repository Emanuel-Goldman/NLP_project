import spacy
import os
from main import tokenize_text_to_sentences
from spacy.language import Language
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


#adding sentence tokenization rule for apostrophes for spacy
@Language.component('set_custom_boundaries')
def set_custom_boundaries(doc):
    predecessor_token = doc[0]
    for token in doc[1:-1]:
        if token.text == "'" and predecessor_token.text == ".":
            doc[token.i+1].is_sent_start = False
        predecessor_token = token
    return doc


#tokenize a chapter to sentences and save as a json file
def save_chaps_tokenized_to_sents(read_path, write_path, num_of_books, nlp, book_name):

    for i in range(num_of_books):

        read_path_for_chap = os.path.join(read_path, book_name + str(i) + ".txt")
        write_path_for_chap = os.path.join(write_path, book_name + str(i) + ".json")

        with open(read_path_for_chap, "r", encoding="utf-8") as file:
            chap = file.read()

        doc = nlp(chap)
        tokenized_chap = tokenize_text_to_sentences(doc)

        with open(write_path_for_chap, "w") as json_file:
            json.dump(tokenized_chap, json_file)


#create a vector of length of sentences for a single chapter
def chap_float_length_of_sentences_vector(chap_sentences, nlp):

    length_of_sentences_vector = []

    for sentence in chap_sentences:
        doc = nlp(sentence)
        length = float(sum(1 for token in doc if not token.is_punct and not token.is_space))
        length_of_sentences_vector.append(length)

    return length_of_sentences_vector


#create a matrix of lengths of sentences of all chapters and save as a json file
def chaps_float_length_of_sentences_matrix(tokenized_chaps_path, output_path, nlp):

    length_of_sentences_matrix = []
    files = sorted(os.listdir(tokenized_chaps_path))

    for filename in files:
        filepath = os.path.join(tokenized_chaps_path, filename)
        with open(filepath, "r") as json_file:
            chap_sentences = json.load(json_file)

        length_of_sentences_matrix.append(chap_float_length_of_sentences_vector(chap_sentences, nlp))

    with open(output_path, "w") as json_file:
        json.dump(length_of_sentences_matrix, json_file)


#pad the matrix of lengths of sentences with zeros to the longest chapter size
def padding_float_length_of_sentences_matrix(read_path, output_path):

    with open(read_path, "r") as json_file:
        length_of_sentences_matrix = json.load(json_file)

    padded_sequences = []
    max_length = max(len(row) for row in length_of_sentences_matrix)

    for row in length_of_sentences_matrix:
        padded_row = np.pad(row, (0, max_length - len(row)), mode='constant', constant_values=0.0)
        padded_sequences.append(padded_row)

    padded_matrix = np.array(padded_sequences)
    padded_matrix = padded_matrix.tolist()

    with open(output_path, "w") as json_file:
        json.dump(padded_matrix, json_file)


#lable chapters to periods vector and save as a json file
def label_chaps_to_periods(tokenized_chaps_path, output_path):

    labels = []

    files = sorted(os.listdir(tokenized_chaps_path))

    for filename in files:
        parts = filename.split('(')
        year = int(parts[-1].split(')')[0])

        if year < 1844:
            labels.append(1)
        elif year < 1854:
            labels.append(2)
        else:
            labels.append(3)

    with open(output_path, "w") as json_file:
        json.dump(labels, json_file)


def chaps_int_length_of_sentences_matrix(read_path, write_path):

    with open(read_path, "r") as json_file:
        float_length_of_sentences_matrix = json.load(json_file)

    int_length_of_sentences_matrix = [[int(value) for value in row] for row in float_length_of_sentences_matrix]

    with open(write_path, "w") as json_file:
        json.dump(int_length_of_sentences_matrix, json_file)


#classifying the periods of the chapters using mlp classifier
def mlp_classifier_length_of_sentences(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)

    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)

    return accuracy #45%


def logistic_regression_length_of_sentences(X, y):

    scaler = preprocessing.StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42, max_iter=10000)

    clf.fit(X_train_scaled, y_train)

    y_predict = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_predict)
    print(accuracy) #46%


def main():

    #loading spacy pipline
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('set_custom_boundaries', before="parser")

    #folder paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    chaps_path = os.path.join(ROOT_DIR, 'chaps')
    tokenized_chaps_path = os.path.join(ROOT_DIR, 'Tokenized_Chaps')

    #AYELET- running AI
    path = os.path.join(ROOT_DIR, 'Input_for_AI')
    path2 = os.path.join(path, "padded_int_length_of_sentences_matrix.json")
    path3 = os.path.join(path, "label_chaps_to_periods.json")
    with open(path2, "r") as json_file:
        padded_int_length_of_sentences_matrix = json.load(json_file)
    with open(path3, "r") as json_file:
        label_chaps_to_periods = json.load(json_file)
    logistic_regression_length_of_sentences(padded_int_length_of_sentences_matrix, label_chaps_to_periods)

if __name__ == "__main__":
    main()