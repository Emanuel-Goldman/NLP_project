from collections import Counter

import spacy
import os
from main import tokenize_text_to_sentences, texts_to_docs, extract_year_from_filename
from spacy.language import Language
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import tree
from main import most_freq_words
from main import load_txt_files


# adding sentence tokenization rule for apostrophes for spacy
@Language.component('set_custom_boundaries')
def set_custom_boundaries(doc):
    predecessor_token = doc[0]
    for token in doc[1:-1]:
        if token.text == "'" and predecessor_token.text == ".":
            doc[token.i + 1].is_sent_start = False
        predecessor_token = token
    return doc


# tokenize a chapter to sentences and save as a json file
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


# create a vector of length of sentences for a single chapter
def chap_float_length_of_sentences_vector(chap_sentences, nlp):
    length_of_sentences_vector = []

    for sentence in chap_sentences:
        doc = nlp(sentence)
        length = float(sum(1 for token in doc if not token.is_punct and not token.is_space))
        length_of_sentences_vector.append(length)

    return length_of_sentences_vector


# create a matrix of lengths of sentences of all chapters and save as a json file
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


# pad the matrix of lengths of sentences with zeros to the longest chapter size
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


# lable chapters to periods vector and save as a json file
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


def chap_pos_percentage(chap_sentences, nlp, pos):
    num_of_pos = 0
    num_of_words = 0

    for sentence in chap_sentences:
        doc = nlp(sentence)
        for token in doc:
            if token.pos_ == pos:
                num_of_pos += 1
            if not token.is_punct and not token.is_space:
                num_of_words += 1

    return (num_of_pos / num_of_words)


def chap_pos_percentage_vector(chap_sentences, nlp):
    pos_percentege = []

    pos_percentege.append(chap_pos_percentage(chap_sentences, nlp, "VERB"))
    pos_percentege.append(chap_pos_percentage(chap_sentences, nlp, "ADJ"))
    pos_percentege.append(chap_pos_percentage(chap_sentences, nlp, "ADV"))
    pos_percentege.append(chap_pos_percentage(chap_sentences, nlp, "INTJ"))

    return pos_percentege


def chap_punctuation_percentage(chap_sentences, nlp):
    num_of_punctuation = 0
    num_of_words = 0

    for sentence in chap_sentences:
        doc = nlp(sentence)
        for token in doc:
            if token.is_punct and token.text != ".":
                num_of_punctuation += 1
            if not token.is_punct and not token.is_space:
                num_of_words += 1

    return (num_of_punctuation / num_of_words)


def chaps_words_frequencies_matrix(chap_list: list[str], nlp):
    docs = texts_to_docs(nlp, chap_list)
    freq_words_matrix = []
    for doc in docs:
        most_common_words_and_freq = most_freq_words(nlp, doc)
        frequencies = [freq for word, freq in most_common_words_and_freq]
        freq_words_matrix.append(frequencies)
    print(freq_words_matrix)
    return freq_words_matrix


def mlp_classifier(X, y):
    scaler = preprocessing.StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)

    clf.fit(X_train_scaled, y_train)

    y_predict = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_predict)

    return accuracy  # 45%


def logistic_regression(X, y):
    scaler = preprocessing.StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42, max_iter=10000)

    clf.fit(X_train_scaled, y_train)

    y_predict = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_predict)
    print(accuracy)  # 46%


def decision_tree(X, y):
    scaler = preprocessing.StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = tree.DecisionTreeClassifier(random_state=42)

    clf.fit(X_train_scaled, y_train)

    y_predict = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_predict)
    print(accuracy)


def create_data_most_freq_words_ai(chap_list: list[tuple[str, str]], chaps_path: str, nlp, ROOT_DIR):
    chaps = [chap for chap, year in chap_list]
    freq_words = chaps_words_frequencies_matrix(chaps, nlp)
    freq_words_labels = extract_year_from_filename(chaps_path)
    path = os.path.join(ROOT_DIR, "input_for_AI")
    # Write JSON data to files in the "input_for_ai" folder
    with open(os.path.join(path, 'freq_words.json'), 'w') as freq_words_file:
        json.dump(freq_words, freq_words_file)

    with open(os.path.join(path, 'freq_words_labels.json'), 'w') as freq_words_labels_file:
        json.dump(freq_words_labels, freq_words_labels_file)


def most_freq_words_ai(ROOT_DIR):
    path = os.path.join(ROOT_DIR, 'Input_for_AI')
    path_freq_words = os.path.join(path, "freq_words.json")
    path_labels = os.path.join(path, "freq_words_labels.json")
    with open(path_freq_words, "r") as json_file:
        freq_words_matrix = json.load(json_file)

    with open(path_labels, "r") as json_file:
        freq_words_labels = json.load(json_file)

    logistic_regression(freq_words_matrix, freq_words_labels)


def main():
    # loading spacy pipline
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('set_custom_boundaries', before="parser")

    # folder paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    chaps_path = os.path.join(ROOT_DIR, 'chaps')
    chap_list = load_txt_files(chaps_path)
    # most_freq_words_ai(chap_list, chaps_path, nlp, ROOT_DIR)

    create_data_most_freq_words_ai(chap_list, chaps_path, nlp, ROOT_DIR)
    most_freq_words_ai(ROOT_DIR)

    # tokenized_chaps_path = os.path.join(ROOT_DIR, 'Tokenized_Chaps')

    # chap1 = os.path.join(tokenized_chaps_path, "A Tale of Two Cities(1859) 4.json")
    #
    # with open(chap1, "r") as json_file:
    #     chap = json.load(json_file)
    # print(chap_pos_percentage_vector(chap, nlp))

    '''
    #AYELET- running AI
    path = os.path.join(ROOT_DIR, 'Input_for_AI')
    path2 = os.path.join(path, "padded_int_length_of_sentences_matrix.json")
    path3 = os.path.join(path, "label_chaps_to_periods.json")
    with open(path2, "r") as json_file:
        padded_int_length_of_sentences_matrix = json.load(json_file)
    with open(path3, "r") as json_file:
        label_chaps_to_periods = json.load(json_file)
    logistic_regression(padded_int_length_of_sentences_matrix, label_chaps_to_periods)
    '''


if __name__ == "__main__":
    main()
