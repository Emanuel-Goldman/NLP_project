import os
import json
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import tree


# using neuron network (mlp) to classify the data, returns the accuracy percent
def mlp_classifier(X, y):
    scaler = preprocessing.StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42, max_iter=10000)

    clf.fit(X_train_scaled, y_train)

    y_predict = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_predict)

    return accuracy


# using logistic regression to classify the data, returns the accuracy percent
def logistic_regression(X, y):
    scaler = preprocessing.StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42, max_iter=10000)

    clf.fit(X_train_scaled, y_train)

    y_predict = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_predict)
    return accuracy


# using decision tree to classify the data, returns the accuracy percent
def decision_tree(X, y):
    scaler = preprocessing.StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = tree.DecisionTreeClassifier(random_state=42)

    clf.fit(X_train_scaled, y_train)

    y_predict = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_predict)
    return accuracy


# run logistic regression on the chapters' lemmas frequencies
def most_freq_lemmas_ai(ROOT_DIR):
    path = os.path.join(ROOT_DIR, 'Input_for_AI')
    path_freq_words = os.path.join(path, "freq_lemmas.json")
    path_labels = os.path.join(path, "freq_lemmas_labels.json")
    with open(path_freq_words, "r") as json_file:
        freq_lemmas_matrix = json.load(json_file)

    with open(path_labels, "r") as json_file:
        freq_lemmas_labels = json.load(json_file)

    logistic_regression(freq_lemmas_matrix, freq_lemmas_labels)


# run logistic regression on the chapters' most common words frequencies
def most_freq_words_ai(ROOT_DIR):
    path = os.path.join(ROOT_DIR, 'Input_for_AI')
    path_freq_words = os.path.join(path, "freq_words.json")
    path_labels = os.path.join(path, "freq_words_labels.json")
    with open(path_freq_words, "r") as json_file:
        freq_words_matrix = json.load(json_file)

    with open(path_labels, "r") as json_file:
        freq_words_labels = json.load(json_file)

    logistic_regression(freq_words_matrix, freq_words_labels)


def run_classifier(path1, path2):
    with open(path2, "r") as json_file:
        labels = json.load(json_file)

    input_path = os.path.join(path1,
                              "chaps_pos_freq_words_tense_entities_lemmas_sents_punct_marks_matrix.json")  # all the features
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("all the featurs: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1,
                              "chaps_pos_freq_words_tense_entities_lemmas_sents_matrix.json")  # all the features besides punctuation marks
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("all the featurs besides punctuation marks: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1,
                              "chaps_pos_percentage_tense_entities_lemmas_sents_punct_marks_matrix.json")  # all the features besides freq words
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("all the featurs besides freq words: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1,
                              "chaps_pos_freq_words_tense_entities_lemmas_punct_marks_matrix.json")  # all the features besides sentiments
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("all the featurs besides sentiments: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1,
                              "chaps_pos_freq_words_entities_lemmas_sents_punct_marks_matrix.json")  # all the features besides verb tenses
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("all the featurs besides verb tenses: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1,
                              "chaps_pos_freq_words_tense_lemmas_sents_punct_marks_matrix.json")  # all the features besides named entities
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("all the featurs besides named entities: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1,
                              "chaps_freq_words_tense_entities_lemmas_sents_punct_marks_matrix.json")  # all the features besides pos
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("all the featurs besides pos: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1,
                              "chaps_pos_freq_words_tense_entities_sents_punct_marks_matrix.json")  # all the features besides freq lemmas
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("all the featurs besides freq lemmas: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1, "chaps_pos_percentage_matrix.json")  # pos
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("pos: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1, "freq_words.json")  # freq words
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("freq words: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1, "freq_lemmas.json")  # freq lemmas
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("freq lemmas: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1, "Average_sentiment.json")  # sentiments
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("sentiments: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1, "tense_matrix.json")  # verb tenses
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("verb tenses: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1, "entities_matrix.json")  # named entities
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("named entities: " + str(logistic_regression(input, labels)))
    input_path = os.path.join(path1, "punctuation_marks_matrix.json")  # punctuation marks
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("punctuation marks: " + str(logistic_regression(input, labels)))


def run_classifier_without_noise(path1, path2):
    with open(path2, "r") as json_file:
        labels = json.load(json_file)

    input_path = os.path.join(path1, "chaps_pos_freq_words_sents_punct_marks_matrix.json")
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("mlp: " + str(mlp_classifier(input, labels)))
    input_path = os.path.join(path1, "chaps_pos_freq_words_tense_entities_sents_matrix.json")
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("decision tree: " + str(decision_tree(input, labels)))
    input_path = os.path.join(path1, "chaps_pos_freq_words_tense_entities_sents_punct_marks_matrix.json")
    with open(input_path, "r") as json_file:
        input = json.load(json_file)
    print("logistic regression: " + str(logistic_regression(input, labels)))


def main():
    # folder paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path1 = os.path.join(ROOT_DIR, 'Input_for_AI')
    path2 = os.path.join(path1, "label_chaps_to_periods.json")
    run_classifier(path1, path2)
    run_classifier_without_noise(path1, path2)


if __name__ == "__main__":
    main()
