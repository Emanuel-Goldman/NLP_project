import spacy
import os
from main import tokenize_text_to_sentences, texts_to_docs, lemmas_freq
from spacy.language import Language
import json
import numpy as np
from main import most_freq_words
from nltk.sentiment import SentimentIntensityAnalyzer


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


# label chapters to periods vector and save as a json file
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


# save the matrix of sentences lengths casting to int instead of float
def chaps_int_length_of_sentences_matrix(read_path, write_path):
    with open(read_path, "r") as json_file:
        float_length_of_sentences_matrix = json.load(json_file)

    int_length_of_sentences_matrix = [[int(value) for value in row] for row in float_length_of_sentences_matrix]

    with open(write_path, "w") as json_file:
        json.dump(int_length_of_sentences_matrix, json_file)


# returns the percentage of a pos in a chapter
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

    return num_of_pos / num_of_words


# returns a vector for a chapter of all poses percentage
def chap_pos_percentage_vector(chap_sentences, nlp):
    pos_percentege = []

    pos_percentege.append(chap_pos_percentage(chap_sentences, nlp, "VERB"))
    pos_percentege.append(chap_pos_percentage(chap_sentences, nlp, "ADJ"))
    pos_percentege.append(chap_pos_percentage(chap_sentences, nlp, "ADV"))
    pos_percentege.append(chap_pos_percentage(chap_sentences, nlp, "INTJ"))
    pos_percentege.append(chap_punctuation_percentage(chap_sentences, nlp))
    pos_percentege.append(chap_pos_percentage(chap_sentences, nlp, "NOUN"))

    return pos_percentege


# returns the percentage of the chapter's punctuation
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

    return num_of_punctuation / num_of_words


# returns the percentage of the chapter's question marks out of the chapter's punctuation
def chap_question_mark_percentage(chap_sentences, nlp):
    num_of_punctuation = 0
    num_of_question_mark = 0

    for sentence in chap_sentences:
        doc = nlp(sentence)
        for token in doc:
            if token.is_punct and token.text != ".":
                num_of_punctuation += 1
            if token.text == "?":
                num_of_question_mark += 1

    return num_of_question_mark / num_of_punctuation


# save a matrix of the chapters' pos percentage
def save_chaps_pos_percentage_matrix(tokenized_chaps_path, output_path, nlp):
    chaps_pos_percentage_matrix = []
    files = sorted(os.listdir(tokenized_chaps_path))
    for filename in files:
        filepath = os.path.join(tokenized_chaps_path, filename)
        with open(filepath, "r") as json_file:
            chap_sentences = json.load(json_file)

        chaps_pos_percentage_matrix.append(chap_pos_percentage_vector(chap_sentences, nlp))

    with open(output_path, "w") as json_file:
        json.dump(chaps_pos_percentage_matrix, json_file)


# returns a matrix with the chapter's most common words' frequencies
def chaps_words_frequencies_matrix(chap_list: list[str], nlp):
    docs = texts_to_docs(nlp, chap_list)
    freq_words_matrix = []
    total_words_per_chap = [len(doc) for doc in docs]
    for doc, total_words in zip(docs, total_words_per_chap):
        most_common_words_and_freq = most_freq_words(nlp, doc)
        frequencies = [(freq / total_words) for word, freq in most_common_words_and_freq]
        freq_words_matrix.append(frequencies)
    return freq_words_matrix


# returns a matrix with the chapter's most common lemmas' frequencies
def chaps_lemmas_frequencies_matrix(chap_list: list[str], nlp):
    docs = texts_to_docs(nlp, chap_list)
    freq_lemmas_matrix = []
    total_words_per_chap = [len(doc) for doc in docs]
    for doc, total_words in zip(docs, total_words_per_chap):
        most_common_lemmas_and_freq = lemmas_freq(nlp, doc)
        frequencies = [(freq / total_words) for word, freq in most_common_lemmas_and_freq.items()]
        freq_lemmas_matrix.append(frequencies)
    return freq_lemmas_matrix


# save the matrix of the chapters' lemmas frequencies
def create_data_most_freq_lemmas_ai(chap_list: list[tuple[str, str]], chaps_path: str, nlp, ROOT_DIR):
    chaps = [chap for chap, year in chap_list]
    freq_lemmas = chaps_lemmas_frequencies_matrix(chaps, nlp)
    path = os.path.join(ROOT_DIR, "input_for_AI")
    # Write JSON data to files in the "input_for_ai" folder
    with open(os.path.join(path, 'freq_lemmas.json'), 'w') as json_file:
        json.dump(freq_lemmas, json_file)

    output_path = os.path.join(path, "freq_lemmas_labels.json")
    label_chaps_to_periods(chaps_path, output_path)


# save the matrix of the chapters' most common words frequencies
def create_data_most_freq_words_ai(chap_list: list[tuple[str, str]], chaps_path: str, nlp, ROOT_DIR):
    chaps = [chap for chap, year in chap_list]
    freq_words = chaps_words_frequencies_matrix(chaps, nlp)
    path = os.path.join(ROOT_DIR, "input_for_AI")
    # Write JSON data to files in the "input_for_ai" folder
    with open(os.path.join(path, 'freq_words.json'), 'w') as json_file:
        json.dump(freq_words, json_file)

    output_path = os.path.join(path, "freq_words_labels.json")
    label_chaps_to_periods(chaps_path, output_path)


# return a score to describe the sentiment of the text
def sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)['compound']


# combine two matrices together
def combine_matrices(matrix1, matrix2, write_path):
    for i in range(len(matrix1)):
        for num in matrix2[i]:
            matrix1[i].append(num)
    with open(write_path, "w") as json_file:
        json.dump(matrix1, json_file)


# returns a list of the tenses' percentage of a chapter
def verbs_tenses_percent(sentences, nlp):
    past_count = 0
    present_count = 0
    future_count = 0

    for sentence in sentences:
        doc = nlp(sentence)

        for token in doc:
            if token.tag_ == "VBD":
                past_count += 1
            elif token.tag_ == "VBP" or token.tag_ == "VBZ" or token.tag_ == "VB":
                present_count += 1
            elif token.tag_ == "MD" and token.text.lower() in ["will", "shall"]:
                future_count += 1
                present_count -= 1

    sum = past_count + present_count + future_count
    output = [past_count / sum, present_count / sum, future_count / sum]
    return output


# save a matrix of the chapters verb tenses and pos
def save_pos_freq_words_verbs_matrix(tokenized_chaps_path, matrix, output_path, nlp):
    files = sorted(os.listdir(tokenized_chaps_path))
    i = 0
    for filename in files:
        filepath = os.path.join(tokenized_chaps_path, filename)
        with open(filepath, "r") as json_file:
            chap_sentences = json.load(json_file)
        tenses = verbs_tenses_percent(chap_sentences, nlp)
        for tense in tenses:
            matrix[i].append(tense)
        i += 1
        print(i)

    with open(output_path, "w") as json_file:
        json.dump(matrix, json_file)


# return the percentage of named entity in a chapter
def named_entity_percentage(chap_sentences, nlp):
    num_of_words = 0
    num_of_entities = 0

    for sentence in chap_sentences:
        doc = nlp(sentence)
        for token in doc:
            if not token.is_punct and not token.is_space:
                num_of_words += 1
        for ent in doc.ents:
            num_of_entities += 1

    return num_of_entities / num_of_words


# save a matrix of the chapters verb tenses named entities and pos
def save_pos_freq_words_verbs_entities_matrix(tokenized_chaps_path, matrix, output_path, nlp):
    files = sorted(os.listdir(tokenized_chaps_path))
    i = 0
    for filename in files:
        filepath = os.path.join(tokenized_chaps_path, filename)
        with open(filepath, "r") as json_file:
            chap_sentences = json.load(json_file)
        matrix[i].append(named_entity_percentage(chap_sentences, nlp))
        i += 1
        print(i)

    with open(output_path, "w") as json_file:
        json.dump(matrix, json_file)


def main():
    # loading spacy pipline
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('set_custom_boundaries', before="parser")

    # folder paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    chaps_path = os.path.join(ROOT_DIR, 'chaps')
    tokenized_chaps_path = os.path.join(ROOT_DIR, 'Tokenized_Chaps')
    path = os.path.join(ROOT_DIR, 'Input_for_AI')


if __name__ == "__main__":
    main()
