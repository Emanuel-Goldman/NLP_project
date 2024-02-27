import os
from collections import Counter

import json
import gensim
import matplotlib.pyplot as plt
import pyLDAvis.gensim
import spacy
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from spacy.tokens import Doc
from spacy.language import Language
from deviding_to_chaps import get_books_names


def most_freq_words(nlp, doc: Doc) -> list[tuple[str, int]]:
    costume_stop_words = [",", " ", ".", "\n", ";", "-", "--", ":", '“', '”', "'", '"', "\n\n", "_", "!", "?"]
    for stopword in costume_stop_words:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

    filtered_tokens = [token.text.lower() for token in doc if
                       token.text.lower() not in nlp.Defaults.stop_words and token.text.lower() not in
                       costume_stop_words and token.text.lower().strip()]
    print(filtered_tokens)
    word_freq = Counter(filtered_tokens)

    most_common_words = word_freq.most_common(10)
    return most_common_words


def most_freq_words_in_all_chaps(nlp, docs: list[Doc]) -> list[tuple[str, int]]:
    # Create a set of custom stop words
    costume_stop_words = {",", " ", ".", "\n", ";", "-", "--", ":", '“', '”', "'", '"', "\n\n", "_", "!", "?", "The",
                          "It", "‘", "’"}

    # Initialize a Counter to store word frequencies
    word_freq = Counter()

    # Loop over each document in the list
    for doc in docs:
        # Update the set of stop words for each document
        for stopword in costume_stop_words:
            lexeme = nlp.vocab[stopword]
            lexeme.is_stop = True

        # Filter tokens based on stop words and additional criteria
        filtered_tokens = [token.text.lower() for token in doc if
                           token.text.lower() not in nlp.Defaults.stop_words and token.text.lower() not in
                           costume_stop_words and token.text.strip()]

        # Update the overall word frequency counter
        word_freq.update(filtered_tokens)

    # Get the 10 most common words
    most_common_words = word_freq.most_common(10)
    # print(most_common_words)
    return most_common_words


def text_to_tokens(text, doc):
    tokens = [token.text for token in doc]
    return tokens


def text_to_clean_tokens(tokens):
    cleaned_tokens = [tokens.text for tokens in tokens if not tokens.is_stop]
    return cleaned_tokens


def classifier_part_of_speech(text):
    return


def plot_pos(docs: list[Doc]):
    pos = []
    for doc in docs:
        pos.extend(text_to_lemma_pos(doc))

    # print(pos)
    pos_frequency = count_pos(pos)
    top_pos_frequency = dict(pos_frequency)
    pos, frequencies = zip(*top_pos_frequency.items())

    plt.bar(pos, frequencies)
    plt.xlabel('Pos')
    plt.ylabel('Frequency')
    plt.title('Pos Frequency')
    plt.xticks(rotation=45, ha="right")
    plt.show()


def plot_lemmas(nlp, docs: list[Doc]):
    lemmas = []
    for doc in docs:
        lemmas.extend(text_to_lemma_pos(doc))

    stop_words = set(nlp.Defaults.stop_words)
    custom_stop_words = [",", " ", ".", "\n", ";", "-", "--", ":", '“', '”', "'", '"', "\n\n", "_", "!", "?"]
    stop_words.update(custom_stop_words)
    lemmas = [(lemma, pos) for lemma, pos in lemmas if lemma.lower() not in stop_words]

    # print(lemmas)
    lemmas_frequency = count_lemmas(lemmas)
    top_lemmas_frequency = dict(lemmas_frequency)
    lemmas, frequencies = zip(*top_lemmas_frequency.items())

    plt.bar(lemmas, frequencies)
    plt.xlabel('Lemmas')
    plt.ylabel('Frequency')
    plt.title('Lemmas Frequency')
    plt.xticks(rotation=45, ha="right")
    plt.show()


def count_lemmas(lemma_pos_list: list[tuple[str, str]], top_n=10) -> list[tuple[str, int]]:
    lemmas_counter = Counter(lemma for lemma, pos in lemma_pos_list)
    return lemmas_counter.most_common(top_n)


def count_pos(lemma_pos_list: list[tuple[str, str]], top_n=10) -> list[tuple[str, int]]:
    pos_counter = Counter(pos for lemma, pos in lemma_pos_list)
    return pos_counter.most_common(top_n)


def text_to_lemma_pos(doc: Doc) -> list[tuple[str, str]]:
    lemmas_and_pos = [(token.lemma_.lower(), token.pos_) for token in doc]
    return lemmas_and_pos


def text_to_clean_lemma(nlp, text):
    tokens = nlp(text)
    lemmas = [token.lemma_ for token in tokens]
    return lemmas


def texts_to_docs(nlp, chap_list: list[str]) -> list[Doc]:
    nlp.max_length = 1500000
    docs = []
    for text in chap_list:
        docs.append(nlp(text))
    return docs


def plot_data_per_year(nlp, chap_list: list[tuple[str, str]], chosen_year: str):
    organized_by_year = organize_by_year(chap_list)
    chaps_in_year = organized_by_year.get(chosen_year)
    docs = texts_to_docs(nlp, chaps_in_year)

    plot_most_freq_words_by_year(nlp, docs, chosen_year)
    plot_average_sentence_length(nlp, docs)
    plot_lemmas(nlp, docs)
    plot_pos(docs)


def plot_average_sentence_length(nlp, docs: list[Doc]):
    average_sentences = []
    for doc in docs:
        average_sentences.append(average_sentence_length(nlp, doc))

    plt.scatter(range(1, len(docs) + 1), average_sentences)
    plt.xlabel('Document')
    plt.ylabel('Average Sentences')
    plt.title('Average Number of Sentences in Different Documents')
    plt.show()


def organize_by_year(chap_list: list[tuple[str, str]]) -> dict[str, list[str]]:
    #TODO:check if works
    organized_by_year = {}
    for chap, year in chap_list:
        organized_by_year.setdefault(year, []).append(chap)

        # filename = f"chaps_in_{year}.json"
        # with open(filename, 'w') as json_file:
        #     json.dump(organized_by_year[year], json_file)
    return organized_by_year


def organize_by_period_of_time(organized_by_year: dict[str, list[str]]) -> dict[str, list[str]]:
    periods = {"period1": [], "period2": [], "period3": []}
    for year, chaps in organized_by_year.items():
        if int(year) >= 1837 & int(year) <= 1841:
            periods["period1"].extend(chaps)
        elif int(year) >= 1843 & int(year) <= 1854:
            periods["period2"].extend(chaps)
        else:
            periods["period3"].extend(chaps)

    return periods


def plot_data_per_period(nlp, chap_list: list[tuple[str, str]], chosen_period: str):
    organized_by_year = organize_by_year(chap_list)
    organize_by_period = organize_by_period_of_time(organized_by_year)
    chaps_in_period = organize_by_period.get(chosen_period)
    docs = texts_to_docs(nlp, chaps_in_period)

    plot_most_freq_words_by_year(nlp, docs, chosen_period)
    plot_average_sentence_length(nlp, docs)
    plot_lemmas(nlp, docs)
    plot_pos(docs)


def plot_most_freq_words_by_year(nlp, docs: list[Doc], chosen_year: str):
    freq_words_list = most_freq_words_in_all_chaps(nlp, docs)
    # Extract words and their frequencies
    words, frequencies = zip(*freq_words_list)

    # Plotting
    plt.bar(words, frequencies)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Most Frequent Words in Books ({chosen_year})')
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better visibility
    plt.show()


def length_of_sentence(nlp, sentence):
    doc = nlp(sentence)
    return sum(1 for token in doc if not token.is_punct and not token.is_space)


def average_sentence_length(nlp, doc):
    sum_len_of_sentences = 0
    sentences = tokenize_text_to_sentences(doc)
    for sentence in sentences:
        sum_len_of_sentences += length_of_sentence(nlp, sentence)

    return sum_len_of_sentences / len(sentences)


#returns an array - not a doc object
def tokenize_text_to_sentences(doc: Doc) -> list[str]:
    sentences = [sentence.text.replace('\n', ' ').strip() for sentence in doc.sents]
    return sentences


def topic_modeling_LDA(doc):
    # We add some words to the stop word list
    texts, article = [], []

    for word in doc:

        if word.text != '\n' and not word.is_stop and not word.is_punct and not word.like_num and word.text != 'I':
            article.append(word.lemma_)

        if word.text == '\n':
            texts.append(article)
            article = []

    bigram = gensim.models.phrases.Phrases(texts)
    texts = [bigram[line] for line in texts]
    texts = [bigram[line] for line in texts]

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
    print(lda_model.show_topics())
    vis_data = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, 'lda_visualization.html')


def text_to_vec(nlp, text):
    doc = nlp(text)
    return doc


def get_max_num_of_sentences(nlp, chap_list: list[tuple[str, str]]):
    only_chaps = []
    for chap, year in chap_list:
        only_chaps.append(chap)
    docs = texts_to_docs(nlp, only_chaps)
    max_num_of_sentences = 0
    for doc in docs:
        sen = tokenize_text_to_sentences(doc)
        max_num_of_sentences = max(max_num_of_sentences, len(sen))
    return max_num_of_sentences


def extract_year_from_filename(directory):
    chaps_names = get_books_names(directory)
    years = []
    for name in chaps_names:
        parts = name.split('(')
        year = parts[-1].split(')')[0]
        years.append(year)
    return years


def load_txt_files(directory) -> list[tuple[str, str]]:
    years = extract_year_from_filename(directory)
    file_contents = []
    i = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            chapter_text = file.read()
            year = years[i]
            file_contents.append((chapter_text, year))
        i += 1
    return file_contents


def print_freq_words(freq_words):
    for word, freq in freq_words:
        print(f'{word}: {freq} times')
    print('\n')

@Language.component('set_custom_boundaries')
def set_custom_boundaries(doc):
    predecessor_token = doc[0]
    for token in doc[1:-1]:
        if token.text == "'" and predecessor_token.text == ".":
            doc[token.i + 1].is_sent_start = False
        predecessor_token = token
    return doc

def main():
    # loading spacy pipline
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe("set_custom_boundaries", before="parser")

    # TODO: don't erase anything! you can put as comment if you don't want to run it all
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    CHAPS_PATH = os.path.join(ROOT_DIR, 'chaps')
    chap_list = load_txt_files(CHAPS_PATH)
    # print(get_max_num_of_sentences(nlp, chap_list))

    # check = [("hadar hadar hadar Hadar lets see if its work, i have a dog and a cat, hadar.", "1843"), ("go to eat", "1843"), ("so pretty", "1843"),
    #          ("have a nice day Hadar", "1837"),
    #          ("great!", "1843")]
    # text =[text for text,year in check]
    # docs = texts_to_docs(nlp, text)
    # print(most_freq_words(nlp, docs[0]))
    # plot_data_per_year(nlp, check,"1843")

    # plot_data_per_period(nlp, chap_list, "period1")
    plot_data_per_year(nlp, chap_list, "1859")

    # organized_by_year = organize_by_year(chap_list)
    # chaps_in_year = organized_by_year.get("1843")
    # docs = texts_to_docs(nlp, chaps_in_year)
    # print(most_freq_words(nlp, docs[0]))

    # topic_modeling_LDA(docs_list)


if __name__ == "__main__":
    main()
