import os
from collections import Counter

import gensim
import matplotlib.pyplot as plt
import pyLDAvis.gensim
import sklearn
import spacy
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from spacy.tokens import Doc

from deviding_to_chaps import get_books_names
import deviding_to_chaps


# Assuming 'df' is your DataFrame with features (X) and target (Y)
# For example, X contains features like 'feature1', 'feature2', etc., and Y contains the target variable.

# Sample data creation

# data = {'feature1': chap_list[0], 'target': np.random.rand(100)}
# df = pd.DataFrame(data)

# Split the data into features (X) and target (Y)
# X = df[['feature1']]
# Y = df['target']

# Split the data into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
# # Standardize the features (important for neural networks)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Build a simple neural network model
# model = keras.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(1)  # Output layer with 1 neuron (regression)
# ])

# Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
# model.fit(X_train_scaled, Y_train, epochs=50, batch_size=16, verbose=2)
#
# # Make predictions on the test set
# predictions = model.predict(X_test_scaled)
#
# # Evaluate the model
# mse = mean_squared_error(Y_test, predictions)
# print(f"Mean Squared Error on Test Set: {mse}")


# ------------------------------- End of AI part ------------------------------

def most_freq_words(doc: Doc) -> list[tuple[str, int]]:
    nlp = spacy.load('en_core_web_sm')
    costume_stop_words = [",", " ", ".", "\n", ";", "-", "--", ":", '“', '”']
    for stopword in costume_stop_words:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

    filtered_tokens = [token.text for token in doc if
                       token.text not in nlp.Defaults.stop_words and token.text not in costume_stop_words and
                       token.text.strip()]
    word_freq = Counter(filtered_tokens)

    most_common_words = word_freq.most_common(10)
    return most_common_words


def most_freq_words_in_all_chaps(chap_list: list[str]) -> list[tuple[str, int]]:
    chap_docs = texts_to_docs(chap_list)
    freq_words = []
    for chap in chap_docs:
        freq_words.extend(most_freq_words(chap))
    return freq_words


def text_to_tokens(text, doc):
    tokens = [token.text for token in doc]
    return tokens


def text_to_clean_tokens(tokens):
    cleaned_tokens = [tokens.text for tokens in tokens if not tokens.is_stop]
    return cleaned_tokens


def classifier_part_of_speech(text):
    return


def text_to_lemma_pos(doc):
    lemmas_and_pos = [(token.lemma_, token.pos_) for token in doc]
    return lemmas_and_pos


def text_to_clean_lemma(text):
    nlp = spacy.load("en_core_web_sm")
    tokens = nlp(text)
    lemmas = [token.lemma_ for token in tokens]
    return lemmas


def texts_to_docs(chap_list: list[str]) -> list[Doc]:
    #TODO: change to loop over all docs instead of 5
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 1500000
    docs = []
    for i in range(5):
        text = chap_list[i]
        docs.append(nlp(text))
    return docs


def plot_average_sentence_length(chap_list: list[tuple[str, str]]):
    # TODO: change to get all chapters per year
    all_chaps = []
    i = 0
    for chap, year in chap_list:
        if i < 5:
            all_chaps.append(chap)

    docs = texts_to_docs(all_chaps)
    average_sentences = []
    for doc in docs:
        average_sentences.append(average_sentence_length(doc))

    plt.bar(range(1, len(docs) + 1), average_sentences)
    plt.xlabel('Document')
    plt.ylabel('Average Sentences')
    plt.title('Average Number of Sentences in Different Documents')
    plt.show()


def organize_by_year(chap_list: list[tuple[str, str]]) -> dict[str, list[str]]:
    organized_by_year = {}
    for chap, year in chap_list:
        organized_by_year.setdefault(year, []).append(chap)
    return organized_by_year


def plot_most_freq_words_by_year(chap_list: list[tuple[str, str]], chosen_year: str):
    organized_by_year = organize_by_year(chap_list)
    freq_words_list = []
    for year, chapters in organized_by_year.items():
        if year == chosen_year:
            # x = most_freq_words_in_all_chaps(chapters)
            # print_freq_words(x)
            freq_words_list.extend(most_freq_words_in_all_chaps(chapters))

        # Extract words and their frequencies
        words, frequencies = zip(*freq_words_list)

        # Plotting
        plt.bar(words, frequencies)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title(f'Most Frequent Words in Books ({year})')
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better visibility
        plt.show()


def length_of_sentence(sentence):
    return len(sentence.split())


def average_sentence_length(doc):
    sum_len_of_sentences = 0
    sentences = classifier_text_to_sentences(doc)
    for sentence in sentences:
        sum_len_of_sentences += length_of_sentence(sentence)

    return sum_len_of_sentences / len(sentences)


def classifier_text_to_sentences(doc: Doc) -> list[str]:
    # TODO: it add the /n to the sentences  - maybe we wont to fix this?
    sentences = [sentence.text for sentence in doc.sents]
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


def text_to_vec(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return doc


def get_max_num_of_sentences(chap_list: list[tuple[str, str]]):
    only_chaps = []
    for chap, year in chap_list:
        only_chaps.append(chap)
    docs = texts_to_docs(only_chaps)
    max_num_of_sentences = 0
    for doc in docs:
        sen = classifier_text_to_sentences(doc)
        max_num_of_sentences = max(max_num_of_sentences, len(sen))
    return max_num_of_sentences


def extract_year_from_filename():
    folder = r"C:\Users\User\PycharmProjects\NLP_project\Chaps"
    chaps_names = get_books_names(folder)
    years = []
    for name in chaps_names:
        parts = name.split('(')
        year = parts[-1].split(')')[0]
        years.append(year)
    return years


def load_txt_files(directory) -> list[tuple[str, str]]:
    years = extract_year_from_filename()
    file_contents = []
    i = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            chapter_text = file.read()
            year = years[i]
            file_contents.append((chapter_text, year))
    return file_contents


def print_freq_words(freq_words):
    for word, freq in freq_words:
        print(f'{word}: {freq} times')
    print('\n')


def main():
    #TODO: don't erase anything! you can put as comment if you don't want to run it all
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    CHAPS_PATH = os.path.join(ROOT_DIR, 'chaps')
    chap_list = load_txt_files(CHAPS_PATH)
    print(get_max_num_of_sentences(chap_list))
    plot_most_freq_words_by_year(chap_list, "1859")
    plot_average_sentence_length(chap_list)

    # topic_modeling_LDA(docs_list)


if __name__ == "__main__":
    main()
