from collections import Counter
import gensim
from gensim import models
import matplotlib.pyplot as plt
import spacy
import sklearn
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis.gensim
import deviding_to_chaps
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Assuming 'df' is your DataFrame with features (X) and target (Y)
# For example, X contains features like 'feature1', 'feature2', etc., and Y contains the target variable.

# Sample data creation

#data = {'feature1': chap_list[0], 'target': np.random.rand(100)}
#df = pd.DataFrame(data)

# Split the data into features (X) and target (Y)
#X = df[['feature1']]
#Y = df['target']

# Split the data into training and testing sets
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
# # Standardize the features (important for neural networks)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# ------------------------------- End of AI part ------------------------------


nlp = spacy.load('en_core_web_sm')


def most_freq_words(doc):  # we have a bug in this function - we get an empty token that we don't want.
    costume_stop_words = [",", " ", ".", "\n", r'[\x00-\x1F]']
    for stopword in costume_stop_words:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

    filtered_tokens = [token.text for token in doc if not token.is_stop]
    word_freq = Counter(filtered_tokens)

    most_common_words = word_freq.most_common(10)
    for word, freq in most_common_words:
        print(f'{word}: {freq} times')


def text_to_tokens(text,doc):
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


def texts_to_docs(text_list):
    return [nlp(text) for text in text_list]


def plot_average_sentence_length(doc_list):
    average_sentences = []

    for doc in doc_list:
        average_sentences.append(average_sentence_length(doc))

    plt.bar(range(1, len(doc_list) + 1), average_sentences)
    plt.xlabel('Document')
    plt.ylabel('Average Sentences')
    plt.title('Average Number of Sentences in Different Documents')
    plt.show()


def length_of_sentence(sentence):
    return len(sentence.split())


def average_sentence_length(doc):
    sum = 0
    for sentence in classifier_text_to_sentences(doc):
        sum += length_of_sentence(sentence)
    return sum / len(classifier_text_to_sentences(doc))


def classifier_text_to_sentences(doc):
    # it add the /n to the sentences  - maybe we wont to fix this?
    sentences = [sentence.text for sentence in doc.sents]
    return sentences


def topic_modeling_LDA(corpus,doc):
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
    doc = nlp(text)


def get_max_num_of_sentences(chap_list):
    max_num_of_sentences = 0
    for pair in chap_list:
        chap = pair[0]
        sen = classifier_text_to_sentences(nlp(chap))
        max_num_of_sentences = max(max_num_of_sentences, len(sen))
    return max_num_of_sentences


def load_txt_files(directory):
    file_contents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            file_contents.append(file.read())
    return file_contents


def main():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    CHAPS_PATH = os.path.join(ROOT_DIR, 'chaps')
    chap_list = load_txt_files(CHAPS_PATH)
    print(chap_list)
    print(get_max_num_of_sentences(chap_list))


if __name__ == "__main__":
    main()
