from collections import Counter
import gensim
from gensim import models
import matplotlib.pyplot as plt
import spacy
import sklearn
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nlp = spacy.load('en_core_web_sm')

doc = nlp(
    """Once upon a time, in the small village of Eldoria, 
nestled between rolling hills and lush forests, there lived a young girl named Elara. 
Elara was known for her boundless curiosity and an insatiable desire for adventure. 
Her eyes sparkled like the morning dew on a flower, and her spirit was as untamed as the winds
that whispered through the ancient trees.
write that with no quotaion matks
ChatGPT
Once upon a time, in the small village of Eldoria, nestled between rolling hills and lush forests, there
 lived a young girl named Elara. Elara was known for her boundless curiosity and an insatiable 
 desire for adventure. Her eyes sparkled like the morning dew on a flower, and her spirit was as 
 untamed as the winds that whispered through the ancient trees.

One day, as Elara roamed the outskirts of the village, she stumbled upon a mysterious, 
overgrown path leading into the heart of the Enchanted Woods. 
The villagers had always spoken of the woods with a mixture of awe and caution, 
for it was said that magical creatures and ancient spirits dwelled within.

Undeterred by the warnings, Elara ventured deeper into the woods, guided by the ethereal glow of fireflies.
 The air was thick with enchantment, and the trees seemed to whisper secrets only the brave could comprehend. 
 As she journeyed further, the forest transformed around her, revealing a hidden realm of wonder.
Suddenly, Elara found herself in a clearing bathed in the soft light of a magical moonflower. 
In the center of the clearing stood a majestic unicorn, its coat shimmering like moonlight. 
The unicorn, named Celestia, spoke in a voice that echoed like a gentle melody.


Elara, captivated by the unicorns presence, shared her longing for adventure and discovery. 
Celestia, wise beyond the ages, saw the pure heart within the young girl and granted her a magical quest.

Brave Elara, embark on a journey to collect the Tears of Stardust, scattered across the realms of imagination. 
With each tear you gather, you will unlock the mysteries of the universe.

With newfound determination, Elara set forth on her quest. She traveled through lands of talking trees, 
danced with the northern lights, and soared on the wings of mythical birds. Along her journey, 
she encountered challenges that tested her courage and kindness, but with each trial, 
Elara grew wiser and more resilient.

As Elara collected the Tears of Stardust, she felt the magic within her heart blossom like a rare flower. 
The celestial tears granted her the ability to understand the language of the wind and the wisdom to mend broken spirits. 
She became a beacon of hope, lighting the way for those lost in the shadows.

In the end, Elara returned to her village, 
not only with the Tears of Stardust but also with a radiant spirit that touched the lives of everyone she met.
The once-small village of Eldoria transformed into a haven of dreams and possibilities, 
all because a young girl dared to embark on a magical quest.

And so, the tale of Elara and the Tears of Stardust became a cherished legend, 
passed down through generations, inspiring others to seek the extraordinary within themselves and the world around them.""")

doc_2 = nlp("hi there. I. am. a. good. boy.")


# we have a bug in this function - we get an empty token that we don't want.
def most_freq_words(doc):
    costume_stop_words = [",", " ", ".", "\n", r'[\x00-\x1F]']
    for stopword in costume_stop_words:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

    filtered_tokens = [token.text for token in doc if not token.is_stop]
    word_freq = Counter(filtered_tokens)

    most_common_words = word_freq.most_common(10)
    for word, freq in most_common_words:
        print(f'{word}: {freq} times')


def text_to_tokens(text):
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


def topic_modeling_LDA(corpus):
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


def text_to_vec(text):
    doc = nlp(text)


def main():
    topic_modeling_LDA(doc)
    print(sklearn.__version__)


if __name__ == "__main__":
    main()
