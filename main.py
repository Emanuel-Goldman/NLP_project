import spacy
from collections import Counter

nlp = spacy.load('en_core_web_sm')

doc = nlp(
    """Once upon a time, in the small village of Eldoria, nestled between rolling hills and lush forests, there lived a young girl named Elara. Elara was known for her boundless curiosity and an insatiable desire for adventure. Her eyes sparkled like the morning dew on a flower, and her spirit was as untamed as the winds that whispered through the ancient trees.

One day, as Elara roamed the outskirts of the village, she stumbled upon a mysterious, overgrown path leading into the heart of the Enchanted Woods. The villagers had always spoken of the woods with a mixture of awe and caution, for it was said that magical creatures and ancient spirits dwelled within.

Undeterred by the warnings, Elara ventured deeper into the woods, guided by the ethereal glow of fireflies. The air was thick with enchantment, and the trees seemed to whisper secrets only the brave could comprehend. As she journeyed further, the forest transformed around her, revealing a hidden realm of wonder.

Suddenly, Elara found herself in a clearing bathed in the soft light of a magical moonflower. In the center of the clearing stood a majestic unicorn, its coat shimmering like moonlight. The unicorn, named Celestia, spoke in a voice that echoed like a gentle melody.

Elara, seeker of the extraordinary, you have entered the realm of dreams and desires. What brings you to this enchanted place?

Elara, captivated by the unicorn's presence, shared her longing for adventure and discovery. Celestia, wise beyond the ages, saw the pure heart within the young girl and granted her a magical quest.

Brave Elara, embark on a journey to collect the Tears of Stardust, scattered across the realms of imagination. With each tear you gather, you will unlock the mysteries of the universe.

With newfound determination, Elara set forth on her quest. She traveled through lands of talking trees, danced with the northern lights, and soared on the wings of mythical birds. Along her journey, she encountered challenges that tested her courage and kindness, but with each trial, Elara grew wiser and more resilient.

As Elara collected the Tears of Stardust, she felt the magic within her heart blossom like a rare flower. The celestial tears granted her the ability to understand the language of the wind and the wisdom to mend broken spirits. She became a beacon of hope, lighting the way for those lost in the shadows.

In the end, Elara returned to her village, not only with the Tears of Stardust but also with a radiant spirit that touched the lives of everyone she met. The once-small village of Eldoria transformed into a haven of dreams and possibilities, all because a young girl dared to embark on a magical quest.

And so, the tale of Elara and the Tears of Stardust became a cherished legend, passed down through generations, inspiring others to seek the extraordinary within themselves and the world around them.




User
write that with no quotaion matks
ChatGPT
Once upon a time, in the small village of Eldoria, nestled between rolling hills and lush forests, there lived a young girl named Elara. Elara was known for her boundless curiosity and an insatiable desire for adventure. Her eyes sparkled like the morning dew on a flower, and her spirit was as untamed as the winds that whispered through the ancient trees.

One day, as Elara roamed the outskirts of the village, she stumbled upon a mysterious, overgrown path leading into the heart of the Enchanted Woods. The villagers had always spoken of the woods with a mixture of awe and caution, for it was said that magical creatures and ancient spirits dwelled within.

Undeterred by the warnings, Elara ventured deeper into the woods, guided by the ethereal glow of fireflies. The air was thick with enchantment, and the trees seemed to whisper secrets only the brave could comprehend. As she journeyed further, the forest transformed around her, revealing a hidden realm of wonder.

Suddenly, Elara found herself in a clearing bathed in the soft light of a magical moonflower. In the center of the clearing stood a majestic unicorn, its coat shimmering like moonlight. The unicorn, named Celestia, spoke in a voice that echoed like a gentle melody.

Elara, captivated by the unicorns presence, shared her longing for adventure and discovery. Celestia, wise beyond the ages, saw the pure heart within the young girl and granted her a magical quest.

Brave Elara, embark on a journey to collect the Tears of Stardust, scattered across the realms of imagination. With each tear you gather, you will unlock the mysteries of the universe.

With newfound determination, Elara set forth on her quest. She traveled through lands of talking trees, danced with the northern lights, and soared on the wings of mythical birds. Along her journey, she encountered challenges that tested her courage and kindness, but with each trial, Elara grew wiser and more resilient.

As Elara collected the Tears of Stardust, she felt the magic within her heart blossom like a rare flower. The celestial tears granted her the ability to understand the language of the wind and the wisdom to mend broken spirits. She became a beacon of hope, lighting the way for those lost in the shadows.

In the end, Elara returned to her village, not only with the Tears of Stardust but also with a radiant spirit that touched the lives of everyone she met. The once-small village of Eldoria transformed into a haven of dreams and possibilities, all because a young girl dared to embark on a magical quest.

And so, the tale of Elara and the Tears of Stardust became a cherished legend, passed down through generations, inspiring others to seek the extraordinary within themselves and the world around them.""")


#תדירות
#נושאים
#אורך משפט
#חלקי דיבר
#למטיזיישן

def most_freq_words(doc):

    costum_stop_words = [","," ",".","\n",r'[\x00-\x1F]']
    for stopword in costum_stop_words:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

    filtered_tokens = [token.text for token in doc if not token.is_stop]
    word_freq = Counter(filtered_tokens)

    most_common_words = word_freq.most_common(10)
    for word, freq in most_common_words:
        print(f'{word}: {freq} times')

def main():
    print(most_freq_words(doc))


if __name__ == "__main__":
    main()
