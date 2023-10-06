import re

from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
              'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
              'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
              'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
              'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
              'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
              'through', 'during', 'before', 'after', 'above', 'below', 'to',
              'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
              'again', 'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
              'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
              'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
              'can', 'will', 'just', 'don', 'should', 'now']
LEMMATIZER = WordNetLemmatizer()

STEMMER = SnowballStemmer("english")

PUNCTUATION  = set('''!()-[]{};:'"\,<>./?@#$%^&*_~''')


# basic stopword removal
def remove_stopwords(text, do_stemming=False):
    word_tokens = word_tokenize(text.lower())
    filtered_text = [w for w in word_tokens if not w in STOP_WORDS and not w in PUNCTUATION]
    filtered_text_stemmed = filtered_text
    if do_stemming:
        filtered_text_stemmed = [STEMMER.stem(w) for w in filtered_text_stemmed]
    return ' '.join(filtered_text_stemmed)


# basic stopword removal
def remove_stopwords_and_lemmatize(text, do_stemming=False, lemmatize=False):
    word_tokens = word_tokenize(text.lower())
    filtered_text = [w for w in word_tokens if not w in STOP_WORDS and not w in PUNCTUATION]

    filtered_text_stemmed = filtered_text
    if do_stemming:
        filtered_text_stemmed = [STEMMER.stem(w) for w in filtered_text_stemmed]

    filtered_text_lemmatized = filtered_text_stemmed
    if lemmatize:
        filtered_text_lemmatized = [LEMMATIZER.lemmatize(w) for w in filtered_text_lemmatized]
    return ' '.join(filtered_text_lemmatized)


def parse_ambiguous_request(request_string):
    output_dict = {}

    # Check if the string starts with "Ambiguous request"
    if request_string.startswith("Ambiguous request"):

        # Split the string into lines
        lines = request_string.split('\n')

        # Iterate through the lines (excluding the first one)
        for line in lines[1:]:
            # Use regular expression to match the index and rest of the string after the ":"
            match = re.match(r'(\d+):\s+(.*)', line)
            if match:
                index, action = match.groups()
                output_dict[int(index)] = action

    return output_dict