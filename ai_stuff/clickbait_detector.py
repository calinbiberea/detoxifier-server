# Import the Word2Vec model and the mean vector representation computed on the train set:
import re
import emoji
import numpy as np
import pandas as pd
import pickle
from gensim.parsing.preprocessing import *

from sklearn.svm import SVC


word2vec = pickle.load(open("ai_stuff/word2vec", "rb"))
mean_title_embedding = pickle.load(open("ai_stuff/mean-title-embedding", "rb"))
# Compute the log of the video metadata or replace the missing values with the mean values obtained
# from the train set:
min_max_scaler = pickle.load(open("ai_stuff/min-max-scaler2", "rb"))
# Import the SVM model:
svm = pickle.load(open("ai_stuff/svm2", "rb"))


def is_clickbait(sample):
    input = {
        "video_title": sample,
    }
    sample = pd.DataFrame([input])

    # Tokenize the title and then compute its embedding:
    sample["video_title"] = sample["video_title"].apply(tokenize)
    sample["video_title"] = sample["video_title"].apply(
        average_embedding, word2vec=word2vec, na_vector=mean_title_embedding)
    sample = sample["video_title"].apply(pd.Series)
    # Replace any -Inf value with 0:
    sample = sample.replace(-np.inf, 0)

    # Import the min-max scaler and apply it to the sample:

    sample = pd.DataFrame(min_max_scaler.transform(
        sample), columns=sample.columns)

    # Print its prediction:
    #print(svm.predict(sample)[0])
    pred = svm.predict(sample)[0]
    if pred == 1:
        return "clickbait"
    else:
        return "normal"


def tokenize(string):
    """ Tokenizes a string.

    Adds a space between numbers and letters, removes punctuation, repeated whitespaces, words
    shorter than 2 characters, and stop-words. Returns a list of stems and, eventually, emojis.

    @param string: String to tokenize.
    @return: A list of stems and emojis.
    """

    # Based on the Ranks NL (Google) stopwords list, but "how" and "will" are not stripped, and
    # words shorter than 2 characters are not checked (since they are stripped):
    stop_words = [
        "about", "an", "are", "as", "at", "be", "by", "com", "for", "from", "in", "is", "it", "of",
        "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "who", "with",
        "the", "www"
    ]

    string = strip_short(
        strip_multiple_whitespaces(
            strip_punctuation(
                split_alphanum(string))),
        minsize=2)
    # Parse emojis:
    emojis = [c for c in string if c in emoji.UNICODE_EMOJI]
    # Remove every non-word character and stem each word:
    string = stem_text(re.sub(r"[^\w\s,]", "", string))
    # List of stems and emojis:
    tokens = string.split() + emojis

    for stop_word in stop_words:
        try:
            tokens.remove(stop_word)
        except:
            pass

    return tokens


def average_embedding(tokens, word2vec, na_vector=None):
    """ Embeds a title with the average representation of its tokens.

    Returns the mean vector representation of the tokens representations. When no token is in the
    Word2Vec model, it can be provided a vector to use instead (for example the mean vector
    representation of the train set titles).

    @param tokens: List of tokens to embed.
    @param word2vec: Word2Vec model.
    @param na_vector: Vector representation to use when no token is in the Word2Vec model.
    @return: A vector representation for the token list.
    """

    vectors = list()

    for token in tokens:
        if token in word2vec:
            vectors.append(word2vec[token])

    if len(vectors) == 0 and na_vector is not None:
        vectors.append(na_vector)

    return np.mean(np.array(vectors), axis=0)
