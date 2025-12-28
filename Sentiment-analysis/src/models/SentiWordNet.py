import nltk
import string

from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer as wnl
from nltk.corpus import stopwords

# nltk.download('all')

english_stopwords = set(stopwords.words("english"))
NEGATIONS = {"not", "no", "nor", "never", "n't"}
clean_stopwords = english_stopwords - NEGATIONS

def text_processing(text, verbose = False):

    """
    Preprocess the input text: tokenize, lemmatize, remove stopwords & punctuation,
    and perform POS tagging.

    Parameters
    ----------
    text : str
        The raw input text to process.

    Returns
    -------
    list of tuple
        A list of (word, POS_tag) pairs after preprocessing.
        Example: [('good', 'JJ'), ('product', 'NN'), ...]
    """

    if verbose:
        print("Tokenization...")
    tokens = nltk.word_tokenize(text)

    lemms = []
    if verbose:
        print("Lemmatization...")
    for tok in tokens:
        lemmatization = wnl().lemmatize(tok.lower())
        if lemmatization.lower() not in clean_stopwords and lemmatization.lower() not in string.punctuation:
            lemms.append(lemmatization)

    if verbose:
        print("POS tagging...")
    pos_tags = nltk.pos_tag(lemms)

    return pos_tags




def sentiwordnet_scores(pos_tags):

    """
    Compute sentiment scores using SentiWordNet for the given POS-tagged tokens.

    The function:
    - maps NLTK POS tags to WordNet POS tags,
    - retrieves WordNet synsets,
    - retrieves SentiWordNet sentiment scores for the first matching synset,
    - accumulates total positive and negative scores.

    Parameters
    ----------
    pos_tags : list of tuple
        Output of text_processing(), i.e., a list of (word, POS_tag) pairs.

    Returns
    -------
    tuple
        (positive_sentiment, negative_sentiment)
        Positive and negative sentiment scores aggregated across all words.
    """

    positive_sentiment = 0
    negative_sentiment = 0


    for element in pos_tags:
        if element[1].startswith('J'):
            tag = wn.ADJ
        elif element[1].startswith('R'):
            tag = wn.ADV
        elif element[1].startswith('N'):
            tag = wn.NOUN
        elif element[1].startswith('V'):
            tag = wn.VERB   
        else:
            continue

        word_synsets = wn.synsets(element[0], tag)

        if len(word_synsets) > 0:    
            word_senti = swn.senti_synset(word_synsets[0].name())
            positive_sentiment += word_senti.pos_score()
            negative_sentiment += word_senti.neg_score()    
        else:
            continue
    
    return positive_sentiment, negative_sentiment



def swn_predict_sentiment(text, pos_threshold = 0.5, neg_threshold = -0.5):
    
    """
    Predict the sentiment of a text as 'positive', 'negative', or 'neutral'.

    Steps:
    - Preprocess the text (tokenize, lemmatize, remove stopwords and punctuation, POS tagging)
    using text_processing().
    - Compute positive and negative sentiment scores using sentiwordnet_scores().
    - Calculate the difference between positive and negative scores.
    - Assign a label: 'positive', 'negative', or 'neutral'.

    Parameters
    ----------
    text : str
        The text to analyze.

    Returns
    -------
    str
        The predicted sentiment: 'positive', 'negative', or 'neutral'.
    """

    pos_tags = text_processing(text)
    positive_sentiment, negative_sentiment = sentiwordnet_scores(pos_tags)
    score_diff = positive_sentiment - negative_sentiment

    if score_diff > pos_threshold:
        return 'positive'
    elif score_diff < neg_threshold:
        return 'negative'
    else:
        return 'neutral'


