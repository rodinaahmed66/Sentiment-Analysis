import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt') 
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
emoticon_meanings = {
    ":)": "Happy",
    ":(": "Sad",
    ":D": "Very Happy",
    ":|": "Neutral",
    ":O": "Surprised",
    "<3": "Love",
    ";)": "Wink",
    ":P": "Playful",
    ":/": "Confused",
    ":*": "Kiss",
    ":')": "Touched",
    "XD": "Laughing",
    ":3": "Cute",
    ">:(": "Angry",
    ":-O": "Shocked",
    ":|]": "Robot",
    ":>": "Sly",
    "^_^": "Happy",
    "O_o": "Confused",
    ":-|": "Straight Face",
    ":X": "Silent",
    "B-)": "Cool",
    "<(‘.'<)": "Dance",
    "(-_-)": "Bored",
    "(>_<)": "Upset",
    "(¬‿¬)": "Sarcastic",
    "(o_o)": "Surprised",
    "(o.O)": "Shocked",
    ":0": "Shocked",
    ":*(": "Crying",
    ":v": "Pac-Man",
    "(^_^)v": "Double Victory",
    ":-D": "Big Grin",
    ":-*": "Blowing a Kiss",
    ":^)": "Nosey",
    ":-((": "Very Sad",
    ":-(": "Frowning",
}

def remove_repeated_characters(text, max_repeats=2):
    """Remove repeated characters, e.g., 'loooove' becomes 'love'."""
    pattern = r"(.)\1{%d,}" % (max_repeats - 1)
    return re.sub(pattern, r"\1", text)

def remove_mentions(text):
    """Remove mentions (e.g., @user) from the text."""
    return re.sub(r"@[\w]*", '', text)

def remove_links(text):
    """Remove URLs from the text."""
    return re.sub(r"https?://\S+|www\.\S+", '', text)


def convert_emoticons(text: str):
    ''' This Function is to replace the emoticons with thier meaning instead '''
    for emoticon, meaning in emoticon_meanings.items():
         text = text.replace(emoticon, '' + meaning + '')
    return text

def remove_punctuations(text):
    """Remove punctuation and non-alphabetic characters, except for apostrophes."""
    return re.sub(r"[^a-zA-Z\s']", '', text)

def tokenize_and_lemmatize(text):
    """Tokenize and lemmatize the text."""
    tokenized_text = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in tokenized_text])


def clean_text(text):
    """
    Apply all text preprocessing steps in order.
    """
    text = text.lower().strip()
    text = remove_links(text)
    text = remove_mentions(text)
    text = remove_repeated_characters(text)
    text = convert_emoticons(text)
    text = remove_punctuations(text)
    text = tokenize_and_lemmatize(text)
    text = text.lower()
    text = ' '.join(text.split())  # remove redundant spaces
    return text
