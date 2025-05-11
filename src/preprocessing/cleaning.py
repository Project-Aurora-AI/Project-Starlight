import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter

# Download necessary NLTK resources (run this once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Advanced cleaning of text including:
    - Lowercasing
    - Removing URLs
    - Removing special characters and punctuation
    - Removing stopwords
    - Lemmatization
    - Removing extra spaces
    """
    # Lowercase text
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove non-alphabetical characters (you can customize this)
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove stopwords
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])

    # Lemmatization (convert to root form)
    text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean_dataset(dataset):
    """
    Apply the `clean_text` function to the dataset.
    Handles both lists of strings or lists of dictionaries with a 'text' field.
    """
    # If dataset is a list of strings
    if isinstance(dataset, list) and isinstance(dataset[0], str):
        return [clean_text(text) for text in dataset]

    # If dataset is a list of dicts (like JSON with 'text' field)
    elif isinstance(dataset, list) and isinstance(dataset[0], dict):
        for entry in dataset:
            if 'text' in entry:
                entry['text'] = clean_text(entry['text'])
        return dataset

    else:
        raise ValueError("Dataset format not recognized. Must be list of strings or list of dicts.")

def expand_contractions(text):
    """
    Expands common contractions like "don't" to "do not" in the text.
    """
    contractions = {
        "don't": "do not", "can't": "cannot", "won't": "will not", "isn't": "is not", 
        "aren't": "are not", "wasn't": "was not", "weren't": "were not", "haven't": "have not", 
        "hasn't": "has not", "hadn't": "had not", "doesn't": "does not", "didn't": "did not",
        "you've": "you have", "you're": "you are", "it's": "it is", "i'm": "i am",
        # Add more contractions as needed
    }
    for contraction, expanded in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expanded, text)
    return text

def remove_emojis(text):
    """
    Remove emojis from text if necessary.
    """
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # Enclosed characters
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

