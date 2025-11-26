import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ps = PorterStemmer()
STEMMED_STOPWORDS = set(ps.stem(w) for w in ENGLISH_STOP_WORDS)

def normalize(text: str) -> str:
    if text is None:
        return ""
    t = re.sub(r"http\S+|www\S+", " ", str(text))
    t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[^A-Za-z]+", " ", t)
    t = t.lower()
    tokens = t.split()
    tokens = [ps.stem(tok) for tok in tokens]
    tokens = [tok for tok in tokens if tok not in STEMMED_STOPWORDS]
    return " ".join(tokens)
