import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class TextProcessor:
    def __init__(self, language='english'):
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()


    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = [self.stemmer.stem(t) for t in text.split() if t not in self.stop_words]
        return ' '.join(tokens)


    def batch_clean(self, texts):
        return [self.clean(t) for t in texts]