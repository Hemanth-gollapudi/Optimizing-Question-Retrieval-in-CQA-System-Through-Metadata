"""
Text processing utilities for ITRLM+RAG
"""

import re
import nltk
from typing import List, Set, Optional
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text processing utilities for cleaning and preprocessing."""
    
    def __init__(self):
        self.stopwords = self._load_stopwords()
        self.stemmer = nltk.stem.PorterStemmer()
        
    def _load_stopwords(self) -> Set[str]:
        """Load English stopwords."""
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Could not load NLTK stopwords: {e}")
            return set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Simple whitespace tokenization
        tokens = text.split()
        
        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens without stopwords
        """
        return [token for token in tokens if token not in self.stopwords]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Stem tokens using Porter stemmer.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text: str, remove_stopwords: bool = True, stem: bool = True) -> List[str]:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            stem: Whether to stem tokens
            
        Returns:
            List of processed tokens
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Stem
        if stem:
            tokens = self.stem_tokens(tokens)
        
        return tokens
    
    def get_vocabulary(self, texts: List[str]) -> Set[str]:
        """
        Get vocabulary from a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Set of unique tokens
        """
        vocabulary = set()
        for text in texts:
            tokens = self.preprocess(text)
            vocabulary.update(tokens)
        
        return vocabulary
    
    def get_word_frequencies(self, texts: List[str]) -> Counter:
        """
        Get word frequencies from a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Counter of word frequencies
        """
        word_freq = Counter()
        for text in texts:
            tokens = self.preprocess(text)
            word_freq.update(tokens)
        
        return word_freq
