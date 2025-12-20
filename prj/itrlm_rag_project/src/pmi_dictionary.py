"""
PMI Dictionary Builder for ITRLM
Builds general and category-specific PMI dictionaries
"""

import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any, Optional
import logging
from pathlib import Path
import pickle

from .text_processing import TextProcessor
from .utils import save_pickle, load_pickle

logger = logging.getLogger(__name__)


class PMIDictionary:
    """Builds and manages PMI dictionaries for translation-based similarity."""
    
    def __init__(self, config: Dict[str, Any], text_processor: TextProcessor):
        self.config = config
        self.text_processor = text_processor
        
        # PMI dictionaries
        self.general_pmi = {}  # General PMI scores
        self.category_pmi = defaultdict(dict)  # Category-specific PMI scores
        
        # Statistics
        self.word_counts = Counter()
        self.category_word_counts = defaultdict(Counter)
        self.total_words = 0
        self.category_total_words = defaultdict(int)
        
    def build_all(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Build both general and category-specific PMI dictionaries.
        
        Args:
            training_data: List of training examples with 'question', 'answer', 'category'
        """
        logger.info("Building PMI dictionaries...")
        
        # Extract all text
        all_texts = []
        category_texts = defaultdict(list)
        
        for item in training_data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            category = item.get('category', 'unknown')
            
            # Combine question and answer
            combined_text = f"{question} {answer}"
            all_texts.append(combined_text)
            category_texts[category].append(combined_text)
        
        # Build general PMI dictionary
        self._build_general_pmi(all_texts)
        
        # Build category-specific PMI dictionaries
        for category, texts in category_texts.items():
            if len(texts) > 10:  # Only build if enough data
                self._build_category_pmi(category, texts)
        
        logger.info(f"Built PMI dictionaries for {len(self.category_pmi)} categories")
    
    def _build_general_pmi(self, texts: List[str]) -> None:
        """Build general PMI dictionary from all texts."""
        logger.info("Building general PMI dictionary...")
        
        # Count word co-occurrences
        cooccurrence_counts = defaultdict(int)
        word_counts = Counter()
        total_words = 0
        
        for text in texts:
            tokens = self.text_processor.preprocess(text)
            total_words += len(tokens)
            
            # Count individual words
            for token in tokens:
                word_counts[token] += 1
            
            # Count co-occurrences within window
            window_size = 5  # Context window size
            for i, token1 in enumerate(tokens):
                for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
                    if i != j:
                        token2 = tokens[j]
                        cooccurrence_counts[(token1, token2)] += 1
        
        # Calculate PMI scores
        self._calculate_pmi_scores(cooccurrence_counts, word_counts, total_words, self.general_pmi)
        
        logger.info(f"General PMI dictionary built with {len(self.general_pmi)} word pairs")
    
    def _build_category_pmi(self, category: str, texts: List[str]) -> None:
        """Build category-specific PMI dictionary."""
        logger.info(f"Building PMI dictionary for category: {category}")
        
        # Count word co-occurrences for this category
        cooccurrence_counts = defaultdict(int)
        word_counts = Counter()
        total_words = 0
        
        for text in texts:
            tokens = self.text_processor.preprocess(text)
            total_words += len(tokens)
            
            # Count individual words
            for token in tokens:
                word_counts[token] += 1
            
            # Count co-occurrences within window
            window_size = 5
            for i, token1 in enumerate(tokens):
                for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
                    if i != j:
                        token2 = tokens[j]
                        cooccurrence_counts[(token1, token2)] += 1
        
        # Calculate PMI scores for this category
        category_pmi = {}
        self._calculate_pmi_scores(cooccurrence_counts, word_counts, total_words, category_pmi)
        self.category_pmi[category] = category_pmi
        
        logger.info(f"Category {category} PMI dictionary built with {len(category_pmi)} word pairs")
    
    def _calculate_pmi_scores(self, cooccurrence_counts: Dict[Tuple[str, str], int], 
                             word_counts: Counter, total_words: int, 
                             pmi_dict: Dict[Tuple[str, str], float]) -> None:
        """Calculate PMI scores from co-occurrence counts."""
        for (word1, word2), cooccur_count in cooccurrence_counts.items():
            if cooccur_count < 2:  # Skip rare co-occurrences
                continue
                
            # Calculate PMI
            p_word1 = word_counts[word1] / total_words
            p_word2 = word_counts[word2] / total_words
            p_cooccur = cooccur_count / total_words
            
            if p_word1 > 0 and p_word2 > 0 and p_cooccur > 0:
                pmi = np.log2(p_cooccur / (p_word1 * p_word2))
                if pmi > 0:  # Only keep positive PMI
                    pmi_dict[(word1, word2)] = pmi
    
    def get_translations(self, word: str, category: Optional[str] = None, 
                        top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get translations for a word using PMI scores.
        
        Args:
            word: Input word
            category: Category for category-specific translations
            top_k: Number of top translations to return
            
        Returns:
            List of (translation, pmi_score) tuples
        """
        translations = []
        
        # Get general translations
        for (w1, w2), pmi_score in self.general_pmi.items():
            if w1 == word:
                translations.append((w2, pmi_score))
        
        # Add category-specific translations if available
        if category and category in self.category_pmi:
            for (w1, w2), pmi_score in self.category_pmi[category].items():
                if w1 == word:
                    translations.append((w2, pmi_score))
        
        # Sort by PMI score and return top-k
        translations.sort(key=lambda x: x[1], reverse=True)
        return translations[:top_k]
    
    def expand_query(self, query_tokens: List[str], category: Optional[str] = None,
                    top_n: int = 100) -> List[str]:
        """
        Expand query using PMI translations.
        
        Args:
            query_tokens: List of query tokens
            category: Category for category-specific expansion
            top_n: Maximum number of expansion terms
            
        Returns:
            List of expanded terms
        """
        expanded_terms = set(query_tokens)  # Start with original terms
        
        for token in query_tokens:
            translations = self.get_translations(token, category, top_k=10)
            for translation, score in translations:
                if score > self.config.get('min_pmi_threshold', 0.1):
                    expanded_terms.add(translation)
        
        # Convert back to list and limit size
        expanded_list = list(expanded_terms)
        return expanded_list[:top_n]
    
    def save(self, filepath: str) -> None:
        """Save PMI dictionaries to file."""
        data = {
            'general_pmi': self.general_pmi,
            'category_pmi': dict(self.category_pmi),
            'config': self.config
        }
        save_pickle(data, filepath)
        logger.info(f"PMI dictionaries saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load PMI dictionaries from file."""
        data = load_pickle(filepath)
        self.general_pmi = data['general_pmi']
        self.category_pmi = defaultdict(dict, data['category_pmi'])
        self.config = data['config']
        logger.info(f"PMI dictionaries loaded from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the PMI dictionaries."""
        return {
            'general_pmi_size': len(self.general_pmi),
            'num_categories': len(self.category_pmi),
            'category_sizes': {cat: len(pmi) for cat, pmi in self.category_pmi.items()}
        }
