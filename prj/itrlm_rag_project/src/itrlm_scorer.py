"""
ITRLM Scorer Implementation
Computes similarity P(Q_exp_new | Q_exp_can) using α, β, λ, γ parameters
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer
import torch

from .text_processing import TextProcessor
from .pmi_dictionary import PMIDictionary

logger = logging.getLogger(__name__)


class ITRLMScorer:
    """ITRLM scorer for computing translation-based similarity."""
    
    def __init__(self, pmi_dict: PMIDictionary, config: Dict[str, Any], text_processor: TextProcessor):
        self.pmi_dict = pmi_dict
        self.config = config
        self.text_processor = text_processor
        
        # Parameters
        self.alpha = config.get('alpha', 0.3)  # Weight for general PMI translations
        self.beta = config.get('beta', 0.8)    # Weight for category-specific PMI translations
        self.lambda_ = config.get('lambda_', 0.2)  # Weight for semantic similarity
        self.gamma = config.get('gamma', 0.8)  # Weight for RAG-generated answers
        
        # Semantic similarity model
        self.semantic_model = None
        self._initialize_semantic_model()
    
    def _initialize_semantic_model(self) -> None:
        """Initialize semantic similarity model."""
        try:
            self.semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logger.info("Semantic similarity model loaded")
        except Exception as e:
            logger.warning(f"Failed to load semantic model: {e}")
            self.semantic_model = None
    
    def compute_similarity(self, query_new: str, query_candidate: str, 
                          category: Optional[str] = None) -> float:
        """
        Compute ITRLM similarity between two queries.
        
        Args:
            query_new: New query
            query_candidate: Candidate query
            category: Category for category-specific scoring
            
        Returns:
            Similarity score
        """
        # Preprocess queries
        tokens_new = self.text_processor.preprocess(query_new)
        tokens_candidate = self.text_processor.preprocess(query_candidate)
        
        if not tokens_new or not tokens_candidate:
            return 0.0
        
        # Compute translation-based similarity
        translation_sim = self._compute_translation_similarity(
            tokens_new, tokens_candidate, category
        )
        
        # Compute semantic similarity
        semantic_sim = self._compute_semantic_similarity(query_new, query_candidate)
        
        # Combine similarities
        total_similarity = (
            self.alpha * translation_sim['general'] +
            self.beta * translation_sim['category'] +
            self.lambda_ * semantic_sim
        )
        
        return total_similarity
    
    def _compute_translation_similarity(self, tokens_new: List[str], 
                                      tokens_candidate: List[str],
                                      category: Optional[str] = None) -> Dict[str, float]:
        """
        Compute translation-based similarity using PMI dictionaries.
        
        Args:
            tokens_new: Tokens from new query
            tokens_candidate: Tokens from candidate query
            category: Category for category-specific scoring
            
        Returns:
            Dictionary with general and category similarity scores
        """
        # General PMI similarity
        general_sim = self._compute_pmi_similarity(
            tokens_new, tokens_candidate, self.pmi_dict.general_pmi
        )
        
        # Category-specific PMI similarity
        category_sim = 0.0
        if category and category in self.pmi_dict.category_pmi:
            category_sim = self._compute_pmi_similarity(
                tokens_new, tokens_candidate, self.pmi_dict.category_pmi[category]
            )
        
        return {
            'general': general_sim,
            'category': category_sim
        }
    
    def _compute_pmi_similarity(self, tokens_new: List[str], tokens_candidate: List[str],
                               pmi_dict: Dict[Tuple[str, str], float]) -> float:
        """
        Compute PMI-based similarity between token sets.
        
        Args:
            tokens_new: Tokens from new query
            tokens_candidate: Tokens from candidate query
            pmi_dict: PMI dictionary to use
            
        Returns:
            PMI similarity score
        """
        if not pmi_dict:
            return 0.0
        
        # Compute bidirectional translation scores
        forward_scores = []
        backward_scores = []
        
        # Forward: new -> candidate
        for token_new in tokens_new:
            max_pmi = 0.0
            for token_candidate in tokens_candidate:
                pmi_score = pmi_dict.get((token_new, token_candidate), 0.0)
                max_pmi = max(max_pmi, pmi_score)
            forward_scores.append(max_pmi)
        
        # Backward: candidate -> new
        for token_candidate in tokens_candidate:
            max_pmi = 0.0
            for token_new in tokens_new:
                pmi_score = pmi_dict.get((token_candidate, token_new), 0.0)
                max_pmi = max(max_pmi, pmi_score)
            backward_scores.append(max_pmi)
        
        # Compute average scores
        forward_avg = np.mean(forward_scores) if forward_scores else 0.0
        backward_avg = np.mean(backward_scores) if backward_scores else 0.0
        
        # Return harmonic mean for balanced similarity
        if forward_avg + backward_avg > 0:
            return 2 * forward_avg * backward_avg / (forward_avg + backward_avg)
        else:
            return 0.0
    
    def _compute_semantic_similarity(self, query_new: str, query_candidate: str) -> float:
        """
        Compute semantic similarity using sentence embeddings.
        
        Args:
            query_new: New query
            query_candidate: Candidate query
            
        Returns:
            Semantic similarity score
        """
        if self.semantic_model is None:
            # Fallback to simple word overlap
            return self._compute_word_overlap_similarity(query_new, query_candidate)
        
        try:
            # Encode queries
            embeddings = self.semantic_model.encode([query_new, query_candidate])
            
            # Compute cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return self._compute_word_overlap_similarity(query_new, query_candidate)
    
    def _compute_word_overlap_similarity(self, query_new: str, query_candidate: str) -> float:
        """
        Compute simple word overlap similarity as fallback.
        
        Args:
            query_new: New query
            query_candidate: Candidate query
            
        Returns:
            Word overlap similarity score
        """
        tokens_new = set(self.text_processor.preprocess(query_new))
        tokens_candidate = set(self.text_processor.preprocess(query_candidate))
        
        if not tokens_new or not tokens_candidate:
            return 0.0
        
        intersection = len(tokens_new.intersection(tokens_candidate))
        union = len(tokens_new.union(tokens_candidate))
        
        return intersection / union if union > 0 else 0.0
    
    def score_query_answer_pair(self, query: str, answer: str, 
                               category: Optional[str] = None) -> float:
        """
        Score a query-answer pair using ITRLM.
        
        Args:
            query: Input query
            answer: Answer text
            category: Category for category-specific scoring
            
        Returns:
            ITRLM score
        """
        # Combine query and answer for scoring
        combined_text = f"{query} {answer}"
        
        # This is a simplified scoring - in practice, you might want to
        # score against a reference query or use other methods
        return self.compute_similarity(query, combined_text, category)
    
    def rank_candidates(self, query: str, candidates: List[str],
                       categories: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Rank candidate answers for a query.
        
        Args:
            query: Input query
            candidates: List of candidate answers
            categories: Optional list of categories for each candidate
            
        Returns:
            List of (candidate, score) tuples sorted by score
        """
        if categories is None:
            categories = [None] * len(candidates)
        
        scored_candidates = []
        for candidate, category in zip(candidates, categories):
            score = self.score_query_answer_pair(query, candidate, category)
            scored_candidates.append((candidate, score))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current ITRLM parameters."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'lambda_': self.lambda_,
            'gamma': self.gamma
        }
    
    def set_parameters(self, **kwargs) -> None:
        """Set ITRLM parameters."""
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        if 'beta' in kwargs:
            self.beta = kwargs['beta']
        if 'lambda_' in kwargs:
            self.lambda_ = kwargs['lambda_']
        if 'gamma' in kwargs:
            self.gamma = kwargs['gamma']
        
        logger.info(f"ITRLM parameters updated: {self.get_parameters()}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the ITRLM scorer."""
        return {
            'parameters': self.get_parameters(),
            'pmi_dict_stats': self.pmi_dict.get_stats(),
            'semantic_model_loaded': self.semantic_model is not None
        }
