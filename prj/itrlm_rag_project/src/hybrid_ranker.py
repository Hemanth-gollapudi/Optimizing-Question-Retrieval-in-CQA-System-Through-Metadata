"""
Hybrid Ranker for ITRLM+RAG
Multi-stage ranking pipeline: semantic → ITRLM → reranking
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer
import torch

from .itrlm_scorer import ITRLMScorer
from .category_predictor import CategoryPredictor
from .rag_generator import RAGAnswerGen

logger = logging.getLogger(__name__)


class HybridRanker:
    """Multi-stage hybrid ranking system combining semantic, ITRLM, and RAG approaches."""
    
    def __init__(self, itrlm_scorer: ITRLMScorer, category_predictor: CategoryPredictor,
                 rag_generator: RAGAnswerGen, config: Dict[str, Any]):
        self.itrlm_scorer = itrlm_scorer
        self.category_predictor = category_predictor
        self.rag_generator = rag_generator
        self.config = config
        
        # Ranking weights
        self.semantic_weight = config.get('semantic_weight', 0.3)
        self.itrlm_weight = config.get('itrlm_weight', 0.5)
        self.rag_weight = config.get('rag_weight', 0.2)
        
        # Semantic similarity model
        self.semantic_model = None
        self._initialize_semantic_model()
    
    def _initialize_semantic_model(self) -> None:
        """Initialize semantic similarity model."""
        try:
            self.semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logger.info("Hybrid ranker semantic model loaded")
        except Exception as e:
            logger.warning(f"Failed to load semantic model: {e}")
            self.semantic_model = None
    
    def rank(self, query: str, candidates: List[Dict[str, Any]], 
             top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rank candidates using hybrid approach.
        
        Args:
            query: Input query
            candidates: List of candidate answers with metadata
            top_k: Number of top results to return
            
        Returns:
            List of ranked candidates with scores
        """
        logger.info(f"Ranking {len(candidates)} candidates for query: {query[:50]}...")
        
        # Stage 1: Semantic ranking
        semantic_scores = self._compute_semantic_scores(query, candidates)
        
        # Stage 2: ITRLM ranking
        itrlm_scores = self._compute_itrlm_scores(query, candidates)
        
        # Stage 3: RAG-based ranking
        rag_scores = self._compute_rag_scores(query, candidates)
        
        # Combine scores
        final_scores = self._combine_scores(semantic_scores, itrlm_scores, rag_scores)
        
        # Sort by final scores
        ranked_candidates = self._sort_candidates(candidates, final_scores)
        
        return ranked_candidates[:top_k]
    
    def _compute_semantic_scores(self, query: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """Compute semantic similarity scores."""
        if self.semantic_model is None:
            # Fallback to simple word overlap
            return [self._compute_word_overlap(query, candidate.get('answer', '')) 
                   for candidate in candidates]
        
        try:
            # Encode query
            query_embedding = self.semantic_model.encode([query])
            
            # Encode candidates
            candidate_texts = [candidate.get('answer', '') for candidate in candidates]
            candidate_embeddings = self.semantic_model.encode(candidate_texts)
            
            # Compute cosine similarities
            similarities = []
            for candidate_embedding in candidate_embeddings:
                similarity = np.dot(query_embedding[0], candidate_embedding) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(candidate_embedding)
                )
                similarities.append(float(similarity))
            
            return similarities
            
        except Exception as e:
            logger.warning(f"Semantic scoring failed: {e}")
            return [0.0] * len(candidates)
    
    def _compute_itrlm_scores(self, query: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """Compute ITRLM scores."""
        scores = []
        
        for candidate in candidates:
            answer = candidate.get('answer', '')
            category = candidate.get('category')
            
            # Predict category if not provided
            if category is None:
                category, _ = self.category_predictor.predict_category(query)
            
            # Compute ITRLM score
            score = self.itrlm_scorer.score_query_answer_pair(query, answer, category)
            scores.append(score)
        
        return scores
    
    def _compute_rag_scores(self, query: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """Compute RAG-based scores."""
        scores = []
        
        # Generate RAG answer for the query
        try:
            rag_answer = self.rag_generator.generate_answer(query)
            
            # Compute similarity between RAG answer and candidates
            for candidate in candidates:
                answer = candidate.get('answer', '')
                similarity = self._compute_text_similarity(rag_answer, answer)
                scores.append(similarity)
                
        except Exception as e:
            logger.warning(f"RAG scoring failed: {e}")
            scores = [0.0] * len(candidates)
        
        return scores
    
    def _combine_scores(self, semantic_scores: List[float], itrlm_scores: List[float],
                       rag_scores: List[float]) -> List[float]:
        """Combine different scoring approaches."""
        # Normalize scores to [0, 1] range
        semantic_norm = self._normalize_scores(semantic_scores)
        itrlm_norm = self._normalize_scores(itrlm_scores)
        rag_norm = self._normalize_scores(rag_scores)
        
        # Weighted combination
        final_scores = []
        for i in range(len(semantic_scores)):
            combined_score = (
                self.semantic_weight * semantic_norm[i] +
                self.itrlm_weight * itrlm_norm[i] +
                self.rag_weight * rag_norm[i]
            )
            final_scores.append(combined_score)
        
        return final_scores
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)  # All equal scores
        
        normalized = [(score - min_score) / (max_score - min_score) for score in scores]
        return normalized
    
    def _sort_candidates(self, candidates: List[Dict[str, Any]], 
                        scores: List[float]) -> List[Dict[str, Any]]:
        """Sort candidates by scores."""
        # Create list of (candidate, score) tuples
        candidate_scores = list(zip(candidates, scores))
        
        # Sort by score (descending)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add scores to candidates and return
        ranked_candidates = []
        for candidate, score in candidate_scores:
            candidate_with_score = candidate.copy()
            candidate_with_score['hybrid_score'] = score
            ranked_candidates.append(candidate_with_score)
        
        return ranked_candidates
    
    def _compute_word_overlap(self, text1: str, text2: str) -> float:
        """Compute word overlap similarity as fallback."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        if self.semantic_model is None:
            return self._compute_word_overlap(text1, text2)
        
        try:
            embeddings = self.semantic_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Text similarity computation failed: {e}")
            return self._compute_word_overlap(text1, text2)
    
    def rerank_with_feedback(self, query: str, candidates: List[Dict[str, Any]],
                           feedback: List[float]) -> List[Dict[str, Any]]:
        """
        Rerank candidates based on user feedback.
        
        Args:
            query: Input query
            candidates: List of candidates
            feedback: List of feedback scores
            
        Returns:
            Reranked candidates
        """
        # Simple feedback integration - in practice, you might use more sophisticated methods
        feedback_weight = 0.3
        
        for i, candidate in enumerate(candidates):
            if i < len(feedback):
                current_score = candidate.get('hybrid_score', 0.0)
                feedback_score = feedback[i]
                
                # Combine current score with feedback
                new_score = (1 - feedback_weight) * current_score + feedback_weight * feedback_score
                candidate['hybrid_score'] = new_score
        
        # Sort by updated scores
        candidates.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
        
        return candidates
    
    def get_ranking_weights(self) -> Dict[str, float]:
        """Get current ranking weights."""
        return {
            'semantic_weight': self.semantic_weight,
            'itrlm_weight': self.itrlm_weight,
            'rag_weight': self.rag_weight
        }
    
    def set_ranking_weights(self, semantic_weight: float = None, 
                           itrlm_weight: float = None, rag_weight: float = None) -> None:
        """Set ranking weights."""
        if semantic_weight is not None:
            self.semantic_weight = semantic_weight
        if itrlm_weight is not None:
            self.itrlm_weight = itrlm_weight
        if rag_weight is not None:
            self.rag_weight = rag_weight
        
        # Normalize weights
        total = self.semantic_weight + self.itrlm_weight + self.rag_weight
        if total > 0:
            self.semantic_weight /= total
            self.itrlm_weight /= total
            self.rag_weight /= total
        
        logger.info(f"Ranking weights updated: {self.get_ranking_weights()}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the hybrid ranker."""
        return {
            'ranking_weights': self.get_ranking_weights(),
            'semantic_model_loaded': self.semantic_model is not None,
            'itrlm_stats': self.itrlm_scorer.get_stats(),
            'rag_stats': self.rag_generator.get_stats()
        }
