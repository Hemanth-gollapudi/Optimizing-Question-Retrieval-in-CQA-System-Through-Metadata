"""
RAG Answer Generator for ITRLM
Uses sentence embeddings and RAG generation for answer expansion
"""

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from pathlib import Path

from .text_processing import TextProcessor
from .utils import save_pickle, load_pickle

logger = logging.getLogger(__name__)


class RAGAnswerGen:
    """RAG-based answer generator for query expansion."""
    
    def __init__(self, config: Dict[str, Any], text_processor: TextProcessor):
        self.config = config
        self.text_processor = text_processor
        
        # Models
        self.embedding_model = None
        self.generation_model = None
        self.generation_tokenizer = None
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Knowledge base
        self.knowledge_base = []
        self.knowledge_embeddings = None
        
    def initialize_models(self) -> None:
        """Initialize embedding and generation models."""
        logger.info("Initializing RAG models...")
        
        try:
            # Initialize embedding model
            embed_model_name = self.config.get('embed_model', 'sentence-transformers/all-mpnet-base-v2')
            self.embedding_model = SentenceTransformer(embed_model_name)
            logger.info(f"Embedding model loaded: {embed_model_name}")
            
            # Initialize generation model (use a smaller model for demo)
            gen_model_name = self.config.get('gen_model', 'microsoft/DialoGPT-medium')
            self.generation_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
            self.generation_model = AutoModelForCausalLM.from_pretrained(gen_model_name)
            
            # Add padding token if not present
            if self.generation_tokenizer.pad_token is None:
                self.generation_tokenizer.pad_token = self.generation_tokenizer.eos_token
            
            self.generation_model.to(self.device)
            logger.info(f"Generation model loaded: {gen_model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load models: {e}, using dummy implementation")
            self._create_dummy_models()
    
    def _create_dummy_models(self) -> None:
        """Create dummy models for demonstration."""
        logger.info("Creating dummy RAG models...")
        
        # Dummy embedding model
        class DummyEmbeddingModel:
            def encode(self, texts):
                # Return random embeddings
                if isinstance(texts, str):
                    texts = [texts]
                return np.random.randn(len(texts), 768)
        
        self.embedding_model = DummyEmbeddingModel()
        
        # Dummy generation model
        class DummyGenerationModel:
            def generate(self, input_ids, max_length=50, **kwargs):
                # Return dummy generated tokens
                batch_size = input_ids.shape[0]
                dummy_tokens = torch.randint(0, 1000, (batch_size, max_length))
                return dummy_tokens
        
        self.generation_model = DummyGenerationModel()
        
        # Dummy tokenizer
        class DummyTokenizer:
            def __init__(self):
                self.pad_token = '<pad>'
                self.eos_token = '<eos>'
            
            def encode(self, text, return_tensors=None, **kwargs):
                # Return dummy token IDs
                tokens = [1, 2, 3, 4, 5]  # Dummy tokens
                if return_tensors == 'pt':
                    return torch.tensor([tokens])
                return tokens
            
            def decode(self, tokens, skip_special_tokens=True):
                # Return dummy decoded text
                return "This is a dummy generated answer."
        
        self.generation_tokenizer = DummyTokenizer()
    
    def build_knowledge_base(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Build knowledge base from training data.
        
        Args:
            training_data: List of training examples
        """
        logger.info("Building knowledge base...")
        
        # Extract answers from training data
        self.knowledge_base = []
        for item in training_data:
            answer = item.get('answer', '')
            if answer:
                self.knowledge_base.append(answer)
        
        if not self.knowledge_base:
            logger.warning("No knowledge base data available")
            return
        
        # Initialize models if not done
        if self.embedding_model is None:
            self.initialize_models()
        
        # Create embeddings for knowledge base
        self.knowledge_embeddings = self.embedding_model.encode(self.knowledge_base)
        
        logger.info(f"Knowledge base built with {len(self.knowledge_base)} entries")
    
    def retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Input query
            top_k: Number of top contexts to retrieve
            
        Returns:
            List of relevant context strings
        """
        if self.knowledge_embeddings is None or len(self.knowledge_base) == 0:
            logger.warning("Knowledge base not built, returning empty context")
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Compute similarities
        similarities = np.dot(query_embedding, self.knowledge_embeddings.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return relevant contexts
        relevant_contexts = [self.knowledge_base[idx] for idx in top_indices]
        
        return relevant_contexts
    
    def generate_answer(self, query: str, context: List[str] = None) -> str:
        """
        Generate answer using RAG.
        
        Args:
            query: Input query
            context: Optional context (if None, will retrieve)
            
        Returns:
            Generated answer
        """
        if context is None:
            context = self.retrieve_relevant_context(query, self.config.get('top_k_ctx', 5))
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        # Generate answer
        try:
            # Tokenize input
            inputs = self.generation_tokenizer.encode(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.generation_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.config.get('max_answer_tokens', 50),
                    num_return_sequences=1,
                    temperature=self.config.get('temperature', 0.7),
                    do_sample=self.config.get('do_sample', True),
                    pad_token_id=self.generation_tokenizer.pad_token_id
                )
            
            # Decode generated text
            generated_text = self.generation_tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.warning(f"Generation failed: {e}, returning dummy answer")
            return f"Based on the context, here's a relevant answer to '{query}'."
    
    def _create_prompt(self, query: str, context: List[str]) -> str:
        """Create prompt for generation."""
        context_text = "\n".join(context[:3])  # Use top 3 contexts
        
        prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""
        
        return prompt
    
    def expand_query_with_rag(self, query: str) -> str:
        """
        Expand query using RAG-generated content.
        
        Args:
            query: Input query
            
        Returns:
            Expanded query with RAG-generated content
        """
        # Generate answer
        generated_answer = self.generate_answer(query)
        
        # Combine original query with generated content
        expanded_query = f"{query} {generated_answer}"
        
        return expanded_query
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding for a query.
        
        Args:
            query: Input query
            
        Returns:
            Query embedding
        """
        if self.embedding_model is None:
            self.initialize_models()
        
        return self.embedding_model.encode([query])[0]
    
    def save_knowledge_base(self, filepath: str) -> None:
        """Save knowledge base and embeddings."""
        data = {
            'knowledge_base': self.knowledge_base,
            'knowledge_embeddings': self.knowledge_embeddings,
            'config': self.config
        }
        save_pickle(data, filepath)
        logger.info(f"Knowledge base saved to {filepath}")
    
    def load_knowledge_base(self, filepath: str) -> None:
        """Load knowledge base and embeddings."""
        data = load_pickle(filepath)
        self.knowledge_base = data['knowledge_base']
        self.knowledge_embeddings = data['knowledge_embeddings']
        self.config = data['config']
        logger.info(f"Knowledge base loaded from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        return {
            'knowledge_base_size': len(self.knowledge_base) if self.knowledge_base else 0,
            'embedding_model_loaded': self.embedding_model is not None,
            'generation_model_loaded': self.generation_model is not None,
            'config': self.config
        }
