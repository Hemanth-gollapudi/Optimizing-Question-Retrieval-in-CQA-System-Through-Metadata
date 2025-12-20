"""
Category Predictor using BERT for ITRLM
Fine-tuned BERT classifier for question categorization
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import numpy as np

from .text_processing import TextProcessor
from .utils import save_pickle, load_pickle

logger = logging.getLogger(__name__)


class CategoryPredictor:
    """BERT-based category predictor for questions."""
    
    def __init__(self, config: Dict[str, Any], text_processor: TextProcessor):
        self.config = config
        self.text_processor = text_processor
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.label_encoder = {}
        self.num_labels = 0
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, training_data: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        Prepare training data for BERT.
        
        Args:
            training_data: List of training examples
            
        Returns:
            Tuple of (texts, labels)
        """
        texts = []
        labels = []
        
        # Get all unique categories
        categories = list(set(item.get('category', 'unknown') for item in training_data))
        self.label_encoder = {cat: idx for idx, cat in enumerate(categories)}
        self.num_labels = len(categories)
        
        logger.info(f"Found {self.num_labels} categories: {categories}")
        
        for item in training_data:
            question = item.get('question', '')
            category = item.get('category', 'unknown')
            
            if question and category in self.label_encoder:
                texts.append(question)
                labels.append(self.label_encoder[category])
        
        return texts, labels
    
    def load_or_train(self, training_data: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Load existing model or train new one.
        
        Args:
            training_data: Training data for new model
        """
        model_path = Path("outputs/checkpoints/category_predictor")
        
        if model_path.exists() and (model_path / "pytorch_model.bin").exists():
            logger.info("Loading existing category predictor...")
            self.load_model(str(model_path))
        else:
            if training_data is None:
                logger.warning("No training data provided and no existing model found")
                self._create_dummy_model()
            else:
                logger.info("Training new category predictor...")
                self.train(training_data)
    
    def _create_dummy_model(self) -> None:
        """Create a dummy model for demonstration."""
        logger.info("Creating dummy category predictor...")
        
        # Create dummy categories
        self.label_encoder = {
            'technology': 0,
            'science': 1,
            'health': 2,
            'food': 3,
            'travel': 4,
            'finance': 5,
            'unknown': 6
        }
        self.num_labels = len(self.label_encoder)
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        
        logger.info("Dummy category predictor created")
    
    def train(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Train the category predictor.
        
        Args:
            training_data: List of training examples
        """
        logger.info("Training category predictor...")
        
        # Prepare data
        texts, labels = self.prepare_data(training_data)
        
        if len(texts) < 10:
            logger.warning("Insufficient training data, creating dummy model")
            self._create_dummy_model()
            return
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=self.num_labels
        )
        
        # Create dataset
        dataset = self._create_dataset(texts, labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='outputs/checkpoints/category_predictor',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='outputs/logs',
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="no",
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained('outputs/checkpoints/category_predictor')
        
        logger.info("Category predictor training completed")
    
    def _create_dataset(self, texts: List[str], labels: List[int]) -> 'Dataset':
        """Create PyTorch dataset for training."""
        from torch.utils.data import Dataset
        
        class CategoryDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=128):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]
                
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
        
        return CategoryDataset(texts, labels, self.tokenizer)
    
    def predict_category(self, question: str) -> Tuple[str, float]:
        """
        Predict category for a question.
        
        Args:
            question: Input question
            
        Returns:
            Tuple of (predicted_category, confidence_score)
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("Model not loaded, returning dummy prediction")
            return 'unknown', 0.5
        
        # Tokenize
        inputs = self.tokenizer(
            question,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get prediction
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class_id].item()
            
            # Convert back to category name
            predicted_category = None
            for cat, idx in self.label_encoder.items():
                if idx == predicted_class_id:
                    predicted_category = cat
                    break
            
            if predicted_category is None:
                predicted_category = 'unknown'
        
        return predicted_category, confidence
    
    def predict_batch(self, questions: List[str]) -> List[Tuple[str, float]]:
        """
        Predict categories for a batch of questions.
        
        Args:
            questions: List of questions
            
        Returns:
            List of (category, confidence) tuples
        """
        predictions = []
        for question in questions:
            pred = self.predict_category(question)
            predictions.append(pred)
        return predictions
    
    def load_model(self, model_path: str) -> None:
        """Load model from checkpoint."""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            
            # Load label encoder
            label_encoder_path = Path(model_path) / "label_encoder.pkl"
            if label_encoder_path.exists():
                self.label_encoder = load_pickle(str(label_encoder_path))
                self.num_labels = len(self.label_encoder)
            
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._create_dummy_model()
    
    def save_model(self, model_path: str) -> None:
        """Save model and label encoder."""
        # Save label encoder
        label_encoder_path = Path(model_path) / "label_encoder.pkl"
        save_pickle(self.label_encoder, str(label_encoder_path))
        
        logger.info(f"Model saved to {model_path}")
    
    def get_categories(self) -> List[str]:
        """Get list of available categories."""
        return list(self.label_encoder.keys())
