"""
Yahoo Answers Dataset Loader for PMI Dictionary Building
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class YahooLoader:
    """Loader for Yahoo Answers dataset."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.yahoo_dir = self.data_dir / "yahoo_answers"
        
    def load_data(self, split: str = "train", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load Yahoo Answers data.
        
        Args:
            split: Dataset split ('train', 'test')
            max_samples: Maximum number of samples to load
            
        Returns:
            List of question-answer pairs with categories
        """
        if split == "train":
            return self._load_train_data(max_samples)
        elif split == "test":
            return self._load_test_data(max_samples)
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def _load_train_data(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load training data from Yahoo Answers."""
        train_file = self.yahoo_dir / "yahoo_answers_train.json"
        
        if not train_file.exists():
            logger.warning(f"Train file not found: {train_file}")
            return self._create_dummy_train_data(max_samples)
        
        data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                try:
                    item = json.loads(line.strip())
                    data.append({
                        'question_id': item.get('question_id', f'q_{i}'),
                        'question': item.get('question', ''),
                        'answer': item.get('answer', ''),
                        'category': item.get('category', 'unknown'),
                        'relevance': 1  # Yahoo Answers are assumed relevant
                    })
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(data)} training samples from Yahoo Answers")
        return data
    
    def _load_test_data(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load test data from Yahoo Answers."""
        test_file = self.yahoo_dir / "yahoo_answers_test.json"
        
        if not test_file.exists():
            logger.warning(f"Test file not found: {test_file}")
            return self._create_dummy_test_data(max_samples)
        
        return self._load_train_data(max_samples)  # Same format
    
    def _create_dummy_train_data(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Create dummy training data for demonstration."""
        dummy_data = [
            {
                'question_id': 'yahoo_1',
                'question': 'How do I cook pasta?',
                'answer': 'Boil water, add salt, add pasta, cook for 8-12 minutes, drain and serve.',
                'category': 'food',
                'relevance': 1
            },
            {
                'question_id': 'yahoo_2',
                'question': 'What is the best programming language?',
                'answer': 'Python is great for beginners, JavaScript for web development, and C++ for performance.',
                'category': 'technology',
                'relevance': 1
            },
            {
                'question_id': 'yahoo_3',
                'question': 'How to lose weight?',
                'answer': 'Eat a balanced diet, exercise regularly, and maintain a calorie deficit.',
                'category': 'health',
                'relevance': 1
            },
            {
                'question_id': 'yahoo_4',
                'question': 'Best travel destinations in Europe?',
                'answer': 'Paris, Rome, Barcelona, Amsterdam, and Prague are popular European destinations.',
                'category': 'travel',
                'relevance': 1
            },
            {
                'question_id': 'yahoo_5',
                'question': 'How to invest in stocks?',
                'answer': 'Research companies, diversify your portfolio, and consider long-term investments.',
                'category': 'finance',
                'relevance': 1
            }
        ]
        
        if max_samples:
            dummy_data = dummy_data[:max_samples]
        
        logger.info(f"Created {len(dummy_data)} dummy training samples")
        return dummy_data
    
    def _create_dummy_test_data(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Create dummy test data."""
        return self._create_dummy_train_data(max_samples)
    
    def get_categories(self) -> List[str]:
        """Get list of available categories."""
        data = self.load_data()
        categories = list(set(item['category'] for item in data))
        return sorted(categories)
    
    def download_data(self) -> None:
        """Download Yahoo Answers data if not present."""
        # This would contain the actual download logic
        logger.info("Yahoo Answers data download not implemented - using dummy data")
        pass
