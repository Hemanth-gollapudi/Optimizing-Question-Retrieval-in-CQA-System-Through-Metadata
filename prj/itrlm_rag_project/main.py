"""
ITRLM+RAG Implementation - Main Entry Point
Improved Translation-based Language Model with RAG enhancements
"""

import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path

from src.utils import set_seed, setup_logging
from src.text_processing import TextProcessor
from src.pmi_dictionary import PMIDictionary
from src.category_predictor import CategoryPredictor
from src.rag_generator import RAGAnswerGen
from src.itrlm_scorer import ITRLMScorer
from src.hybrid_ranker import HybridRanker
from src.evaluation import Evaluator
from data.semeval_loader import SemEvalLoader
from data.yahoo_loader import YahooLoader


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    """Main pipeline for ITRLM+RAG training and evaluation."""
    
    # Setup
    set_seed(cfg.random_seed)
    setup_logging(cfg.paths.logs)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting {cfg.project_name} pipeline")
    logger.info(f"Configuration: {cfg}")
    
    # Create output directories
    for path in [cfg.paths.checkpoints, cfg.paths.logs, cfg.paths.results]:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    try:
        print("[1] Initializing text processor...")
        text_processor = TextProcessor()
        
        print("[2] Loading datasets...")
        semeval_loader = SemEvalLoader(cfg.paths.data_dir)
        yahoo_loader = YahooLoader(cfg.paths.data_dir)
        
        print("[3] Building/Loading PMI dictionaries...")
        pmi = PMIDictionary(cfg.model, text_processor)
        pmi.build_all(yahoo_loader.load_data())
        
        print("[4] Loading category predictor...")
        cat_pred = CategoryPredictor(cfg.model, text_processor)
        cat_pred.load_or_train(yahoo_loader.load_data())
        
        print("[5] Initializing RAG generator...")
        rag = RAGAnswerGen(cfg.rag, text_processor)
        
        print("[6] Initializing ITRLM scorer...")
        itrlm = ITRLMScorer(pmi, cfg.model, text_processor)
        
        print("[7] Setting up hybrid ranker...")
        ranker = HybridRanker(itrlm, cat_pred, rag, cfg.model)
        
        print("[8] Loading test data...")
        test_data = semeval_loader.load_data()
        
        print("[9] Evaluating on SemEval test set...")
        evaluator = Evaluator(cfg.model.evaluation)
        results = evaluator.run(ranker, test_data)
        
        print("✅ Results:", results)
        logger.info(f"Final results: {results}")
        
        # Save results
        results_path = Path(cfg.paths.results) / "evaluation_results.json"
        evaluator.save_results(results, results_path)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
