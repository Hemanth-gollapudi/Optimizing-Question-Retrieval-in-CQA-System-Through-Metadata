"""
FastAPI Backend for ITRLM+RAG System

This API provides endpoints for:
- Language detection
- Text translation
- Text processing
- Category prediction
- RAG-based answer generation
- End-to-end multilingual query processing
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmr.lang_pipeline import LanguagePipeline
from hmr.text_processing import TextProcessor
from hmr.category_predictor import CategoryPredictor
from hmr.rag_generator import RAGAnswerGen
from data.semeval_loader import SemEvalLoader
from data.yahoo_loader import YahooLoader

# Initialize FastAPI app
app = FastAPI(
    title="ITRLM+RAG API",
    description="Multilingual Question Answering System with RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
lang_pipeline: Optional[LanguagePipeline] = None
text_processor: Optional[TextProcessor] = None
category_predictor: Optional[CategoryPredictor] = None
rag_generator: Optional[RAGAnswerGen] = None


# ==================== Request/Response Models ====================

class TextInput(BaseModel):
    text: str = Field(..., description="Input text to process", min_length=1)


class LanguageDetectionResponse(BaseModel):
    detected_language: str = Field(..., description="ISO 639-1 language code")
    text: str = Field(..., description="Original input text")


class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate", min_length=1)
    source_lang: Optional[str] = Field(None, description="Source language code (auto-detect if not provided)")
    target_lang: str = Field("en", description="Target language code")


class TranslationResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str


class ProcessTextResponse(BaseModel):
    original_text: str
    processed_text: str


class CategoryPredictionResponse(BaseModel):
    category: str
    confidence: float


class GenerateAnswerRequest(BaseModel):
    question: str = Field(..., description="Question to answer", min_length=1)


class GenerateAnswerResponse(BaseModel):
    question: str
    answer: str


class MultilingualQueryRequest(BaseModel):
    question: str = Field(..., description="Question in any language", min_length=1)
    return_english: bool = Field(False, description="Return answer in English instead of original language")


class MultilingualQueryResponse(BaseModel):
    original_question: str
    detected_language: str
    question_in_english: str
    answer_in_english: str
    answer_in_original_language: Optional[str] = None


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize all models and pipelines on server startup (load-only mode)"""
    global lang_pipeline, text_processor, category_predictor, rag_generator
    
    print("🚀 Initializing ITRLM+RAG Backend...")
    print("📌 LOAD-ONLY MODE: Will not train models, only load existing ones\n")
    
    # Initialize language pipeline
    print("📚 Loading language detection and translation models...")
    lang_pipeline = LanguagePipeline()
    print("✅ Language pipeline ready\n")
    
    # Initialize text processor
    print("🔤 Initializing text processor...")
    text_processor = TextProcessor()
    print("✅ Text processor ready\n")
    
    # Initialize category predictor (LOAD ONLY - NO TRAINING)
    print("🏷️  Loading category predictor...")
    category_predictor = CategoryPredictor()
    try:
        category_predictor.load_only()  # Changed from load_or_train()
        print("✅ Category predictor loaded successfully\n")
    except FileNotFoundError as e:
        print(f"⚠️  {str(e)}")
        print("   Category prediction endpoint will not be available")
        print("   Train the model using: notebooks/exploration.ipynb\n")
        category_predictor = None
    except Exception as e:
        print(f"⚠️  Could not load category predictor: {e}")
        print("   Category prediction endpoint will not be available\n")
        category_predictor = None
    
    # Initialize RAG generator
    print("🤖 Loading RAG generator...")
    cfg_rag = {
        "embed_model": "sentence-transformers/all-mpnet-base-v2",
        "gen_model": "google/flan-t5-base",
        "top_k_ctx": 5,
        "max_answer_tokens": 50
    }
    rag_generator = RAGAnswerGen(cfg_rag)
    
    # Try to load existing FAISS index (NO BUILDING/TRAINING)
    try:
        print("📊 Loading FAISS index...")
        rag_generator.load_index()
        print("✅ FAISS index loaded successfully\n")
    except FileNotFoundError as e:
        print(f"⚠️  {str(e)}")
        print("   RAG answer generation endpoints will not be available")
        print("   Build the index using: notebooks/exploration.ipynb\n")
        rag_generator = None
    except Exception as e:
        print(f"⚠️  Could not load FAISS index: {e}")
        print("   RAG answer generation endpoints will not be available\n")
        rag_generator = None
    
    print("=" * 60)
    print("✅ Backend initialization complete!")
    print("=" * 60)
    print("\n📋 Component Status:")
    print(f"  - Language Pipeline: {'✅ Ready' if lang_pipeline else '❌ Failed'}")
    print(f"  - Text Processor: {'✅ Ready' if text_processor else '❌ Failed'}")
    print(f"  - Category Predictor: {'✅ Ready' if category_predictor else '⚠️  Not Available'}")
    print(f"  - RAG Generator: {'✅ Ready' if rag_generator else '⚠️  Not Available'}")
    print("\n📖 API Documentation: http://localhost:8000/docs")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    print("👋 Shutting down ITRLM+RAG Backend...")


# ==================== Health Check ====================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API status"""
    return {
        "status": "online",
        "message": "ITRLM+RAG API is running",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "lang_pipeline": lang_pipeline is not None,
            "text_processor": text_processor is not None,
            "category_predictor": category_predictor is not None,
            "rag_generator": rag_generator is not None,
        }
    }


# ==================== Language Detection ====================

@app.post("/detect-language", response_model=LanguageDetectionResponse, tags=["Language"])
async def detect_language(input_data: TextInput):
    """
    Detect the language of input text
    
    Returns ISO 639-1 language code (e.g., 'en', 'es', 'te', 'hi')
    """
    if not lang_pipeline:
        raise HTTPException(status_code=503, detail="Language pipeline not initialized")
    
    try:
        detected_lang = lang_pipeline.detect_language(input_data.text)
        return LanguageDetectionResponse(
            detected_language=detected_lang,
            text=input_data.text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")


# ==================== Translation ====================

@app.post("/translate", response_model=TranslationResponse, tags=["Language"])
async def translate_text(request: TranslationRequest):
    """
    Translate text from source language to target language
    
    If source_lang is not provided, it will be auto-detected
    """
    if not lang_pipeline:
        raise HTTPException(status_code=503, detail="Language pipeline not initialized")
    
    try:
        # Detect source language if not provided
        source_lang = request.source_lang
        if not source_lang:
            source_lang = lang_pipeline.detect_language(request.text)
        
        # Translate based on target language
        if request.target_lang == "en":
            translated = lang_pipeline.translate_to_english(request.text, source_lang)
        else:
            # First translate to English, then to target language
            if source_lang != "en":
                english_text = lang_pipeline.translate_to_english(request.text, source_lang)
            else:
                english_text = request.text
            translated = lang_pipeline.translate_from_english(english_text, request.target_lang)
        
        return TranslationResponse(
            translated_text=translated,
            source_lang=source_lang,
            target_lang=request.target_lang
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


# ==================== Text Processing ====================

@app.post("/process-text", response_model=ProcessTextResponse, tags=["Text Processing"])
async def process_text(input_data: TextInput):
    """
    Clean and process text (lowercasing, removing special characters, etc.)
    """
    if not text_processor:
        raise HTTPException(status_code=503, detail="Text processor not initialized")
    
    try:
        processed = text_processor.clean(input_data.text)
        return ProcessTextResponse(
            original_text=input_data.text,
            processed_text=processed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")


# ==================== Category Prediction ====================

@app.post("/predict-category", response_model=CategoryPredictionResponse, tags=["Classification"])
async def predict_category(input_data: TextInput):
    """
    Predict the category/topic of a question
    
    Returns category name and confidence score
    
    **Requires trained model:** This endpoint needs a pre-trained category predictor.
    Train it by running: `notebooks/exploration.ipynb`
    """
    if not category_predictor:
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "Category predictor not available",
                "reason": "Model checkpoint not found",
                "solution": "Train the model using notebooks/exploration.ipynb",
                "required_files": [
                    "outputs/checkpoints/bert_category_predictor/model.pt",
                    "outputs/checkpoints/bert_category_predictor/label_map.json"
                ]
            }
        )
    
    try:
        category, confidence = category_predictor.predict(input_data.text)
        return CategoryPredictionResponse(
            category=category,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Category prediction failed: {str(e)}")


# ==================== RAG Answer Generation ====================

@app.post("/generate-answer", response_model=GenerateAnswerResponse, tags=["RAG"])
async def generate_answer(request: GenerateAnswerRequest):
    """
    Generate an answer to a question using RAG (Retrieval-Augmented Generation)
    
    The question should be in English
    
    **Requires trained model:** This endpoint needs a pre-built FAISS index.
    Build it by running: `notebooks/exploration.ipynb`
    """
    if not rag_generator:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "RAG generator not available",
                "reason": "FAISS index not found",
                "solution": "Build the FAISS index using notebooks/exploration.ipynb",
                "required_files": [
                    "outputs/faiss_index.index",
                    "outputs/faiss_index_texts.json"
                ]
            }
        )
    
    try:
        answer = rag_generator.generate(request.question)
        return GenerateAnswerResponse(
            question=request.question,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")


# ==================== Multilingual Query Pipeline ====================

@app.post("/multilingual-query", response_model=MultilingualQueryResponse, tags=["Multilingual"])
async def multilingual_query(request: MultilingualQueryRequest):
    """
    Complete multilingual query pipeline:
    1. Detect language of input question
    2. Translate question to English
    3. Generate answer using RAG
    4. Translate answer back to original language
    
    This endpoint handles questions in any language and returns answers in the same language
    
    **Requires trained model:** This endpoint needs a pre-built FAISS index.
    Build it by running: `notebooks/exploration.ipynb`
    """
    if not lang_pipeline:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Language pipeline not available",
                "reason": "Translation service not initialized"
            }
        )
    
    if not rag_generator:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "RAG generator not available",
                "reason": "FAISS index not found",
                "solution": "Build the FAISS index using notebooks/exploration.ipynb",
                "required_files": [
                    "outputs/faiss_index.index",
                    "outputs/faiss_index_texts.json"
                ]
            }
        )
    
    try:
        # Step 1: Detect language
        detected_lang = lang_pipeline.detect_language(request.question)
        
        # Step 2: Translate to English if needed
        if detected_lang == "en":
            question_en = request.question
        else:
            question_en = lang_pipeline.translate_to_english(request.question, detected_lang)
        
        # Step 3: Generate answer in English
        answer_en = rag_generator.generate(question_en)
        
        # Step 4: Translate answer back to original language if needed
        if request.return_english or detected_lang == "en":
            answer_original = None
        else:
            answer_original = lang_pipeline.translate_from_english(answer_en, detected_lang)
        
        return MultilingualQueryResponse(
            original_question=request.question,
            detected_language=detected_lang,
            question_in_english=question_en,
            answer_in_english=answer_en,
            answer_in_original_language=answer_original if not request.return_english else answer_en
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multilingual query processing failed: {str(e)}")


# ==================== Additional Info Endpoints ====================

@app.get("/supported-languages", tags=["Info"])
async def get_supported_languages():
    """Get list of commonly supported languages"""
    return {
        "languages": {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh-cn": "Chinese (Simplified)",
            "zh-tw": "Chinese (Traditional)",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi",
            "te": "Telugu",
            "ta": "Tamil",
            "bn": "Bengali",
            "mr": "Marathi",
        },
        "note": "Many more languages supported via Google Translate"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

