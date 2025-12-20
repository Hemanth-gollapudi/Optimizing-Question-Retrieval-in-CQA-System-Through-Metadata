#!/usr/bin/env python3
"""
Test script for ITRLM+RAG API endpoints

Usage:
    python test_api.py
"""

import requests
import time
import json
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"


def print_response(endpoint: str, response: requests.Response):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"Endpoint: {endpoint}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print(f"{'='*60}")


def test_health_check():
    """Test health check endpoint"""
    print("\n🏥 Testing Health Check...")
    response = requests.get(f"{API_BASE_URL}/health")
    print_response("/health", response)
    return response.status_code == 200


def test_language_detection():
    """Test language detection endpoint"""
    print("\n🌍 Testing Language Detection...")
    
    test_cases = [
        {"text": "Hello, how are you?"},
        {"text": "¿Dónde puedo comprar billetes de avión baratos?"},
        {"text": "भारत की राजधानी क्या है?"},
        {"text": "తెలుగు టెస్ట్"},
    ]
    
    for test_case in test_cases:
        response = requests.post(
            f"{API_BASE_URL}/detect-language",
            json=test_case
        )
        print_response("/detect-language", response)


def test_translation():
    """Test translation endpoint"""
    print("\n🔄 Testing Translation...")
    
    test_cases = [
        {
            "text": "¿Dónde puedo comprar billetes de avión baratos?",
            "target_lang": "en"
        },
        {
            "text": "Hello, how are you?",
            "target_lang": "es"
        },
        {
            "text": "భారతదేశ రాజధాని నగరం ఏది?",
            "target_lang": "en"
        },
    ]
    
    for test_case in test_cases:
        response = requests.post(
            f"{API_BASE_URL}/translate",
            json=test_case
        )
        print_response("/translate", response)


def test_text_processing():
    """Test text processing endpoint"""
    print("\n🧹 Testing Text Processing...")
    
    test_cases = [
        {"text": "Where can I buy CHEAP airline tickets???"},
        {"text": "  Hello    World!!! @#$%  "},
        {"text": "THIS IS ALL CAPS WITH NUMBERS 123456"},
    ]
    
    for test_case in test_cases:
        response = requests.post(
            f"{API_BASE_URL}/process-text",
            json=test_case
        )
        print_response("/process-text", response)


def test_category_prediction():
    """Test category prediction endpoint"""
    print("\n🏷️ Testing Category Prediction...")
    
    test_cases = [
        {"text": "How do I invest in the stock market?"},
        {"text": "What's the best way to learn Python programming?"},
        {"text": "Where can I find cheap airline tickets?"},
    ]
    
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict-category",
                json=test_case
            )
            print_response("/predict-category", response)
        except Exception as e:
            print(f"⚠️ Category prediction test failed: {e}")


def test_generate_answer():
    """Test RAG answer generation endpoint"""
    print("\n🤖 Testing RAG Answer Generation...")
    
    test_cases = [
        {"question": "Where can I buy cheap airline tickets?"},
        {"question": "What is the best way to learn programming?"},
    ]
    
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate-answer",
                json=test_case
            )
            print_response("/generate-answer", response)
        except Exception as e:
            print(f"⚠️ Answer generation test failed: {e}")


def test_multilingual_query():
    """Test complete multilingual query pipeline"""
    print("\n🌐 Testing Multilingual Query Pipeline...")
    
    test_cases = [
        {
            "question": "¿Dónde puedo comprar billetes de avión baratos?",
            "return_english": False
        },
        {
            "question": "भारत की राजधानी क्या है?",
            "return_english": False
        },
        {
            "question": "Where can I find information about Python?",
            "return_english": True
        },
    ]
    
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{API_BASE_URL}/multilingual-query",
                json=test_case
            )
            print_response("/multilingual-query", response)
        except Exception as e:
            print(f"⚠️ Multilingual query test failed: {e}")


def test_supported_languages():
    """Test supported languages endpoint"""
    print("\n🗣️ Testing Supported Languages...")
    response = requests.get(f"{API_BASE_URL}/supported-languages")
    print_response("/supported-languages", response)


def main():
    """Run all tests"""
    print("🚀 Starting API Tests...")
    print(f"API Base URL: {API_BASE_URL}")
    
    # Wait for server to be ready
    print("\n⏳ Waiting for server to be ready...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                print(f"Retrying... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                print("❌ Could not connect to server. Please start the server first:")
                print("   python run_server.py")
                return
    
    # Run tests
    try:
        test_health_check()
        test_supported_languages()
        test_language_detection()
        test_translation()
        test_text_processing()
        test_category_prediction()
        test_generate_answer()
        test_multilingual_query()
        
        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

