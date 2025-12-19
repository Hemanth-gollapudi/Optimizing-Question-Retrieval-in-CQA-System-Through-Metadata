#!/usr/bin/env python3
"""
Comprehensive test suite for CQA Chat System
Tests various questions and calculates evaluation metrics
"""
import requests
import json
import time
import sys
import os
from typing import List, Dict, Tuple
from collections import defaultdict

# Import project configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

API_ENDPOINT = config.API_ENDPOINT

# Test questions covering different categories and question types
TEST_QUESTIONS = [
    # Technology - WiFi
    ("How to fix WiFi disconnecting?", "Technology", "WiFi"),
    ("My internet keeps dropping", "Technology", "WiFi"),
    ("WiFi connection problems", "Technology", "WiFi"),
    ("Laptop WiFi keeps disconnecting", "Technology", "WiFi"),
    
    # Technology - Bluetooth
    ("Bluetooth not detecting devices", "Technology", "Bluetooth"),
    ("Bluetooth pairing issues", "Technology", "Bluetooth"),
    ("Can't find Bluetooth devices", "Technology", "Bluetooth"),
    
    # Technology - Storage
    ("Storage is full", "Technology", "Storage"),
    ("Phone storage full", "Technology", "Storage"),
    ("How to free up storage?", "Technology", "Storage"),
    
    # Technology - App crashing
    ("App keeps crashing", "Technology", "App"),
    ("Application closes suddenly", "Technology", "App"),
    ("App not working", "Technology", "App"),
    
    # Technology - Phone overheating
    ("Phone gets hot while charging", "Technology", "Phone"),
    ("Device overheating", "Technology", "Phone"),
    
    # Finance - UPI
    ("UPI transaction failed but money deducted", "Finance", "UPI"),
    ("UPI payment failed", "Finance", "UPI"),
    ("Money debited but UPI failed", "Finance", "UPI"),
    
    # Finance - Budget
    ("How to track spending?", "Finance", "Budget"),
    ("Budget planning tips", "Finance", "Budget"),
    ("Want to save more money", "Finance", "Budget"),
    
    # Finance - Credit score
    ("Credit score dropped", "Finance", "Credit"),
    ("How to improve credit score?", "Finance", "Credit"),
    
    # Finance - Investing
    ("How to start investing?", "Finance", "Investing"),
    ("Beginner investment advice", "Finance", "Investing"),
    
    # Health - Headache
    ("Wake up with headache", "Health", "Headache"),
    ("Morning headaches", "Health", "Headache"),
    
    # Health - Stomach pain
    ("Stomach pain after meals", "Health", "Stomach"),
    ("Bloating and pain", "Health", "Stomach"),
    
    # Health - Cold
    ("Sore throat and cough", "Health", "Cold"),
    ("Cold symptoms", "Health", "Cold"),
    
    # Education - Exam anxiety
    ("Nervous during exams", "Education", "Exam"),
    ("Exam anxiety", "Education", "Exam"),
    ("Forget answers in exam", "Education", "Exam"),
    
    # Education - Concentration
    ("Can't concentrate while studying", "Education", "Concentration"),
    ("Get distracted easily", "Education", "Concentration"),
    
    # Education - Programming
    ("Confused about loops and functions", "Education", "Programming"),
    ("Programming basics help", "Education", "Programming"),
    
    # Out-of-dataset questions (should use LLM or fail gracefully)
    ("What is quantum computing?", "General", "Science"),
    ("Explain machine learning", "General", "Science"),
    ("Who won the World Cup 2022?", "General", "Sports"),
    ("What is the capital of Mars?", "General", "Geography"),
]

def test_question(question: str, category: str, topic: str) -> Dict:
    """Test a single question and return results"""
    try:
        start_time = time.time()
        response = requests.post(
            API_ENDPOINT,
            json={"query": question},
            timeout=180  # Increased to 3 minutes to allow time for LLM responses
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "question": question,
                "category": category,
                "topic": topic,
                "success": True,
                "source": data.get("source", "unknown"),
                "similarity_score": data.get("similarity_score"),
                "response_time": response_time,
                "has_response": bool(data.get("response")),
                "response_length": len(data.get("response", "")),
                "error": None
            }
        else:
            error_data = response.json() if response.content else {}
            return {
                "question": question,
                "category": category,
                "topic": topic,
                "success": False,
                "source": "error",
                "similarity_score": None,
                "response_time": response_time,
                "has_response": False,
                "response_length": 0,
                "error": error_data.get("detail", f"HTTP {response.status_code}")
            }
    except Exception as e:
        return {
            "question": question,
            "category": category,
            "topic": topic,
            "success": False,
            "source": "error",
            "similarity_score": None,
            "response_time": 0,
            "has_response": False,
            "response_length": 0,
            "error": str(e)
        }

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics"""
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    dataset_hits = sum(1 for r in results if r["source"] == "dataset")
    llm_fallback = sum(1 for r in results if r["source"] == "llm")
    errors = sum(1 for r in results if not r["success"])
    
    # Similarity scores (only for dataset hits)
    similarity_scores = [r["similarity_score"] for r in results 
                        if r["similarity_score"] is not None]
    
    # Response times
    response_times = [r["response_time"] for r in results if r["response_time"] > 0]
    
    # By category
    by_category = defaultdict(lambda: {"total": 0, "success": 0, "dataset": 0, "llm": 0, "errors": 0})
    for r in results:
        cat = r["category"]
        by_category[cat]["total"] += 1
        if r["success"]:
            by_category[cat]["success"] += 1
        if r["source"] == "dataset":
            by_category[cat]["dataset"] += 1
        elif r["source"] == "llm":
            by_category[cat]["llm"] += 1
        else:
            by_category[cat]["errors"] += 1
    
    # By topic
    by_topic = defaultdict(lambda: {"total": 0, "success": 0, "avg_similarity": []})
    for r in results:
        top = r["topic"]
        by_topic[top]["total"] += 1
        if r["success"]:
            by_topic[top]["success"] += 1
        if r["similarity_score"] is not None:
            by_topic[top]["avg_similarity"].append(r["similarity_score"])
    
    for top in by_topic:
        if by_topic[top]["avg_similarity"]:
            by_topic[top]["avg_similarity"] = sum(by_topic[top]["avg_similarity"]) / len(by_topic[top]["avg_similarity"])
        else:
            by_topic[top]["avg_similarity"] = None
    
    metrics = {
        "total_questions": total,
        "success_rate": successful / total if total > 0 else 0,
        "dataset_hit_rate": dataset_hits / total if total > 0 else 0,
        "llm_fallback_rate": llm_fallback / total if total > 0 else 0,
        "error_rate": errors / total if total > 0 else 0,
        "similarity_stats": {
            "mean": sum(similarity_scores) / len(similarity_scores) if similarity_scores else None,
            "min": min(similarity_scores) if similarity_scores else None,
            "max": max(similarity_scores) if similarity_scores else None,
            "above_threshold": sum(1 for s in similarity_scores if s >= 0.55) if similarity_scores else 0,
            "count": len(similarity_scores)
        },
        "response_time_stats": {
            "mean": sum(response_times) / len(response_times) if response_times else None,
            "min": min(response_times) if response_times else None,
            "max": max(response_times) if response_times else None,
            "count": len(response_times)
        },
        "by_category": dict(by_category),
        "by_topic": dict(by_topic)
    }
    
    return metrics

def print_report(results: List[Dict], metrics: Dict):
    """Print a formatted evaluation report"""
    print("=" * 80)
    print("CQA CHAT SYSTEM - COMPREHENSIVE EVALUATION REPORT")
    print("=" * 80)
    print()
    
    print("📊 OVERALL METRICS")
    print("-" * 80)
    print(f"Total Questions Tested:     {metrics['total_questions']}")
    print(f"Success Rate:                {metrics['success_rate']:.2%}")
    print(f"Dataset Hit Rate:            {metrics['dataset_hit_rate']:.2%}")
    print(f"LLM Fallback Rate:           {metrics['llm_fallback_rate']:.2%}")
    print(f"Error Rate:                  {metrics['error_rate']:.2%}")
    print()
    
    if metrics['similarity_stats']['count'] > 0:
        print("🎯 SIMILARITY SCORE STATISTICS")
        print("-" * 80)
        sim_stats = metrics['similarity_stats']
        print(f"Mean Similarity:            {sim_stats['mean']:.4f}")
        print(f"Min Similarity:              {sim_stats['min']:.4f}")
        print(f"Max Similarity:              {sim_stats['max']:.4f}")
        print(f"Above Threshold (≥0.55):     {sim_stats['above_threshold']}/{sim_stats['count']} ({sim_stats['above_threshold']/sim_stats['count']:.2%})")
        print()
    
    if metrics['response_time_stats']['count'] > 0:
        print("⏱️  RESPONSE TIME STATISTICS")
        print("-" * 80)
        rt_stats = metrics['response_time_stats']
        print(f"Mean Response Time:          {rt_stats['mean']:.3f}s")
        print(f"Min Response Time:            {rt_stats['min']:.3f}s")
        print(f"Max Response Time:           {rt_stats['max']:.3f}s")
        print()
    
    print("📁 PERFORMANCE BY CATEGORY")
    print("-" * 80)
    for cat, stats in sorted(metrics['by_category'].items()):
        success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        dataset_rate = stats['dataset'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{cat:20s} | Total: {stats['total']:2d} | Success: {stats['success']:2d} ({success_rate:.1%}) | Dataset: {stats['dataset']:2d} ({dataset_rate:.1%}) | Errors: {stats['errors']:2d}")
    print()
    
    print("🔍 TOP TOPICS - AVERAGE SIMILARITY SCORES")
    print("-" * 80)
    topic_sims = [(top, stats['avg_similarity']) for top, stats in metrics['by_topic'].items() 
                  if stats['avg_similarity'] is not None]
    topic_sims.sort(key=lambda x: x[1] or 0, reverse=True)
    for topic, avg_sim in topic_sims[:15]:
        stats = metrics['by_topic'][topic]
        print(f"{topic:25s} | Avg Similarity: {avg_sim:.4f} | Tests: {stats['total']:2d} | Success: {stats['success']:2d}")
    print()
    
    print("❌ ERRORS AND FAILURES")
    print("-" * 80)
    errors = [r for r in results if not r['success'] or r.get('error')]
    if errors:
        for err in errors[:10]:  # Show first 10 errors
            print(f"Question: {err['question'][:60]}")
            print(f"  Error: {err.get('error', 'Unknown error')}")
            print()
    else:
        print("No errors found! ✅")
    print()
    
    print("=" * 80)

def main():
    print("Starting comprehensive system evaluation...")
    print(f"Testing {len(TEST_QUESTIONS)} questions...")
    print()
    
    results = []
    for i, (question, category, topic) in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] Testing: {question[:60]}...", end=" ")
        result = test_question(question, category, topic)
        results.append(result)
        
        if result["success"]:
            source = result["source"]
            sim = result.get("similarity_score")
            if sim:
                print(f"✅ {source} (similarity: {sim:.3f})")
            else:
                print(f"✅ {source}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown')}")
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.1)
    
    print()
    print("Calculating metrics...")
    metrics = calculate_metrics(results)
    
    print_report(results, metrics)
    
    # Save detailed results to JSON
    output_file = "test_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "results": results,
            "metrics": metrics,
            "test_questions": TEST_QUESTIONS
        }, f, indent=2)
    print(f"\n📄 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()

