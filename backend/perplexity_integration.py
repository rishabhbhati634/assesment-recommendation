
# perplexity_integration.py
import os
import json
import requests
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from collections import Counter
from dotenv import load_dotenv

from balanced_retrieval import (
    adaptive_search, 
    classify_query_intent,
    balance_recommendations
)

# ============================================
# Perplexity API Configuration
# ============================================

load_dotenv()

# Get Perplexity API credentials and configuration from environment
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

import math
import numpy as np

def _is_nan(value):
    try:
        return (isinstance(value, float) and math.isnan(value)) or (isinstance(value, (np.floating,)) and np.isnan(value))
    except Exception:
        return False

def sanitize_for_json(obj):
    """
    Recursively convert numpy types to built-ins and replace NaN/Inf with None
    so json.dumps produces valid JSON for the browser.
    """
    if obj is None:
        return None
    # numpy scalar -> python native
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [sanitize_for_json(x) for x in list(obj)]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = sanitize_for_json(v)
        return out
    # float NaN/Inf -> None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    # other primitives
    return obj

# ============================================
# Perplexity API Functions
# ============================================

def call_perplexity_api(messages: List[Dict], temperature=0.2, max_tokens=2000):
    """
    Call Perplexity API
    
    Args:
        messages: list of message dicts with 'role' and 'content'
        temperature: sampling temperature (0-2)
        max_tokens: maximum tokens to generate
    
    Returns:
        API response dict
    """
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "return_citations": False,  # We don't need web citations for catalog data
        "return_images": False
    }
    
    try:
        resp = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers, timeout=30)
    except Exception as e:
        print(f"❌ Perplexity request failed: {e}")
        return None

    if not (200 <= resp.status_code < 300):
        # Print full server error for debugging — copy/paste this output when asking for help
        print(f"❌ Perplexity API Error: {resp.status_code} {resp.reason} for url: {PERPLEXITY_API_URL}")
        # try to pretty-print JSON body if possible
        try:
            print("Response body (json):", resp.json())
        except Exception:
            print("Response body (text):", resp.text)
        return None

    try:
        return resp.json()
    except Exception as e:
        print(f"❌ Failed to decode Perplexity JSON response: {e}")
        print("Raw response text:", resp.text[:2000])
        return None

def analyze_query_with_perplexity(query_text: str, catalog_summary: str = None) -> Dict:
    """
    Use Perplexity to analyze the query and extract assessment requirements
    
    Args:
        query_text: user query or job description
        catalog_summary: optional summary of available assessment types
    
    Returns:
        dict with analysis results
    """
    
    system_prompt = """You are an expert HR assessment consultant specializing in SHL assessments. 
Your task is to analyze job descriptions or hiring queries and determine what types of assessments are needed.

Available assessment categories:
1. Knowledge & Skills (K) - Technical abilities, programming, job-specific skills
2. Personality & Behavior (P) - Soft skills, teamwork, communication, leadership
3. Cognitive (C) - Problem-solving, analytical thinking, reasoning abilities

Analyze the given text and respond ONLY in JSON format with this structure:
{
    "assessment_needs": {
        "technical_skills": ["skill1", "skill2"],
        "behavioral_traits": ["trait1", "trait2"],
        "cognitive_abilities": ["ability1", "ability2"]
    },
    "priority_categories": ["K", "P", "C"],
    "job_level": "Entry/Mid/Senior/Executive",
    "balance_recommendation": "Description of how to balance test types",
    "key_requirements": ["requirement1", "requirement2"]
}"""

    user_prompt = f"""Analyze this hiring requirement and extract assessment needs:

{query_text}

Provide your analysis in the JSON format specified."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print("🤖 Calling Perplexity API for query analysis...")
    response = call_perplexity_api(messages, temperature=0.2)
    
    if not response or 'choices' not in response:
        print("❌ Failed to get response from Perplexity")
        return None
    
    content = response['choices'][0]['message']['content']
    
    # Parse JSON response
    try:
        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content.strip()
        
        analysis = json.loads(json_str)
        print("✅ Successfully analyzed query with Perplexity")
        return analysis
    except json.JSONDecodeError as e:
        print(f"⚠️ Failed to parse JSON response: {e}")
        print(f"Raw response: {content[:200]}...")
        return None

def get_assessment_recommendations_with_perplexity(
    query_text: str, 
    df: pd.DataFrame,
    top_candidates: List[Dict],
    max_results: int = 10
) -> Dict:
    """
    Use Perplexity to re-rank and select the best assessments from candidates
    
    Args:
        query_text: original query
        df: assessments dataframe
        top_candidates: list of candidate assessments from semantic search
        max_results: number of final recommendations
    
    Returns:
        dict with refined recommendations
    """
    
    # Prepare assessment catalog info for Perplexity
    candidates_info = []
    for i, candidate in enumerate(top_candidates[:20], 1):  # Limit to top 20 for context
        info = f"{i}. {candidate['name']}\n"
        info += f"   Categories: {', '.join(candidate['categories'])}\n"
        info += f"   Test Types: {candidate['test_types']}\n"
        info += f"   Job Level: {candidate.get('job_level', 'N/A')}\n"
        info += f"   Description: {candidate.get('description', 'N/A')[:150]}...\n"
        info += f"   Semantic Score: {candidate['score']:.3f}\n"
        candidates_info.append(info)
    
    catalog_text = "\n".join(candidates_info)
    
    system_prompt = """You are an expert HR assessment consultant. You will be given:
1. A hiring requirement (job description or query)
2. A list of candidate assessments ranked by semantic similarity

Your task is to select the BEST 5-10 assessments that:
- Match the job requirements accurately
- Are balanced across categories when needed (technical, behavioral, cognitive)
- Consider job level appropriateness
- Prioritize high semantic similarity scores

"""

    user_prompt = f"""Hiring Requirement:
{query_text}

Available Candidate Assessments (ranked by semantic similarity):
{catalog_text}

Select the best 5-10 assessments from the list above. Ensure balanced coverage if the query requires multiple skill types.
Also Ensure the response is provided in the below JSON format.
Respond ONLY in JSON format:
{
    "selected_assessments": [
        {
            "assessment_number": 1,
            "assessment_name": "name",
            "reason": "why this is relevant",
            "priority": "high/medium/low"
        }
    ],
    "balance_achieved": "Description of category balance",
    "recommendations": "Any additional advice"
}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print("🤖 Calling Perplexity API for assessment selection...")
    response = call_perplexity_api(messages, temperature=0.1, max_tokens=2000)
    
    if not response or 'choices' not in response:
        print("❌ Failed to get response from Perplexity")
        return None
    
    content = response['choices'][0]['message']['content']
    
    # Parse JSON response
    try:
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content.strip()
        
        selection = json.loads(json_str)
        print("✅ Successfully got recommendations from Perplexity")
        return selection
    except json.JSONDecodeError as e:
        print(f"⚠️ Failed to parse JSON response: {e}")
        print(f"Raw response: {content[:200]}...")
        return None

# ============================================
# Hybrid Approach: Semantic + Perplexity
# ============================================

def hybrid_recommendation_system(
    query_text: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    index,
    min_results: int = 5,
    max_results: int = 10,
    use_perplexity: bool = True
) -> Dict:
    """
    Complete hybrid system combining semantic search + Perplexity reasoning
    
    Pipeline:
    1. Semantic search to get top 20-50 candidates
    2. Perplexity analyzes query to understand requirements
    3. Perplexity re-ranks and selects best assessments
    4. Fallback to pure semantic if Perplexity fails
    
    Args:
        query_text: user query or JD
        df: assessments dataframe
        embeddings: numpy array
        index: FAISS index
        min_results: minimum recommendations
        max_results: maximum recommendations
        use_perplexity: whether to use Perplexity (can disable for testing)
    
    Returns:
        dict with recommendations and metadata
    """
    
    """
    Complete hybrid system with automatic chunking for long JDs
    
    Pipeline:
    1. Adaptive semantic search (auto-chunks if long JD)
    2. Perplexity analyzes query (optional)
    3. Perplexity re-ranks and selects best assessments (optional)
    4. Fallback to balanced semantic if Perplexity fails
    """
    
    print(f"\n{'='*80}")
    print(f"HYBRID RECOMMENDATION SYSTEM")
    print(f"{'='*80}")
    
    # Step 1: Adaptive semantic search (with automatic chunking)
    print("\n📊 Step 1: Adaptive Semantic Search")
    initial_candidates, used_chunking = adaptive_search(
        query_text, index, df, embeddings, top_k=50
    )
    print(f"   ✅ Retrieved {len(initial_candidates)} candidates")
    
    if not use_perplexity or not PERPLEXITY_API_KEY:
        print("\n⚠️ Perplexity disabled - using pure semantic search")
        query_intent = classify_query_intent(query_text)
        balanced = balance_recommendations(initial_candidates, query_intent, max_results)
        
        return {
            'query': query_text,
            'method': 'semantic_only',
            'recommendations': balanced,
            'count': len(balanced),
            'used_chunking': used_chunking,
            'query_length_words': len(query_text.split())
        }
    

    # Step 2: Perplexity query analysis (optional but helpful)
    print("\n🧠 Step 2: Query Analysis with Perplexity")
    query_analysis = analyze_query_with_perplexity(query_text)
    
    # Step 3: Perplexity-guided selection
    print("\n🎯 Step 3: Assessment Selection with Perplexity")
    perplexity_selection = get_assessment_recommendations_with_perplexity(
        query_text, 
        df, 
        initial_candidates,
        max_results
    )
    
    query_intent = classify_query_intent(query_text)
    balanced = balance_recommendations(initial_candidates, query_intent, max_results)
    
    category_dist = Counter()
    for rec in balanced:
        for cat in rec['categories']:
            category_dist[cat] += 1
    
    return {
        'query': query_text,
        'method': 'hybrid_semantic',
        'recommendations': balanced,
        'count': len(balanced),
        'category_distribution': dict(category_dist),
        'used_chunking': used_chunking,
        'query_length_words': len(query_text.split())
    }