import os
import re
import json
import requests
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import ast

# ============================================
# Configuration
# ============================================

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = "llama-3.1-sonar-large-128k-online"

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# ============================================
# Entity Extraction
# ============================================

def extract_skills(query_text: str) -> List[str]:
    """
    Extract technical skills from query using pattern matching
    
    Examples:
    - "Java developers" → ["Java"]
    - "Python, SQL and JavaScript" → ["Python", "SQL", "JavaScript"]
    """
    skills_keywords = [
        'java', 'python', 'javascript', 'sql', 'c++', 'c#', '.net', 'ruby', 'php',
        'golang', 'rust', 'kotlin', 'swift', 'typescript', 'r programming',
        'html', 'css', 'react', 'angular', 'vue', 'node', 'django', 'flask',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
        'machine learning', 'data science', 'artificial intelligence', 'ai',
        'tableau', 'power bi', 'excel', 'sap', 'salesforce'
    ]
    
    query_lower = query_text.lower()
    found_skills = []
    
    for skill in skills_keywords:
        # Match whole words
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, query_lower):
            found_skills.append(skill.title())
    
    return found_skills

def extract_time_duration(query_text: str) -> Optional[int]:
    """
    Extract time duration in minutes from query
    
    Examples:
    - "40 minutes" → 40
    - "1 hour" → 60
    - "30 mins" → 30
    - "1.5 hours" → 90
    """
    query_lower = query_text.lower()
    
    # Pattern 1: "X minutes" or "X mins"
    minutes_pattern = r'(\d+)\s*(minute|minutes|mins|min)\b'
    minutes_match = re.search(minutes_pattern, query_lower)
    if minutes_match:
        return int(minutes_match.group(1))
    
    # Pattern 2: "X hours"
    hours_pattern = r'(\d+\.?\d*)\s*(hour|hours|hrs|hr)\b'
    hours_match = re.search(hours_pattern, query_lower)
    if hours_match:
        hours = float(hours_match.group(1))
        return int(hours * 60)
    
    # Pattern 3: "X-Y minutes" (take the max)
    range_pattern = r'(\d+)\s*-\s*(\d+)\s*(minute|minutes|mins)'
    range_match = re.search(range_pattern, query_lower)
    if range_match:
        return int(range_match.group(2))  # Return max duration
    
    return None

def extract_job_level(query_text: str) -> Optional[str]:
    """
    Extract job level from query
    
    Examples:
    - "senior developer" → "Senior"
    - "entry-level" → "Entry"
    - "mid-level" → "Mid"
    """
    query_lower = query_text.lower()
    
    level_mapping = {
        'entry': 'Entry',
        'junior': 'Entry',
        'mid': 'Mid',
        'intermediate': 'Mid',
        'senior': 'Senior',
        'lead': 'Senior',
        'principal': 'Senior',
        'executive': 'Executive',
        'manager': 'Executive',
        'director': 'Executive'
    }
    
    for keyword, level in level_mapping.items():
        if re.search(r'\b' + keyword + r'\b', query_lower):
            return level
    
    return None

def extract_entities(query_text: str) -> Dict:
    """
    Extract all entities from query
    
    Returns:
        dict with 'skills', 'duration_minutes', 'job_level'
    """
    return {
        'skills': extract_skills(query_text),
        'duration_minutes': extract_time_duration(query_text),
        'job_level': extract_job_level(query_text),
        'query_length': len(query_text.split())
    }

# ============================================
# Helper Functions
# ============================================

def parse_test_type(test_type_value):
    """Parse test_type column"""
    if pd.isna(test_type_value):
        return []
    
    test_type_str = str(test_type_value).strip()
    
    if test_type_str.startswith('[') and test_type_str.endswith(']'):
        try:
            return ast.literal_eval(test_type_str)
        except:
            pass
    
    if ',' in test_type_str:
        return [t.strip() for t in test_type_str.split(',')]
    
    return [test_type_str]

def categorize_test_type(test_types):
    """Categorize test types into K/P/C"""
    categories = set()
    
    for tt in test_types:
        tt_lower = str(tt).lower()
        
        if any(kw in tt_lower for kw in ['knowledge', 'skill', 'technical', 'job-specific', 'coding']):
            categories.add('K')
        
        if any(kw in tt_lower for kw in ['personality', 'behavior', 'behavioural']):
            categories.add('P')
        
        if any(kw in tt_lower for kw in ['cognitive', 'ability', 'aptitude', 'reasoning', 'intelligence']):
            categories.add('C')
    
    return list(categories) if categories else ['U']

def parse_duration(duration_str):
    """
    Parse duration string to minutes
    
    Examples:
    - "40 minutes" → 40
    - "1 hour" → 60
    - "30-40 minutes" → 40 (max)
    """
    if pd.isna(duration_str):
        return None
    
    duration_str = str(duration_str).lower()
    
    # Extract numbers
    minutes_match = re.search(r'(\d+)\s*minute', duration_str)
    if minutes_match:
        return int(minutes_match.group(1))
    
    hours_match = re.search(r'(\d+\.?\d*)\s*hour', duration_str)
    if hours_match:
        return int(float(hours_match.group(1)) * 60)
    
    # Range pattern (take max)
    range_match = re.search(r'(\d+)\s*-\s*(\d+)', duration_str)
    if range_match:
        return int(range_match.group(2))
    
    # Just a number
    number_match = re.search(r'(\d+)', duration_str)
    if number_match:
        return int(number_match.group(1))
    
    return None

# ============================================
# Vector Search with FAISS
# ============================================

def load_resources():
    """Load assessment data, embeddings, and FAISS index"""
    from config import Config
    
    df = pd.read_csv(Config.PROCESSED_CSV)
    embeddings = np.load(Config.EMBEDDINGS_FILE)
    index = faiss.read_index(Config.FAISS_INDEX)
    
    # Pre-process duration column
    if 'length' in df.columns:
        df['duration_minutes'] = df['length'].apply(parse_duration)
    
    return df, embeddings, index

def query_to_embedding(query_text: str, normalize_vec=True):
    """Convert query to normalized embedding"""
    q = str(query_text).strip()
    q_emb = model.encode([q], convert_to_numpy=True, show_progress_bar=False)
    if normalize_vec:
        q_emb = normalize(q_emb, axis=1)
    return q_emb.astype(np.float32)

def vector_search(
    query_text: str,
    index,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 30,
    filters: Dict = None
) -> List[Dict]:
    """
    Vector search with optional filtering
    
    Args:
        query_text: search query
        index: FAISS index
        df: assessments dataframe
        embeddings: numpy embeddings array
        top_k: number of candidates to retrieve
        filters: dict with 'skills', 'duration_minutes', 'job_level'
    
    Returns:
        list of candidate assessments
    """
    # Step 1: Vector search
    q_emb = query_to_embedding(query_text)
    distances, ids = index.search(q_emb, min(top_k * 3, len(df)))  # Get extra for filtering
    
    distances = distances.flatten().tolist()
    ids = ids.flatten().tolist()
    
    # Step 2: Build candidates
    candidates = []
    for idx, score in zip(ids, distances):
        if idx < 0 or idx >= len(df):
            continue
        
        row = df.iloc[idx]
        
        test_types = parse_test_type(row.get("test_type", ""))
        categories = categorize_test_type(test_types)
        
        candidate = {
            "name": row.get("name", "N/A"),
            "url": row.get("url", "N/A"),
            "score": float(score),
            "idx": int(idx),
            "test_types": test_types,
            "categories": categories,
            "description": row.get("description", ""),
            "job_level": row.get("Job level", row.get("Job_level", "N/A")),
            "length": row.get("length", "N/A"),
            "duration_minutes": row.get("duration_minutes")
        }
        
        candidates.append(candidate)
    
    # Step 3: Apply filters if provided
    if filters:
        filtered_candidates = []
        
        for candidate in candidates:
            # Filter by duration
            if filters.get('duration_minutes'):
                target_duration = filters['duration_minutes']
                candidate_duration = candidate.get('duration_minutes')
                
                if candidate_duration:
                    # Allow ±10 minutes tolerance
                    if abs(candidate_duration - target_duration) > 10:
                        continue
            
            # Filter by job level (optional - can be relaxed)
            # if filters.get('job_level'):
            #     if candidate['job_level'] != filters['job_level']:
            #         continue
            
            # Filter by skills (check if mentioned in name or description)
            if filters.get('skills'):
                text_to_check = (candidate['name'] + ' ' + candidate['description']).lower()
                skill_match = any(skill.lower() in text_to_check for skill in filters['skills'])
                
                # If skills mentioned, prioritize but don't exclude others
                if skill_match:
                    candidate['score'] += 0.1  # Boost score for skill match
            
            filtered_candidates.append(candidate)
        
        candidates = filtered_candidates
    
    # Sort by score and return top_k
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_k]

# ============================================
# LLM Re-ranking with Perplexity
# ============================================

def call_perplexity_api(messages: List[Dict], temperature=0.2, max_tokens=3000):
    """Call Perplexity API"""
    if not PERPLEXITY_API_KEY:
        print("⚠️ PERPLEXITY_API_KEY not set - skipping LLM re-ranking")
        return None
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "return_citations": False,
        "return_images": False
    }
    
    try:
        response = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Perplexity API Error: {e}")
        return None

def llm_rerank(
    query_text: str,
    candidates: List[Dict],
    extracted_entities: Dict,
    max_results: int = 10
) -> List[Dict]:
    """
    Use LLM to re-rank candidates based on query context
    
    UPDATED: Returns candidates with ranking scores, not final JSON
    The API endpoint will format the final response
    """
    # Prepare candidate summary for LLM
    candidates_text = []
    for i, cand in enumerate(candidates[:20], 1):  # Limit to top 20 for context
        text = f"{i}. {cand['name']}\n"
        text += f"   URL: {cand['url']}\n"
        text += f"   Categories: {', '.join(cand['categories'])}\n"
        text += f"   Test Types: {', '.join(cand['test_types'])}\n"
        text += f"   Duration: {cand['length']}\n"
        text += f"   Job Level: {cand['job_level']}\n"
        text += f"   Semantic Score: {cand['score']:.3f}\n"
        text += f"   Description: {cand['description'][:150]}...\n"
        candidates_text.append(text)
    
    candidates_summary = "\n".join(candidates_text)
    
    # Build prompt - CRITICAL: Must specify output format exactly
    system_prompt = """You are an expert HR assessment consultant for SHL assessments.

Your task: Given a hiring query and a list of pre-ranked candidate assessments, select the BEST 5-10 assessments that match the requirements.

CRITICAL SELECTION RULES:
1. **Duration Matching**: If duration is specified in extracted entities, ONLY select assessments with matching duration (±5 minutes tolerance)
2. **Skill Matching**: Prioritize assessments that test the mentioned skills (e.g., Java, Python, SQL)
3. **Balance Test Types**: For mixed queries (technical + soft skills), ensure a balanced mix:
   - Knowledge & Skills (K) assessments for technical abilities
   - Personality & Behavior (P) assessments for soft skills
   - Cognitive (C) assessments for analytical abilities
4. **Relevance**: Consider semantic score but don't rely on it alone
5. **Count**: Select between 5-10 assessments (prefer 10 if available)

RESPONSE FORMAT - You MUST respond in this exact JSON format:
{
    "selected_assessment_numbers": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    "reasoning": {
        "1": "Reason why assessment 1 was selected",
        "3": "Reason why assessment 3 was selected",
        ...
    },
    "balance_note": "Explanation of how test types are balanced (e.g., 5 technical + 5 behavioral)"
}

IMPORTANT:
- "selected_assessment_numbers" should be a list of integers (the assessment numbers from the candidate list)
- Only include assessment numbers that exist in the provided list (1-20)
- Select between 5-10 assessments
- Provide reasoning for each selected assessment"""

    # Add extracted entity context
    entity_context = ""
    if extracted_entities['skills']:
        entity_context += f"\n\n**EXTRACTED SKILLS**: {', '.join(extracted_entities['skills'])}"
        entity_context += "\n→ PRIORITIZE assessments that test these specific skills"
    
    if extracted_entities['duration_minutes']:
        entity_context += f"\n\n**REQUIRED DURATION**: {extracted_entities['duration_minutes']} minutes"
        entity_context += f"\n→ ONLY select assessments with duration between {extracted_entities['duration_minutes']-5} and {extracted_entities['duration_minutes']+5} minutes"
    
    if extracted_entities['job_level']:
        entity_context += f"\n\n**JOB LEVEL**: {extracted_entities['job_level']}"
        entity_context += "\n→ Consider job level when selecting assessments"

    user_prompt = f"""HIRING QUERY:
{query_text}
{entity_context}

CANDIDATE ASSESSMENTS (pre-ranked by semantic similarity):
{candidates_summary}

Based on the hiring query and extracted requirements, select the best 5-10 assessments from the list above.

Remember:
- Use ONLY assessment numbers from the list (1-20)
- If duration is specified ({extracted_entities['duration_minutes']} minutes), ONLY select matching duration assessments
- Balance test types if query requires multiple skill domains
- Provide clear reasoning for each selection

Respond in the exact JSON format specified."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Call API
    print("   🤖 Calling Perplexity for LLM re-ranking...")
    response = call_perplexity_api(messages, temperature=0.1, max_tokens=2000)
    
    if not response or 'choices' not in response:
        print("   ⚠️ LLM re-ranking failed, using vector search results")
        return candidates[:max_results]
    
    content = response['choices'][0]['message']['content']
    
    # Parse response
    try:
        # Extract JSON from markdown code blocks
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content.strip()
        
        llm_result = json.loads(json_str)
        
        # Validate response structure
        if 'selected_assessment_numbers' not in llm_result:
            print("   ⚠️ Invalid LLM response structure, using fallback")
            return candidates[:max_results]
        
        selected_numbers = llm_result['selected_assessment_numbers']
        reasoning = llm_result.get('reasoning', {})
        
        # Map selections back to candidates
        final_results = []
        for assess_num in selected_numbers:
            if 1 <= assess_num <= len(candidates):
                candidate = candidates[assess_num - 1].copy()
                
                # Add LLM reasoning if available
                reason_key = str(assess_num)
                if reason_key in reasoning:
                    candidate['llm_reason'] = reasoning[reason_key]
                else:
                    candidate['llm_reason'] = "Selected by AI as highly relevant"
                
                candidate['llm_confidence'] = 0.9  # Default confidence
                final_results.append(candidate)
        
        # Ensure we have enough results
        if len(final_results) < 5:
            print(f"   ⚠️ LLM only selected {len(final_results)}, adding top vector results")
            for candidate in candidates:
                if candidate['url'] not in [r['url'] for r in final_results]:
                    final_results.append(candidate)
                    if len(final_results) >= max_results:
                        break
        
        print(f"   ✅ LLM re-ranked {len(final_results)} assessments")
        return final_results[:max_results]
        
    except json.JSONDecodeError as e:
        print(f"   ⚠️ Failed to parse LLM response: {e}")
        print(f"   Raw response: {content[:200]}...")
        return candidates[:max_results]
    except Exception as e:
        print(f"   ⚠️ Error processing LLM response: {e}")
        return candidates[:max_results]

# ============================================
# Complete RAG Pipeline
# ============================================

def rag_search(
    query_text: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    index,
    min_results: int = 5,
    max_results: int = 10,
    use_llm_rerank: bool = True
) -> Dict:
    """
    Complete RAG pipeline: Extract → Retrieve → Re-rank
    
    Pipeline:
    1. Extract entities (skills, duration, job level)
    2. Vector search with filters (retrieve top candidates)
    3. LLM re-ranking (final selection)
    
    Args:
        query_text: user query
        df: assessments dataframe
        embeddings: numpy embeddings
        index: FAISS index
        min_results: minimum recommendations
        max_results: maximum recommendations
        use_llm_rerank: whether to use LLM for re-ranking
    
    Returns:
        dict with recommendations and metadata
    """
    print(f"\n{'='*80}")
    print(f"RAG SEARCH PIPELINE")
    print(f"{'='*80}")
    
    # Step 1: Entity Extraction
    print("\n🔍 Step 1: Entity Extraction")
    entities = extract_entities(query_text)
    print(f"   Skills: {entities['skills']}")
    print(f"   Duration: {entities['duration_minutes']} minutes" if entities['duration_minutes'] else "   Duration: Not specified")
    print(f"   Job Level: {entities['job_level']}" if entities['job_level'] else "   Job Level: Not specified")
    
    # Step 2: Vector Search with Filters
    print("\n📊 Step 2: Vector Search (Retrieval)")
    candidates = vector_search(
        query_text=query_text,
        index=index,
        df=df,
        embeddings=embeddings,
        top_k=30,
        filters=entities
    )
    print(f"   Retrieved {len(candidates)} candidates from vector search")
    
    # Step 3: LLM Re-ranking
    if use_llm_rerank and PERPLEXITY_API_KEY:
        print("\n🧠 Step 3: LLM Re-ranking (Generation)")
        final_results = llm_rerank(query_text, candidates, entities, max_results)
    else:
        print("\n⚠️ Step 3: Skipping LLM re-ranking")
        final_results = candidates[:max_results]
    
    # Analyze category distribution
    category_dist = Counter()
    for rec in final_results:
        for cat in rec['categories']:
            category_dist[cat] += 1
    
    return {
        'query': query_text,
        'method': 'rag_vector_llm' if use_llm_rerank else 'rag_vector_only',
        'extracted_entities': entities,
        'recommendations': final_results,
        'count': len(final_results),
        'category_distribution': dict(category_dist),
        'candidates_retrieved': len(candidates)
    }

# ============================================
# Display Functions
# ============================================

def print_rag_results(result: Dict):
    """Pretty print RAG results"""
    print(f"\n{'='*80}")
    print(f"RAG SEARCH RESULTS")
    print(f"{'='*80}")
    print(f"Query: {result['query']}")
    print(f"Method: {result['method']}")
    
    print(f"\n🔍 Extracted Entities:")
    entities = result['extracted_entities']
    if entities['skills']:
        print(f"   Skills: {', '.join(entities['skills'])}")
    if entities['duration_minutes']:
        print(f"   Duration: {entities['duration_minutes']} minutes")
    if entities['job_level']:
        print(f"   Job Level: {entities['job_level']}")
    
    print(f"\n📊 Search Stats:")
    print(f"   Candidates Retrieved: {result['candidates_retrieved']}")
    print(f"   Final Recommendations: {result['count']}")
    
    print(f"\n📈 Category Distribution:")
    cat_names = {'K': 'Knowledge & Skills', 'P': 'Personality & Behavior', 'C': 'Cognitive', 'U': 'Unknown'}
    for cat, count in result['category_distribution'].items():
        print(f"   {cat_names.get(cat, cat)}: {count}")
    
    print(f"\n📋 Recommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"\n{i}. {rec['name']}")
        print(f"   Vector Score: {rec['score']:.4f}")
        print(f"   Categories: {', '.join(rec['categories'])}")
        print(f"   Duration: {rec['length']}")
        print(f"   Job Level: {rec['job_level']}")
        
        if 'llm_reason' in rec and rec['llm_reason']:
            print(f"   🤖 LLM Reason: {rec['llm_reason']}")
        if 'llm_confidence' in rec:
            print(f"   Confidence: {rec['llm_confidence']:.2f}")
        
        print(f"   URL: {rec['url'][:70]}...")

# ============================================
# Format Output for API Specification
# ============================================

def format_recommendations_for_api(recommendations: List[Dict], catalog_df: pd.DataFrame) -> List[Dict]:
    """
    Format recommendations to match the API specification exactly
    
    Required output format per the assignment:
    {
        "url": "string",
        "name": "string",
        "adaptive_support": "Yes" or "No",
        "description": "string",
        "duration": integer (minutes),
        "remote_support": "Yes" or "No",
        "test_type": ["list", "of", "strings"]
    }
    """
    formatted_results = []
    
    for rec in recommendations:
        # Get full row from catalog for complete data
        try:
            # Find by URL (most reliable)
            matching_rows = catalog_df[catalog_df['url'] == rec['url']]
            
            if len(matching_rows) == 0:
                # Fallback: try finding by name
                matching_rows = catalog_df[catalog_df['name'].str.lower() == rec['name'].lower()]
            
            if len(matching_rows) > 0:
                row = matching_rows.iloc[0]
            else:
                print(f"   ⚠️ Warning: Could not find catalog entry for {rec['name']}")
                row = None
        except Exception as e:
            print(f"   ⚠️ Error finding catalog entry: {e}")
            row = None
        
        # ============================================
        # Extract and format each field
        # ============================================
        
        # 1. URL (required, string)
        url = rec.get('url', '')
        
        # 2. Name (required, string)
        name = rec.get('name', 'Unknown Assessment')
        
        # 3. Adaptive Support (required, "Yes" or "No")
        adaptive_support = "No"  # Default
        if row is not None and 'adaptive_support' in catalog_df.columns:
            adaptive_val = str(row.get('adaptive_support', '')).strip().lower()
            if adaptive_val in ['yes', 'true', '1', 'y']:
                adaptive_support = "Yes"
        
        # 4. Description (required, string)
        description = ""
        if row is not None and 'description' in catalog_df.columns:
            description = str(row.get('description', ''))
        elif 'description' in rec:
            description = str(rec.get('description', ''))
        
        # Truncate description if too long (keep it reasonable for API response)
        if len(description) > 250:
            description = description[:247] + "..."
        
        # 5. Duration (required, integer in minutes)
        duration = None
        
        # Try multiple sources for duration
        if row is not None and 'duration_minutes' in catalog_df.columns:
            duration_val = row.get('duration_minutes')
            if pd.notna(duration_val):
                try:
                    duration = int(float(duration_val))
                except (ValueError, TypeError):
                    pass
        
        # Fallback: try from recommendation
        if duration is None and rec.get('duration_minutes'):
            try:
                duration = int(rec['duration_minutes'])
            except (ValueError, TypeError):
                pass
        
        # Last resort: parse from length string
        if duration is None:
            length_str = ''
            if row is not None and 'length' in catalog_df.columns:
                length_str = str(row.get('length', ''))
            elif 'length' in rec:
                length_str = str(rec.get('length', ''))
            
            if length_str:
                duration = parse_duration(length_str)
        
        # 6. Remote Support (required, "Yes" or "No")
        remote_support = "No"  # Default
        if row is not None and 'remote_support' in catalog_df.columns:
            remote_val = str(row.get('remote_support', '')).strip().lower()
            if remote_val in ['yes', 'true', '1', 'y']:
                remote_support = "Yes"
        
        # 7. Test Type (required, list of strings)
        test_type = []
        
        if row is not None and 'test_type' in catalog_df.columns:
            test_type_val = row.get('test_type')
            if pd.notna(test_type_val):
                test_type = parse_test_type(test_type_val)
        
        # Fallback to recommendation data
        if not test_type and 'test_types' in rec:
            test_type = rec['test_types']
        
        # Ensure test_type is a list of strings
        if not isinstance(test_type, list):
            test_type = [str(test_type)] if test_type else []
        
        # Clean up test_type - remove empty strings
        test_type = [str(t).strip() for t in test_type if str(t).strip()]
        
        if not test_type:
            test_type = ["General Assessment"]  # Fallback
        
        # ============================================
        # Build final formatted result
        # ============================================
        
        formatted_result = {
            "url": url,
            "name": name,
            "adaptive_support": adaptive_support,
            "description": description,
            "duration": duration,
            "remote_support": remote_support,
            "test_type": test_type
        }
        
        formatted_results.append(formatted_result)
    
    return formatted_results

# ============================================
# Testing
# ============================================

if __name__ == "__main__":
    from config import Config
    
    print("Loading resources...")
    df, embeddings, index = load_resources()
    print(f"✓ Loaded {len(df)} assessments\n")
    
    # Test queries
    test_queries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
        
        "Need Python and SQL skills assessment for mid-level analyst. Duration should be around 1 hour.",
        
        "Senior software engineer with leadership skills"  # No duration specified
    ]
    
    for query in test_queries:
        result = rag_search(query, df, embeddings, index, use_llm_rerank=True)
        print_rag_results(result)
        print("\n" + "="*80 + "\n")