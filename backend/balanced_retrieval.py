# backend/balanced_retrieval.py
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import Counter, defaultdict
import ast
from config import Config

# ============================================
# Load Model
# ============================================
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def load_resources():
    """Load all necessary resources using Config paths"""
    df = pd.read_csv(Config.PROCESSED_CSV)
    embeddings = np.load(Config.EMBEDDINGS_FILE)
    index = faiss.read_index(Config.FAISS_INDEX)
    return df, embeddings, index

# ============================================
# Query Understanding & Classification
# ============================================

def classify_query_intent(query_text):
    """
    Classify if query requires technical, behavioral, or mixed assessments
    Returns: dict with 'technical_score', 'behavioral_score', 'cognitive_score'
    """
    query_lower = query_text.lower()
    
    # Technical/Knowledge keywords
    technical_keywords = [
        'java', 'python', 'sql', 'javascript', 'coding', 'programming', 'developer',
        'engineer', 'technical', 'software', 'data', 'algorithm', 'api', 'database',
        'web', 'mobile', 'cloud', 'devops', 'frontend', 'backend', 'full stack',
        'react', 'angular', 'node', 'c++', 'c#', '.net', 'aws', 'azure', 'html',
        'css', 'typescript', 'kotlin', 'swift', 'ruby', 'php', 'golang', 'rust'
    ]
    
    # Behavioral/Personality keywords
    behavioral_keywords = [
        'teamwork', 'collaboration', 'collaborate', 'communicate', 'communication', 'leadership',
        'interpersonal', 'soft skills', 'personality', 'behavior', 'behaviour', 'attitude',
        'cultural fit', 'work style', 'motivat', 'social', 'emotional intelligence',
        'stakeholder', 'team player', 'adaptability', 'resilience', 'integrity',
        'mentor', 'cross-functional', 'business', 'client', 'customer'
    ]
    
    # Cognitive/Analytical keywords
    cognitive_keywords = [
        'analytical', 'problem solving', 'critical thinking', 'reasoning',
        'cognitive', 'logical', 'aptitude', 'numerical', 'verbal', 'abstract',
        'intelligence', 'mental ability', 'thinking skills', 'decision making',
        'analysis', 'evaluate', 'assess'
    ]
    
    # Count matches
    technical_score = sum(1 for kw in technical_keywords if kw in query_lower)
    behavioral_score = sum(1 for kw in behavioral_keywords if kw in query_lower)
    cognitive_score = sum(1 for kw in cognitive_keywords if kw in query_lower)
    
    return {
        'technical': technical_score,
        'behavioral': behavioral_score,
        'cognitive': cognitive_score,
        'is_mixed': (technical_score > 0 and behavioral_score > 0) or 
                    (technical_score > 0 and cognitive_score > 0) or
                    (behavioral_score > 0 and cognitive_score > 0)
    }

# ============================================
# Test Type Parsing
# ============================================

def parse_test_type(test_type_value):
    """
    Parse test_type column which might be:
    - A list string: "['Knowledge & Skills', 'Personality & Behavior']"
    - A single value: "Knowledge & Skills"
    - Multiple values separated by comma
    
    Returns: list of test type categories
    """
    if pd.isna(test_type_value):
        return []
    
    test_type_str = str(test_type_value).strip()
    
    # Try to parse as Python list
    if test_type_str.startswith('[') and test_type_str.endswith(']'):
        try:
            return ast.literal_eval(test_type_str)
        except:
            pass
    
    # Split by comma if multiple values
    if ',' in test_type_str:
        return [t.strip() for t in test_type_str.split(',')]
    
    # Single value
    return [test_type_str]

def categorize_test_type(test_types):
    """
    Categorize test types into broad categories: K (Knowledge), P (Personality), C (Cognitive)
    
    Args:
        test_types: list of test type strings
    
    Returns:
        list of categories (e.g., ['K', 'P'])
    """
    categories = set()
    
    for tt in test_types:
        tt_lower = str(tt).lower()
        
        # Knowledge & Skills
        if any(kw in tt_lower for kw in ['knowledge', 'skill', 'technical', 'job-specific', 'coding']):
            categories.add('K')
        
        # Personality & Behavior
        if any(kw in tt_lower for kw in ['personality', 'behavior', 'behavioural']):
            categories.add('P')
        
        # Cognitive
        if any(kw in tt_lower for kw in ['cognitive', 'ability', 'aptitude', 'reasoning', 'intelligence']):
            categories.add('C')
    
    return list(categories) if categories else ['U']  # U = Unknown

# ============================================
# Text Chunking for Long JDs
# ============================================

def is_long_text(text, word_threshold=100):
    """Check if text is long enough to benefit from chunking"""
    word_count = len(str(text).split())
    return word_count > word_threshold

def chunk_text(text, chunk_size=150, overlap=50):
    """
    Split text into overlapping chunks
    
    Args:
        text: input text
        chunk_size: number of words per chunk
        overlap: number of overlapping words between chunks
    
    Returns:
        list of text chunks
    """
    words = str(text).split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        # Move start forward, accounting for overlap
        start = end - overlap
        
        # Ensure we don't get stuck in infinite loop
        if start >= len(words) - overlap:
            break
    
    return chunks

def extract_key_sections(text):
    """
    Extract key sections from JD (requirements, skills, qualifications)
    This helps focus on most relevant parts
    """
    text_lower = text.lower()
    
    # Common JD section headers
    section_patterns = [
        r'requirements?:?\s*(.*?)(?=\n\n|\Z)',
        r'qualifications?:?\s*(.*?)(?=\n\n|\Z)',
        r'skills?:?\s*(.*?)(?=\n\n|\Z)',
        r'responsibilities:?\s*(.*?)(?=\n\n|\Z)',
        r'must have:?\s*(.*?)(?=\n\n|\Z)',
        r'nice to have:?\s*(.*?)(?=\n\n|\Z)'
    ]
    
    key_sections = []
    for pattern in section_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            section_text = match.group(1).strip()
            if section_text and len(section_text) > 20:
                key_sections.append(section_text)
    
    # If no sections found, return original text
    return key_sections if key_sections else [text]

# ============================================
# Core Retrieval Functions
# ============================================

def query_to_embedding(query_text, normalize_vec=True):
    """Convert query to normalized embedding"""
    q = str(query_text).strip()
    q_emb = model.encode([q], convert_to_numpy=True, show_progress_bar=False)
    if normalize_vec:
        q_emb = normalize(q_emb, axis=1)
    return q_emb.astype(np.float32)

def search_single_query(query_text, index, df, top_k=50):
    """
    Basic search for a single query
    """
    q_emb = query_to_embedding(query_text)
    distances, ids = index.search(q_emb, top_k)
    
    distances = distances.flatten().tolist()
    ids = ids.flatten().tolist()
    
    results = {}  # Use dict to track max score per item
    
    for idx, score in zip(ids, distances):
        if idx < 0 or idx >= len(df):
            continue
        
        # Keep only the highest score for each item
        if idx not in results or score > results[idx]['score']:
            row = df.iloc[idx]
            
            # Parse test types
            test_type_raw = row.get("test_type", "")
            test_types = parse_test_type(test_type_raw)
            categories = categorize_test_type(test_types)
            
            results[idx] = {
                "name": row.get("name", "N/A"),
                "url": row.get("url", "N/A"),
                "score": float(score),
                "idx": int(idx),
                "test_type_raw": test_type_raw,
                "test_types": test_types,
                "categories": categories,
                "description": row.get("description", ""),
                "job_level": row.get("Job level", row.get("Job_level", "N/A")),
                "length": row.get("length", "N/A")
            }
    
    return list(results.values())

def search_with_chunking(query_text, index, df, embeddings, top_k=50, 
                         chunk_size=150, overlap=50):
    """
    Search using chunked queries for long JDs
    Aggregates scores across chunks using max pooling
    
    Args:
        query_text: long job description
        index: FAISS index
        df: assessments dataframe
        embeddings: numpy array of embeddings
        top_k: number of candidates to retrieve
        chunk_size: words per chunk
        overlap: overlapping words between chunks
    
    Returns:
        list of aggregated results
    """
    # Step 1: Extract key sections (optional but helpful)
    key_sections = extract_key_sections(query_text)
    
    # Step 2: Create chunks
    all_chunks = []
    for section in key_sections:
        section_chunks = chunk_text(section, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(section_chunks)
    
    # Deduplicate chunks
    all_chunks = list(set(all_chunks))
    
    print(f"   📄 Created {len(all_chunks)} chunks from JD")
    
    # Step 3: Search with each chunk and aggregate scores
    aggregated_scores = {}  # idx -> max score
    
    for chunk_idx, chunk in enumerate(all_chunks):
        q_emb = query_to_embedding(chunk)
        
        # Compute similarity with all embeddings
        scores = (embeddings @ q_emb.T).flatten()
        
        # Aggregate: keep max score for each assessment
        for idx, score in enumerate(scores):
            if idx not in aggregated_scores or score > aggregated_scores[idx]:
                aggregated_scores[idx] = float(score)
    
    # Step 4: Sort and get top_k
    sorted_items = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Step 5: Build result objects
    results = []
    for idx, score in sorted_items:
        row = df.iloc[idx]
        
        test_type_raw = row.get("test_type", "")
        test_types = parse_test_type(test_type_raw)
        categories = categorize_test_type(test_types)
        
        results.append({
            "name": row.get("name", "N/A"),
            "url": row.get("url", "N/A"),
            "score": score,
            "idx": int(idx),
            "test_type_raw": test_type_raw,
            "test_types": test_types,
            "categories": categories,
            "description": row.get("description", ""),
            "job_level": row.get("Job level", row.get("Job_level", "N/A")),
            "length": row.get("length", "N/A")
        })
    
    return results

def adaptive_search(query_text, index, df, embeddings, top_k=50, 
                    long_text_threshold=100):
    """
    Automatically choose between single query search and chunked search
    based on text length
    
    Args:
        query_text: user query or JD
        index: FAISS index
        df: assessments dataframe
        embeddings: numpy array
        top_k: candidates to retrieve
        long_text_threshold: word count threshold for chunking
    
    Returns:
        tuple: (list of search results, bool used_chunking)
    """
    if is_long_text(query_text, word_threshold=long_text_threshold):
        print(f"📄 Long JD detected ({len(query_text.split())} words) - using chunked search")
        results = search_with_chunking(query_text, index, df, embeddings, top_k=top_k)
        return results, True
    else:
        print(f"📝 Short query ({len(query_text.split())} words) - using standard search")
        results = search_single_query(query_text, index, df, top_k=top_k)
        return results, False

# ============================================
# Balanced Recommendation Logic
# ============================================

def balance_recommendations(results, query_intent, target_count=10, min_per_category=3):
    """
    Balance recommendations across test type categories based on query intent
    
    Args:
        results: list of search results with 'categories' field
        query_intent: dict from classify_query_intent()
        target_count: total number of recommendations to return (default: 10)
        min_per_category: minimum items per category for mixed queries (default: 3)
    
    Returns:
        list of balanced results
    """
    
    # Group results by category
    category_results = defaultdict(list)
    for r in results:
        for cat in r['categories']:
            category_results[cat].append(r)
    
    # If not a mixed query, return top results as-is
    if not query_intent['is_mixed']:
        return results[:target_count]
    
    # Determine which categories are needed based on query intent
    needed_categories = []
    if query_intent['technical'] > 0:
        needed_categories.append('K')
    if query_intent['behavioral'] > 0:
        needed_categories.append('P')
    if query_intent['cognitive'] > 0:
        needed_categories.append('C')
    
    # If we couldn't detect categories from keywords, use what we have in results
    if not needed_categories:
        needed_categories = list(category_results.keys())
    
    # Calculate how many items to take from each category
    num_categories = len(needed_categories)
    if num_categories == 0:
        return results[:target_count]
    
    items_per_category = max(min_per_category, target_count // num_categories)
    
    # Collect balanced results
    balanced = []
    seen_urls = set()
    
    # First pass: Get min_per_category from each needed category
    for cat in needed_categories:
        cat_results = category_results.get(cat, [])
        count = 0
        for r in cat_results:
            if r['url'] not in seen_urls and count < items_per_category:
                balanced.append(r)
                seen_urls.add(r['url'])
                count += 1
    
    # Second pass: Fill remaining slots with highest scoring items
    remaining_slots = target_count - len(balanced)
    if remaining_slots > 0:
        for r in results:
            if r['url'] not in seen_urls:
                balanced.append(r)
                seen_urls.add(r['url'])
                if len(balanced) >= target_count:
                    break
    
    # Sort by score to maintain quality
    balanced.sort(key=lambda x: x['score'], reverse=True)
    
    return balanced[:target_count]