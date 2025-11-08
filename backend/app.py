from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import traceback

# Add backend to path
sys.path.append(os.path.dirname(__file__))

from rag_system import rag_search, load_resources, extract_entities
from balanced_retrieval import load_resources, adaptive_search, classify_query_intent, balance_recommendations
from config import Config
from flask import Flask, request, jsonify, Response
import json
# and import sanitize_for_json from perplexity_integration
from perplexity_integration import hybrid_recommendation_system, sanitize_for_json

from rag_system import (
    rag_search, 
    load_resources, 
    extract_entities,
    format_recommendations_for_api
)
from config import Config

app = Flask(__name__)
CORS(app)

# Load resources at startup
print("🚀 Loading RAG system...")
df, embeddings, index = load_resources()
print(f"✅ Loaded {len(df)} assessments")

# ============================================
# API v1 Endpoints (As per specification)
# ============================================

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """
    Health Check Endpoint
    
    Method: GET
    Endpoint: /api/v1/health
    
    Response:
    {
        "status": "healthy"
    }
    """
    return jsonify({
        "status": "healthy"
    }), 200

@app.route('/api/v1/recommend', methods=['POST'])
def recommend():
    """
    Assessment Recommendation Endpoint
    
    Method: POST
    Endpoint: /api/v1/recommend
    Content-Type: application/json
    
    Request Body:
    {
        "query": "string"
    }
    
    Response:
    {
        "recommended_assessments": [
            {
                "url": "string",
                "name": "string",
                "adaptive_support": "string",
                "description": "string",
                "duration": integer,
                "remote_support": "string",
                "test_type": ["list of string"]
            }
        ]
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json"
            }), 400
        
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing required field 'query' in request body"
            }), 400
        
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return jsonify({
                "error": "Query cannot be empty"
            }), 400
        
        print(f"\n📥 Received query: {query_text[:100]}...")
        
        # Run RAG search
        result = rag_search(
            query_text=query_text,
            df=df,
            embeddings=embeddings,
            index=index,
            min_results=1,
            max_results=10,
            use_llm_rerank=True
        )
        
        # Format recommendations according to API specification
        formatted_recommendations = format_recommendations_for_api(
            result['recommendations'],
            df
        )
        
        # Ensure we return at least 1 and at most 10 assessments
        formatted_recommendations = formatted_recommendations[:10]
        
        if len(formatted_recommendations) == 0:
            # If no results, return at least 1 (fallback to top scoring)
            print("⚠️ No recommendations found, using fallback")
            # This shouldn't happen with RAG, but safety check
            return jsonify({
                "recommended_assessments": []
            }), 200
        
        response = {
            "recommended_assessments": formatted_recommendations
        }
        
        print(f"✅ Returning {len(formatted_recommendations)} recommendations")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        
        # Return 500 with error details (helpful for debugging)
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

# ============================================
# Additional Endpoints (Not required but useful)
# ============================================

@app.route('/api/v1/extract', methods=['POST'])
def extract():
    """
    Entity Extraction Endpoint (Optional - for testing)
    
    Extracts entities like skills, duration, job level from query
    """
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        entities = extract_entities(query)
        
        return jsonify(entities), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/stats', methods=['GET'])
def stats():
    """
    System Statistics Endpoint (Optional - for monitoring)
    """
    try:
        return jsonify({
            "total_assessments": len(df),
            "embedding_dimension": embeddings.shape[1],
            "perplexity_enabled": bool(os.getenv("PERPLEXITY_API_KEY"))
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================
# Error Handlers
# ============================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "message": "Please check the API documentation for valid endpoints"
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "error": "Method not allowed",
        "message": "Please check the HTTP method (GET/POST) for this endpoint"
    }), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "error": "Internal server error",
        "message": str(e)
    }), 500

# ============================================
# Run Application
# ============================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("SHL ASSESSMENT RAG API")
    print("="*80)
    print("\nAPI Endpoints:")
    print("  GET  /api/v1/health      - Health check")
    print("  POST /api/v1/recommend   - Get recommendations")
    print("  POST /api/v1/extract     - Extract entities (optional)")
    print("  GET  /api/v1/stats       - System stats (optional)")
    print("\nAPI is running on: http://0.0.0.0:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)