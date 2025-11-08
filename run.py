import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app import app

if __name__ == '__main__':
    print("="*60)
    print("🚀 SHL Assessment Recommendation System")
    print("="*60)
    print("Backend API: http://localhost:5000")
    print("Frontend: Open frontend/index.html in browser")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)